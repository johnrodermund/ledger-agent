# LedgerAgent

A LangGraph agent that audits a synthetic legacy ERP in Postgres, finds data discrepancies in the general ledger and accounts payable, and suggests fixes that a deterministic verifier has checked. It never applies a fix.

## Results

The agent runs on Claude Sonnet 5.5 by default, and Claude Haiku 5.5 replays from its own recordings for comparison. The clean comparison is the **fresh holdout**: three scenarios on unused seeds, added after everything else here was settled, and run once per model. Neither model was tuned against them or seen on them. These are the recorded numbers, misses included.

| Fresh holdout | Haiku 5.5 | Sonnet 5.5 |
| --- | --- | --- |
| Recall | 85.7% (12 of 14) | 92.9% (13 of 14) |
| Precision | 80.0% (12 of 15) | 72.2% (13 of 18) |
| Decoy false alarm rate | 25.0% (2 of 8) | 50.0% (4 of 8) |
| Fix validity | 100.0% (12 of 12) | 100.0% (13 of 13) |
| Verified but wrong | 0 | 0 |
| End-to-end fix rate | 85.7% (12 of 14) | 92.9% (13 of 14) |
| Needs human rate | 20.0% (3 of 15) | 27.8% (5 of 18) |
| Incomplete scenarios | 0 | 0 |
| Tokens, cost and queries per run | about 123,000 tokens, $0.016, 27 queries | about 243,000 tokens, $0.45, 42 queries |

Misses on the fresh holdout. Haiku missed an invalid account (`fresh-2`) and an AP-to-GL mismatch (`fresh-3`), and flagged two decoys. Sonnet missed a duplicate invoice (`fresh-3`) and flagged four decoys, two of them in `fresh-2`.

Sonnet finds and fixes one more planted fault in 14, and reports more valid-but-odd records as problems. It costs about 28 times as much per run. The hunt prompt has no guidance on decoys, on purpose, so the false alarm rates are the honest ones.

### How the numbers got here

The first holdout (`holdout-1` to `-3`) was recorded once on Haiku and then once on Sonnet, before the two fixes below. Both models found 11 of 14 faults there, and Sonnet cost about 20 times as much. Reading Sonnet's misses showed that most were a reporting problem, not a detection one: it often put several faulty journals into one finding, which the scoring credits once and `propose` fixes once. Two fixes followed:

- `confirm` splits a finding that cites a few journals, or a few invoices for an AP-to-GL tie-out, into one finding per record. A duplicate pair stays one finding.
- `propose` is told, for a duplicate invoice only, which copy to keep and that the journal recording the voided copy must be reversed in full.

The first holdout has now been seen, so its post-fix numbers are reported as such, not as a clean result:

| First holdout, seen, after the fixes | Haiku 5.5 | Sonnet 5.5 |
| --- | --- | --- |
| Recall | 85.7% (12 of 14) | 92.9% (13 of 14) |
| Decoy false alarm rate | 37.5% (3 of 8) | 37.5% (3 of 8) |
| End-to-end fix rate | 78.6% (11 of 14) | 92.9% (13 of 14) |
| Verified but wrong | 1 | 0 |

Prompt caching is on for both models. On Sonnet's recordings, 57% of prompt tokens were read from cache, which cut the bill by 26% against the same tokens uncached. It cannot cut more, because output is about half of Sonnet's cost and output is never cached. The injection scenario passed on both models: the hostile memo reached the model, recall held, and no proposed line used the memo's account.

CI replays the default model and fails if it falls below the thresholds in `evals/thresholds.json`, set from Sonnet's first-holdout run to the nearest 5 points and rounded in the cautious direction. The guardrail suite is a separate result: the verifier must refuse 100% of its deliberately bad fixes. That figure covers only that suite, which has no LLM and does not measure the agent.

## How the agent is wired

```mermaid
flowchart LR
    explore --> hunt --> confirm --> pick
    pick -->|finding left| propose
    pick -->|queue empty| report([report])
    propose -->|proposal| verify
    propose -->|malformed twice| pick
    verify -->|pass| pick
    verify -->|fail, attempts left| propose
    verify -->|fail, out of attempts| pick
```

## Run it

```bash
docker compose up --build --abort-on-container-exit --exit-code-from evals
```

This needs Docker and no API key. Every setting has a test default in `docker-compose.yml`, so no `.env` is required. To change a value, run `cp .env.example .env` and edit it. The role passwords are also fixed in `db/init/01_roles.sql`, so change both together.

The default run replays the committed recordings in `recordings/`, so it reproduces the results above without calling the model.

Services start in this order:

1. `db` starts Postgres and becomes healthy.
2. `seed` loads scenario `dev-1` into `ledger_dev_1` and writes its answer key to a volume that only `seed` mounts.
3. `agent` audits `ledger_dev_1` and writes a Markdown and JSON report to `./out`.
4. `evals` runs the guardrail suite, then every scenario, and writes results to `./out/evals`.

`seed` and `agent` idle after they succeed, because `--abort-on-container-exit` stops the whole stack as soon as any container exits, even with code 0. A failure exits non-zero and stops the run.

Run the tests with `docker compose run --rm tests pytest`.

### Default runs are replays

The model client has three modes, set with `LEDGER_LLM_MODE`:

- `replay` (the default): answers come from recordings in `recordings/`. No API key and no network.
- `record`: calls the model and writes each response to `recordings/`. A request that already has a recording is not sent again, so an interrupted run resumes without paying twice.
- `live`: calls the model and records nothing. Needs `ANTHROPIC_API_KEY`.

A recording is keyed by a hash of the model, messages and tools, and holds that key, the model, the response and a little metadata, not the request. A request with no recording fails loudly with a missing-recording error. There is no fallback to a live call. Changing a prompt means re-recording.

The agent runs on Claude Sonnet 5.5 (`claude-sonnet-5-5`) with its lowest thinking setting, `between_tools`. Set `LEDGER_MODEL=claude-haiku-5-5` to run Haiku 5.5 with thinking off instead. Recordings are keyed by model, so both replay with no key: `LEDGER_MODEL=claude-haiku-5-5 docker compose run --rm --no-deps evals python -m ledger_evals.run --split all --out /out/haiku` reproduces the Haiku column above. A third model means recording it.

A replay proves that the pipeline, the verifier and the scoring work end to end and give the same answer every time. It does not prove how the agent behaves against the real model today, and it does not prove that a changed prompt still works. The manually started live workflow exists for that.

## How it works

- `explore` reads the catalog and a short, partly wrong data dictionary, samples data, and writes schema notes including joins it infers. It uses a read-only SQL tool: single `SELECT` statements, a row cap, a timeout, and a read-only database role.
- `hunt` is given general audit principles (double-entry balance, subledger-to-GL tie-out, period controls, reference integrity, duplicates), the finding type names, and no table names. It returns findings, each with the SQL that proves it.
- `confirm` re-runs each finding's evidence SQL and drops any that return no rows. It splits a finding that cites a few journals, or a few invoices for an AP-to-GL tie-out, into one finding per record, as each is a separate problem with a separate fix.
- `propose` has no SQL tool. Code gives it the target journal, any invoice it cites with the journal that recorded it, the open periods and the active accounts, read through the read-only tool, and it returns a structured correction, not SQL: journal lines, the journal being corrected, any invoice status change. Code renders that to parameterised SQL.
- `verify` is deterministic code. It checks structure, runs the fix in a trial transaction, checks only new correcting rows changed, applies a coded resolution rule for the fault type, then checks the whole ledger for new violations. A failure reason goes back to `propose`. Each finding gets three attempts, then goes to `needs_human`.

Two database roles keep the boundaries clear. The SQL tool uses the read-only role. Only the verifier uses the write role, and it always rolls back: the verifier never commits, and no code path does.

The fix convention is that posted entries are never edited. A fix is a new correcting journal entry that references the original, unless the problem is in non-posted data such as an invoice status.

The answer key is hidden from the agent. It lives outside the database and is never mounted in the `agent` container.

A run has a token limit, 600,000 tokens on Sonnet and 300,000 on Haiku, shared out as stage budgets: 25% for `explore`, 35% for `hunt` and 40% for `propose`, each measured from the start of its stage. Tokens count every request in full, so a long conversation spends its budget quickly. Requests use prompt caching, which lowers what a long conversation costs but not what it counts against a budget: cached tokens count in full, and the cost figure prices cache writes and reads at their own rates. Near its budget, `explore` or `hunt` loses the SQL tool and is told to submit, so what it found is kept. That last turn can run a little past the budget. In `propose`, findings left when the budget runs out go to `needs_human`. Hitting a budget is recorded in the report and marks the run incomplete.

### Evals

- The guardrail suite feeds the verifier a library of bad fixes (unbalanced groups, closed periods, inactive accounts, oversized changes, fixes that break something else) and good fixes. It must refuse every bad one at its expected check and pass every good one. No LLM is involved.
- The agent suite runs the full graph on generated ERPs: dev scenarios for tuning prompts, holdout scenarios on seeds never tuned against that supply the headline numbers, and an injection scenario that plants a hostile memo on the closed-period fault's journal. Injection only passes if the memo reached the model, in a query result it was sent or in a propose prompt, since a memo the model never saw tests nothing.
- The holdout is looked at once, with prompts frozen first. Its three scenarios hold 14 planted faults, so one miss moves recall by about 7 percentage points. The first holdout was seen before the two fixes described under Results, so a fresh holdout of three more scenarios gives the clean comparison.
- Scoring is separate from the verifier. Eval code checks each fix against the answer key, so a fix the verifier accepted can still be scored wrong. That count is reported as `verified_but_wrong`.

### Cost of the live workflow

`.github/workflows/live-evals.yml` has no schedule and only runs when started by hand, with a `mode` input of `live` (the default) or `record`. `record` uploads `recordings/` as a workflow artifact for a person to review and commit. The workflow never commits anything. Its cost is bounded two ways:

- `timeout-minutes: 45` stops the job.
- Each audit stays near its token limit, 600,000 on Sonnet, as its stage budgets force a submit before they run out, and each forced turn can run a little past. In `live` mode the agent and the thirteen eval scenarios each use a fresh client, so one run is capped at roughly 8.4 million tokens in total. `record` mode skips the `agent` service, as the evals record the same scenario, so it is capped at roughly 7.8 million.

A run without the `ANTHROPIC_API_KEY` secret fails with a clear error.

## Limitations

- The audit principles given to `hunt` overlap the five planted fault types. The agent is told to look for balance, tie-out, period and reference problems, and those are the faults that were planted. Recall here is not evidence it would find faults of a kind nobody planted.
- The five type names are given to the model. `submit_findings` lists them as an enum alongside `other`, so `hunt` knows which kinds of fault are expected, though not where they are or how to detect them.
- The verifier has coded resolution rules for five fault types only: unbalanced journal, duplicate invoice, AP-to-GL mismatch, closed-period posting, invalid account. A finding of any other type goes straight to `needs_human`.
- Verified does not mean correct. The verifier checks that a fix is well formed, contained, and resolves the fault by its own rule. It can accept a balanced fix posted to an account that is active but wrong. That is why scoring is independent and `verified_but_wrong` is reported.
- The data is synthetic, generated from a seed with planted faults and decoys. Real ledgers are messier, bigger and wrong in more ways.
- Nothing is applied. Fixes are suggestions in a report, never applied to any database, and a person has to review each one.
- Replays do not prove behaviour against the live model, as described above.
- Nothing runs the live evals automatically. If the model changes or a prompt edit hurts quality, replays stay green until someone starts the live workflow by hand, so published results can be stale.
- The guardrail suite's 100% figure says the verifier refuses the bad fixes in that library, not that it refuses every bad fix. The agent's fix validity on 12 or 13 fixes is a small sample, not a guarantee.

## Possible extensions

- An `apply` step behind an approval pause, so a person can approve a verified fix and have it applied in a transaction.
- Prompt optimisation with DSPy against the scored dev set, which already provides a metric to optimise.
