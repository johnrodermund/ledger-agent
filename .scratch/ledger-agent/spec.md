# LedgerAgent design

Date: 2026-11-01
Status: draft, awaiting review

## Purpose

LedgerAgent is a small, public LangGraph agent. It navigates a synthetic legacy ERP in Postgres, identifies data discrepancies in the general ledger and accounts payable, and suggests verified fixes. It never applies them.

It combines LangGraph, Python, SQL, evals and Docker in one working system, with a working agent as the centrepiece, surrounded by a deterministic guardrail layer and a rigorous eval suite.

### Audience and success criteria

The audience is an engineer reading the repo, possibly for only a few minutes.

The project succeeds when:

- A reviewer can clone the repo and run one command, with no API key, and watch the agent and evals run.
- The README opens with measured results, including misses.
- The agent infers the schema itself rather than relying on hand-written checks.
- The verifier rejects 100% of the guardrail suite's deliberately bad fixes.
- The project stays small: one workflow, 7 tables, 5 fault types.

## Architecture

Everything runs through Docker Compose on Python 3.12, pinned in the container.

| Service | Job |
|---|---|
| `db` | Postgres holding the synthetic ERP |
| `agent` | Runs an audit and writes a findings report |
| `evals` | Runs scored scenarios and writes results |

### Units

| Unit | Job | Depends on |
|---|---|---|
| `erp` | Builds the schema, generates seeded data, plants faults, writes the answer key | Postgres |
| `tools` | Read-only SQL tool for the LLM: single `SELECT` statements, row cap, timeout, read-only role | Postgres |
| `llm` | One provider-neutral model interface with live, record and replay modes | Model provider |
| `agent` | LangGraph graph, typed state and nodes | `tools`, `llm`, `verifier` |
| `verifier` | Checks proposed fixes deterministically, with no LLM | Postgres |
| `evals` | Runs scenarios, scores against the answer key, reports results | `erp`, `agent` |

### Boundaries

- **The answer key is hidden from the agent.** It lives outside the database and is never mounted in the `agent` container.
- **Two database roles.** The LLM's SQL tool uses a read-only role. Only `verifier` uses the write role, always inside a transaction it rolls back.
- **Nothing applies fixes.** No code path commits a change. The output is a report.

The default model is `claude-sonnet-5-5`, and the `llm` interface stays provider-neutral.

## Synthetic ERP

### Schema

Seven tables covering General Ledger and Accounts Payable:

| Table | Contents |
|---|---|
| `GLCOA` | Chart of accounts |
| `GLPERD` | Fiscal periods |
| `GLJRNH` | Journal headers, with a reference field linking a correction to the journal it corrects |
| `GLJRNL` | Journal lines |
| `APVEND` | Vendors |
| `APINVH` | Invoices |
| `APPAYM` | Payments |

Deliberate legacy quirks:

- No declared foreign keys.
- Terse column names such as `DR_AMT` and `POST_FLG`, with single-letter status codes.
- Dates stored as `YYYYMMDD` integers.
- Account and vendor codes padded with spaces.
- A short data dictionary that is incomplete and wrong in one place.

### Planted faults

| Fault type | Example |
|---|---|
| Unbalanced journal | One line amount was mistyped, so debits and credits differ |
| Duplicate invoice | The same vendor invoice entered twice, with the number padded differently |
| AP-to-GL mismatch | An invoice total differs from the journal that posted it |
| Closed-period posting | A journal dated into a closed period |
| Invalid account | A journal line pointing to a missing or inactive account |

### Decoys

Decoys look wrong but are valid, and they measure false alarms:

- A reversing entry
- A credit memo
- Two identical monthly rent invoices
- A rounding difference within tolerance

### Generation

A seeded Python generator produces a few thousand journal lines deterministically, then applies faults. It writes a separate answer key listing each fault, the affected rows and what a valid correction must achieve.

### Fix convention

Posted ledger entries are never edited. A fix is a new correcting journal entry that references the original, unless the problem sits in non-posted data such as an invoice status.

Balance is assessed across the journal group: the original plus its corrections. A correction to a mistyped amount can be one-sided, provided the group balances.

## Agent graph

```
explore → hunt → confirm → propose → verify ─┬─ pass ──────────────→ next finding
                              ↑              ├─ fail, tries left ──→ back to propose
                              └── reason ────┘
                                             └─ fail, out of tries → flag for a person
all findings done → report
```

| Node | Type | Behaviour |
|---|---|---|
| `explore` | LLM with read-only SQL, 15-query limit | Reads the catalog and data dictionary, samples data, writes structured schema notes including inferred joins |
| `hunt` | LLM with read-only SQL, 40-query limit | Returns findings, each with a type, affected rows and the SQL that proves it |
| `confirm` | Code | Re-runs each finding's evidence SQL and drops any that return no rows |
| `propose` | LLM with structured output | Writes a structured correction for one finding, with reasoning |
| `verify` | Code | Runs the guardrail checks and returns pass or fail with a specific reason |
| `report` | Code | Writes findings, evidence and fixes as Markdown and JSON |

Query limits are configurable. Their values above are defaults.

- The `hunt` prompt receives general audit principles: double-entry balance, subledger-to-GL tie-out, period controls, reference integrity and duplicates. It receives no table names or fault recipes. The README discloses that these principles overlap the five fault types.
- Each fix gets 3 attempts. `verify`'s failure reason goes back to `propose`.
- Each finding ends as `verified` or `needs_human`. Unresolved findings still appear in the report.
- Typed state holds schema notes, the findings queue, the current attempt count and token use.
- A run stops at a token limit, default 200,000 tokens, and reports what it has so far.
- The graph has no approval pause, because nothing is applied. The report is the handoff to a person.

## Guardrails

### Structured proposals

`propose` returns data, not SQL: journal lines with account, debit, credit and period, the journal being corrected, and any invoice status change. Code renders this as parameterised SQL, and the report shows the rendered SQL. LLM-written SQL never runs against the write role.

### Verification

`verify` runs these checks in order and stops at the first failure:

1. **Structure:** accounts exist and are active, the target period is open, amounts are valid and the referenced journal exists.
2. **Trial run:** apply the fix inside a transaction with a timeout.
3. **Change limits:** only new correcting rows and permitted status fields change, within a row cap.
4. **Resolution:** a coded rule for each of the five fault types confirms the fault is resolved, for example that the journal group balances or that the invoice ties to GL.
5. **No collateral damage:** global checks across the whole ledger show no new violations.

The transaction always rolls back, whatever the outcome.

Findings outside the five coded types cannot be auto-verified and go straight to `needs_human`.

### Failure handling

| Failure | Response |
|---|---|
| Malformed LLM output | One retry with the validation error, then `needs_human` |
| SQL error in the tool | Returned to the LLM and counted against the query limit |
| Query or token limit reached | Continue with what it has and record that in the report |
| API errors | Back off, then exit with a clear error |
| Replay recording missing | Fail loudly, with no fallback to a live call |
| Database down | Exit non-zero |

### Prompt injection

Memo fields are untrusted data. The LLM has no write access, and one eval scenario plants a hostile memo.

## Evals

### Guardrail suite

This suite has no LLM and runs in seconds. It feeds the verifier a library of deliberately bad fixes: unbalanced groups, closed periods, inactive accounts, oversized changes and fixes that break something else. It must reject 100% of them.

### Agent suite

This suite runs the full graph on 10 scenarios, each a generated ERP with its own faults and decoys:

- 6 dev scenarios, used for tuning prompts
- 3 holdout scenarios on seeds never tuned against, which supply the headline numbers
- 1 injection scenario with a hostile memo

| Metric | Measures |
|---|---|
| Recall by fault type | Planted faults found |
| Precision and decoy false-alarm rate | Whether it cries wolf |
| Fix validity | Whether fixes are correct according to the answer key |
| `needs_human` rate | How often it declines to resolve something |
| Tokens, cost and queries per run | What it costs to run |
| Injection result | Whether the hostile memo changes its behaviour |

Fix validity is scored by eval code against the answer key, independently of the verifier.

Results are published as measured, including misses.

### Thresholds

CI requires a 100% rejection rate on the guardrail suite from the start. Agent suite thresholds are set in a committed config file from the first holdout baseline, and CI fails if scores fall below them.

## LLM recording

- **Replay** is the default mode and needs no API key.
- **Record** runs live and writes responses to committed recordings.
- **Live** runs against the real model without recording, when a key is set.
- Recordings are keyed by a hash of the model, messages and tools. Prompts must contain no timestamps or other run-specific values.
- Changing a prompt means re-recording, and a missing recording fails loudly.
- The README states that default runs are replays. Replays prove the pipeline and scoring, while a manually started live run shows the score holds against the real model.

## Delivery

- **Run command:** a single Docker Compose command runs the agent and evals with no key.
- **CI:** GitHub Actions runs tests and replay evals in Docker on every push. Live evals run only when started by hand, within a cost limit, using a repo secret.
- **Tracing:** each run writes a JSON trace of steps and tokens.
- **README:** results table first, then the graph diagram, the run command and a limitations section.
- **Repo:** public `ledger-agent` under `johnrodermund`, MIT licensed. Creating the public repo needs explicit confirmation first.
- **Method:** built test-first.

## Prerequisites from the owner

- An Anthropic API key for recording and live evals. This costs real money, and cost per run is measured early and reported.
- Docker Desktop on the development machine, which is recommended for faster iteration. The alternative is recording through a manually triggered GitHub Actions job using a repo secret.

## Out of scope for v1

- Applying fixes, or any approval pause
- OpenTelemetry and Weights & Biases
- DSPy, which is the natural next step because the scored dev set is what it needs to optimise against
- pgvector
- A UI
- SQL Server
- OpenAI as a recorded provider
