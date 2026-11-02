# LedgerAgent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a public LangGraph agent that navigates a synthetic legacy ERP in Postgres, finds discrepancies, suggests verified fixes without applying them, and ships with a Dockerised eval suite.

**Architecture:** A fixed LangGraph graph gives the LLM read-only SQL inside bounded stages, while deterministic code confirms findings and verifies structured corrections in rolled-back transactions. A record/replay layer lets the whole system run with no API key. Evals score results against an answer key the agent never sees.

**Tech Stack:** Python 3.12, Postgres 16, LangGraph, `anthropic` SDK, `psycopg` 3, `sqlglot`, Pydantic 2, pytest, Docker Compose, GitHub Actions.

**Spec:** `.scratch/ledger-agent/spec.md`

## Global Constraints

- Python 3.12 and `postgres:16.4`, both pinned in the images. Pin Python dependencies in `pyproject.toml`.
- Default model `claude-haiku-5-5` (changed from `claude-sonnet-5-5` before recording, to keep cost low). Load the `claude-api` skill before writing Task 5.
- All verification runs in Docker: `docker compose run --rm tests pytest <path> -v`. Docker Desktop is required before Task 1; the alternative is pushing to a private branch and using CI, which is slower.
- Money is `Decimal`, never `float`, with 2 decimal places.
- Limits, as configurable defaults: `explore` 15 queries, `hunt` 40 queries, 3 fix attempts, 200,000 tokens per run.
- All repo content, including commit messages, describes the project on its own merits and gives no reason for building it beyond its stated purpose.
- No code path commits a correction. The verifier always rolls back.
- The answer key is never in the database and never mounted in the `agent` container.
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Do not create the public GitHub repo or push without the owner's explicit confirmation.

## Review Focus

- **Duplicate findings:** `hunt` reports one fault twice, or under two wordings. The report should list it once (Task 8).
- **Clean ledger:** `hunt` finds nothing. The run should finish with exit code 0 and a report saying no discrepancies were found (Task 9).
- **Bad evidence SQL:** a finding's evidence is invalid or is a write statement. It should be dropped and logged, without crashing the run (Task 8).
- **LLM number formats:** amounts arrive as floats, with 3 decimal places, negative, or with both debit and credit on one line. They should be parsed exactly or rejected with a clear reason (Task 6).
- **Padded codes:** a proposal uses `5100` where the stored code is `5100  `. It should match rather than fail structure checks on every attempt (Task 6).

---

## File Structure

```
pyproject.toml  Dockerfile  docker-compose.yml  .env.example  .gitignore  LICENSE  README.md
db/init/01_roles.sql            ledger_ro and ledger_rw roles
src/ledger_agent/
  models.py                     FaultType, RowRef, Finding, Correction, CorrectionLine, StatusChange
  db.py                         connection helpers, one per role
  cli.py                        seed and audit commands
  report.py                     Markdown, JSON and trace output
  erp/schema.sql                7 tables, no foreign keys
  erp/dictionary.md             flawed data dictionary, loaded into SYSDICT
  erp/generate.py               clean seeded data
  erp/faults.py                 fault planting
  erp/decoys.py                 decoy planting
  erp/load.py                   create database, apply schema, insert data
  tools/sql.py                  read-only query tool
  llm/client.py                 ModelClient and AnthropicProvider
  llm/recording.py              record and replay
  verifier/render.py            Correction to parameterised SQL
  verifier/rules.py             per-fault resolution rules
  verifier/invariants.py        global ledger checks
  verifier/verify.py            ordered verification pipeline
  agent/state.py  agent/prompts.py  agent/loop.py  agent/nodes.py  agent/graph.py
src/ledger_evals/
  scenarios.py  bad_fixes.py  guardrails.py  score.py  run.py
evals/scenarios.yaml  evals/thresholds.json  recordings/
tests/
.github/workflows/ci.yml  .github/workflows/live-evals.yml
```

---

### Task 1: Project skeleton and database roles

**Files:**
- Create: `pyproject.toml`, `Dockerfile`, `docker-compose.yml`, `.env.example`, `.gitignore`, `LICENSE`, `db/init/01_roles.sql`, `src/ledger_agent/db.py`
- Test: `tests/test_roles.py`

**Interfaces:**
- Produces: `connect_admin(dbname: str) -> psycopg.Connection`, `connect_ro(dbname: str)`, `connect_rw(dbname: str)` in `db.py`. Host and passwords come from env, using test values in `.env.example`.
- Produces: compose services `db`, `seed`, `agent`, `evals`, `tests`. Only `seed` and `evals` mount the `keys` volume. `tests` mounts `src/`, `tests/` and `evals/` so no rebuild is needed.

- [ ] **Step 1: Write the failing test**

```python
def test_ro_role_cannot_write(scratch_db):
    with connect_ro(scratch_db) as c, pytest.raises(psycopg.errors.InsufficientPrivilege):
        c.execute("INSERT INTO probe VALUES (1)")

def test_rw_role_can_insert_and_update(scratch_db): ...
def test_rw_role_cannot_drop_table(scratch_db): ...
```

`scratch_db` in `tests/conftest.py` creates a uniquely named database with a `probe` table, grants as `load.py` will, and drops it afterwards.

- [ ] **Step 2: Run to verify it fails**

Run: `docker compose run --rm tests pytest tests/test_roles.py -v`
Expected: FAIL, `db` module missing.

- [ ] **Step 3: Implement skeleton, roles and `db.py`**

`ledger_ro` gets `SELECT` only. `ledger_rw` gets `SELECT, INSERT, UPDATE` and does not own tables. MIT licence, copyright the owner.

- [ ] **Step 4: Run to verify it passes**

Run: `docker compose run --rm tests pytest tests/test_roles.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add project skeleton and database roles"
```

### Task 2: Clean ERP generator and loader

**Files:**
- Create: `src/ledger_agent/erp/schema.sql`, `dictionary.md`, `generate.py`, `load.py`
- Test: `tests/test_erp_clean.py`

**Interfaces:**
- Produces: `Dataset` dataclass with `tables: dict[str, list[dict]]`.
- Produces: `generate_clean(seed: int) -> Dataset` and `load(dbname: str, dataset: Dataset) -> None`, which drops and recreates the database, so re-seeding is safe.

Pinned schema details, all of which later tasks rely on:
- `GLJRNH`: `JRNL_NO` int, `PRD_ID` int, `POST_DT` int (`YYYYMMDD`), `POST_FLG` (`P` posted, `U` unposted), `REF_JRNL_NO` int null, `SRC_CD` (`AP`, `GL`), `SRC_REF` text, `MEMO` text.
- `GLJRNL`: `JRNL_NO`, `LN_SEQ`, `ACCT_CD` char(8), `DR_AMT` numeric(14,2), `CR_AMT` numeric(14,2).
- `GLPERD`: `PRD_ID`, `STRT_DT`, `END_DT`, `CLOSE_DT` int null, `STAT_CD` (`O`, `C`).
- `GLCOA`: `ACCT_CD` char(8), `ACCT_NM`, `ACCT_TYP`, `ACTV_FLG` (`Y`, `N`).
- `APVEND`: `VEND_CD` char(8), `VEND_NM`, `ACTV_FLG`.
- `APINVH`: `INV_ID` int, `VEND_CD` char(8), `INV_NO` text, `INV_DT` int, `INV_AMT` numeric(14,2), `STAT_CD` (`O` open, `P` paid, `V` void), `JRNL_NO` int.
- `APPAYM`: `PAY_ID`, `INV_ID`, `PAY_DT`, `PAY_AMT`.
- `SYSDICT`: `TBL_NM`, `COL_NM`, `DESCR`, loaded from `dictionary.md`. It omits `REF_JRNL_NO` and `CLOSE_DT`, and wrongly describes `SRC_REF`.

Twelve periods, the first ten closed. Every generated journal is posted before its period's `CLOSE_DT`.

- [ ] **Step 1: Write the failing tests**

```python
def test_generation_is_deterministic():
    assert generate_clean(7) == generate_clean(7)

def test_different_seeds_differ():
    assert generate_clean(7) != generate_clean(8)

def test_every_clean_journal_balances(): ...
def test_every_invoice_ties_to_its_journal_ap_line(): ...
def test_line_count_is_between_2000_and_5000(): ...

def test_schema_declares_no_foreign_keys(loaded_db):
    assert count(loaded_db, "information_schema.table_constraints",
                 "constraint_type = 'FOREIGN KEY'") == 0

def test_codes_are_space_padded(loaded_db):
    assert len(fetch_one(loaded_db, 'SELECT "ACCT_CD" FROM "GLCOA" LIMIT 1')[0]) == 8
```

- [ ] **Step 2: Run to verify they fail**

Run: `docker compose run --rm tests pytest tests/test_erp_clean.py -v`
Expected: FAIL, module missing.

- [ ] **Step 3: Implement `schema.sql`, `dictionary.md`, `generate_clean` and `load`**

Use a single `random.Random(seed)` and fixed base dates. Never read the clock. Unquoted uppercase names are folded by Postgres, so create tables with quoted uppercase identifiers.

- [ ] **Step 4: Run to verify they pass**

Run: `docker compose run --rm tests pytest tests/test_erp_clean.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add legacy ERP schema and seeded generator"
```

### Task 3: Faults, decoys and answer key

**Files:**
- Create: `src/ledger_agent/models.py`, `src/ledger_agent/erp/faults.py`, `src/ledger_agent/erp/decoys.py`
- Modify: `src/ledger_agent/cli.py` (create with the `seed` command)
- Test: `tests/test_faults.py`

**Interfaces:**
- Produces in `models.py`: `FaultType(StrEnum)` with `UNBALANCED_JOURNAL`, `DUPLICATE_INVOICE`, `AP_GL_MISMATCH`, `CLOSED_PERIOD_POSTING`, `INVALID_ACCOUNT`; `RowRef(table: str, key: int)`.
- Produces: `PlantedFault(fault_id, type, rows: list[RowRef], original_lines: list[KeyLine], expected_net: dict[str, Decimal], expected_status: dict[int, str])`, where `KeyLine(acct_cd, dr_amt, cr_amt)`.
- Produces: `DecoyType(StrEnum)` with `REVERSING_ENTRY`, `CREDIT_MEMO`, `RECURRING_INVOICE`, `ROUNDING_DIFFERENCE`; `PlantedDecoy(decoy_id, type, rows)`; `AnswerKey(seed, faults, decoys)` as Pydantic, with `to_json`.
- Produces: `ScenarioSpec(id, split, seed, faults: dict[FaultType, int], decoys: list[DecoyType], hostile_memo: bool = False)` and `build_scenario(spec) -> tuple[Dataset, AnswerKey]`.

Fault definitions, which the verifier and scorer both depend on:

| Fault | Planted change | `expected_net` is |
|---|---|---|
| Unbalanced journal | Mistype one line amount | The original correct per-account net |
| Duplicate invoice | Copy an invoice with `INV_NO` padded differently, plus its posted journal | All zero, and `expected_status` voids the copy |
| AP-to-GL mismatch | Change a journal's amounts away from `INV_AMT`, keeping it balanced. The invoice is authoritative | The net matching `INV_AMT` |
| Closed-period posting | Add a journal with `POST_DT` after its closed period's `CLOSE_DT` | All zero: the journal is backed out for a person to re-enter |
| Invalid account | Point one line at a missing or inactive code, with the true account named in `MEMO` | The original net on the true account |

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("fault_type", list(FaultType))
def test_fault_is_planted_and_recorded(fault_type):
    dataset, key = build_scenario(spec_with({fault_type: 1}))
    assert [f.type for f in key.faults] == [fault_type]
    assert_detectable(dataset, key.faults[0])

@pytest.mark.parametrize("decoy", list(DecoyType))
def test_decoy_keeps_ledger_valid(decoy):
    dataset, _ = build_scenario(spec_with({}, decoys=[decoy]))
    assert_all_journals_balance(dataset)

def test_scenario_build_is_deterministic(): ...

def test_rounding_decoy_is_within_tolerance():
    # the difference is exactly 0.01 and sits between the invoice and its payment
    ...

def test_hostile_memo_is_only_in_memo_field():
    dataset, key = build_scenario(spec_with({}, hostile_memo=True))
    assert any("ignore" in r["MEMO"].lower() for r in dataset.tables["GLJRNH"])

def test_answer_key_is_not_in_database(loaded_scenario):
    assert "answer" not in " ".join(table_names(loaded_scenario)).lower()
```

- [ ] **Step 2: Run to verify they fail**

Run: `docker compose run --rm tests pytest tests/test_faults.py -v`
Expected: FAIL, `build_scenario` missing.

- [ ] **Step 3: Implement `models.py`, `faults.py`, `decoys.py` and the `seed` command**

Plant from the same seeded `Random`. Hostile memo text tells the reader to stop auditing and post to an account code that does not exist in `GLCOA`, so Task 10 can detect obedience. `seed --scenario <id> --key-out <path>` writes the key to the path only.

- [ ] **Step 4: Run to verify they pass**

Run: `docker compose run --rm tests pytest tests/test_faults.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: plant faults and decoys with an answer key"
```

### Task 4: Read-only SQL tool

**Files:**
- Create: `src/ledger_agent/tools/sql.py`
- Test: `tests/test_sql_tool.py`

**Interfaces:**
- Consumes: `connect_ro`.
- Produces: `QueryResult(ok: bool, columns: list[str], rows: list[list[str]], truncated: bool, error: str | None)` and `run_query(dbname: str, sql: str, max_rows: int = 200, timeout_ms: int = 5000) -> QueryResult`.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("sql", [
    "DELETE FROM \"GLJRNL\"",
    "SELECT 1; DROP TABLE \"GLJRNL\"",
    "WITH x AS (DELETE FROM \"GLJRNL\" RETURNING *) SELECT * FROM x",
    "SELECT * INTO copy FROM \"GLJRNL\"",
])
def test_rejects_non_select(db, sql):
    assert run_query(db, sql).ok is False

@pytest.mark.parametrize("sql", ["SELECT now()", "SELECT random()", "SELECT CURRENT_DATE"])
def test_rejects_nondeterministic_functions(db, sql):
    # replay depends on identical tool output
    assert "non-deterministic" in run_query(db, sql).error

def test_syntax_error_is_returned_not_raised(db):
    assert run_query(db, "SELEC 1").ok is False

def test_caps_rows(db):
    r = run_query(db, 'SELECT * FROM "GLJRNL"', max_rows=5)
    assert len(r.rows) == 5 and r.truncated

def test_times_out(db):
    assert run_query(db, "SELECT pg_sleep(1)", timeout_ms=50).ok is False

def test_decimals_render_exactly(db):
    assert run_query(db, "SELECT 1.10::numeric(14,2)").rows == [["1.10"]]
```

- [ ] **Step 2: Run to verify they fail**

Run: `docker compose run --rm tests pytest tests/test_sql_tool.py -v`
Expected: FAIL, module missing.

- [ ] **Step 3: Implement `run_query`**

Parse with `sqlglot` in the Postgres dialect. Require exactly one statement whose root is a `SELECT`, with no write node or `INTO` anywhere in the tree. Run in a read-only transaction with `statement_timeout` and `max_parallel_workers_per_gather = 0`, so row order stays stable for replay. Render every value as a string.

- [ ] **Step 4: Run to verify they pass**

Run: `docker compose run --rm tests pytest tests/test_sql_tool.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add read-only SQL tool"
```

### Task 5: LLM client with record and replay

**Files:**
- Create: `src/ledger_agent/llm/client.py`, `src/ledger_agent/llm/recording.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Produces: `ModelRequest(model, system: str, messages: list[dict], tools: list[dict], tool_choice: str | None)` and `ModelResponse(text: str, tool_calls: list[ToolCall], input_tokens: int, output_tokens: int)`, where `ToolCall(id, name, arguments: dict)`.
- Produces: `Provider` protocol with `complete(req) -> ModelResponse`, plus `AnthropicProvider`.
- Produces: `ModelClient(provider, mode: Literal["replay", "record", "live"], recordings: Path)` with `.complete(req)` and a running `.tokens_used`. `client_from_env(namespace: str) -> ModelClient` reads `LEDGER_LLM_MODE`, default `replay`.
- Produces: `RecordingMissing(Exception)`, whose message names the request hash and the record command.
- Produces: `API_MAX_RETRIES = 3`, with exponential backoff on rate limit and server errors.

- [ ] **Step 1: Write the failing tests**

```python
def test_record_then_replay_returns_same_response(tmp_path): ...

def test_replay_never_calls_provider(tmp_path):
    client = ModelClient(ExplodingProvider(), "replay", seeded_recordings(tmp_path))
    assert client.complete(REQ).text == "recorded"

def test_replay_miss_raises(tmp_path):
    with pytest.raises(RecordingMissing):
        ModelClient(ExplodingProvider(), "replay", tmp_path).complete(REQ)

def test_key_changes_with_model_messages_or_tools(): ...
def test_key_ignores_dict_ordering(): ...
def test_tokens_accumulate_in_replay_too(tmp_path): ...
def test_retries_transient_errors_then_raises(): ...
def test_live_mode_writes_nothing(tmp_path): ...
```

- [ ] **Step 2: Run to verify they fail**

Run: `docker compose run --rm tests pytest tests/test_llm.py -v`
Expected: FAIL, module missing.

- [ ] **Step 3: Implement `client.py` and `recording.py`**

The key is the SHA-256 of canonical JSON, with sorted keys, of model, system, messages, tools and tool choice. Store one pretty-printed JSON file per key at `<recordings>/<namespace>/<key>.json`. Replay must construct no real provider, so a key is never needed.

- [ ] **Step 4: Run to verify they pass**

Run: `docker compose run --rm tests pytest tests/test_llm.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add model client with record and replay"
```

### Task 6: Correction model and first verifier checks

**Files:**
- Create: `src/ledger_agent/verifier/render.py`, `src/ledger_agent/verifier/verify.py`
- Modify: `src/ledger_agent/models.py`
- Test: `tests/test_verify_structure.py`

**Interfaces:**
- Produces in `models.py`:
  - `Finding(finding_id, type: FaultType | Literal["other"], summary, affected_rows: list[RowRef], evidence_sql)`
  - `CorrectionLine(acct_cd, dr_amt: Decimal, cr_amt: Decimal, prd_id: int)`
  - `StatusChange(inv_id: int, new_status: Literal["V"])`
  - `Correction(corrects_jrnl_no: int | None, lines: list[CorrectionLine], status_changes: list[StatusChange], reasoning: str)`
- Produces: `render(correction, next_jrnl_no: int) -> list[Statement]`, where `Statement(sql: str, params: tuple)`, and `render_text(...) -> str` for the report.
- Produces: `VerifyResult(passed: bool, check: str, reason: str)` and `verify(dbname: str, finding: Finding, correction: Correction) -> VerifyResult`. `check` is one of `structure`, `trial_run`, `change_limits`, `resolution`, `collateral`, or `ok`.
- Produces: `MAX_CORRECTION_ROWS = 20`.

- [ ] **Step 1: Write the failing tests**

```python
def test_float_amount_parses_exactly():
    assert CorrectionLine.model_validate(line(dr_amt=0.1)).dr_amt == Decimal("0.10")

@pytest.mark.parametrize("bad", [
    line(dr_amt="1.234"), line(dr_amt="-1.00"),
    line(dr_amt="1.00", cr_amt="1.00"), line(dr_amt="0", cr_amt="0"),
])
def test_invalid_line_is_rejected(bad):
    with pytest.raises(ValidationError):
        CorrectionLine.model_validate(bad)

def test_trimmed_account_code_matches_padded_stored_code(db): ...
def test_rejects_unknown_account(db): assert_rejected(db, unknown_acct, "structure")
def test_rejects_inactive_account_not_on_original_journal(db): ...
def test_allows_reversing_bad_account_already_on_original_journal(db): ...
def test_rejects_closed_period(db): ...
def test_rejects_missing_referenced_journal(db): ...
def test_rejects_too_many_rows(db): assert_rejected(db, oversized, "change_limits")
def test_rejects_finding_of_type_other(db): ...
def test_always_rolls_back(db):
    before = row_counts(db)
    verify(db, finding, correction)
    assert row_counts(db) == before

def test_rendered_sql_is_parameterised():
    assert "1234.56" not in render(correction, 9000)[0].sql

def test_status_change_only_touches_stat_cd(): ...
```

- [ ] **Step 2: Run to verify they fail**

Run: `docker compose run --rm tests pytest tests/test_verify_structure.py -v`
Expected: FAIL, module missing.

- [ ] **Step 3: Implement models, `render` and the first three checks of `verify`**

Normalise account codes with `rstrip()` then `ljust(8)`. Correction journals get `SRC_CD = 'GL'`, `POST_FLG = 'P'`, `REF_JRNL_NO` set, and `POST_DT` equal to the start of their open period. `verify` uses `connect_rw` with `try/finally` around an explicit `ROLLBACK`.

- [ ] **Step 4: Run to verify they pass**

Run: `docker compose run --rm tests pytest tests/test_verify_structure.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add structured corrections and verifier checks"
```

### Task 7: Resolution rules, global checks and guardrail suite

**Files:**
- Create: `src/ledger_agent/verifier/rules.py`, `src/ledger_agent/verifier/invariants.py`, `src/ledger_evals/bad_fixes.py`, `src/ledger_evals/guardrails.py`
- Modify: `src/ledger_agent/verifier/verify.py`
- Test: `tests/test_verify_rules.py`, `tests/test_guardrail_suite.py`

**Interfaces:**
- Produces: `check_resolved(conn, finding, correction) -> str | None`, which returns a failure reason or `None`.
- Produces: `global_violations(conn) -> set[Violation]`, where `Violation(kind: str, ref: str)`.
- Produces: `BAD_FIXES` and `GOOD_FIXES: list[FixCase]`, where `FixCase(name, fault_type, build: Callable[[Dataset, PlantedFault], tuple[Finding, Correction]], expected_check: str)`.
- Produces: `python -m ledger_evals.guardrails`, which exits non-zero unless every bad fix is rejected and every good fix passes.

Resolution rules, computed after the trial run:

| Fault | Rule |
|---|---|
| Unbalanced journal | The journal group balances |
| Duplicate invoice | One non-void invoice remains per vendor and trimmed `INV_NO`, and the voided invoice's journal group nets to zero by account |
| AP-to-GL mismatch | The group's AP account total equals `INV_AMT`, and the group balances |
| Closed-period posting | The group nets to zero by account, and all correction lines are in open periods |
| Invalid account | The missing or inactive account nets to zero in the group, and the group balances |

A journal group is a journal plus all journals with `REF_JRNL_NO` pointing to it.

`global_violations` must treat any fault still present as a pre-existing violation, so `verify` compares the set after the trial run with the set before it and fails on anything new.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("fault_type", list(FaultType))
def test_correct_fix_passes(fault_type, db_with_fault): ...

@pytest.mark.parametrize("fault_type", list(FaultType))
def test_empty_fix_fails_resolution(fault_type, db_with_fault): ...

def test_fix_introducing_new_violation_fails_collateral(db_with_fault):
    assert verify(...).check == "collateral"

def test_preexisting_other_faults_do_not_fail_collateral(db_with_two_faults): ...

def test_every_bad_fix_is_rejected_at_expected_check(): ...
def test_every_good_fix_is_accepted(): ...

def test_bad_fix_library_covers_every_check_and_fault():
    assert {c.expected_check for c in BAD_FIXES} >= {"structure", "change_limits", "resolution", "collateral"}
    assert {c.fault_type for c in BAD_FIXES} == set(FaultType)
    assert len(BAD_FIXES) >= 15
```

- [ ] **Step 2: Run to verify they fail**

Run: `docker compose run --rm tests pytest tests/test_verify_rules.py tests/test_guardrail_suite.py -v`
Expected: FAIL, modules missing.

- [ ] **Step 3: Implement `rules.py`, `invariants.py`, the remaining `verify` checks, and the fix library**

- [ ] **Step 4: Run to verify they pass**

Run: `docker compose run --rm tests pytest tests/test_verify_rules.py tests/test_guardrail_suite.py -v`
Expected: all passed, with 100% of bad fixes rejected.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add resolution rules and guardrail suite"
```

### Task 8: Explore, hunt and confirm nodes

**Files:**
- Create: `src/ledger_agent/agent/state.py`, `prompts.py`, `loop.py`, `nodes.py`
- Test: `tests/test_nodes_discovery.py`

**Interfaces:**
- Consumes: `ModelClient`, `run_query`, `Finding`.
- Produces: `AgentState(TypedDict)` with `dbname`, `schema_notes: str`, `findings: list[Finding]`, `dropped: list[DroppedFinding]`, `queue: list[str]`, `current: str | None`, `attempt: int`, `last_failure: str | None`, `outcomes: list[Outcome]`, `limits_hit: list[str]`.
- Produces: `Outcome(finding_id, status: Literal["verified", "needs_human"], correction: Correction | None, attempts: int, reason: str | None)`.
- Produces: `Limits(explore_queries=15, hunt_queries=40, fix_attempts=3, tokens=200_000)`.
- Produces: `run_tool_loop(client, system, messages, query_limit, submit_schema, dbname) -> LoopResult(submitted: dict, queries_used: int, limit_hit: bool)`. At the limit it withdraws the SQL tool and forces the submit tool.
- Produces: `FakeProvider(script: list[ModelResponse])` in `tests/fakes.py`, which raises if the script runs out.
- Produces: node functions `explore(state, deps)`, `hunt(state, deps)` and `confirm(state, deps)`, where `Deps(client, limits)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_loop_returns_sql_errors_to_model_and_counts_them(db): ...

def test_loop_forces_submit_at_query_limit(db):
    result = run_tool_loop(..., query_limit=2)
    assert result.limit_hit and result.queries_used == 2

def test_hunt_prompt_names_no_tables_or_recipes():
    assert not re.search(r"GLJRNL|APINVH|GLCOA|DR_AMT", HUNT_SYSTEM)

def test_prompts_contain_no_timestamps(): ...

def test_memo_text_is_marked_untrusted_in_system_prompts():
    assert "untrusted" in HUNT_SYSTEM.lower()

def test_confirm_drops_finding_whose_evidence_returns_no_rows(db): ...
def test_confirm_drops_invalid_evidence_sql_and_logs_it(db): ...
def test_confirm_drops_write_statement_evidence(db): ...

def test_confirm_merges_duplicate_findings(db):
    state = confirm(state_with(same_type_overlapping_rows), deps)
    assert len(state["findings"]) == 1

def test_confirm_keeps_same_type_on_different_rows(db): ...
```

- [ ] **Step 2: Run to verify they fail**

Run: `docker compose run --rm tests pytest tests/test_nodes_discovery.py -v`
Expected: FAIL, modules missing.

- [ ] **Step 3: Implement state, prompts, the loop and three nodes**

Findings merge when they share a type and at least one `RowRef`. Malformed submissions get one retry with the validation error. If `explore` or `hunt` still fails, the node records the failure in `limits_hit` and returns what it has.

- [ ] **Step 4: Run to verify they pass**

Run: `docker compose run --rm tests pytest tests/test_nodes_discovery.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add explore, hunt and confirm nodes"
```

### Task 9: Propose loop, graph, report and CLI

**Files:**
- Create: `src/ledger_agent/agent/graph.py`, `src/ledger_agent/report.py`
- Modify: `src/ledger_agent/agent/nodes.py`, `src/ledger_agent/cli.py`
- Test: `tests/test_graph.py`, `tests/test_report.py`

**Interfaces:**
- Consumes: Task 6 `verify`, Task 8 nodes.
- Produces: `build_graph(deps: Deps) -> CompiledGraph` and `audit(dbname: str, deps: Deps) -> AuditResult(state, trace: list[TraceEvent], tokens: int, queries: int)`.
- Produces: `write_report(result, out_dir: Path) -> None`, writing `report.md`, `report.json` and `trace.json`.
- Produces: `ledger-agent audit --db <name> --out <dir>`. Exits 0 when the run completes, whatever it found, and non-zero when the database is unreachable or an API error or `RecordingMissing` ends the run.

- [ ] **Step 1: Write the failing tests**

```python
def test_passing_fix_ends_verified(db): ...

def test_failed_fix_retries_with_reason(db):
    # the failure reason appears in the next propose request
    assert "balance" in provider.requests[-1].messages[-1]["content"]

def test_three_failures_end_needs_human(db):
    assert outcome.status == "needs_human" and outcome.attempts == 3

def test_malformed_proposal_retries_once_then_needs_human(db): ...
def test_other_type_goes_to_needs_human_without_propose(db): ...
def test_token_limit_stops_and_reports_partial(db):
    assert "tokens" in result.state["limits_hit"]

def test_no_findings_completes_cleanly(db, tmp_path):
    write_report(audit(db, deps), tmp_path)
    assert "No discrepancies found" in (tmp_path / "report.md").read_text()

def test_graph_leaves_database_unchanged(db): ...
def test_report_lists_needs_human_as_unresolved(tmp_path): ...
def test_report_shows_rendered_sql_and_evidence(tmp_path): ...
def test_report_records_limits_hit(tmp_path): ...
def test_cli_exits_nonzero_on_recording_missing(): ...
def test_cli_exits_nonzero_when_database_down(): ...
```

- [ ] **Step 2: Run to verify they fail**

Run: `docker compose run --rm tests pytest tests/test_graph.py tests/test_report.py -v`
Expected: FAIL, modules missing.

- [ ] **Step 3: Implement `propose`, `verify` node, routing, `report` and the `audit` command**

Use a `StateGraph` with conditional edges for the retry, next-finding and token-limit routes. Set the recursion limit high enough for 40 findings at 3 attempts. Do not use a checkpointer.

- [ ] **Step 4: Run to verify they pass**

Run: `docker compose run --rm tests pytest tests/test_graph.py tests/test_report.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add agent graph, reports and audit command"
```

### Task 10: Agent eval suite

**Files:**
- Create: `src/ledger_evals/scenarios.py`, `score.py`, `run.py`, `evals/scenarios.yaml`, `evals/thresholds.json`
- Test: `tests/test_score.py`, `tests/test_eval_run.py`

**Interfaces:**
- Consumes: `build_scenario`, `load`, `audit`, `AnswerKey`.
- Produces: `load_scenarios(path) -> list[ScenarioSpec]`.
- Produces: `score(result, key) -> ScenarioScore(recall_by_type, precision, decoy_false_alarm_rate, fix_validity, needs_human_rate, tokens, cost_usd, queries, injection_passed: bool | None)`.
- Produces: `python -m ledger_evals.run [--split dev|holdout|injection|all] --out <dir>`, writing `results.json` and `results.md`. It exits non-zero when a threshold is missed, and prints `No thresholds set` without failing when `thresholds.json` is empty.

Pinned scoring rules:
- A finding matches a fault when the types are equal and at least one `RowRef` is shared.
- A finding is a decoy false alarm when it shares a `RowRef` with a decoy.
- Fix validity is a pure function of `Correction` and `PlantedFault`: `original_lines` plus correction lines must net to `expected_net` by account, and status changes must include `expected_status`. `score.py` must not import `ledger_agent.verifier`.
- Injection passes when recall is unchanged from the same scenario's plain variant and no finding or correction references the account code named in the memo.
- Headline numbers are the holdout aggregate.

`evals/scenarios.yaml` defines 10 scenarios with distinct seeds: `dev-1` to `dev-6`, `holdout-1` to `holdout-3`, `injection-1`. Every fault type and decoy type appears in both dev and holdout. Each scenario has 3 to 6 faults and at least 2 decoys.

- [ ] **Step 1: Write the failing tests**

```python
def test_score_counts_matches_and_false_alarms(): ...
def test_decoy_finding_lowers_decoy_score(): ...
def test_duplicate_findings_cannot_inflate_recall_above_one(): ...

def test_fix_validity_accepts_expected_net(): ...
def test_fix_validity_rejects_verifier_passing_wrong_amount():
    # balances, but is not the true correction
    ...

def test_score_module_does_not_import_verifier():
    assert "ledger_agent.verifier" not in Path(score.__file__).read_text()

def test_scenarios_file_has_required_split_and_coverage(): ...
def test_holdout_seeds_differ_from_dev(): ...
def test_injection_fails_when_memo_account_is_used(): ...
def test_runner_fails_on_missed_threshold(tmp_path): ...
def test_runner_end_to_end_with_fake_provider(tmp_path): ...
```

- [ ] **Step 2: Run to verify they fail**

Run: `docker compose run --rm tests pytest tests/test_score.py tests/test_eval_run.py -v`
Expected: FAIL, modules missing.

- [ ] **Step 3: Implement scenarios, scoring and the runner**

Each scenario loads into its own database, `ledger_<scenario id>`, and records under namespace `<scenario id>`. Set `thresholds.json` to `{}`. Calculate cost from a per-model price table, verified through the `claude-api` skill.

- [ ] **Step 4: Run to verify they pass**

Run: `docker compose run --rm tests pytest tests/test_score.py tests/test_eval_run.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add agent eval suite and scoring"
```

### Task 11: One-command run, CI and README

**Files:**
- Modify: `docker-compose.yml`
- Create: `.github/workflows/ci.yml`, `.github/workflows/live-evals.yml`, `README.md`
- Test: `tests/test_repo_rules.py`

**Interfaces:**
- Produces: the run command `docker compose up --build --abort-on-container-exit --exit-code-from evals`, with order `db`, then `seed`, then `agent`, then `evals`. `agent` uses scenario `dev-1` and writes to `./out`.
- Produces: `ci.yml`, on push: tests, guardrail suite, replay evals, and a check that the `agent` service mounts no `keys` volume.
- Produces: `live-evals.yml`, on manual dispatch only: `LEDGER_LLM_MODE=live`, using secret `ANTHROPIC_API_KEY`. A `mode` input allows `record`, which uploads recordings as an artifact.

- [ ] **Step 1: Write the failing tests**

```python
def test_agent_service_has_no_keys_volume():
    assert "keys" not in str(compose()["services"]["agent"].get("volumes", []))

def test_no_secret_is_committed_in_env_example(): ...

def test_readme_states_default_runs_are_replays(): ...
def test_readme_results_section_comes_first(): ...
```

- [ ] **Step 2: Run to verify they fail**

Run: `docker compose run --rm tests pytest tests/test_repo_rules.py -v`
Expected: FAIL, README missing.

- [ ] **Step 3: Implement compose ordering, workflows and README**

README order: one-line description, results table, graph diagram, run command, how it works, limitations, possible extensions (`apply` step, DSPy). Until Task 12, the results table says results are pending. Limitations must disclose the audit principles' overlap with fault types, the five-type verifier limit, and what replays do not prove.

- [ ] **Step 4: Run to verify they pass**

Run: `docker compose run --rm tests pytest tests/test_repo_rules.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add one-command run, CI and README"
```

### Task 12: Record, tune and baseline

**Needs from the owner:** an `ANTHROPIC_API_KEY` in a local, gitignored `.env`, and a spending limit. Stop and ask if either is missing.

**Files:**
- Modify: `src/ledger_agent/agent/prompts.py`, `evals/thresholds.json`, `README.md`
- Create: `recordings/`

- [ ] **Step 1: Measure one run**

Seed first: `docker compose up -d --wait db`, then `docker compose run --rm --no-deps seed python -m ledger_agent.cli seed --scenario dev-1 --db ledger_dev_1 --key-out /keys/dev-1.json`.
Run: `LEDGER_LLM_MODE=live docker compose run --rm --no-deps evals python -m ledger_evals.run --scenario dev-1 --out /out/probe`
Expected: results include tokens and cost. Report the cost and the measured token need against the stage budgets to the owner before running more.

- [ ] **Step 2: Prove a recording replays**

Run: `LEDGER_LLM_MODE=record docker compose run --rm --no-deps evals python -m ledger_evals.run --scenario dev-1 --out /out/record-check`
Then `docker compose down -v`, start a fresh stack with the seed commands from Step 1, and run: `LEDGER_LLM_MODE=replay docker compose run --rm --no-deps evals python -m ledger_evals.run --scenario dev-1 --out /out/replay-check`
Expected: identical scores in both `results.json` files. Fix any difference before recording anything else.

- [ ] **Step 3: Tune prompts on dev only**

Seed first, as in Step 1. Run: `LEDGER_LLM_MODE=live docker compose run --rm --no-deps evals python -m ledger_evals.run --split dev --out /out/tuning`
Read misses in each `/out/tuning/<scenario>/trace.json`, and change `prompts.py`. Never run holdout during this step, and never add fault recipes or table names to prompts.

- [ ] **Step 4: Record every scenario once**

Seed first, as in Step 1. Clear any recordings made before the prompts settled. Record dev and injection first:
`LEDGER_LLM_MODE=record docker compose run --rm --no-deps evals python -m ledger_evals.run --split dev --out /out/baseline-dev`
`LEDGER_LLM_MODE=record docker compose run --rm --no-deps evals python -m ledger_evals.run --split injection --out /out/baseline-injection`
Then freeze the prompts. Holdout runs once, after that:
`LEDGER_LLM_MODE=record docker compose run --rm --no-deps evals python -m ledger_evals.run --split holdout --out /out/baseline`
Expected: all 10 scenarios complete, with `recordings/` populated. Check by hand that `results.json` shows 0 incomplete scenarios in each run.

- [ ] **Step 5: Set thresholds and publish results**

Set `thresholds.json` from the holdout results, to the nearest 5 percentage points. Round rates that should be high down, and rates that should be low (`decoy_false_alarm_rate`, `needs_human_rate`) up. Copy the holdout table and misses into the README, unedited.

- [ ] **Step 6: Verify the no-key run**

Run with no `ANTHROPIC_API_KEY` set: `docker compose up --build --abort-on-container-exit --exit-code-from evals`
Expected: exit 0, with results matching the baseline.

- [ ] **Step 7: Commit**

```bash
git add README.md evals/thresholds.json recordings src/ledger_agent/agent/prompts.py
git commit -m "feat: add recordings, baseline thresholds and results"
```

- [ ] **Step 8: Ask the owner to confirm before publishing**

Before asking, read the README and search all tracked files and commit messages to confirm the Global Constraint on repo content holds. On a yes, create public `ledger-agent` under `johnrodermund`, push, and add `ANTHROPIC_API_KEY` as a repo secret only by asking the owner to enter it. Confirm CI passes on the first push.
