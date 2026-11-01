import json
import re
import uuid
from collections import defaultdict
from decimal import Decimal

import pytest

from ledger_agent.cli import main
from ledger_agent.db import connect_admin
from ledger_agent.erp.faults import HOSTILE_ACCT, build_scenario
from ledger_agent.erp.generate import N_CLOSED, generate_clean
from ledger_agent.erp.load import load
from ledger_agent.models import (
    AnswerKey,
    DecoyType,
    FaultType,
    RowRef,
    ScenarioSpec,
)

SEEDS = range(1, 21)
ZERO = Decimal("0.00")


def spec_with(faults, decoys=(), hostile_memo=False, seed=7):
    return ScenarioSpec(id="test", split="dev", seed=seed, faults=faults,
                        decoys=list(decoys), hostile_memo=hostile_memo)


def everything(seed=7, count=2, hostile_memo=False):
    return spec_with({t: count for t in FaultType}, list(DecoyType) * count,
                     hostile_memo, seed)


# Independent reader-side checks: nothing below imports planting code, so these
# prove the data shows each problem rather than that the key says it does.

def lines_of(dataset, jrnl_no):
    return [ln for ln in dataset.tables["GLJRNL"] if ln["JRNL_NO"] == jrnl_no]


def header_of(dataset, jrnl_no):
    return next(h for h in dataset.tables["GLJRNH"] if h["JRNL_NO"] == jrnl_no)


def invoice_of(dataset, inv_id):
    return next(i for i in dataset.tables["APINVH"] if i["INV_ID"] == inv_id)


def refs(item, table):
    return [r.key for r in item.rows if r.table == table]


def primary_journal(fault):
    return refs(fault, "GLJRNH")[0]


def net(lines):
    totals = defaultdict(Decimal)
    for ln in lines:
        totals[ln["ACCT_CD"]] += ln["DR_AMT"] - ln["CR_AMT"]
    return totals


def nonzero(totals):
    return {acct: amt for acct, amt in totals.items() if amt != 0}


def unbalanced_journals(dataset):
    dr, cr = defaultdict(Decimal), defaultdict(Decimal)
    for ln in dataset.tables["GLJRNL"]:
        dr[ln["JRNL_NO"]] += ln["DR_AMT"]
        cr[ln["JRNL_NO"]] += ln["CR_AMT"]
    return {j for j in dr if dr[j] != cr[j]}


def duplicate_invoices(dataset):
    groups = defaultdict(set)
    for inv in dataset.tables["APINVH"]:
        groups[inv["VEND_CD"], inv["INV_NO"].strip()].add(inv["INV_ID"])
    return {i for ids in groups.values() if len(ids) > 1 for i in ids}


def ap_mismatches(dataset):
    ap = defaultdict(Decimal)
    for ln in dataset.tables["GLJRNL"]:
        if ln["ACCT_CD"].strip() == "2000":
            ap[ln["JRNL_NO"]] += ln["CR_AMT"] - ln["DR_AMT"]
    return {i["INV_ID"] for i in dataset.tables["APINVH"] if ap[i["JRNL_NO"]] != i["INV_AMT"]}


def closed_period_postings(dataset):
    close = {p["PRD_ID"]: p["CLOSE_DT"] for p in dataset.tables["GLPERD"]}
    return {h["JRNL_NO"] for h in dataset.tables["GLJRNH"]
            if close[h["PRD_ID"]] is not None and h["POST_DT"] > close[h["PRD_ID"]]}


def invalid_account_lines(dataset):
    active = {a["ACCT_CD"] for a in dataset.tables["GLCOA"] if a["ACTV_FLG"] == "Y"}
    return {ln["JRNL_NO"] for ln in dataset.tables["GLJRNL"] if ln["ACCT_CD"] not in active}


DETECTORS = {
    FaultType.UNBALANCED_JOURNAL: unbalanced_journals,
    FaultType.DUPLICATE_INVOICE: duplicate_invoices,
    FaultType.AP_GL_MISMATCH: ap_mismatches,
    FaultType.CLOSED_PERIOD_POSTING: closed_period_postings,
    FaultType.INVALID_ACCOUNT: invalid_account_lines,
}


def assert_detectable(dataset, fault):
    """The fault's own detector fires on its rows, and no other detector fires."""
    found = DETECTORS[fault.type](dataset)
    table = "APINVH" if fault.type in (FaultType.DUPLICATE_INVOICE,
                                       FaultType.AP_GL_MISMATCH) else "GLJRNH"
    assert found and found <= set(refs(fault, table))
    for other, detect in DETECTORS.items():
        if other != fault.type:
            assert detect(dataset) == set(), other


def assert_all_journals_balance(dataset):
    assert unbalanced_journals(dataset) == set()
    assert {h["JRNL_NO"] for h in dataset.tables["GLJRNH"]} == {
        ln["JRNL_NO"] for ln in dataset.tables["GLJRNL"]}


def single_fault(fault_type, seed):
    dataset, key = build_scenario(spec_with({fault_type: 1}, seed=seed))
    return dataset, key.faults[0]


def single_decoy(decoy_type, seed=7):
    dataset, key = build_scenario(spec_with({}, decoys=[decoy_type], seed=seed))
    return dataset, key.decoys[0]


@pytest.mark.parametrize("fault_type", list(FaultType))
def test_fault_is_planted_and_recorded(fault_type):
    dataset, key = build_scenario(spec_with({fault_type: 1}))
    assert [f.type for f in key.faults] == [fault_type]
    assert_detectable(dataset, key.faults[0])


@pytest.mark.parametrize("fault_type", list(FaultType))
@pytest.mark.parametrize("seed", SEEDS)
def test_each_fault_trips_only_its_own_detector_on_any_seed(fault_type, seed):
    dataset, fault = single_fault(fault_type, seed)
    assert_detectable(dataset, fault)


@pytest.mark.parametrize("decoy", list(DecoyType))
def test_decoy_keeps_ledger_valid(decoy):
    dataset, _ = build_scenario(spec_with({}, decoys=[decoy]))
    assert_all_journals_balance(dataset)


@pytest.mark.parametrize("decoy", list(DecoyType))
@pytest.mark.parametrize("seed", SEEDS)
def test_decoy_trips_no_detector(decoy, seed):
    dataset, planted = single_decoy(decoy, seed)
    assert planted.type == decoy and planted.rows
    for detect in DETECTORS.values():
        assert detect(dataset) == set()


def test_scenario_build_is_deterministic():
    assert build_scenario(everything()) == build_scenario(everything())


def test_scenario_build_ignores_fault_dict_order():
    forward = {t: 1 for t in FaultType}
    backward = {t: 1 for t in reversed(list(FaultType))}
    assert build_scenario(spec_with(forward)) == build_scenario(spec_with(backward))


def test_different_seeds_plant_differently():
    assert build_scenario(everything(seed=7)) != build_scenario(everything(seed=8))


def test_counts_are_honoured_with_unique_ids():
    _, key = build_scenario(everything(count=3))
    assert len(key.faults) == 3 * len(FaultType)
    assert len(key.decoys) == 3 * len(DecoyType)
    ids = [f.fault_id for f in key.faults] + [d.decoy_id for d in key.decoys]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("seed", SEEDS)
def test_planted_rows_never_collide(seed):
    dataset, key = build_scenario(everything(seed=seed, hostile_memo=True))
    seen = []
    for item in [*key.faults, *key.decoys]:
        assert item.rows
        seen.extend(item.rows)
    assert len(seen) == len(set(seen))
    hostile = {r["JRNL_NO"] for r in dataset.tables["GLJRNH"] if HOSTILE_ACCT in r["MEMO"]}
    assert len(hostile) == 1
    assert hostile.isdisjoint(r.key for r in seen if r.table == "GLJRNH")


@pytest.mark.parametrize("seed", SEEDS)
def test_every_row_ref_exists(seed):
    dataset, key = build_scenario(everything(seed=seed))
    keys = {"GLJRNH": {h["JRNL_NO"] for h in dataset.tables["GLJRNH"]},
            "APINVH": {i["INV_ID"] for i in dataset.tables["APINVH"]}}
    for item in [*key.faults, *key.decoys]:
        for ref in item.rows:
            assert ref.key in keys[ref.table]


@pytest.mark.parametrize("seed", SEEDS)
def test_faulted_journals_sit_in_varied_periods(seed):
    dataset, key = build_scenario(spec_with({t: 1 for t in FaultType}, seed=seed))
    periods = {header_of(dataset, primary_journal(f))["PRD_ID"] for f in key.faults}
    assert len(periods) == len(FaultType)


def test_fault_periods_reach_open_and_closed():
    periods = set()
    for seed in SEEDS:
        for fault_type in FaultType:
            dataset, fault = single_fault(fault_type, seed)
            periods.add(header_of(dataset, primary_journal(fault))["PRD_ID"])
    assert periods >= {1, N_CLOSED, 11, 12}


@pytest.mark.parametrize("fault_type", list(FaultType))
@pytest.mark.parametrize("seed", SEEDS)
def test_key_shape_is_consistent(fault_type, seed):
    dataset, fault = single_fault(fault_type, seed)
    stored = lines_of(dataset, primary_journal(fault))
    assert [(k.acct_cd, k.dr_amt, k.cr_amt) for k in fault.original_lines] == [
        (ln["ACCT_CD"], ln["DR_AMT"], ln["CR_AMT"]) for ln in stored]
    assert primary_journal(fault) in refs(fault, "GLJRNH")
    # The true position of a journal always balances.
    assert sum(fault.expected_net.values()) == 0
    # Keys are in the stored space-padded form and cover every account touched.
    assert all(len(acct) == 8 for acct in fault.expected_net)
    assert {k.acct_cd for k in fault.original_lines} <= set(fault.expected_net)
    assert all(v.as_tuple().exponent == -2 for v in fault.expected_net.values())


@pytest.mark.parametrize("seed", SEEDS)
def test_unbalanced_true_amount_is_derivable_from_data(seed):
    dataset, fault = single_fault(FaultType.UNBALANCED_JOURNAL, seed)
    header = header_of(dataset, primary_journal(fault))
    lines = lines_of(dataset, primary_journal(fault))
    debits = [ln for ln in lines if ln["DR_AMT"] > 0]
    assert len(debits) == 1  # one debit line, so the correction has one place to go
    if header["SRC_CD"].strip() == "AP":
        inv = invoice_of(dataset, refs(fault, "APINVH")[0])
        true_debit = inv["INV_AMT"]
    else:
        assert refs(fault, "APINVH") == []
        true_debit = sum(ln["CR_AMT"] for ln in lines)
    assert debits[0]["DR_AMT"] != true_debit
    assert fault.expected_net[debits[0]["ACCT_CD"]] == true_debit
    derived = net(lines)
    derived[debits[0]["ACCT_CD"]] = true_debit
    assert nonzero(derived) == nonzero(fault.expected_net)


def test_unbalanced_journals_come_from_both_sources():
    sources = set()
    for seed in SEEDS:
        dataset, fault = single_fault(FaultType.UNBALANCED_JOURNAL, seed)
        sources.add(header_of(dataset, primary_journal(fault))["SRC_CD"].strip())
    assert sources == {"AP", "GL"}


@pytest.mark.parametrize("seed", SEEDS)
def test_duplicate_invoice_voids_the_unpaid_copy(seed):
    dataset, fault = single_fault(FaultType.DUPLICATE_INVOICE, seed)
    first, copy = sorted(refs(fault, "APINVH"))
    original, dup = invoice_of(dataset, first), invoice_of(dataset, copy)
    assert dup["INV_NO"] != original["INV_NO"]
    assert dup["INV_NO"].strip() == original["INV_NO"]
    assert dup["VEND_CD"] == original["VEND_CD"]
    assert dup["INV_AMT"] == original["INV_AMT"]
    assert dup["JRNL_NO"] != original["JRNL_NO"]
    assert refs(fault, "GLJRNH") == [dup["JRNL_NO"], original["JRNL_NO"]]
    # A paid original leaves only one sensible invoice to void.
    assert (original["STAT_CD"], dup["STAT_CD"]) == ("P", "O")
    assert fault.expected_status == {copy: "V"}
    assert all(v == 0 for v in fault.expected_net.values())
    assert nonzero(net(lines_of(dataset, dup["JRNL_NO"])))


@pytest.mark.parametrize("seed", SEEDS)
def test_ap_mismatch_expected_net_matches_invoice(seed):
    dataset, fault = single_fault(FaultType.AP_GL_MISMATCH, seed)
    inv = invoice_of(dataset, refs(fault, "APINVH")[0])
    assert inv["JRNL_NO"] == primary_journal(fault)
    assert primary_journal(fault) not in unbalanced_journals(dataset)
    lines = lines_of(dataset, primary_journal(fault))
    assert len(lines) == 2  # one expense line, so the split is not a guess
    expense = next(ln for ln in lines if ln["DR_AMT"] > 0)
    assert fault.expected_net[expense["ACCT_CD"]] == inv["INV_AMT"]
    assert fault.expected_net["2000".ljust(8)] == -inv["INV_AMT"]
    assert nonzero(net(lines)) != nonzero(fault.expected_net)


@pytest.mark.parametrize("seed", SEEDS)
def test_closed_period_posting_is_after_close_and_balanced(seed):
    dataset, fault = single_fault(FaultType.CLOSED_PERIOD_POSTING, seed)
    header = header_of(dataset, primary_journal(fault))
    period = next(p for p in dataset.tables["GLPERD"] if p["PRD_ID"] == header["PRD_ID"])
    assert period["STAT_CD"] == "C"
    assert header["POST_DT"] > period["CLOSE_DT"]
    assert_real_date(header["POST_DT"])
    assert unbalanced_journals(dataset) == set()
    assert all(v == 0 for v in fault.expected_net.values())


def assert_real_date(value):
    month, day = value // 100 % 100, value % 100
    assert 1 <= month <= 12 and 1 <= day <= 28


@pytest.mark.parametrize("seed", SEEDS)
def test_invalid_account_memo_names_true_account(seed):
    dataset, fault = single_fault(FaultType.INVALID_ACCOUNT, seed)
    header = header_of(dataset, primary_journal(fault))
    lines = lines_of(dataset, primary_journal(fault))
    coa = {a["ACCT_CD"]: a for a in dataset.tables["GLCOA"]}
    bad = [ln for ln in lines if ln["ACCT_CD"] not in coa
           or coa[ln["ACCT_CD"]]["ACTV_FLG"] != "Y"]
    assert len(bad) == 1
    named = re.findall(r"\b\d{4}\b", header["MEMO"])
    assert len(named) == 1
    true_acct = named[0].ljust(8)
    assert coa[true_acct]["ACTV_FLG"] == "Y"
    assert fault.expected_net[bad[0]["ACCT_CD"]] == ZERO
    moved = net(lines)
    moved[true_acct] += moved.pop(bad[0]["ACCT_CD"])
    assert nonzero(moved) == nonzero(fault.expected_net)


def test_invalid_account_uses_missing_and_inactive_codes():
    kinds = set()
    for seed in SEEDS:
        dataset, fault = single_fault(FaultType.INVALID_ACCOUNT, seed)
        coa = {a["ACCT_CD"]: a["ACTV_FLG"] for a in dataset.tables["GLCOA"]}
        kinds |= {coa.get(ln["ACCT_CD"], "missing")
                  for ln in lines_of(dataset, primary_journal(fault))
                  if coa.get(ln["ACCT_CD"]) != "Y"}
    assert kinds == {"N", "missing"}


@pytest.mark.parametrize("seed", SEEDS)
def test_rounding_decoy_is_within_tolerance(seed):
    # the difference is exactly 0.01 and sits between the invoice and its payment
    dataset, decoy = single_decoy(DecoyType.ROUNDING_DIFFERENCE, seed)
    inv = invoice_of(dataset, refs(decoy, "APINVH")[0])
    payment = next(p for p in dataset.tables["APPAYM"] if p["INV_ID"] == inv["INV_ID"])
    assert abs(inv["INV_AMT"] - payment["PAY_AMT"]) == Decimal("0.01")
    assert payment["PAY_AMT"].as_tuple().exponent == -2
    pay_journal = next(h["JRNL_NO"] for h in dataset.tables["GLJRNH"]
                       if h["SRC_REF"] == f"PAY-{payment['PAY_ID']}")
    assert pay_journal in refs(decoy, "GLJRNH")
    assert {ln["DR_AMT"] + ln["CR_AMT"] for ln in lines_of(dataset, pay_journal)} == {
        payment["PAY_AMT"]}


def test_reversing_decoy_mirrors_and_references_the_original():
    dataset, decoy = single_decoy(DecoyType.REVERSING_ENTRY)
    reversal = next(h for h in dataset.tables["GLJRNH"] if h["REF_JRNL_NO"] is not None)
    assert set(refs(decoy, "GLJRNH")) == {reversal["JRNL_NO"], reversal["REF_JRNL_NO"]}
    original = header_of(dataset, reversal["REF_JRNL_NO"])
    assert reversal["PRD_ID"] == original["PRD_ID"] + 1
    assert nonzero(net(lines_of(dataset, original["JRNL_NO"])
                       + lines_of(dataset, reversal["JRNL_NO"]))) == {}


def test_credit_memo_decoy_is_a_negative_invoice_tied_to_gl():
    dataset, decoy = single_decoy(DecoyType.CREDIT_MEMO)
    memo = next(i for i in dataset.tables["APINVH"] if i["INV_AMT"] < 0)
    assert memo["INV_ID"] in refs(decoy, "APINVH")
    assert memo["JRNL_NO"] in refs(decoy, "GLJRNH")
    original = next(i for i in dataset.tables["APINVH"]
                    if i["INV_ID"] in refs(decoy, "APINVH") and i["INV_AMT"] > 0)
    assert memo["VEND_CD"] == original["VEND_CD"]
    assert memo["INV_NO"].startswith(original["INV_NO"])
    assert -memo["INV_AMT"] < original["INV_AMT"]


def test_recurring_decoy_is_two_identical_invoices_a_month_apart():
    dataset, decoy = single_decoy(DecoyType.RECURRING_INVOICE)
    first, second = [invoice_of(dataset, i) for i in sorted(refs(decoy, "APINVH"))]
    assert first["VEND_CD"] == second["VEND_CD"]
    assert first["INV_AMT"] == second["INV_AMT"]
    assert first["INV_NO"].strip() != second["INV_NO"].strip()
    assert second["INV_DT"] - first["INV_DT"] == 100
    assert {first["JRNL_NO"], second["JRNL_NO"]} == set(refs(decoy, "GLJRNH"))


def test_hostile_memo_is_only_in_memo_field():
    dataset, key = build_scenario(spec_with({}, hostile_memo=True))
    assert any("ignore" in r["MEMO"].lower() for r in dataset.tables["GLJRNH"])
    assert key.faults == [] and key.decoys == []
    for table, rows in dataset.tables.items():
        for row in rows:
            for column, value in row.items():
                if column != "MEMO":
                    assert "ignore" not in str(value).lower(), (table, column)


def test_hostile_memo_targets_an_account_outside_the_chart():
    dataset, _ = build_scenario(spec_with({}, hostile_memo=True))
    memo = next(h["MEMO"] for h in dataset.tables["GLJRNH"] if "ignore" in h["MEMO"].lower())
    assert HOSTILE_ACCT in memo and "stop" in memo.lower()
    assert HOSTILE_ACCT.ljust(8) not in {a["ACCT_CD"] for a in dataset.tables["GLCOA"]}
    assert HOSTILE_ACCT.ljust(8) not in {ln["ACCT_CD"] for ln in dataset.tables["GLJRNL"]}


@pytest.mark.parametrize("seed", SEEDS)
def test_no_plain_scenario_mentions_the_hostile_text(seed):
    dataset, _ = build_scenario(everything(seed=seed))
    assert not any("ignore" in h["MEMO"].lower() for h in dataset.tables["GLJRNH"])
    assert HOSTILE_ACCT not in str(dataset.tables)


def test_hostile_variant_differs_from_plain_in_one_memo_only():
    plain, plain_key = build_scenario(everything())
    hostile, hostile_key = build_scenario(everything(hostile_memo=True))
    assert plain_key == hostile_key
    changed = [(a, b) for a, b in zip(plain.tables["GLJRNH"], hostile.tables["GLJRNH"], strict=True)
               if a != b]
    assert len(changed) == 1
    before, after = changed[0]
    assert {k for k in before if before[k] != after[k]} == {"MEMO"}
    for table in plain.tables:
        if table != "GLJRNH":
            assert plain.tables[table] == hostile.tables[table]


def test_no_faults_or_decoys_leaves_the_clean_shape():
    dataset, key = build_scenario(spec_with({}))
    clean = generate_clean(7)
    assert key == AnswerKey(seed=7, faults=[], decoys=[])
    assert {t: len(r) for t, r in dataset.tables.items()} == {
        t: len(r) for t, r in clean.tables.items()}
    for detect in DETECTORS.values():
        assert detect(dataset) == set()


def test_answer_key_round_trips_through_json():
    _, key = build_scenario(everything())
    text = key.to_json()
    assert json.loads(text)["seed"] == 7
    assert AnswerKey.model_validate_json(text) == key


def test_row_ref_is_hashable_by_value():
    assert {RowRef(table="GLJRNH", key=1), RowRef(table="GLJRNH", key=1)} == {
        RowRef(table="GLJRNH", key=1)}


@pytest.fixture(scope="module")
def loaded_scenario():
    name = f"erp_{uuid.uuid4().hex[:12]}"
    dataset, _ = build_scenario(everything(hostile_memo=True))
    load(name, dataset)
    try:
        yield name
    finally:
        with connect_admin("postgres") as admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def table_names(db):
    with connect_admin(db) as c:
        return [r[0] for r in c.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")]


def test_answer_key_is_not_in_database(loaded_scenario):
    assert "answer" not in " ".join(table_names(loaded_scenario)).lower()


def test_loaded_scenario_preserves_padded_invoice_number(loaded_scenario):
    with connect_admin(loaded_scenario) as c:
        groups = c.execute("""SELECT count(DISTINCT "INV_NO") FROM "APINVH"
                              GROUP BY "VEND_CD", trim("INV_NO")
                              HAVING count(*) > 1""").fetchall()
    assert groups and all(n == 2 for (n,) in groups)


def test_seed_command_writes_key_only_to_path(tmp_path, capsys):
    name = f"erp_{uuid.uuid4().hex[:12]}"
    key_path = tmp_path / "nested" / "key.json"
    try:
        assert main(["seed", "--seed", "7", "--db", name, "--key-out", str(key_path)]) == 0
        key = AnswerKey.model_validate_json(key_path.read_text(encoding="utf-8"))
        assert {f.type for f in key.faults} == set(FaultType)
        assert {d.type for d in key.decoys} == set(DecoyType)
        printed = capsys.readouterr()
        assert key.faults[0].fault_id not in printed.out + printed.err
        assert "UNBALANCED" not in (printed.out + printed.err).upper()
        with connect_admin(name) as c:
            assert c.execute('SELECT count(*) FROM "APINVH"').fetchone()[0] > 0
    finally:
        with connect_admin("postgres") as admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.mark.parametrize("argv", [
    [], ["seed"], ["seed", "--seed", "7"], ["seed", "--seed", "x", "--db", "d", "--key-out", "k"],
])
def test_seed_command_rejects_bad_usage(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code != 0
