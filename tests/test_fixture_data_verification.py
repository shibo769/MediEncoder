"""Exact archived-data guards: small synthetic fixtures, no training."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from mediencoder.simulation import dgp, runner


SPEC = importlib.util.spec_from_file_location(
    "verify_fixture_data", Path(__file__).parents[1] / "scripts/verify_fixture_data.py")
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


@pytest.fixture
def bundle(tmp_path):
    params = dgp.draw_parameters(dgp.DGPConfig(p=8, q=6, bar_p=2, bar_q=2, pilot_size=64), 910000)
    prefix = tmp_path / "mechanism"
    dgp.save_mechanism(params, prefix)
    reference = dict(mechanism_hash=params.mechanism_hash, seed_base=880000,
                     n_values=[40, 44], replications=2, source_run_id="37174159429", records=[])
    for rep in range(2):
        for n in reference["n_values"]:
            data_seed, training_seed = runner.task_seeds(reference["seed_base"], n, rep)
            reference["records"].append(dict(n=n, rep=rep, data_seed=data_seed,
                training_seed=training_seed,
                observed_data_sha256=runner.observed_data_hash(dgp.sample_data(params, n, data_seed))))
    path = tmp_path / "reference.json"
    runner.atomic_json(path, reference)
    return prefix, path, reference


def test_exact_complete_inventory_passes_and_writes_audit_without_redraw(bundle, tmp_path, monkeypatch):
    prefix, path, reference = bundle
    monkeypatch.setattr(dgp, "draw_parameters", lambda *a, **kw: pytest.fail("Unexpected parameter redraw"))
    monkeypatch.setattr(dgp, "population_truth", lambda *a, **kw: pytest.fail("Unexpected integration"))
    output = tmp_path / "audit.json"
    assert guard.main(["--mechanism", str(prefix), "--reference", str(path), "--output", str(output)]) == 0
    audit = json.loads(output.read_text())
    assert audit["exact_match"]
    assert audit["checked_records"] == audit["matched_records"] == audit["expected_records"] == 4
    assert audit["numerical_tolerance"] == audit["models_fitted"] == 0
    assert audit["failures"] == []
    assert audit["reference_sha256"] == runner.file_hash(path)


def test_changed_hash_fails_and_identifies_exact_pair(bundle, tmp_path):
    prefix, path, reference = bundle
    changed = deepcopy(reference)
    changed["records"][2]["observed_data_sha256"] = "0" * 64
    runner.atomic_json(path, changed)
    output = tmp_path / "audit.json"
    assert guard.main(["--mechanism", str(prefix), "--reference", str(path), "--output", str(output)]) == 2
    audit = json.loads(output.read_text())
    assert audit["checked_records"] == 4 and audit["matched_records"] == 3
    assert [(row["n"], row["rep"]) for row in audit["failures"]] == [(40, 1)]


@pytest.mark.parametrize("case", ("missing", "duplicate", "extra", "wrong_seed", "wrong_training_seed", "wrong_metadata"))
def test_invalid_inventory_fails_before_generating_any_data(bundle, case, monkeypatch):
    prefix, path, reference = bundle
    changed = deepcopy(reference)
    if case == "missing":
        changed["records"].pop()
    elif case == "duplicate":
        changed["records"][-1] = deepcopy(changed["records"][0])
    elif case == "extra":
        changed["records"][0]["rep"] = 2
    elif case == "wrong_seed":
        changed["records"][0]["data_seed"] += 1
    elif case == "wrong_training_seed":
        changed["records"][0]["training_seed"] += 1
    else:
        changed["replications"] = 3
    runner.atomic_json(path, changed)
    monkeypatch.setattr(dgp, "sample_data", lambda *a, **kw: pytest.fail("Invalid inventory was sampled"))
    with pytest.raises(ValueError):
        guard.verify_fixture_data(prefix, path)


def test_mechanism_or_metadata_tampering_is_rejected(bundle, monkeypatch):
    prefix, path, reference = bundle
    reference["mechanism_hash"] = "0" * 64
    runner.atomic_json(path, reference)
    monkeypatch.setattr(dgp, "sample_data", lambda *a, **kw: pytest.fail("Wrong mechanism was sampled"))
    with pytest.raises(ValueError, match="mechanism hash differs"):
        guard.verify_fixture_data(prefix, path)
    for name, bad in (("replications", True), ("seed_base", -1), ("n_values", [40, 40])):
        changed = deepcopy(reference)
        changed[name] = bad
        with pytest.raises(ValueError):
            guard.validate_reference(changed)


def test_cli_validation_error_is_nonzero_and_cannot_overwrite_reference(bundle, tmp_path):
    prefix, path, reference = bundle
    before = path.read_bytes()
    with pytest.raises(SystemExit):
        guard.main(["--mechanism", str(prefix), "--reference", str(path), "--output", str(path)])
    assert path.read_bytes() == before
    reference["records"].pop()
    runner.atomic_json(path, reference)
    output = tmp_path / "failed-audit.json"
    assert guard.main(["--mechanism", str(prefix), "--reference", str(path), "--output", str(output)]) == 2
    audit = json.loads(output.read_text())
    assert audit["error_type"] == "ValueError" and not audit["exact_match"]
