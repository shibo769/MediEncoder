"""Small CPU checks of comparison targets, pairing and interval accounting."""
from dataclasses import replace
import json

import numpy as np
import pytest

from mediencoder.comparison.dgp import ComparisonConfig, draw_mechanism, load_mechanism, sample_data, save_mechanism
from mediencoder.comparison.reporting import summarize, markdown_table, validate_result
from mediencoder.comparison.runner import run_replicate, merge_records, method_seed, scientific_fingerprints


def _row(rep, estimate=1.0, interval=(0.9, 1.1), status="ok", method="example"):
    effects = {k: estimate for k in ("NIE", "NDE", "TE")}
    return {"mechanism_hash": "mechanism", "dataset_hash": f"data-{rep}", "n": 100,
            "replicate": rep, "data_seed": rep, "method": method, "status": status,
            "effects": effects, "truth": {k: 1.0 for k in effects},
            "effect_ci": {k: interval for k in effects}, "inference_available": True,
            "B_requested": 200, "study_fingerprint": "study", "estimator_fingerprint": "estimator-" + method}


def test_polynomial_mechanism_has_exact_effect_truth_and_fixed_parameters(tmp_path):
    config = ComparisonConfig(p=6, q=5, bar_p=2, bar_q=3)
    params = draw_mechanism(config, 17)
    first = sample_data(params, 30, 70)
    second = sample_data(params, 50, 70)
    other = sample_data(params, 30, 71)
    for key in ("X", "M", "A", "Y"):
        np.testing.assert_allclose(first[key], second[key][:30], rtol=0, atol=1e-12)
    assert not np.array_equal(first["X"], other["X"])
    for data in (first, other):
        oracle = data["oracle"]
        np.testing.assert_allclose(oracle["mu11"] - oracle["mu10"], params.truth["NIE"])
        np.testing.assert_allclose(oracle["mu10"] - oracle["mu00"], params.truth["NDE"])
    np.testing.assert_allclose(params.truth["NIE"], params.arrays["alpha_M"] @ params.arrays["gamma"])
    save_mechanism(params, tmp_path / "mechanism")
    restored = load_mechanism(tmp_path / "mechanism")
    assert restored.mechanism_hash == params.mechanism_hash
    assert restored.truth == params.truth
    changed_dimensions = draw_mechanism(replace(config, p=8), 17)
    assert changed_dimensions.truth == params.truth
    metadata = json.loads((tmp_path / "mechanism.json").read_text())
    metadata["truth"]["NIE"] += 1
    (tmp_path / "mechanism.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="integrity"):
        load_mechanism(tmp_path / "mechanism")


def test_summary_uses_each_reported_interval_not_monte_carlo_sd():
    rows = [_row(0, 1.2, (1.15, 1.25)), _row(1, 1.2, (0.8, 1.6))]
    result = summarize(rows)[0]
    assert result["SD_error"] == 0
    assert result["Coverage"] == 0.5
    assert result["CI_Length"] == pytest.approx(0.45)
    assert result["B_requested"] == 200
    assert result["B_attempted"] == 2 and result["B_pending"] == 198
    rows[0]["effect_ci"] = {effect: (0.1, 2.3) for effect in rows[0]["effects"]}
    assert summarize(rows)[0]["Coverage"] == 1


def test_failures_and_absent_inference_are_never_hidden():
    rows = [_row(0), _row(1, status="failed")]
    result = summarize(rows)[0]
    assert result["B_failed"] == 1 and result["B_intervals"] == 1
    assert result["Coverage"] == 1 and result["Coverage_all_attempts"] == 0.5
    point = _row(0, interval=None, method="point")
    point["inference_available"] = False
    result = summarize([point])[0]
    assert result["Coverage"] is None and result["CI_Length"] is None
    assert result["Coverage_all_attempts"] is None
    assert "Unavailable" in markdown_table([result])
    with pytest.raises(ValueError, match="Duplicate"):
        summarize([rows[0], rows[0]])
    rows[0]["effects"]["NIE"] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        summarize(rows)


def test_replicate_pairs_data_and_records_failure_without_resampling():
    params = draw_mechanism(ComparisonConfig(p=4, q=3, bar_p=2, bar_q=2), 9)
    seen = []
    def fit(method, X, M, A, Y, *, seed, options, external_adapter):
        seen.append((X.copy(), M.copy(), A.copy(), Y.copy(), seed))
        X[:] = 123  # must not alter the next method's paired observations
        if method == "failed-method":
            raise RuntimeError("planned test failure")
        return {"effects": {e: 0.0 for e in ("NIE", "NDE", "TE")},
                "effect_ci": {}, "inference_available": False}
    rows = run_replicate(params, 30, 0, 99, ["ok-method", "failed-method"], fit=fit)
    assert [row["status"] for row in rows] == ["ok", "failed"]
    assert rows[0]["dataset_hash"] == rows[1]["dataset_hash"]
    for left, right in zip(seen[0][:4], seen[1][:4]):
        np.testing.assert_array_equal(left, right)
    assert seen[0][-1] != seen[1][-1]
    assert method_seed(99, "ok-method") == seen[0][-1]
    assert "planned test failure" in rows[1]["error"]


def test_merge_rejects_unpaired_and_duplicate_records(tmp_path):
    a, b = _row(0, method="a"), _row(0, method="b")
    path = tmp_path / "shard.jsonl"
    path.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n")
    assert len(merge_records([path])) == 2
    b["dataset_hash"] = "different"
    path.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n")
    with pytest.raises(ValueError, match="unpaired"):
        merge_records([path])


def test_merge_rejects_config_drift_and_missing_scientific_provenance(tmp_path):
    a, b = _row(0), _row(1)
    b["estimator_fingerprint"] = "different-epochs"
    path = tmp_path / "shards.jsonl"
    path.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n")
    with pytest.raises(ValueError, match="different scientific"):
        merge_records([path])
    del b["estimator_fingerprint"]
    path.write_text(json.dumps(b) + "\n")
    with pytest.raises(ValueError, match="lacks scientific"):
        merge_records([path])


def test_scientific_fingerprint_ignores_shard_bounds_but_tracks_training_budget():
    manifest = dict(mode="pilot", config={}, mechanism_hash="fixed", device="cpu", threads=1,
                    python="3.12", numpy="2", torch="2", source_sha256={"core": "same"},
                    options={"method": {"epochs": 10}}, replications_requested=200, n_grid=[100])
    original = scientific_fingerprints(manifest, "method")
    manifest["n_grid"] = [200]
    manifest["replications_requested"] = 10
    assert scientific_fingerprints(manifest, "method") == original
    manifest["options"]["method"]["epochs"] = 20
    changed = scientific_fingerprints(manifest, "method")
    assert changed[0] == original[0] and changed[1] != original[1]
    manifest["source_sha256"]["core"] = "edited"
    assert scientific_fingerprints(manifest, "method")[0] != original[0]


def test_invalid_intervals_and_zero_epoch_budgets_fail():
    with pytest.raises(ValueError, match="Invalid NIE"):
        validate_result({"effects": {e: 1 for e in ("NIE", "NDE", "TE")}, "effect_ci": {"NIE": [2, 1]}})
    from mediencoder.comparison.adapters import fit_method
    params = draw_mechanism(ComparisonConfig(p=4, q=3, bar_p=2, bar_q=2), 9)
    data = sample_data(params, 20, 2)
    with pytest.raises(ValueError, match="budget"):
        fit_method("nath-adapted", *(data[k] for k in ("X", "M", "A", "Y")), seed=1, options={"epochs": 0})


def test_ridge_baseline_produces_points_without_fabricated_intervals():
    from mediencoder.comparison.adapters import fit_method
    params = draw_mechanism(ComparisonConfig(p=5, q=4, bar_p=2, bar_q=2), 9)
    data = sample_data(params, 40, 112)
    result = fit_method("lsem-ridge", *(data[k] for k in ("X", "M", "A", "Y")), seed=3,
                        options={"tilde_p": 2, "tilde_q": 2})
    assert np.isfinite(list(result["effects"].values())).all()
    assert result["effects"]["TE"] == pytest.approx(result["effects"]["NIE"] + result["effects"]["NDE"])
    assert not result["inference_available"]
    assert all(value is None for value in result["effect_ci"].values())


def test_declared_canonical_budget_is_applied_and_restored_even_on_failure():
    import mediencoder.estimation as estimation
    from mediencoder.comparison.adapters import _canonical_budget
    before, hidden = dict(estimation.SHARED_TRAIN_CFG), estimation.SHARED_HIDDEN
    with pytest.raises(RuntimeError, match="test"):
        with _canonical_budget(1, 1, (4,)):
            assert estimation.SHARED_TRAIN_CFG["epochs"] == 1
            assert estimation.SHARED_TRAIN_CFG["patience"] == 1
            assert estimation.SHARED_HIDDEN == (4,)
            raise RuntimeError("test")
    assert estimation.SHARED_TRAIN_CFG == before
    assert estimation.SHARED_HIDDEN == hidden
