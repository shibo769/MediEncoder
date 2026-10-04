"""Shared regularization and fixed-mechanism reuse, without fitting a model."""
from copy import deepcopy
from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from mediencoder.simulation import dgp, runner


@pytest.fixture(scope="module")
def mechanism():
    return dgp.draw_parameters(dgp.DGPConfig(p=5, q=4, bar_p=2, bar_q=2, pilot_size=64), 910000)


@pytest.fixture
def mechanism_source(mechanism, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    dgp.save_mechanism(mechanism, source / "mechanism")
    return source


def mechanism_config(mechanism):
    return dict(dgp=asdict(mechanism.config), mechanism_seed=mechanism.parameter_seed)


def test_regularization_cli_rejects_invalid_values():
    for value in ("-0.01", "nan", "inf", "-inf"):
        with pytest.raises(SystemExit):
            runner.parse_args(["--output-dir", "unused", "--weight-decay=" + value])
        with pytest.raises(ValueError, match="finite and nonnegative"):
            runner.validate_weight_decay(value)
    assert runner.parse_args(["--output-dir", "unused"]).weight_decay == 0.0


def test_regularization_reaches_shared_configs_and_survives_worker_reset():
    runner.configure_environment("cpu")
    from mediencoder.estimation import SHARED_TRAIN_CFG, _merge_method_cfg, _shared_cfg
    original = deepcopy(SHARED_TRAIN_CFG)
    try:
        args = runner.parse_args(["--output-dir", "unused", "--weight-decay", ".01",
                                  "--reps", "200", "--target-reps", "100",
                                  "--arms", "mediencoder,mediencoder_l3zero"])
        config = runner.make_config(args)
        assert config["B_requested"] == 200
        assert len(config["lambda_grid"]) == 36 and len(config["lambda_grid_zero"]) == 9
        # A spawned process starts with the module's zero-decay defaults.
        SHARED_TRAIN_CFG["weight_decay"] = 0.0
        runner.configure_shared_training(config["training"])
        runtime = runner.runtime_training_configuration(config["training"])
        for name in ("nn_cfg", "ae_cfg", "me_cfg"):
            assert config["training"][name]["weight_decay"] == .01
            effective = _merge_method_cfg(runtime[name], _shared_cfg(), method=name)
            assert effective["weight_decay"] == .01
            assert effective["epochs"] == 300
        # Tuning auxiliary regressions receive nn_cfg; all representations use
        # the same shared override, so no opt-out or silent budget drift occurs.
        bad = deepcopy(config["training"])
        bad["nn_cfg"]["weight_decay"] = 0.0
        with pytest.raises(ValueError, match="same shared weight decay"):
            runner.configure_shared_training(bad)
        default = runner.make_config(runner.parse_args(["--output-dir", "unused"]))
        assert all(default["training"][name]["weight_decay"] == 0.0
                   for name in ("nn_cfg", "ae_cfg", "me_cfg"))
    finally:
        SHARED_TRAIN_CFG.clear()
        SHARED_TRAIN_CFG.update(original)


def test_regularization_changes_identity_but_preserves_data_and_training_seeds():
    runner.configure_environment("cpu")
    from mediencoder.estimation import SHARED_TRAIN_CFG
    original = deepcopy(SHARED_TRAIN_CFG)
    try:
        common = ["--output-dir", "unused", "--reps", "200",
                  "--arms", "mediencoder,mediencoder_l3zero"]
        zero = runner.make_config(runner.parse_args(common))
        regularized = runner.make_config(runner.parse_args(common + ["--weight-decay", ".01"]))
        restored = deepcopy(regularized)
        for name in ("nn_cfg", "ae_cfg", "me_cfg"):
            restored["training"][name]["weight_decay"] = 0.0
        assert restored == zero
        assert runner.digest(zero) != runner.digest(regularized)
        old_tasks = list(runner.build_tasks(runner.execution_config(zero, 50)))
        new_tasks = list(runner.build_tasks(runner.execution_config(regularized, 100)))
        assert old_tasks == new_tasks[:len(old_tasks)]
        imported_args = runner.parse_args(common + ["--mechanism-from", "a-different-local-path"])
        assert runner.make_config(imported_args) == zero
    finally:
        SHARED_TRAIN_CFG.clear()
        SHARED_TRAIN_CFG.update(original)


@pytest.mark.parametrize("suffix", (None, "", ".json", ".npz"))
def test_import_roundtrip_and_resume_do_not_redraw_or_reintegrate(
        mechanism, mechanism_source, tmp_path, monkeypatch, suffix):
    source_arg = mechanism_source if suffix is None else mechanism_source / ("mechanism" + suffix)
    output = tmp_path / "new-run"
    output.mkdir()
    source_hashes = runner.mechanism_artifact_hashes(mechanism_source / "mechanism")
    monkeypatch.setattr(dgp, "draw_parameters", lambda *a, **kw: pytest.fail("Mechanism was redrawn"))
    monkeypatch.setattr(dgp, "population_truth", lambda *a, **kw: pytest.fail("Truth was recomputed"))
    loaded, provenance = runner.prepare_mechanism(
        output, mechanism_config(mechanism), mechanism_from=source_arg)
    assert loaded.mechanism_hash == mechanism.mechanism_hash
    assert loaded.truth == mechanism.truth
    assert provenance["kind"] == "imported"
    assert provenance["source_artifact_sha256"] == source_hashes
    assert provenance["saved_artifact_sha256"] == source_hashes
    assert runner.mechanism_artifact_hashes(mechanism_source / "mechanism") == source_hashes
    existing = {"mechanism_hash": loaded.mechanism_hash, "mechanism_provenance": provenance}
    for resume_source in (None, source_arg):
        resumed, _ = runner.prepare_mechanism(
            output, mechanism_config(mechanism), existing, resume_source)
        assert resumed.mechanism_hash == mechanism.mechanism_hash
        for name, values in mechanism.arrays.items():
            np.testing.assert_array_equal(resumed.arrays[name], values)
    original_data = dgp.sample_data(mechanism, 12, 987)
    imported_data = dgp.sample_data(loaded, 12, 987)
    for key in ("X", "M", "A", "Y"):
        np.testing.assert_array_equal(original_data[key], imported_data[key])


def test_import_rejects_dgp_or_seed_mismatch_before_writing(
        mechanism, mechanism_source, tmp_path):
    output = tmp_path / "new-run"
    output.mkdir()
    config = mechanism_config(mechanism)
    bad_config = deepcopy(config)
    bad_config["dgp"]["p"] += 1
    with pytest.raises(ValueError, match="DGP configuration differs"):
        runner.prepare_mechanism(output, bad_config, mechanism_from=mechanism_source)
    with pytest.raises(ValueError, match="parameter seed differs"):
        runner.prepare_mechanism(output, dict(config, mechanism_seed=17), mechanism_from=mechanism_source)
    assert list(output.iterdir()) == []


def test_new_run_never_overwrites_existing_unmanifested_files(
        mechanism, mechanism_source, tmp_path):
    output = tmp_path / "new-run"
    output.mkdir()
    existing = output / "mechanism.npz"
    existing.write_bytes(b"do not replace this")
    with pytest.raises(ValueError, match="empty output directory"):
        runner.prepare_mechanism(output, mechanism_config(mechanism), mechanism_from=mechanism_source)
    assert existing.read_bytes() == b"do not replace this"


def test_monitor_startup_files_are_allowed_but_scientific_files_are_not(
        mechanism, mechanism_source, tmp_path):
    output = tmp_path / "monitored-run"
    output.mkdir()
    allowed = (".runner.lock", "run.log", "progress.html", "supervisor.json",
               "progress.html.tmp", "supervisor.json.tmp")
    for name in allowed:
        (output / name).write_bytes(b"monitor owns this file")
    loaded, _ = runner.prepare_mechanism(
        output, mechanism_config(mechanism), mechanism_from=mechanism_source)
    assert loaded.mechanism_hash == mechanism.mechanism_hash
    for name in allowed:
        assert (output / name).read_bytes() == b"monitor owns this file"
    for name in ("status.json", "summary.csv", "tasks", "progress.html"):
        blocked = tmp_path / ("blocked-" + name)
        blocked.mkdir()
        if name in ("tasks", "progress.html"):
            (blocked / name).mkdir()  # Allowed names must be regular files.
        else:
            (blocked / name).write_text("must be preserved")
        with pytest.raises(ValueError, match="empty output directory"):
            runner.prepare_mechanism(blocked, mechanism_config(mechanism), mechanism_from=mechanism_source)


def test_custom_named_bundle_is_rebased_without_changing_numerical_mechanism(
        mechanism, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    dgp.save_mechanism(mechanism, source / "custom-bundle")
    output = tmp_path / "new-run"
    output.mkdir()
    loaded, _ = runner.prepare_mechanism(
        output, mechanism_config(mechanism), mechanism_from=source / "custom-bundle.json")
    assert loaded.mechanism_hash == mechanism.mechanism_hash
    assert dgp.load_mechanism(output / "mechanism").mechanism_hash == mechanism.mechanism_hash


def test_resume_rejects_a_different_import_or_changed_saved_artifacts(
        mechanism, mechanism_source, tmp_path):
    output = tmp_path / "new-run"
    output.mkdir()
    _, provenance = runner.prepare_mechanism(output, mechanism_config(mechanism), mechanism_from=mechanism_source)
    manifest = {"mechanism_hash": mechanism.mechanism_hash, "mechanism_provenance": provenance}
    arrays = {key: value.copy() for key, value in mechanism.arrays.items()}
    arrays["Lambda_X_coef"] *= 2
    changed = replace(mechanism, arrays=arrays)
    other = tmp_path / "other"
    other.mkdir()
    dgp.save_mechanism(changed, other / "mechanism")
    with pytest.raises(ValueError, match="differs from the original imported mechanism"):
        runner.prepare_mechanism(output, mechanism_config(mechanism), manifest, other)
    # Even an otherwise harmless byte change is detected once a bundle is saved.
    saved_json = output / "mechanism.json"
    saved_json.write_text(saved_json.read_text() + "\n")
    with pytest.raises(ValueError, match="artifacts changed"):
        runner.prepare_mechanism(output, mechanism_config(mechanism), manifest)


def test_summarize_only_import_records_provenance_and_resumes_without_source(
        mechanism, mechanism_source, tmp_path, monkeypatch):
    config = dict(mechanism_config(mechanism), n_values=[100], methods=["mediencoder"],
                  B_requested=200, seed_base=880000, run_kind="FORMAL", device="cpu")
    monkeypatch.setattr(runner, "make_config", lambda args: config)
    monkeypatch.setattr(runner, "environment_identity", lambda device: {"test": "no fitting"})
    monkeypatch.setattr(runner, "collect_code_hashes", lambda: {"runner.py": "test-source-hash"})
    monkeypatch.setattr(dgp, "draw_parameters", lambda *a, **kw: pytest.fail("Mechanism was redrawn"))
    output = tmp_path / "run"
    common = ["--output-dir", str(output), "--reps", "200", "--target-reps", "100", "--summarize-only"]
    assert runner.main(common + ["--mechanism-from", str(mechanism_source)]) == 0
    manifest_path = output / "manifest.json"
    initial_bytes = manifest_path.read_bytes()
    manifest = json.loads(initial_bytes)
    assert manifest["mechanism_hash"] == mechanism.mechanism_hash
    assert manifest["mechanism_provenance"]["kind"] == "imported"
    assert manifest["truth"]["value"] == mechanism.truth.value
    assert runner.main(common) == 0
    assert manifest_path.read_bytes() == initial_bytes
    assert not (output / "tasks").exists()
