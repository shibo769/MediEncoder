"""Default cubic maps and explicit legacy compatibility; no model training."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pytest

from mediencoder.simulation import dgp, runner


@pytest.fixture(scope="module")
def cubic():
    return dgp.draw_parameters(dgp.DGPConfig(p=7, q=6, bar_p=2, bar_q=2,
                                           sigma_eps_X=0, sigma_eps_M=0), 1234)


def test_default_measurements_are_actual_additive_cubics(cubic):
    assert dgp.DGPConfig().loading_family == "polynomial"
    assert cubic.metadata["polynomial_degree"] == 3
    assert "r_list" not in cubic.arrays and "support_bounds" not in cubic.metadata
    data = dgp.sample_data(cubic, 13, 19)
    for observed, latent, coeff in (("X", "f_X", "polynomial_X"),
                                    ("M", "f_M", "polynomial_M")):
        f, c = data["oracle"][latent], cubic.arrays[coeff]
        expected = np.array([[sum(c[k, j, r-1] * f[i, j]**r
                                 for j in range(f.shape[1]) for r in (1, 2, 3))
                              for k in range(c.shape[0])] for i in range(len(f))])
        np.testing.assert_allclose(data[observed], expected, rtol=1e-13, atol=1e-13)


def test_changing_loading_family_preserves_structural_draws_truth_and_latents(cubic):
    haar = dgp.draw_parameters(replace(cubic.config, loading_family="haar"), 1234)
    for name in ("alpha", "delta_0", "delta_1", "Sigma_U", "beta_0", "beta_1",
                 "gamma_0", "gamma_1", "kappa_0", "kappa_1"):
        np.testing.assert_array_equal(cubic.arrays[name], haar.arrays[name])
    assert cubic.truth == haar.truth
    assert cubic.config.delta1_contrast == haar.config.delta1_contrast == 2.0
    poly_data, haar_data = (dgp.sample_data(m, 21, 141) for m in (cubic, haar))
    for name in ("A", "Y"):
        np.testing.assert_array_equal(poly_data[name], haar_data[name])
    for name in poly_data["oracle"]:
        np.testing.assert_array_equal(poly_data["oracle"][name], haar_data["oracle"][name])
    assert not np.array_equal(poly_data["X"], haar_data["X"])


def test_separate_loading_streams_and_data_are_reproducible(cubic):
    again = dgp.draw_parameters(cubic.config, 1234)
    larger = dgp.draw_parameters(replace(cubic.config, p=9, q=8), 1234)
    assert again.mechanism_hash == cubic.mechanism_hash
    for name in ("polynomial_X", "polynomial_M"):
        np.testing.assert_array_equal(cubic.arrays[name], larger.arrays[name][:len(cubic.arrays[name])])
    for key, array in dgp.sample_data(cubic, 11, 204)["oracle"].items():
        np.testing.assert_array_equal(array, dgp.sample_data(again, 11, 204)["oracle"][key])
    for key in ("X", "M", "A", "Y"):
        np.testing.assert_array_equal(dgp.sample_data(cubic, 11, 204)[key],
                                      dgp.sample_data(again, 11, 204)[key])


def test_cubic_schema_roundtrip_records_family_and_detects_tampering(cubic, tmp_path):
    prefix = tmp_path / "mechanism"
    dgp.save_mechanism(cubic, prefix)
    loaded = dgp.load_mechanism(prefix)
    payload = json.loads(prefix.with_suffix(".json").read_text())
    assert payload["schema_version"] == 2
    assert payload["config"]["loading_family"] == "polynomial"
    assert loaded.mechanism_hash == cubic.mechanism_hash
    for name in ("X", "M", "A", "Y"):
        np.testing.assert_array_equal(dgp.sample_data(loaded, 9, 52)[name],
                                      dgp.sample_data(cubic, 9, 52)[name])
    payload["config"]["loading_family"] = "haar"
    prefix.with_suffix(".json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        dgp.load_mechanism(prefix)


def test_schema_one_haar_artifact_preserves_original_hash_and_sampling(tmp_path):
    prefix = Path(__file__).parents[1] / "experiments/mechanisms/wavelet_seed910000/mechanism"
    payload = json.loads(prefix.with_suffix(".json").read_text())
    old = dgp.load_mechanism(prefix)
    assert old.schema_version == 1 and old.config.loading_family == "haar"
    assert old.mechanism_hash == payload["mechanism_hash"]
    data = dgp.sample_data(old, 7, 444)
    a, c = old.arrays, old.config
    noise_seed = dgp._stream_seeds(444, 0x4D454153, 2)[0]
    expected = dgp._loading(data["oracle"]["f_X"], a["Lambda_X_coef"], a["r_list"], a["s_list"])
    expected += np.random.default_rng(noise_seed).normal(0, c.sigma_eps_X, (7, c.p))
    np.testing.assert_array_equal(data["X"], expected)
    dgp.save_mechanism(old, tmp_path / "copied")
    saved = json.loads((tmp_path / "copied.json").read_text())
    assert saved["schema_version"] == 1 and "loading_family" not in saved["config"]
    assert dgp.load_mechanism(tmp_path / "copied").mechanism_hash == payload["mechanism_hash"]


def test_schema_one_explicit_polynomial_metadata_stays_polynomial(cubic, tmp_path):
    # The isolated B20 snapshot predates the config field but records this marker.
    old = replace(cubic, schema_version=1,
                  metadata=dict(exploration_measurement="polynomial_degree3", polynomial_degree=3))
    dgp.save_mechanism(old, tmp_path / "old-cubic")
    loaded = dgp.load_mechanism(tmp_path / "old-cubic")
    assert loaded.config.loading_family == "polynomial"
    assert loaded.mechanism_hash == old.mechanism_hash
    np.testing.assert_array_equal(dgp.sample_data(loaded, 9, 52)["M"],
                                  dgp.sample_data(old, 9, 52)["M"])


def test_cli_default_and_explicit_legacy_import_are_unambiguous(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "lambda_grids", lambda: ([[.2, .5, .3]], [[.5, .5, 0]]))
    monkeypatch.setattr(runner, "training_configuration", lambda *args: {})
    default = runner.make_config(runner.parse_args(["--output-dir", str(tmp_path)]))
    legacy = runner.make_config(runner.parse_args(["--output-dir", str(tmp_path),
                                                  "--loading-family", "haar"]))
    assert default["dgp"]["loading_family"] == "polynomial"
    assert default["loss_normalization"] == "none"
    source = Path(__file__).parents[1] / "experiments/mechanisms/wavelet_seed910000"
    with pytest.raises(ValueError, match="DGP configuration differs"):
        runner.prepare_mechanism(tmp_path, default, mechanism_from=source)
    assert list(tmp_path.iterdir()) == []
    imported, _ = runner.prepare_mechanism(tmp_path, legacy, mechanism_from=source)
    assert imported.config.loading_family == "haar"
    assert asdict(imported.config) == legacy["dgp"]


def test_invalid_loading_family_is_rejected():
    with pytest.raises(ValueError, match="loading_family"):
        dgp.draw_parameters(dgp.DGPConfig(loading_family="unknown"))
    with pytest.raises(SystemExit):
        runner.parse_args(["--output-dir", "unused", "--loading-family", "unknown"])
