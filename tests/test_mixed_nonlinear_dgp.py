"""Independent-pilot injective nonlinear measurement maps; no model training."""
from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from mediencoder.simulation import dgp


@pytest.fixture(scope="module")
def mechanisms():
    base = dgp.DGPConfig(p=7, q=6, bar_p=2, bar_q=2, pilot_size=128,
                         sigma_eps_X=0, sigma_eps_M=0)
    return {(family, strength): dgp.draw_parameters(
        replace(base, loading_family=family, loading_strength=strength), 1234)
        for family in dgp.NONLINEAR_LOADING_FAMILIES for strength in (2.0, 6.0)}


@pytest.mark.parametrize("family", dgp.NONLINEAR_LOADING_FAMILIES)
def test_nonlinear_maps_preserve_causal_mechanism_and_paired_latent_noise_streams(mechanisms, family):
    mixed = mechanisms[family, 2.0]
    cubic = dgp.draw_parameters(replace(mixed.config, loading_family="polynomial"), 1234)
    for name in ("alpha", "delta_0", "delta_1", "Sigma_U", "beta_0", "beta_1",
                 "gamma_0", "gamma_1", "kappa_0", "kappa_1"):
        np.testing.assert_array_equal(mixed.arrays[name], cubic.arrays[name])
    assert mixed.truth == cubic.truth
    left, right = (dgp.sample_data(mechanism, 11, 44) for mechanism in (mixed, cubic))
    for name in ("A", "Y"):
        np.testing.assert_array_equal(left[name], right[name])
    for name in left["oracle"]:
        np.testing.assert_array_equal(left["oracle"][name], right["oracle"][name])
    for name in ("X", "M"):
        assert not np.array_equal(left[name], right[name])


@pytest.mark.parametrize("family", dgp.NONLINEAR_LOADING_FAMILIES)
def test_matrices_are_full_rank_reproducible_and_identical_across_strengths(mechanisms, family):
    weak, strong = (mechanisms[family, strength] for strength in (2.0, 6.0))
    assert weak.mechanism_hash != strong.mechanism_hash
    again = dgp.draw_parameters(weak.config, weak.parameter_seed)
    assert again.mechanism_hash == weak.mechanism_hash
    for label, observed_dim, latent_dim in (("X", weak.config.p, weak.config.bar_p),
                                          ("M", weak.config.q, weak.config.bar_q)):
        for suffix in ("weight1", "bias1", "weight2", "bias2", "input_mean", "input_scale"):
            key = f"mixed_{label}_{suffix}"
            if key in weak.arrays:
                np.testing.assert_array_equal(weak.arrays[key], strong.arrays[key])
        for suffix in ("weight1", "weight2"):
            key = f"mixed_{label}_{suffix}"
            if key not in weak.arrays:
                continue
            weight = weak.arrays[key]
            expected = (weight.shape[0] / weight.shape[1]) * np.eye(weight.shape[1])
            np.testing.assert_allclose(weight.T @ weight, expected, rtol=1e-12, atol=1e-12)
        assert weak.arrays[f"mixed_{label}_weight1"].shape[1] == latent_dim
        final = "weight2" if family == "deep_tanh" else "weight1"
        assert weak.arrays[f"mixed_{label}_{final}"].shape[0] == observed_dim
    for key in ("X", "M", "A", "Y"):
        np.testing.assert_array_equal(dgp.sample_data(weak, 11, 55)[key],
                                      dgp.sample_data(again, 11, 55)[key])


@pytest.mark.parametrize("family", dgp.NONLINEAR_LOADING_FAMILIES)
@pytest.mark.parametrize("strength", (2.0, 6.0))
def test_independent_pilot_matches_cubic_signal_mean_variance(mechanisms, family, strength):
    mechanism = mechanisms[family, strength]
    pilot = dgp._sample_latent(mechanism, mechanism.config.pilot_size,
                               mechanism.metadata["pilot_seed"])
    for label in ("X", "M"):
        factors = pilot[f"f_{label}"]
        reference = dgp._polynomial_loading(factors, mechanism.arrays[f"polynomial_reference_{label}"])
        actual = dgp._mixed_loading(factors, mechanism.arrays, label, strength)
        np.testing.assert_allclose(actual.mean(axis=0), reference.mean(axis=0), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual.std(axis=0), reference.std(axis=0), rtol=1e-12, atol=1e-12)
        assert np.all(mechanism.arrays[f"mixed_{label}_output_scale"] > 0)
    assert mechanism.metadata["calibration_rule"].startswith("independent-fixed-latent-pilot")
    before = mechanism.mechanism_hash
    dgp.sample_data(mechanism, 29, 888)
    assert mechanism.mechanism_hash == before


@pytest.mark.parametrize("family", dgp.NONLINEAR_LOADING_FAMILIES)
def test_mixed_maps_roundtrip_preserves_strength_hash_and_data(mechanisms, family, tmp_path):
    mechanism = mechanisms[family, 6.0]
    prefix = tmp_path / family
    dgp.save_mechanism(mechanism, prefix)
    loaded = dgp.load_mechanism(prefix)
    payload = json.loads(prefix.with_suffix(".json").read_text())
    assert payload["config"]["loading_strength"] == 6.0
    assert loaded.mechanism_hash == mechanism.mechanism_hash
    for name in ("X", "M", "A", "Y"):
        np.testing.assert_array_equal(dgp.sample_data(mechanism, 13, 67)[name],
                                      dgp.sample_data(loaded, 13, 67)[name])
    payload["config"]["loading_strength"] = 2.0
    prefix.with_suffix(".json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        dgp.load_mechanism(prefix)


@pytest.mark.parametrize("family", ("polynomial", "haar"))
def test_legacy_serialized_configs_and_draws_ignore_new_inactive_strength(family):
    config = dgp.DGPConfig(p=7, q=6, bar_p=2, bar_q=2, pilot_size=64, loading_family=family)
    expected = asdict(config)
    expected.pop("loading_strength")
    expected.pop("loading_frequency")
    assert config.to_dict() == expected
    original = dgp.draw_parameters(config, 1234)
    alternative = dgp.draw_parameters(replace(config, loading_strength=6.0, loading_frequency=12.0), 1234)
    assert original.mechanism_hash == alternative.mechanism_hash
    assert original.serialized_config() == expected
    for schema_version in (1, 2, 3):
        serialized = replace(original, schema_version=schema_version).serialized_config()
        assert "loading_strength" not in serialized
        assert "loading_frequency" not in serialized
        assert ("alpha_upper" in serialized) == (schema_version >= 3)
        assert ("loading_family" in serialized) == (schema_version >= 2)
    for key in original.arrays:
        np.testing.assert_array_equal(original.arrays[key], alternative.arrays[key])


@pytest.mark.parametrize("strength", (0.0, 2.0, 6.0))
def test_scalar_map_is_strictly_monotone_with_derivative_bounded_below(strength):
    values = np.linspace(-20, 20, 401)
    actual = dgp._mixed_nonlinearity(values, strength)
    assert np.all(np.diff(actual) >= np.diff(values) - 1e-13)
    if strength == 0:
        np.testing.assert_array_equal(actual, values)


@pytest.mark.parametrize("strength", (-1, float("nan"), float("inf"), True))
def test_invalid_strength_is_rejected(strength):
    with pytest.raises(ValueError, match="loading_strength"):
        dgp.DGPConfig(loading_family="mixed_tanh", loading_strength=strength).validate()
    with pytest.raises(ValueError, match="loading_strength"):
        dgp._mixed_nonlinearity(np.ones((2, 3)), strength)


def test_injective_map_dimension_pilot_and_reference_variance_guards():
    for changes in (dict(p=1), dict(q=1), dict(pilot_size=1),
                    dict(coef_scale_X=0), dict(coef_scale_M=0)):
        config = dgp.DGPConfig(bar_p=2, bar_q=2, loading_family="deep_tanh", **changes)
        with pytest.raises(ValueError):
            config.validate()
    with pytest.raises(ValueError, match="Degenerate"):
        dgp._positive_pilot_scale(np.ones((3, 2)), "test")


@pytest.mark.parametrize("family", dgp.NONLINEAR_LOADING_FAMILIES)
def test_frequency_one_preserves_original_formula_payload_and_map(mechanisms, family):
    mechanism = mechanisms[family, 2.0]
    assert mechanism.config.loading_frequency == 1.0
    assert "loading_frequency" not in mechanism.config.to_dict()
    assert "loading_frequency" not in mechanism.metadata
    assert mechanism.metadata["nonlinearity"] == "h_c(t) = t + c*tanh(t)"
    factors = np.array([[-.2, .4], [.9, -.1]])
    for label in ("X", "M"):
        np.testing.assert_array_equal(
            dgp._mixed_loading(factors, mechanism.arrays, label, 2.0),
            dgp._mixed_loading(factors, mechanism.arrays, label, 2.0, frequency=1.0))
    values = np.linspace(-4, 4, 101)
    np.testing.assert_array_equal(dgp._mixed_nonlinearity(values, 2.0),
                                  values + 2.0 * np.tanh(values))


@pytest.mark.parametrize("family", dgp.NONLINEAR_LOADING_FAMILIES)
def test_frequency_is_recorded_calibrated_and_reproducible(mechanisms, family, tmp_path):
    original = mechanisms[family, 2.0]
    altered = dgp.draw_parameters(replace(original.config, loading_frequency=4.0),
                                  original.parameter_seed)
    assert altered.config.to_dict()["loading_frequency"] == 4.0
    assert altered.metadata["loading_frequency"] == 4.0
    assert altered.mechanism_hash != original.mechanism_hash
    assert altered.truth == original.truth
    pilot = dgp._sample_latent(altered, altered.config.pilot_size, altered.metadata["pilot_seed"])
    for label in ("X", "M"):
        for suffix in ("weight1", "bias1", "weight2", "bias2", "input_mean", "input_scale"):
            key = f"mixed_{label}_{suffix}"
            if key in original.arrays:
                np.testing.assert_array_equal(original.arrays[key], altered.arrays[key])
        actual = dgp._mixed_loading(pilot[f"f_{label}"], altered.arrays, label, 2.0, frequency=4.0)
        reference = dgp._polynomial_loading(pilot[f"f_{label}"], altered.arrays[f"polynomial_reference_{label}"])
        np.testing.assert_allclose(actual.mean(axis=0), reference.mean(axis=0), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual.std(axis=0), reference.std(axis=0), rtol=1e-12, atol=1e-12)
    prefix = tmp_path / "frequency"
    dgp.save_mechanism(altered, prefix)
    restored = dgp.load_mechanism(prefix)
    assert restored.mechanism_hash == altered.mechanism_hash
    for name in ("X", "M", "A", "Y"):
        np.testing.assert_array_equal(dgp.sample_data(altered, 13, 77)[name],
                                      dgp.sample_data(restored, 13, 77)[name])


@pytest.mark.parametrize("frequency", (1.0, 4.0, 12.0))
def test_frequency_changes_curvature_without_losing_scalar_injectivity(frequency):
    values = np.linspace(-5, 5, 401)
    actual = dgp._mixed_nonlinearity(values, 6.0, frequency)
    assert np.all(np.diff(actual) >= np.diff(values) - 1e-13)
    np.testing.assert_array_equal(dgp._mixed_nonlinearity(values, 0.0, frequency), values)


@pytest.mark.parametrize("frequency", (0, -1, float("nan"), float("inf"), True))
def test_invalid_frequency_is_rejected(frequency):
    with pytest.raises(ValueError, match="loading_frequency"):
        dgp.DGPConfig(loading_family="mixed_tanh", loading_frequency=frequency).validate()
    with pytest.raises(ValueError, match="loading_frequency"):
        dgp._mixed_nonlinearity(np.ones((2, 3)), 2.0, frequency)
