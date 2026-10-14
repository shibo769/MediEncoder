"""Assignment-link and covariance alternatives; numerical tests run on SCC only."""
from dataclasses import asdict, replace
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import expit

from mediencoder.simulation import dgp


def small_config(**changes):
    values = dict(p=7, q=6, bar_p=2, bar_q=2, pilot_size=128)
    values.update(changes)
    return dgp.DGPConfig(**values)


@pytest.fixture(scope="module")
def frozen_v3_dgp():
    baseline_path = os.environ.get("MEDIENC_V4_BASELINE_DGP")
    if not baseline_path:
        pytest.skip("SCC compatibility test: set MEDIENC_V4_BASELINE_DGP to frozen V3/612 dgp.py")
    path = Path(baseline_path)
    assert path.is_file() and path.resolve() != Path(dgp.__file__).resolve()
    name = "frozen_v3_assignment_dgp"
    spec = importlib.util.spec_from_file_location(name, path)
    baseline = importlib.util.module_from_spec(spec)
    # dataclasses resolves postponed annotations through the module registry.
    sys.modules[name] = baseline
    spec.loader.exec_module(baseline)
    yield baseline
    sys.modules.pop(name, None)


@pytest.mark.parametrize("family", dgp.LOADING_FAMILIES)
def test_default_linear_matches_frozen_v3_arrays_metadata_hash_and_data(frozen_v3_dgp, family):
    config = small_config(loading_family=family)
    assert config.treatment_link == "linear" and "treatment_link" not in config.to_dict()
    original = frozen_v3_dgp.draw_parameters(config.to_dict(), parameter_seed=1234)
    current = dgp.draw_parameters(config, parameter_seed=1234)
    explicit = dgp.draw_parameters(replace(config, treatment_link="linear"), parameter_seed=1234)
    assert current.serialized_config() == original.serialized_config()
    assert current.metadata == original.metadata
    assert "treatment_assignment" not in current.metadata
    assert current.truth.to_dict() == original.truth.to_dict()
    assert current.mechanism_hash == original.mechanism_hash == explicit.mechanism_hash
    assert current.arrays.keys() == original.arrays.keys()
    for key in current.arrays:
        np.testing.assert_array_equal(current.arrays[key], original.arrays[key])
    reference, candidate = (module.sample_data(mechanism, 31, 81)
                            for module, mechanism in ((frozen_v3_dgp, original), (dgp, current)))
    for key in ("X", "M", "A", "Y"):
        np.testing.assert_array_equal(reference[key], candidate[key])
    for key in reference["oracle"]:
        np.testing.assert_array_equal(reference["oracle"][key], candidate["oracle"][key])


@pytest.mark.parametrize("family", dgp.LOADING_FAMILIES)
def test_linear_config_omission_is_explicit_and_roundtrips_without_schema_changes(family, tmp_path):
    config = small_config(loading_family=family)
    expected = asdict(config)
    expected.pop("treatment_link")
    if family not in dgp.NONLINEAR_LOADING_FAMILIES:
        expected.pop("loading_strength")
    expected.pop("loading_frequency")
    assert config.to_dict() == expected
    mechanism = dgp.draw_parameters(config, 1234)
    prefix = tmp_path / family
    dgp.save_mechanism(mechanism, prefix)
    payload = json.loads(prefix.with_suffix(".json").read_text())
    assert "treatment_link" not in payload["config"]
    restored = dgp.load_mechanism(prefix)
    assert restored.config.treatment_link == "linear"
    assert restored.mechanism_hash == mechanism.mechanism_hash


@pytest.mark.parametrize("mu", (0., .4, -1.5))
def test_quadratic_propensity_uses_population_centering_and_exact_support_bounds(mu):
    mechanism = dgp.draw_parameters(small_config(mu=mu, treatment_link="quadratic_centered"), 1234)
    data = dgp.sample_data(mechanism, 257, 77)
    x, alpha = data["oracle"]["f_X"], mechanism.arrays["alpha"]
    population_m2 = mu ** 2 + 1 / 3
    expected = expit((x ** 2 - population_m2) @ alpha)
    np.testing.assert_array_equal(data["oracle"]["e_true_vals"], expected)
    square_min = max(abs(mu) - 1, 0.) ** 2
    square_max = (abs(mu) + 1) ** 2
    lower = expit((square_min - population_m2) * alpha.sum())
    upper = expit((square_max - population_m2) * alpha.sum())
    assert 0. < lower < upper < 1.
    assert np.all(expected >= lower) and np.all(expected <= upper)
    metadata = mechanism.metadata["treatment_assignment"]
    assert metadata["link"] == "quadratic_centered"
    assert metadata["population_second_moment"] == population_m2
    assert metadata["centering_rule"] == "population-second-moment-independent-of-sampled-data"
    singleton = dgp.sample_data(mechanism, 1, 77)
    np.testing.assert_array_equal(singleton["oracle"]["f_X"][0], x[0])
    np.testing.assert_allclose(singleton["oracle"]["e_true_vals"][0], expected[0], rtol=0, atol=1e-15)


def test_assignment_links_keep_parameter_and_random_draw_streams_paired():
    linear = dgp.draw_parameters(small_config(), 1234)
    quadratic = dgp.draw_parameters(replace(linear.config, treatment_link="quadratic_centered"), 1234)
    assert linear.mechanism_hash != quadratic.mechanism_hash
    for key in linear.arrays:
        np.testing.assert_array_equal(linear.arrays[key], quadratic.arrays[key])
    old, new = (dgp.sample_data(mechanism, 512, 80) for mechanism in (linear, quadratic))
    for key in ("f_X", "mu00_true_vals", "mu10_true_vals", "mu11_true_vals"):
        np.testing.assert_array_equal(old["oracle"][key], new["oracle"][key])
    np.testing.assert_array_equal(old["X"], new["X"])
    uniform_seed = dgp._stream_seeds(80, 0x4C415445, 6)[1]
    uniforms = np.random.default_rng(uniform_seed).random(512)
    for data in (old, new):
        np.testing.assert_array_equal(data["A"], (uniforms < data["oracle"]["e_true_vals"]).astype(np.int64))
    same_arm = old["A"] == new["A"]
    assert same_arm.any() and not same_arm.all()
    for key in ("Y", "M"):
        np.testing.assert_array_equal(old[key][same_arm], new[key][same_arm])
    np.testing.assert_array_equal(old["oracle"]["f_M"][same_arm], new["oracle"]["f_M"][same_arm])


@pytest.mark.parametrize("link", dgp.TREATMENT_LINKS)
@pytest.mark.parametrize("scale", (.05, .2, .3))
def test_assignment_and_mediator_covariance_leave_population_theta_and_effect_means_unchanged(link, scale):
    base = dgp.draw_parameters(small_config(mu=.2, delta1_shift=.1, delta1_contrast=.02), 1234)
    changed = dgp.draw_parameters(replace(base.config, treatment_link=link, sigma_U_scale=scale), 1234)
    assert changed.truth == base.truth
    for key in base.arrays:
        if key != "Sigma_U":
            np.testing.assert_array_equal(changed.arrays[key], base.arrays[key])
    np.testing.assert_allclose(changed.arrays["Sigma_U"], base.arrays["Sigma_U"] * (scale / .05),
                               rtol=5e-15, atol=1e-15)
    eigenvalues = np.linalg.eigvalsh(changed.arrays["Sigma_U"])
    assert np.all(eigenvalues >= scale) and np.all(eigenvalues <= 2 * scale)
    x = np.array([[-.4, .1], [.8, -.6], [1., 1.]])
    original_means, altered_means = (dgp.conditional_counterfactual_means(mechanism, x)
                                    for mechanism in (base, changed))
    for key in original_means:
        np.testing.assert_array_equal(original_means[key], altered_means[key])


@pytest.mark.parametrize("family", dgp.LOADING_FAMILIES)
def test_quadratic_link_is_saved_loaded_and_part_of_mechanism_identity(family, tmp_path):
    mechanism = dgp.draw_parameters(small_config(loading_family=family,
        treatment_link="quadratic_centered", sigma_U_scale=.2), 1234)
    prefix = tmp_path / family
    dgp.save_mechanism(mechanism, prefix)
    payload = json.loads(prefix.with_suffix(".json").read_text())
    assert payload["config"]["treatment_link"] == "quadratic_centered"
    assert payload["config"]["sigma_U_scale"] == .2
    restored = dgp.load_mechanism(prefix)
    assert restored.config == mechanism.config and restored.mechanism_hash == mechanism.mechanism_hash
    original, loaded = (dgp.sample_data(params, 23, 90) for params in (mechanism, restored))
    for key in ("X", "M", "A", "Y"):
        np.testing.assert_array_equal(original[key], loaded[key])
    payload["config"]["treatment_link"] = "linear"
    prefix.with_suffix(".json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        dgp.load_mechanism(prefix)


@pytest.mark.parametrize("link", ("quadratic", "", None, False))
def test_unknown_treatment_link_is_rejected(link):
    with pytest.raises(ValueError, match="treatment_link"):
        small_config(treatment_link=link).validate()
