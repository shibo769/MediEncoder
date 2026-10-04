"""Mathematical and reproducibility checks; no network training is performed."""

from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.integrate import quad

from mediencoder.simulation import dgp


@pytest.fixture(scope="module")
def mechanism():
    return dgp.draw_parameters(dgp.DGPConfig(p=7, q=6, bar_p=2, bar_q=2, pilot_size=128), 910000)


def test_one_dimensional_population_truth_against_independent_quadrature():
    params = dgp.draw_parameters(dgp.DGPConfig(p=3, q=4, bar_p=1, bar_q=1, mu=0.3, pilot_size=64), 4)
    def target(x):
        return float(dgp.conditional_counterfactual_means(params, np.array([[x]]))["mu10_true_vals"][0])
    value, error = quad(target, -0.7, 1.3, epsabs=1e-11, epsrel=1e-11)
    assert params.truth.value == pytest.approx(value / 2, abs=1e-9)
    assert error < 1e-8
    assert params.truth.converged
    assert len(params.truth.successive_differences) >= 2


def test_population_moments_closed_form_when_softplus_is_constant(mechanism):
    arrays = {k: v.copy() for k, v in mechanism.arrays.items()}
    arrays["kappa_1"][:] = 0
    arrays["beta_1"][:] = 0
    arrays["delta_0"][:] = np.eye(2)
    arrays["gamma_1"][:] = [2, 3]
    params = replace(mechanism, arrays=arrays, truth=None)
    truth = dgp.population_truth(params)
    assert truth.value == pytest.approx(np.log(2) + (2 + 3) / 5, abs=1e-12)


def test_data_seed_and_sample_size_do_not_change_mechanism(mechanism):
    before = mechanism.mechanism_hash
    a = dgp.sample_data(mechanism, 15, 42)
    b = dgp.sample_data(mechanism, 15, 42)
    longer = dgp.sample_data(mechanism, 21, 42)
    other = dgp.sample_data(mechanism, 15, 43)
    for key in ("X", "M", "A", "Y"):
        np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_allclose(a[key], longer[key][:15], rtol=0, atol=1e-14)
    assert not np.array_equal(a["X"], other["X"])
    assert before == mechanism.mechanism_hash
    assert a["theta_population"] == other["theta_population"] == mechanism.truth.value
    assert "f_X" not in a and "mu10_true_vals" not in a
    assert "f_X" in a["oracle"] and "mu10_true_vals" in a["oracle"]


def test_fixed_support_and_complete_serialization(mechanism, tmp_path, monkeypatch):
    artifacts = dgp.save_mechanism(mechanism, tmp_path / "mechanism")
    monkeypatch.setattr(dgp, "population_truth", lambda *args, **kwargs: pytest.fail("Unexpected repeated quadrature"))
    loaded = dgp.load_mechanism(artifacts["json"])
    assert loaded.mechanism_hash == mechanism.mechanism_hash
    assert loaded.metadata["pilot_size"] == 128
    assert loaded.metadata["support_rule"] == dgp.SUPPORT_RULE
    assert loaded.metadata["pilot_seed"] != loaded.parameter_seed
    for key in ("X", "M", "A", "Y"):
        np.testing.assert_array_equal(dgp.sample_data(loaded, 13, 9)[key], dgp.sample_data(mechanism, 13, 9)[key])
    for array in loaded.arrays.values():
        assert not array.flags.writeable
    payload = json.loads((tmp_path / "mechanism.json").read_text())
    payload["truth"]["value"] += 1
    (tmp_path / "mechanism.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        dgp.load_mechanism(tmp_path / "mechanism")


def test_oracle_conditional_truth_averages_to_population(mechanism):
    rng = np.random.default_rng(11993)
    x = rng.uniform(-1, 1, size=(100000, 2))
    values = dgp.conditional_counterfactual_means(mechanism, x)["mu10_true_vals"]
    mc_se = values.std(ddof=1) / np.sqrt(len(values))
    assert abs(values.mean() - mechanism.truth.value) < 5 * mc_se


def test_latent_and_outcome_distribution_sanity(mechanism):
    data = dgp.sample_data(mechanism, 20000, 718)
    oracle, a = data["oracle"], mechanism.arrays
    x, m, treatment = oracle["f_X"], oracle["f_M"], data["A"]
    mean0, mean1 = (x ** 2) @ a["delta_0"].T, (x ** 2) @ a["delta_1"].T
    residual_m = m - np.where(treatment[:, None] == 1, mean1, mean0)
    np.testing.assert_allclose(residual_m.mean(axis=0), 0, atol=0.01)
    np.testing.assert_allclose(np.cov(residual_m, rowvar=False), a["Sigma_U"], rtol=0.06, atol=0.002)
    residual_y = data["Y"] - np.where(treatment == 1, oracle["mu1_true_vals"], oracle["mu0_true_vals"])
    assert abs(residual_y.mean()) < 0.03
    assert residual_y.std(ddof=1) == pytest.approx(mechanism.config.sigma_y, rel=0.03)


def test_parameter_draw_is_reproducible_and_separate_from_observed_dimensions(mechanism):
    again = dgp.draw_parameters(mechanism.config, mechanism.parameter_seed)
    assert again.mechanism_hash == mechanism.mechanism_hash
    other = dgp.draw_parameters(replace(mechanism.config, p=8, q=5), mechanism.parameter_seed)
    for name in ("delta_0", "delta_1", "Sigma_U", "kappa_1", "r_list", "s_list"):
        np.testing.assert_array_equal(other.arrays[name], mechanism.arrays[name])
    assert other.truth.value == mechanism.truth.value


def test_invalid_config_and_missing_cached_truth(mechanism):
    with pytest.raises(ValueError, match="bar_p == bar_q"):
        dgp.draw_parameters(dgp.DGPConfig(bar_p=2, bar_q=3))
    with pytest.raises(ValueError, match="cached"):
        dgp.sample_data(replace(mechanism, truth=None), 10, 1)
    with pytest.raises(ValueError, match="increasing"):
        dgp.population_truth(mechanism, orders=(16, 12, 24))
