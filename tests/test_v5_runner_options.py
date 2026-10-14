"""Existing true-dimension and noise-SD CLI options; run numerical checks on SCC."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import quad

from mediencoder.simulation import dgp, runner


@pytest.fixture
def mocked_training(monkeypatch):
    template = dict(nn_cfg=dict(hidden_dims=[300, 300], epochs=300, weight_decay=0.),
        ae_cfg=dict(epochs=300, weight_decay=0.),
        me_cfg=dict(epochs=300, weight_decay=0., lambda1=.45, lambda2=.45, lambda3=.1),
        encode_cfg=dict(batch_size=4096))
    def configuration(epochs, weight_decay=0.):
        settings = deepcopy(template)
        for name in ("nn_cfg", "ae_cfg", "me_cfg"):
            settings[name].update(epochs=epochs, weight_decay=weight_decay)
        return settings
    monkeypatch.setattr(runner, "training_configuration", configuration)
    return template


def make_config(*options):
    return runner.make_config(runner.parse_args(["--output-dir", "unused", *options]))


def test_defaults_and_old_programmatic_arguments_keep_existing_schema_and_fields(mocked_training):
    args = runner.parse_args(["--output-dir", "unused"])
    assert (args.factor_dim, args.sigma_eps_x_sd, args.sigma_eps_m_sd, args.sigma_y_sd) == (5, 2., 1., 1.)
    implicit = runner.make_config(args)
    explicit = make_config("--factor-dim", "5", "--sigma-eps-x-sd", "2", "--sigma-eps-m-sd", "1", "--sigma-y-sd", "1")
    assert runner.canonical_json(implicit) == runner.canonical_json(explicit)
    assert implicit["schema_version"] == 2 and implicit["dgp"] == dgp.DGPConfig().to_dict()
    assert implicit["training"] == mocked_training
    assert implicit["tilde_p"] == implicit["tilde_q"] == 10
    for name in ("factor_dim", "sigma_eps_x_sd", "sigma_eps_m_sd", "sigma_y_sd"):
        assert name not in implicit
        delattr(args, name)
    assert runner.make_config(args) == implicit


@pytest.mark.parametrize("dimension", (1, 2, 3, 4, 5))
def test_true_dimension_changes_both_factor_fields_without_changing_learned_dimensions(dimension, mocked_training):
    default = make_config()
    actual = make_config("--factor-dim", str(dimension))
    expected = deepcopy(default)
    expected["dgp"].update(bar_p=dimension, bar_q=dimension)
    assert actual == expected
    assert actual["tilde_p"] == actual["tilde_q"] == 10


@pytest.mark.parametrize("option,field,value", (("--sigma-eps-x-sd", "sigma_eps_X", .75),
    ("--sigma-eps-m-sd", "sigma_eps_M", .25), ("--sigma-y-sd", "sigma_y", .5)))
def test_noise_SD_switch_changes_only_its_existing_DGP_field(option, field, value, mocked_training):
    default = make_config()
    actual = make_config(option, str(value))
    expected = deepcopy(default); expected["dgp"][field] = value
    assert actual == expected and actual["training"] == default["training"]


def test_zero_noise_is_supported_and_distinct_from_covariance_scale(mocked_training):
    config = make_config("--sigma-eps-x-sd", "0", "--sigma-eps-m-sd", "0", "--sigma-y-sd", "0")
    assert all(config["dgp"][field] == 0 for field in ("sigma_eps_X", "sigma_eps_M", "sigma_y"))
    assert config["dgp"]["sigma_U_scale"] == .05
    dgp.DGPConfig(**config["dgp"]).validate()


@pytest.mark.parametrize("value", ("0", "6", "-1", "1.5", "nan", "True"))
def test_invalid_or_unsupported_true_factor_dimension_is_rejected(value):
    with pytest.raises(SystemExit):
        runner.parse_args(["--output-dir", "unused", "--factor-dim=" + value])


@pytest.mark.parametrize("option", ("--sigma-eps-x-sd", "--sigma-eps-m-sd", "--sigma-y-sd"))
@pytest.mark.parametrize("value", ("-.1", "nan", "inf", "-inf", "not-a-number"))
def test_noise_SD_must_be_finite_and_nonnegative(option, value):
    with pytest.raises(SystemExit):
        runner.parse_args(["--output-dir", "unused", option + "=" + value])


def test_true_dimensions_still_obey_nonlinear_map_observed_dimension_guard():
    runner.parse_args(["--output-dir", "unused", "--factor-dim", "2", "--p", "2", "--q", "2", "--loading-family", "mixed_tanh"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--output-dir", "unused", "--factor-dim", "3", "--p", "2", "--q", "3", "--loading-family", "mixed_tanh"])


def test_new_options_keep_all_five_methods_seeds_and_data_pairing_common(mocked_training):
    default = make_config("--n", "100,300", "--reps", "2")
    changed = make_config("--n", "100,300", "--reps", "2", "--factor-dim", "2",
                         "--sigma-eps-x-sd", ".5", "--sigma-eps-m-sd", ".5", "--sigma-y-sd", ".5")
    assert list(runner.build_tasks(default)) == list(runner.build_tasks(changed))
    tasks = list(runner.build_tasks(changed))
    assert len(tasks) == 20
    for first in range(0, len(tasks), 5):
        group = tasks[first:first+5]
        assert tuple(task["method"] for task in group) == runner.METHODS
        assert len({(task["data_seed"], task["training_seed"], task["rep"], task["n"]) for task in group}) == 1


def test_actual_nondefault_dimension_noise_mechanism_roundtrips_and_rejects_incompatible_imports(tmp_path, mocked_training):
    config = make_config("--p", "7", "--q", "6", "--factor-dim", "2", "--sigma-eps-x-sd", ".5",
                         "--sigma-eps-m-sd", ".25", "--sigma-y-sd", ".75")
    mechanism = dgp.draw_parameters(config["dgp"], config["mechanism_seed"])
    assert mechanism.arrays["alpha"].shape == (2,)
    assert mechanism.arrays["delta_0"].shape == mechanism.arrays["Sigma_U"].shape == (2, 2)
    data = dgp.sample_data(mechanism, 19, 73)
    assert data["X"].shape == (19, 7) and data["M"].shape == (19, 6)
    assert data["oracle"]["f_X"].shape == data["oracle"]["f_M"].shape == (19, 2)
    bundle = tmp_path / "bundle"; bundle.mkdir()
    dgp.save_mechanism(mechanism, bundle / "mechanism")
    loaded = dgp.load_mechanism(bundle / "mechanism")
    assert loaded.config.to_dict() == config["dgp"] and loaded.mechanism_hash == mechanism.mechanism_hash
    matching = tmp_path / "matching"; matching.mkdir()
    imported, provenance = runner.prepare_mechanism(matching, config, mechanism_from=bundle)
    assert imported.mechanism_hash == mechanism.mechanism_hash and provenance["kind"] == "imported"
    assert runner.mechanism_artifact_hashes(matching / "mechanism") == runner.mechanism_artifact_hashes(bundle / "mechanism")
    changed_fields = ({"bar_p": 3, "bar_q": 3}, {"sigma_eps_X": 2.}, {"sigma_eps_M": 1.}, {"sigma_y": 1.})
    for index, difference in enumerate(changed_fields):
        mismatch = deepcopy(config); mismatch["dgp"].update(difference)
        output = tmp_path / f"mismatch-{index}"; output.mkdir()
        with pytest.raises(ValueError, match="DGP configuration differs"):
            runner.prepare_mechanism(output, mismatch, mechanism_from=bundle)
        assert list(output.iterdir()) == []
        with pytest.raises(ValueError, match="DGP configuration differs"):
            runner.prepare_mechanism(matching, mismatch, existing={"mechanism_hash": imported.mechanism_hash})


def test_noise_switches_are_SDs_and_leave_structural_parameters_truth_and_latent_draws_unchanged():
    config = dgp.DGPConfig(p=7, q=6, bar_p=2, bar_q=2)
    original = dgp.draw_parameters(config, 1234)
    half = dgp.draw_parameters(replace(config, sigma_eps_X=1., sigma_eps_M=.5, sigma_y=.25), 1234)
    zero = dgp.draw_parameters(replace(config, sigma_eps_X=0., sigma_eps_M=0., sigma_y=0.), 1234)
    assert original.truth == half.truth == zero.truth
    assert len({item.mechanism_hash for item in (original, half, zero)}) == 3
    for key in original.arrays:
        np.testing.assert_array_equal(original.arrays[key], half.arrays[key])
        np.testing.assert_array_equal(original.arrays[key], zero.arrays[key])
    base, altered, no_noise = (dgp.sample_data(item, 71, 81) for item in (original, half, zero))
    np.testing.assert_array_equal(base["A"], altered["A"])
    for key in base["oracle"]:
        np.testing.assert_array_equal(base["oracle"][key], altered["oracle"][key])
        np.testing.assert_array_equal(base["oracle"][key], no_noise["oracle"][key])
    for key, ratio in (("X", .5), ("M", .5), ("Y", .25)):
        np.testing.assert_allclose(altered[key]-no_noise[key], ratio*(base[key]-no_noise[key]), rtol=1e-12, atol=1e-12)
        assert np.any(base[key] != no_noise[key])


def test_population_theta10_for_one_true_factor_agrees_with_independent_scalar_quadrature():
    mechanism = dgp.draw_parameters(dgp.DGPConfig(p=7, q=6, bar_p=1, bar_q=1), 1234)
    arrays = mechanism.arrays
    # mu=0 removes the sine term. E[X^4]=1/5 supplies the mediator interaction.
    polynomial = float(arrays["gamma_1"][0]*arrays["delta_0"][0, 0]/5)
    kappa = float(arrays["kappa_1"][0])
    softplus = quad(lambda value: float(np.logaddexp(0, kappa*value))/2, -1, 1, epsabs=1e-12, epsrel=1e-12)[0]
    assert mechanism.truth.converged
    assert mechanism.truth.value == pytest.approx(polynomial+softplus, abs=1e-10, rel=1e-10)


@pytest.fixture(scope="module")
def frozen2c_source():
    location = os.environ.get("MEDIENC_V5_BASELINE_SOURCE")
    if not location:
        pytest.skip("SCC default-compatibility check: set MEDIENC_V5_BASELINE_SOURCE to the actual frozen source2c root")
    root = Path(location)
    assert root.resolve() != Path(runner.__file__).resolve().parents[3]
    assert (root / "src/mediencoder/simulation/runner.py").is_file()
    deployment = json.loads((root / "SCC_DEPLOYMENT_MANIFEST.json").read_text())
    assert deployment["source_commit"] == "2c88cf3238ca8f5c8d605e48e2f305fb9685d498"
    assert len(deployment["files"]) == 80
    for relative, metadata in deployment["files"].items():
        path = root / relative
        assert path.resolve().is_relative_to(root.resolve()) and not path.is_symlink()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]
    return root


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_default_full_config_and_unchanged_DGP_source_are_exactly_compatible_with_frozen2c(frozen2c_source):
    baseline_path = frozen2c_source / "src/mediencoder/simulation/runner.py"
    old = load_module("frozen2c_v5_cli_runner", baseline_path)
    try:
        reference = old.make_config(old.parse_args(["--output-dir", "unused"]))
        current = runner.make_config(runner.parse_args(["--output-dir", "unused"]))
        assert runner.canonical_json(current) == old.canonical_json(reference)
        assert runner.digest(current) == old.digest(reference)
        assert Path(dgp.__file__).read_bytes() == baseline_path.with_name("dgp.py").read_bytes()
    finally:
        sys.modules.pop("frozen2c_v5_cli_runner", None)


def test_default_mechanism_arrays_schema_hash_and_observed_RNG_match_actual_frozen2c(frozen2c_source, tmp_path):
    baseline_path = frozen2c_source / "src/mediencoder/simulation/dgp.py"
    old = load_module("frozen2c_v5_default_dgp", baseline_path)
    try:
        current = dgp.draw_parameters(dgp.DGPConfig(), 1234)
        reference = old.draw_parameters(old.DGPConfig(), 1234)
        assert current.schema_version == reference.schema_version
        assert current.serialized_config() == reference.serialized_config()
        assert current.metadata == reference.metadata and current.truth.to_dict() == reference.truth.to_dict()
        assert current.mechanism_hash == reference.mechanism_hash
        assert current.arrays.keys() == reference.arrays.keys()
        for key in current.arrays:
            np.testing.assert_array_equal(current.arrays[key], reference.arrays[key])
        before, after = (module.sample_data(params, 31, 81) for module, params in ((old, reference), (dgp, current)))
        for key in ("X", "M", "A", "Y"):
            np.testing.assert_array_equal(before[key], after[key])
        for key in before["oracle"]:
            np.testing.assert_array_equal(before["oracle"][key], after["oracle"][key])
        old.save_mechanism(reference, tmp_path / "reference/mechanism")
        dgp.save_mechanism(current, tmp_path / "candidate/mechanism")
        for suffix in ("json", "npz"):
            before = tmp_path / "reference" / ("mechanism." + suffix)
            after = tmp_path / "candidate" / ("mechanism." + suffix)
            assert hashlib.sha256(before.read_bytes()).hexdigest() == hashlib.sha256(after.read_bytes()).hexdigest()
    finally:
        sys.modules.pop("frozen2c_v5_default_dgp", None)
