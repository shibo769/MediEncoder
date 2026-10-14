"""Prespecified main-simulation settings and fixed-lambda execution, without fitting."""
from copy import deepcopy

import pytest

from mediencoder.simulation import runner


@pytest.fixture
def shared_training():
    from mediencoder.estimation import SHARED_TRAIN_CFG
    original = deepcopy(SHARED_TRAIN_CFG)
    try:
        yield
    finally:
        SHARED_TRAIN_CFG.clear()
        SHARED_TRAIN_CFG.update(original)


def test_main_default_configuration_records_the_approved_scientific_settings(shared_training):
    config = runner.make_config(runner.parse_args(["--output-dir", "unused"]))
    assert config["n_values"] == [100, 300, 800, 1200, 2000, 3000]
    assert config["B_requested"] == 200 and len(list(runner.build_tasks(config))) == 6000
    assert config["dgp"]["p"] == config["dgp"]["q"] == 500
    assert config["dgp"]["bar_p"] == config["dgp"]["bar_q"] == 5
    assert config["tilde_p"] == config["tilde_q"] == 10
    assert config["preprocessing"] == "none"
    assert config["dgp"]["alpha_upper"] == 1.0
    assert config["dgp"]["delta1_spread"] == .2 and config["dgp"]["delta1_contrast"] == 0.0
    assert config["mechanism_seed"] == 910000 and config["seed_base"] == 20261019
    assert config["lambda_mode"] == "fixed"
    assert config["lambda_grid"] == [[.45, .45, .10]]
    assert config["lambda_grid_zero"] == [[.5, .5, 0.0]]
    assert config["training"]["nn_cfg"]["hidden_dims"] == [300, 300]
    assert all(config["training"][name]["weight_decay"] == 0.0
               for name in ("nn_cfg", "ae_cfg", "me_cfg"))
    assert config["alignment_normalization"] == "row_l2" and config["alignment_norm_eps"] == 1e-8
    assert config["reconstruction_loss_normalization"] == "none" and config["stop_gradient"] is False


@pytest.mark.parametrize("family", ["mixed_tanh", "deep_tanh"])
@pytest.mark.parametrize("strength", [2.0, 6.0])
def test_nonlinear_measurement_configuration_is_explicit(family, strength, shared_training):
    config = runner.make_config(runner.parse_args([
        "--output-dir", "unused", "--loading-family", family,
        "--loading-strength", str(strength),
    ]))
    assert config["dgp"]["loading_family"] == family
    assert config["dgp"]["loading_strength"] == strength
    assert config["alignment_normalization"] == "row_l2"
    assert config["lambda_grid"] == [[.45, .45, .10]]
    assert config["stop_gradient"] is False


@pytest.mark.parametrize("family", ["mixed_tanh", "deep_tanh"])
def test_nonlinear_frequency_and_shared_preprocessing_are_recorded(family, shared_training):
    config = runner.make_config(runner.parse_args([
        "--output-dir", "unused", "--loading-family", family,
        "--loading-frequency", "4", "--preprocessing", "standardize",
    ]))
    assert config["dgp"]["loading_frequency"] == 4.0
    assert config["preprocessing"] == "standardize"
    training = runner.runtime_training_configuration(config["training"])
    for method in config["methods"]:
        _, _, options = runner.task_estimation_options(config, {"method": method}, training)
        assert options["preprocessing"] == "standardize"


@pytest.mark.parametrize("family", ["polynomial", "haar"])
def test_legacy_measurement_configuration_omits_unused_strength(family, shared_training):
    config = runner.make_config(runner.parse_args([
        "--output-dir", "unused", "--loading-family", family,
        "--loading-strength", "6",
    ]))
    assert "loading_strength" not in config["dgp"]
    assert "loading_frequency" not in config["dgp"]


def test_fixed_lambda_reaches_each_arm_without_a_tuning_grid_or_shared_config_mutation(shared_training):
    config = runner.make_config(runner.parse_args(["--output-dir", "unused", "--lambda-fixed", ".4,.4,.2"]))
    training = runner.runtime_training_configuration(config["training"])
    before = deepcopy(training)
    for method, weights in (("mediencoder", (.4, .4, .2)),
                            ("mediencoder_l3zero", (.5, .5, 0.0))):
        base_method, grid, options = runner.task_estimation_options(config, {"method": method}, training)
        assert base_method == "mediencoder" and grid is None
        assert options["fixed_lambda"] == weights
        assert tuple(options["me_cfg"][key] for key in ("lambda1", "lambda2", "lambda3")) == weights
    assert training == before
    assert tuple(config["training"]["me_cfg"][key] for key in ("lambda1", "lambda2", "lambda3")) == (.4, .4, .2)
    _, grid, options = runner.task_estimation_options(config, {"method": "projection"}, training)
    assert grid is None and "fixed_lambda" not in options


def test_explicit_grid_mode_preserves_the_original_candidate_search(shared_training):
    config = runner.make_config(runner.parse_args(["--output-dir", "unused", "--lambda-mode", "grid"]))
    assert len(config["lambda_grid"]) == 36 and len(config["lambda_grid_zero"]) == 9
    training = runner.runtime_training_configuration(config["training"])
    _, grid, options = runner.task_estimation_options(config, {"method": "mediencoder"}, training)
    assert grid == config["lambda_grid"] and "fixed_lambda" not in options


@pytest.mark.parametrize("argument,value", [("--lambda-fixed", ".4,.4,.4"),
    ("--lambda-fixed", ".5,.5,0"), ("--lambda-fixed-zero", ".4,.4,.2"),
    ("--lambda-fixed", "nan,.4,.6"), ("--alpha-upper", "0"),
    ("--delta1-spread", "-.2"), ("--loading-strength", "-1"),
    ("--loading-strength", "nan"), ("--loading-strength", "inf"),
    ("--loading-frequency", "0"), ("--loading-frequency", "-1"),
    ("--loading-frequency", "nan"), ("--loading-frequency", "inf")])
def test_invalid_prespecified_settings_are_rejected(argument, value):
    with pytest.raises(SystemExit):
        runner.parse_args(["--output-dir", "unused", argument + "=" + value])
