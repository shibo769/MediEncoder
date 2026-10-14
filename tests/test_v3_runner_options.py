"""V3 opt-in nuisance scaling and learned dimensions, without fitting models."""
from copy import deepcopy

import pytest

from mediencoder.simulation import runner


@pytest.fixture
def mocked_training(monkeypatch):
    """Resolve CLI routing without importing or fitting the nuisance learners."""
    template = {
        "nn_cfg": {"hidden_dims": [300, 300], "epochs": 300, "weight_decay": 0.0},
        "ae_cfg": {"hidden_dims_X": [300, 200], "hidden_dims_M": [300, 200],
                   "epochs": 300, "weight_decay": 0.0, "beta_kl": 0.5},
        "me_cfg": {"hidden_dims_X": [300, 200], "hidden_dims_M": [300, 200],
                   "hidden_dims_XM": [50, 50], "epochs": 300, "weight_decay": 0.0,
                   "lambda1": 0.45, "lambda2": 0.45, "lambda3": 0.10},
        "encode_cfg": {"batch_size": 4096},
    }
    calls = []

    def configuration(epochs, weight_decay=0.0):
        calls.append((epochs, weight_decay))
        settings = deepcopy(template)
        for name in ("nn_cfg", "ae_cfg", "me_cfg"):
            settings[name].update(epochs=epochs, weight_decay=weight_decay)
        return settings

    monkeypatch.setattr(runner, "training_configuration", configuration)
    return template, calls


def make_config(*options):
    return runner.make_config(runner.parse_args(["--output-dir", "unused", *options]))


def test_new_switches_preserve_default_configuration(mocked_training):
    template, calls = mocked_training
    args = runner.parse_args(["--output-dir", "unused"])
    assert args.nuisance_standardize is False
    assert (args.latent_p, args.latent_q) == (10, 10)
    config = runner.make_config(args)
    assert (config["tilde_p"], config["tilde_q"]) == (10, 10)
    assert "nuisance_standardize" not in config
    assert "standardize_inputs" not in config["training"]["nn_cfg"]
    assert config["training"] == template
    assert calls == [(300, 0.0)]
    assert config == make_config("--latent-p", "10", "--latent-q", "10")


@pytest.mark.parametrize("preprocessing", ["none", "standardize"])
def test_nuisance_scaling_is_the_same_opt_in_for_every_method(preprocessing, mocked_training):
    baseline = make_config("--preprocessing", preprocessing)
    enabled = make_config("--preprocessing", preprocessing, "--nuisance-standardize")
    assert enabled["nuisance_standardize"] is True
    assert enabled["training"]["nn_cfg"]["standardize_inputs"] is True
    expected = deepcopy(baseline)
    expected["nuisance_standardize"] = True
    expected["training"]["nn_cfg"]["standardize_inputs"] = True
    assert enabled == expected
    assert enabled["training"]["ae_cfg"] == baseline["training"]["ae_cfg"]
    assert enabled["training"]["me_cfg"] == baseline["training"]["me_cfg"]
    assert all("standardize_inputs" not in enabled["training"][name]
               for name in ("ae_cfg", "me_cfg", "encode_cfg"))

    training = runner.runtime_training_configuration(enabled["training"])
    before = deepcopy(training)
    for method in runner.METHODS:
        _, _, options = runner.task_estimation_options(enabled, {"method": method}, training)
        assert options["nn_cfg"] == training["nn_cfg"]
        assert options["nn_cfg"]["standardize_inputs"] is True
        assert options["nn_cfg"] is not training["nn_cfg"]
        assert options["preprocessing"] == preprocessing
        assert "standardize_inputs" not in options["ae_cfg"]
        assert "standardize_inputs" not in options["me_cfg"]
        options["nn_cfg"]["standardize_inputs"] = False
    assert training == before
    assert "standardize_inputs" not in make_config()["training"]["nn_cfg"]


def test_learned_dimensions_change_only_the_manifest_representation_dimensions(mocked_training):
    baseline = make_config()
    changed = make_config("--latent-p", "5", "--latent-q", "7")
    assert (changed["tilde_p"], changed["tilde_q"]) == (5, 7)
    expected = deepcopy(baseline)
    expected.update(tilde_p=5, tilde_q=7)
    assert changed == expected
    assert changed["dgp"]["bar_p"] == changed["dgp"]["bar_q"] == 5
    training = runner.runtime_training_configuration(changed["training"])
    for method in runner.METHODS:
        _, _, options = runner.task_estimation_options(changed, {"method": method}, training)
        assert options["nn_cfg"] == training["nn_cfg"]
        assert "standardize_inputs" not in options["nn_cfg"]


@pytest.mark.parametrize("option", ["--latent-p", "--latent-q"])
@pytest.mark.parametrize("value", ["0", "-1", "1.5", "nan"])
def test_learned_dimensions_must_be_positive_integers(option, value):
    with pytest.raises(SystemExit):
        runner.parse_args(["--output-dir", "unused", option + "=" + value])
