"""V4 treatment and mediator-covariance options, without fitting models."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from mediencoder.simulation import runner


@pytest.fixture
def mocked_training(monkeypatch):
    settings = {
        "nn_cfg": {"hidden_dims": [300, 300], "epochs": 300, "weight_decay": 0.0},
        "ae_cfg": {"hidden_dims_X": [300, 200], "hidden_dims_M": [300, 200],
                   "epochs": 300, "weight_decay": 0.0},
        "me_cfg": {"hidden_dims_X": [300, 200], "hidden_dims_M": [300, 200],
                   "hidden_dims_XM": [50, 50], "epochs": 300, "weight_decay": 0.0,
                   "lambda1": 0.45, "lambda2": 0.45, "lambda3": 0.10},
        "encode_cfg": {"batch_size": 4096},
    }

    def configuration(epochs, weight_decay=0.0):
        result = deepcopy(settings)
        for name in ("nn_cfg", "ae_cfg", "me_cfg"):
            result[name].update(epochs=epochs, weight_decay=weight_decay)
        return result

    monkeypatch.setattr(runner, "training_configuration", configuration)
    return settings


def make_config(*options):
    return runner.make_config(runner.parse_args(["--output-dir", "unused", *options]))


def test_v4_default_switches_preserve_canonical_configuration(mocked_training):
    args = runner.parse_args(["--output-dir", "unused"])
    assert args.treatment_link == "linear" and args.sigma_u_scale == 0.05
    default = runner.make_config(args)
    assert "treatment_link" not in default["dgp"]
    assert default["dgp"]["sigma_U_scale"] == 0.05
    assert "treatment_link" not in default and "sigma_u_scale" not in default
    assert default["training"] == mocked_training
    explicit = make_config("--treatment-link", "linear", "--sigma-u-scale", ".05")
    assert runner.canonical_json(default) == runner.canonical_json(explicit)
    # Old programmatic callers without either newly parsed attribute keep
    # the same scientific configuration through the make_config defaults.
    del args.treatment_link
    del args.sigma_u_scale
    assert runner.make_config(args) == default


@pytest.mark.parametrize("options,changed_fields", [
    (("--treatment-link", "quadratic_centered"), {"treatment_link": "quadratic_centered"}),
    (("--sigma-u-scale", ".2"), {"sigma_U_scale": 0.2}),
    (("--treatment-link", "quadratic_centered", "--sigma-u-scale", ".1"),
     {"treatment_link": "quadratic_centered", "sigma_U_scale": 0.1}),
])
def test_v4_switches_change_only_the_requested_dgp_fields(options, changed_fields, mocked_training):
    default = make_config()
    changed = make_config(*options)
    expected = deepcopy(default)
    expected["dgp"].update(changed_fields)
    assert changed == expected
    assert changed["training"] == default["training"]
    assert changed["mechanism_seed"] == default["mechanism_seed"]
    assert changed["seed_base"] == default["seed_base"]


def test_v4_keeps_common_method_seeds_and_paired_observed_data_identity(mocked_training):
    common = ("--n", "100,800", "--reps", "2")
    default = make_config(*common)
    changed = make_config(*common, "--treatment-link", "quadratic_centered", "--sigma-u-scale", ".2")
    tasks = list(runner.build_tasks(changed))
    assert tasks == list(runner.build_tasks(default))
    assert len(tasks) == 2 * 2 * len(runner.METHODS)
    mechanism = SimpleNamespace(mechanism_hash="fixed-v4-mechanism")
    training = runner.runtime_training_configuration(changed["training"])
    for start in range(0, len(tasks), len(runner.METHODS)):
        group = tasks[start:start + len(runner.METHODS)]
        assert tuple(task["method"] for task in group) == runner.METHODS
        assert len({(task["n"], task["rep"], task["data_seed"], task["training_seed"])
                    for task in group}) == 1
        identities = [runner.observed_dataset_identity(mechanism, task, changed) for task in group]
        assert all(identity == identities[0] for identity in identities)
        for task in group:
            _, _, options = runner.task_estimation_options(changed, task, training)
            assert options["nn_cfg"] == training["nn_cfg"]
            assert options["ae_cfg"] == training["ae_cfg"]


@pytest.mark.parametrize("value", ["0", "-0.1", "nan", "inf", "-inf", "not-a-number"])
def test_sigma_u_scale_must_be_finite_and_strictly_positive(value):
    with pytest.raises(SystemExit):
        runner.parse_args(["--output-dir", "unused", "--sigma-u-scale=" + value])


@pytest.mark.parametrize("value", ["quadratic", "centered", "Linear", ""])
def test_unknown_treatment_link_is_rejected(value):
    with pytest.raises(SystemExit):
        runner.parse_args(["--output-dir", "unused", "--treatment-link=" + value])
