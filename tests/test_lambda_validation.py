"""Reject malformed tuning inputs before fits while preserving the existing grid."""
import hashlib
import json
from unittest.mock import patch

import numpy as np
import pytest

from mediencoder import estimation, training
from mediencoder.simulation.runner import lambda_grids


@pytest.mark.parametrize("validator", [training.validate_lambdas, training.validate_lambdas_unbalanced])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("position", range(3))
def test_nonfinite_weight_is_rejected_in_each_position(validator, bad, position):
    weights = [.2, .5, .3]
    weights[position] = bad
    assert not validator(*weights)


@pytest.mark.parametrize("validator", [training.validate_lambdas, training.validate_lambdas_unbalanced])
@pytest.mark.parametrize("kwargs", [
    {"C": np.nan}, {"C": np.inf}, {"C": 0.},
    {"tol": np.nan}, {"tol": np.inf}, {"tol": -1e-6},
])
def test_invalid_validation_parameters_cannot_bypass_simplex(validator, kwargs):
    assert not validator(.2, .5, .3, **kwargs)


def test_generic_validator_preserves_total_ordering_and_ablation_contracts():
    assert training.validate_lambdas(.25, .75, 0., tol=0.)
    assert training.validate_lambdas(.4, 1., .6, C=2.)
    assert training.validate_lambdas_unbalanced(.5, .5, 0.)
    assert not training.validate_lambdas(.5, .5, 0.)
    assert not training.validate_lambdas_unbalanced(.5, .5, .25)
    assert not training.validate_lambdas_unbalanced(.5, .5, -1e-12)


@pytest.mark.parametrize("candidate", [
    (.2, .8), (.2, .5, .3, 0.), .5,
    np.array([[.2], [.5], [.3]]), np.array([[.2, .5, .3]]),
    (.2, np.array([.5]), .3), (".2", ".5", ".3"),
    (.2, .5, np.nan), (.2, np.inf, .3),
    (.5, .5, .25), (.5, .5, -1e-12), (0., .5, .5),
    (.5, 1e-6, .499999),
])
def test_bad_grid_candidate_fails_before_splitting_or_fitting(candidate):
    x = np.ones((8, 2))
    with patch.object(estimation, "_split_indices_4fold") as split, \
         patch.object(estimation, "_learn_representations_fixed_split") as fit:
        with pytest.raises(ValueError, match="lambda_grid candidate"):
            estimation.estimate_triply_IF(
                x, x, np.arange(8) % 2, np.arange(8, dtype=float),
                tilde_p=1, tilde_q=1, factor_method="mediencoder",
                lambda_grid=[candidate],
            )
        split.assert_not_called()
        fit.assert_not_called()


def test_fold_tuning_rejects_invalid_later_candidate_before_any_fit():
    x = np.ones((8, 2))
    with patch.object(estimation, "_learn_representations_fixed_split") as fit:
        with pytest.raises(ValueError, match="candidate 1"):
            estimation._select_lambda_for_fold(
                x, x, np.arange(8) % 2, np.arange(8, dtype=float),
                tilde_p=1, tilde_q=1, factor_method="mediencoder",
                lambda_grid=[(.2, .5, .3), (.5, .5, .25)],
                subtrain_idx=np.arange(4), val_idx=np.arange(4, 8),
            )
        fit.assert_not_called()


def test_valid_default_grid_keeps_exact_values_order_and_input():
    tune, zero = lambda_grids()
    assert (len(tune), len(zero)) == (36, 9)
    before = json.dumps((tune, zero), separators=(",", ":"))
    assert hashlib.sha256(before.encode()).hexdigest() == (
        "022fae04670538828a298868d786b768d50f5fd2a7f94904e676221920d67707"
    )
    estimation._check_lambda_grid_wellposed(tune)
    estimation._check_lambda_grid_wellposed(zero)
    estimation._check_lambda_grid_wellposed(np.asarray([[.5, .5, 0.], [.1, .2, .7]]))
    assert json.dumps((tune, zero), separators=(",", ":")) == before


def test_sum_tolerance_is_absolute_and_tiny_negative_weight_is_not_accepted():
    estimation._check_lambda_grid_wellposed([(.2, .5, .3000005)])
    with pytest.raises(ValueError, match="absolute tolerance"):
        estimation._check_lambda_grid_wellposed([(.2, .5, .300002)])
    with pytest.raises(ValueError, match="nonnegative"):
        estimation._check_lambda_grid_wellposed([(.5, .5, -1e-12)])
