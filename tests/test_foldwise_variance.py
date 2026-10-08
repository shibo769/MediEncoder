"""Hand-calculated variance checks; no experimental training or simulation runs."""
import numpy as np
import pytest

from mediencoder import estimation


def aggregate(scores, folds):
    outputs = [dict(phi=scores[i], theta_hat_IF=float(scores[i].mean())) for i in folds]
    return estimation._aggregate_crossfit_scores(len(scores), folds, outputs)


def test_equal_folds_center_separately_without_changing_point_estimate():
    # Every pair has sample variance 2, irrespective of its fold's offset.
    scores = np.array([0., 2., 10., 12., 100., 102., 1000., 1002.])
    folds = np.array_split(np.arange(8), 4)
    result = aggregate(scores, folds)
    assert result["theta_hat_IF"] == 278.5
    assert result["se_IF"] == pytest.approx(.5)  # sqrt(2 / 8)
    np.testing.assert_array_equal(result["crossfit_scores"], scores)
    assert not np.isclose(result["se_IF"], scores.std(ddof=1) / np.sqrt(8))


def test_unequal_folds_use_nk_weights_not_unweighted_fold_average():
    # Fold variances: 1 and 8; variance of the overall mean is (3*1+2*8)/25.
    scores = np.array([-1., 0., 1., 9., 13.])
    folds = [np.arange(3), np.arange(3, 5)]
    result = aggregate(scores, folds)
    assert result["theta_hat_IF"] == pytest.approx(4.4)
    assert result["se_IF"] ** 2 == pytest.approx(19 / 25)
    assert result["se_IF"] ** 2 != pytest.approx((1 + 8) / (2 * 5))


@pytest.mark.parametrize("folds", [
    [np.array([0]), np.array([1, 2, 3])],
    [np.array([0, 1]), np.array([1, 2])],
    [np.array([0, 1])],
    [np.array([0., 1.]), np.array([2., 3.])],
    [np.array([-1, 1]), np.array([2, 3])],
    [],
])
def test_invalid_partitions_fail_instead_of_reusing_or_dropping_subjects(folds):
    with pytest.raises(ValueError):
        estimation._foldwise_score_covariance(np.arange(4.)[:, None], folds)


def test_effect_contrasts_keep_covariance_with_the_same_fold_weights():
    s11 = np.array([0., 2., 100., 104.])
    s10 = np.array([0., 1., 10., 12.])
    s00 = np.array([0., 0., 1., 2.])
    folds = [np.array([0, 1]), np.array([2, 3])]
    result = estimation.summarize_effect_scores(s11, s10, s00, estimation_indices=folds)
    # Two-subject covariance is outer(difference, difference)/2. Multiplying
    # each by n_k/n^2 leaves (outer(d1,d1)+outer(d2,d2))/16.
    d1, d2 = np.array([1., 1., 2.]), np.array([2., 1., 3.])
    expected = (np.outer(d1, d1) + np.outer(d2, d2)) / 16
    np.testing.assert_allclose(result["effect_covariance"], expected)
    np.testing.assert_allclose([result["effect_se"][key] ** 2 for key in ("NIE", "NDE", "TE")],
                               np.diag(expected))
    # Using separate component variances would give NIE variance (4+16+1+4)/16.
    assert result["effect_se"]["NIE"] ** 2 == pytest.approx(5 / 16)
    assert result["effect_se"]["TE"] ** 2 == pytest.approx(
        result["effect_se"]["NIE"] ** 2 + result["effect_se"]["NDE"] ** 2 + 2 * expected[0, 1])
    assert result["effects"]["TE"] == pytest.approx(result["effects"]["NIE"] + result["effects"]["NDE"])


def test_joint_contrast_variance_equals_direct_scalar_aggregation():
    scores = np.array([-1., 0., 1., 9., 13.])
    folds = [np.arange(3), np.arange(3, 5)]
    result = estimation.summarize_effect_scores(scores, np.zeros(5), np.zeros(5),
                                               estimation_indices=folds)
    assert result["effect_se"]["NIE"] == aggregate(scores, folds)["se_IF"]


@pytest.mark.parametrize("use_pooled_se", [False, True])
def test_simulation_worker_checks_fold_variance_and_saves_failure(monkeypatch, tmp_path, use_pooled_se):
    from types import SimpleNamespace
    from mediencoder.simulation import dgp, runner

    scores = np.array([0., 2., 10., 12., 100., 102., 1000., 1002.])
    folds = np.array_split(np.arange(8), 4)
    se = float(scores.std(ddof=1) / np.sqrt(8)) if use_pooled_se else .5
    mean, z = float(scores.mean()), 1.959963984540054
    result = dict(crossfit_scores=scores, theta_hat_IF=mean, se_IF=se,
                  ci_lower=mean-z*se, ci_upper=mean+z*se,
                  fold_indices=[{"estimation": indices} for indices in folds],
                  variance_estimator="within_fold_size_weighted", fold_score_variances=[2.] * 4)
    data = dict(X=np.zeros((8, 2)), M=np.zeros((8, 2)), A=np.arange(8) % 2,
                Y=np.arange(8.), oracle={"forbidden": "latent information"})
    mechanism = SimpleNamespace(mechanism_hash="fixture", truth=SimpleNamespace(value=mean))
    config = dict(tilde_p=1, tilde_q=1)
    monkeypatch.setattr(runner, "_WORKER", dict(config=config, output_dir=tmp_path,
                                               mechanism=mechanism, run_hash="fixture", training={}))
    monkeypatch.setattr(dgp, "sample_data", lambda *args: data.copy())
    monkeypatch.setattr(estimation, "estimate_triply_IF", lambda *args, **kwargs: result)
    task = dict(task_id="fixture", method="projection", n=8, data_seed=10, training_seed=20, rep=0)
    record = runner._run_task(task)
    assert (tmp_path / "tasks" / "fixture.json").is_file()
    if use_pooled_se:
        assert record["status"] == "failed"
        assert "within-fold" in record["error_message"]
        assert not (tmp_path / "scores" / "fixture.npz").exists()
    else:
        assert record["status"] == "complete"
        assert record["se_IF"] == .5
        assert record["estimator_metadata"]["variance_estimator"] == "within_fold_size_weighted"
        with np.load(tmp_path / record["score_artifact"]) as saved:
            np.testing.assert_array_equal(saved["crossfit_scores"], scores)
