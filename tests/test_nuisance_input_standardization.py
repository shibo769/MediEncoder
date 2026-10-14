"""Training-subset nuisance scaling; numerical tests are intended for SCC only."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from mediencoder import estimation, models


@pytest.fixture(autouse=True)
def one_torch_thread():
    torch.set_num_threads(1)


def small_fit_kwargs(**extra):
    return dict(epochs=2, hidden_dims=(4,), batch_size=5, lr_init=0.,
                early_stop=False, verbose=False, **extra)


def expected_moments(values):
    values = torch.as_tensor(values, dtype=torch.float32).to(torch.float64)
    mean = values.mean(dim=0)
    scale = torch.sqrt(torch.mean((values - mean) ** 2, dim=0)).to(torch.float32)
    scale = torch.where(scale > models.NUISANCE_INPUT_CONSTANT_SD_THRESHOLD,
                        scale, torch.ones_like(scale))
    return mean.to(torch.float32), scale


def assert_training_statistics(model, values):
    mean, scale = expected_moments(values)
    torch.testing.assert_close(model.input_mean.cpu(), mean, rtol=0, atol=0)
    torch.testing.assert_close(model.input_scale.cpu(), scale, rtol=0, atol=0)
    assert model.input_count.item() == len(values)
    assert not model.input_mean.requires_grad and not model.input_scale.requires_grad


@pytest.mark.parametrize("binary", (False, True))
def test_statistics_use_only_this_models_training_inputs(binary):
    X = np.array([[10., 100., 3.], [20., 150., 3.], [30., 120., 3.], [40., 170., 3.]])
    Y = np.array([0., 1., 0., 1.]) if binary else np.array([1000., 2000., 1500., 2500.])
    validation = np.array([[10000., -5000., 3.], [-7000., 3000., 3.]])
    target = np.array([0., 1.]) if binary else np.array([1400., 2300.])
    torch.manual_seed(12)
    fitted, _, _ = models.train_nuisance_nn(X, Y, validation, target, binary=binary,
                                            standardize_inputs=True, **small_fit_kwargs())
    assert isinstance(fitted, models.StandardizedDeepNN)
    assert_training_statistics(fitted, X)
    assert fitted.input_scale[-1].item() == 1.
    torch.manual_seed(12)
    changed, _, _ = models.train_nuisance_nn(X, Y, validation * 2., target[::-1].copy(), binary=binary,
                                            standardize_inputs=True, **small_fit_kwargs())
    assert_training_statistics(changed, X)
    before = {key: value.clone() for key, value in fitted.named_buffers()}
    prediction = models.predict_nn(fitted, np.array([[1234., -9876., 3.]]), binary=binary)
    assert prediction.shape == (1,) and np.isfinite(prediction).all()
    for key, value in fitted.named_buffers():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize("binary", (False, True))
def test_prediction_applies_the_stored_transform_exactly_once(binary):
    X = np.array([[1., 10.], [2., 30.], [4., 50.], [8., 70.]])
    Y = np.array([0., 1., 0., 1.]) if binary else np.array([100., 210., 320., 430.])
    fitted, _, _ = models.train_nuisance_nn(X, Y, X[:2], Y[:2], binary=binary,
                                            standardize_inputs=True, **small_fit_kwargs())
    values = np.array([[9., -30.], [5., 90.], [-2., 120.]])
    tensor = torch.as_tensor(values, dtype=torch.float32, device=fitted.input_mean.device)
    with torch.no_grad():
        manual = fitted.net((tensor - fitted.input_mean) / fitted.input_scale).squeeze(-1)
        if binary:
            manual = torch.sigmoid(manual)
    np.testing.assert_array_equal(models.predict_nn(fitted, values, binary=binary), manual.cpu().numpy())
    batchwise = np.concatenate([models.predict_nn(fitted, values[:1], binary=binary),
                                models.predict_nn(fitted, values[1:], binary=binary)])
    np.testing.assert_allclose(batchwise, manual.cpu().numpy(), rtol=1e-6, atol=1e-7)


def test_buffers_roundtrip_and_follow_network_device_and_dtype():
    X, Y = np.arange(24, dtype=float).reshape(8, 3), np.arange(8, dtype=float)
    fitted, _, _ = models.train_nuisance_nn(X[:6], Y[:6], X[6:], Y[6:],
                                            standardize_inputs=True, **small_fit_kwargs())
    state = {key: value.detach().cpu().clone() for key, value in fitted.state_dict().items()}
    restored = models.StandardizedDeepNN(3, input_mean=torch.zeros(3), input_scale=torch.ones(3),
                                         input_count=1, hidden_dims=(4,))
    restored.load_state_dict(state)
    restored.to(next(fitted.parameters()).device)
    np.testing.assert_allclose(models.predict_nn(fitted, X), models.predict_nn(restored, X), rtol=0, atol=0)
    restored.to(dtype=torch.float64)
    assert restored.input_mean.dtype == restored.input_scale.dtype == torch.float64
    assert restored.input_count.dtype == torch.int64 and restored.input_count.item() == 6


def test_training_targets_are_passed_through_without_rescaling(monkeypatch):
    captured = []
    original_dataset = models.TensorDataset

    def dataset(*tensors):
        captured.append(tensors[1].clone())
        return original_dataset(*tensors)

    monkeypatch.setattr(models, "TensorDataset", dataset)
    X, Y = np.arange(24, dtype=float).reshape(8, 3), np.array([1000., 1200., 1700., 4000., 2000., 8000., 2000., 9000.])
    models.train_nuisance_nn(X[:6], Y[:6], X[6:], Y[6:], standardize_inputs=True, **small_fit_kwargs())
    assert len(captured) == 1
    torch.testing.assert_close(captured[0], torch.as_tensor(Y[:6], dtype=torch.float32).unsqueeze(1), rtol=0, atol=0)


@pytest.mark.parametrize("binary", (False, True))
def test_default_and_explicit_false_preserve_plain_network_weights_predictions_and_rng(binary):
    X = np.arange(36, dtype=float).reshape(12, 3) / 10
    Y = np.arange(12) % 2 if binary else np.linspace(1., 5., 12)
    kwargs = dict(epochs=3, hidden_dims=(4,), batch_size=5, binary=binary, early_stop=False)
    torch.manual_seed(57)
    default, _, _ = models.train_nuisance_nn(X[:8], Y[:8], X[8:], Y[8:], **kwargs)
    default_rng = torch.get_rng_state().clone()
    torch.manual_seed(57)
    explicit, _, _ = models.train_nuisance_nn(X[:8], Y[:8], X[8:], Y[8:], standardize_inputs=False, **kwargs)
    assert type(default) is type(explicit) is models.DeepNN
    assert not list(default.named_buffers())
    assert default.state_dict().keys() == explicit.state_dict().keys()
    for key, value in default.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[key])
    np.testing.assert_array_equal(models.predict_nn(default, X, binary=binary),
                                  models.predict_nn(explicit, X, binary=binary))
    assert torch.equal(default_rng, torch.get_rng_state())


@pytest.mark.parametrize("binary", (False, True))
def test_disabled_branch_matches_frozen_v2_models_when_provided(binary):
    baseline_path = os.environ.get("MEDIENC_V3_BASELINE_MODELS")
    if not baseline_path:
        pytest.skip("SCC compatibility test: set MEDIENC_V3_BASELINE_MODELS to frozen 612da44 models.py")
    path = Path(baseline_path)
    assert path.is_file() and path.resolve() != Path(models.__file__).resolve()
    spec = importlib.util.spec_from_file_location("frozen_v2_nuisance_models", path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    X = np.arange(36, dtype=float).reshape(12, 3) / 10
    Y = np.arange(12) % 2 if binary else np.linspace(1., 5., 12)
    kwargs = dict(epochs=3, hidden_dims=(4,), batch_size=5, binary=binary, early_stop=False)
    torch.manual_seed(18)
    original, _, _ = baseline.train_nuisance_nn(X[:8], Y[:8], X[8:], Y[8:], **kwargs)
    original_rng = torch.get_rng_state().clone()
    torch.manual_seed(18)
    candidate, _, _ = models.train_nuisance_nn(X[:8], Y[:8], X[8:], Y[8:], **kwargs)
    assert original.state_dict().keys() == candidate.state_dict().keys()
    for key, value in original.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[key])
    np.testing.assert_array_equal(baseline.predict_nn(original, X, binary=binary),
                                  models.predict_nn(candidate, X, binary=binary))
    assert torch.equal(original_rng, torch.get_rng_state())


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), -float("inf")))
def test_nonfinite_training_validation_and_prediction_inputs_are_rejected(bad):
    X, Y = np.arange(24, dtype=float).reshape(8, 3), np.arange(8, dtype=float)
    for invalid_training in (False, True):
        training, validation = X[:6].copy(), X[6:].copy()
        (training if invalid_training else validation)[0, 0] = bad
        with pytest.raises(ValueError, match="finite"):
            models.train_nuisance_nn(training, Y[:6], validation, Y[6:],
                                     standardize_inputs=True, **small_fit_kwargs())
    fitted, _, _ = models.train_nuisance_nn(X[:6], Y[:6], X[6:], Y[6:],
                                            standardize_inputs=True, **small_fit_kwargs())
    prediction = X[:1].copy(); prediction[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        models.predict_nn(fitted, prediction)


def test_finite_inputs_that_overflow_float32_are_rejected_after_conversion():
    X, Y = np.arange(24, dtype=float).reshape(8, 3), np.arange(8, dtype=float)
    for invalid_training in (False, True):
        training, validation = X[:6].copy(), X[6:].copy()
        (training if invalid_training else validation)[0, 0] = 1e40
        assert np.isfinite(training).all() and np.isfinite(validation).all()
        with pytest.raises(ValueError, match="finite"):
            models.train_nuisance_nn(training, Y[:6], validation, Y[6:],
                standardize_inputs=True, **small_fit_kwargs())
    fitted, _, _ = models.train_nuisance_nn(X[:6], Y[:6], X[6:], Y[6:],
        standardize_inputs=True, **small_fit_kwargs())
    prediction = np.array([[1e40, 2., 3.]])
    assert np.isfinite(prediction).all()
    with pytest.raises(ValueError, match="finite"):
        models.predict_nn(fitted, prediction)


def test_single_training_row_uses_population_sd_and_constant_scale_one():
    X, Y = np.array([[12., -5.]]), np.array([20.])
    fitted, _, _ = models.train_nuisance_nn(X, Y, X, Y, standardize_inputs=True,
                                            **small_fit_kwargs())
    assert_training_statistics(fitted, X)
    torch.testing.assert_close(fitted.input_scale.cpu(), torch.ones(2), rtol=0, atol=0)
    assert np.isfinite(models.predict_nn(fitted, X)).all()


def test_near_constant_features_are_centered_without_amplifying_validation_drift():
    X = np.array([[0., 1.], [1e-10, 2.], [2e-10, 3.], [3e-10, 4.]])
    Y = np.array([100., 200., 300., 400.])
    validation = np.array([[1e-5, 5.], [-1e-5, 6.]])
    fitted, _, _ = models.train_nuisance_nn(X, Y, validation, Y[:2],
        standardize_inputs=True, **small_fit_kwargs())
    assert_training_statistics(fitted, X)
    assert fitted.input_scale[0].item() == 1.
    transformed = ((torch.as_tensor(validation, dtype=torch.float32,
                    device=fitted.input_mean.device) - fitted.input_mean) / fitted.input_scale)
    assert torch.all(torch.abs(transformed[:, 0]) < 2e-5)
    assert np.isfinite(models.predict_nn(fitted, validation)).all()


@pytest.mark.parametrize("flag", ("yes", 1, None))
def test_non_boolean_standardization_switch_is_rejected(flag):
    with pytest.raises(ValueError, match="boolean"):
        models.train_nuisance_nn(np.ones((2, 2)), np.ones(2), np.ones((2, 2)), np.ones(2),
                                 standardize_inputs=flag, **small_fit_kwargs())


def test_every_crossfit_nuisance_and_marginal_model_uses_its_own_training_subset(monkeypatch):
    n = 100
    X = np.column_stack((np.arange(n, dtype=float), np.arange(n, dtype=float) ** 2 / 10))
    M = np.column_stack((np.arange(n, dtype=float) * 3 + 100, np.sin(np.arange(n))))
    A, Y = np.arange(n) % 2, 1000. + np.arange(n) * 10
    captured = []
    original_fit = models.train_nuisance_nn

    def tracked_fit(training, target, validation, validation_target, **kwargs):
        assert kwargs["standardize_inputs"] is True
        fitted = original_fit(training, target, validation, validation_target, **kwargs)
        assert_training_statistics(fitted[0], training)
        captured.append(dict(training=np.asarray(training).copy(), model=fitted[0]))
        return fitted

    def representation(*args, **kwargs):
        return dict(f_X_all=X, f_M_all=M, rep_fit_info=None, resolved_config={})

    monkeypatch.setattr(estimation, "train_nuisance_nn", tracked_fit)
    monkeypatch.setattr(estimation, "_learn_representations_fixed_split", representation)
    monkeypatch.setattr(estimation, "_fit_outcome_predictor_and_eval",
                        lambda *args, **kwargs: dict(prediction_mse=1., prediction_rmse=1.))
    config = small_fit_kwargs(standardize_inputs=True)
    result = estimation.estimate_triply_IF(X, M, A, Y, tilde_p=2, tilde_q=2,
                                           nn_cfg=config, seed=77, return_effects=True)
    assert len(captured) == 24  # Four final nuisances + two marginal outcomes per fold.
    for fold, roles in enumerate(result["fold_indices"]):
        indices = np.asarray(roles["nuisance_train"])
        treated, control = indices[A[indices] == 1], indices[A[indices] == 0]
        expected = (X[indices], np.hstack((X[treated], M[treated])), X[control],
                    np.hstack((X[indices], M[indices])))
        for actual, values in zip(captured[4*fold:4*fold+4], expected):
            np.testing.assert_array_equal(actual["training"], values)
        np.testing.assert_array_equal(captured[16+2*fold]["training"], X[treated])
        np.testing.assert_array_equal(captured[17+2*fold]["training"], X[control])
    assert result["estimator_contract"]["nuisance_input_standardization"] == "each_model_training_subset_mean_population_sd"
    assert result["estimator_contract"]["nuisance_input_constant_sd_threshold"] == 1e-8
    assert result["resolved_config"]["nuisance"]["standardize_inputs"] is True


def test_auxiliary_outcome_prediction_uses_treated_training_subset_statistics(monkeypatch):
    X, M = np.arange(24, dtype=float).reshape(12, 2), 10. + np.arange(24, dtype=float).reshape(12, 2)
    A, Y = np.arange(12) % 2, np.linspace(100., 200., 12)
    original_fit, captured = models.train_nuisance_nn, []

    def tracked(training, target, validation, validation_target, **kwargs):
        assert kwargs["standardize_inputs"] is True
        result = original_fit(training, target, validation, validation_target, **kwargs)
        captured.append((np.asarray(training).copy(), result[0]))
        return result

    monkeypatch.setattr(estimation, "train_nuisance_nn", tracked)
    result = estimation._fit_outcome_predictor_and_eval(X, M, A, Y, subtrain_idx=np.arange(8),
        val_idx=np.arange(8, 12), nn_cfg=small_fit_kwargs(standardize_inputs=True))
    assert len(captured) == 1 and np.isfinite(result["prediction_mse"])
    expected = np.hstack((X[[1, 3, 5, 7]], M[[1, 3, 5, 7]]))
    np.testing.assert_array_equal(captured[0][0], expected)
    assert_training_statistics(captured[0][1], expected)
