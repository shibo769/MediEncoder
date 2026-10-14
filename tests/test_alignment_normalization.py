"""Check scale invariance, gradient routing, and actual coupled-fit histories."""
from unittest.mock import patch

import numpy as np
import pytest
import torch

from mediencoder import training


def test_alignment_normalizes_each_subject_and_each_branch_independently():
    target = torch.tensor([[3., 4., 0.], [0., 5., 12.]], dtype=torch.float64)
    prediction = torch.tensor([[6., 0., 8.], [4., 3., 0.]], dtype=torch.float64)
    expected = torch.mean((target / torch.tensor([[5.], [13.]])
                           - prediction / torch.tensor([[10.], [5.]])) ** 2)
    actual = training._alignment_loss(target, prediction)
    torch.testing.assert_close(actual, expected)
    for target_scale, prediction_scale in ((.001, 500.), (71., .003)):
        torch.testing.assert_close(
            training._alignment_loss(target * target_scale, prediction * prediction_scale),
            actual,
        )
    # Different positive rescalings on each subject must also leave directions unchanged.
    torch.testing.assert_close(
        training._alignment_loss(target * torch.tensor([[7.], [.04]]),
                                 prediction * torch.tensor([[.05], [90.]])),
        actual,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_alignment_zero_and_near_zero_rows_have_finite_loss_and_gradients(dtype):
    target = torch.tensor([[0., 0., 0.], [1e-12, 0., 0.], [1., 2., 3.]],
                          dtype=dtype, requires_grad=True)
    prediction = torch.tensor([[0., 0., 0.], [0., 1e-12, 0.], [3., 1., 2.]],
                              dtype=dtype, requires_grad=True)
    loss = training._alignment_loss(target, prediction)
    expected = torch.mean((target / target.norm(dim=1, keepdim=True).clamp_min(1e-8)
                           - prediction / prediction.norm(dim=1, keepdim=True).clamp_min(1e-8)) ** 2)
    torch.testing.assert_close(loss, expected)
    expected_gradients = torch.autograd.grad(expected, (target, prediction))
    loss.backward()
    assert torch.isfinite(loss)
    for values, expected_gradient in zip((target, prediction), expected_gradients):
        assert values.grad is not None and torch.isfinite(values.grad).all()
        torch.testing.assert_close(values.grad, expected_gradient)


@pytest.mark.parametrize("variational", [False, True])
def test_alignment_trains_both_encoders_and_prediction_branch(variational):
    torch.manual_seed(12)
    cls = training.MediEncoderVAE if variational else training.MediEncoder
    model = cls(4, 3, 2, 2, hidden_dims_X=(4,), hidden_dims_M=(4,),
                hidden_dims_XM=(4,), activation="tanh")
    output = model(torch.randn(5, 4), torch.randn(5, 3), torch.arange(5) % 2)
    training._alignment_loss(output["z_M"], output["z_M_pred"]).backward()
    named = dict(model.named_parameters())
    groups = [("encoder_M",), ("encoder_X",), ("g_XM",)]
    if variational:
        groups += [("mu_M",), ("logvar_M",), ("mu_X",), ("logvar_X",)]
    for prefixes in groups:
        parameters = [p for name, p in named.items() if name.startswith(prefixes)]
        assert parameters
        assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in parameters)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)
    # Alignment acts on the codes, so it does not train reconstruction decoders.
    decoder_parameters = [p for name, p in named.items() if name.startswith(("decoder_X", "decoder_M"))]
    assert decoder_parameters and all(p.grad is None for p in decoder_parameters)


@pytest.mark.parametrize("variational", [False, True])
def test_zero_alignment_weight_preserves_reconstruction_only_gradients(variational):
    torch.manual_seed(12)
    cls = training.MediEncoderVAE if variational else training.MediEncoder
    model = cls(4, 3, 2, 2, hidden_dims_X=(4,), hidden_dims_M=(4,),
                hidden_dims_XM=(4,), activation="tanh")
    X, M = torch.randn(5, 4), torch.randn(5, 3)
    output = model(X, M, torch.arange(5) % 2)
    reconstruction = (.5 * torch.mean((output["X_recon"] - X) ** 2)
                      + .5 * torch.mean((output["M_recon"] - M) ** 2))
    if variational:
        reconstruction = reconstruction + .5 * (
            training.kl_divergence_standard_normal(output["mu_X"], output["logvar_X"])
            + training.kl_divergence_standard_normal(output["mu_M"], output["logvar_M"]))
    parameters = tuple(model.parameters())
    expected = torch.autograd.grad(reconstruction, parameters, retain_graph=True, allow_unused=True)
    objective = reconstruction + 0. * training._alignment_loss(output["z_M"], output["z_M_pred"])
    observed = torch.autograd.grad(objective, parameters, allow_unused=True)
    for actual, reference in zip(observed, expected):
        if reference is None:
            assert actual is None or torch.count_nonzero(actual) == 0
        else:
            assert actual is not None
            torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize("variational", [False, True])
def test_actual_fit_records_normalized_alignment_and_original_raw_mse(variational):
    torch.set_num_threads(1)
    torch.manual_seed(33)
    rng = np.random.default_rng(72)
    X, M, A = rng.normal(size=(7, 4)), rng.normal(size=(7, 3)), np.arange(7) % 2
    observed = []
    original_alignment = training._alignment_loss

    def capture_alignment(target, prediction):
        value = original_alignment(target, prediction)
        observed.append((value.detach().item(),
                         torch.mean((target.detach() - prediction.detach()) ** 2).item()))
        return value

    fit = training.train_mediencoder_vae if variational else training.train_mediencoder
    with patch.object(training, "_alignment_loss", side_effect=capture_alignment):
        _, history, info = fit(X, M, A, latent_p=2, latent_q=2,
                               hidden_dims_X=(4,), hidden_dims_M=(4,), hidden_dims_XM=(4,),
                               epochs=1, batch_size=7, lambda1=.2, lambda2=.5, lambda3=.3)
    assert len(observed) == 1
    assert history["loss_align"][0] == pytest.approx(observed[0][0])
    assert history["raw_loss_align"][0] == pytest.approx(observed[0][1])
    assert observed[0][0] != pytest.approx(observed[0][1])
    assert info["best_loss_align"] == pytest.approx(observed[0][0])
    assert info["best_raw_loss_align"] == pytest.approx(observed[0][1])
    assert info["loss_normalization"] == "alignment_row_l2"
    assert info["alignment_stop_gradient"] is False
