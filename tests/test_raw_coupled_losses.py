"""Actual tiny fits check raw losses, checkpointing, and retained simplex rules."""
import numpy as np
import pytest
import torch

from mediencoder import training
from mediencoder.simulation import runner


@pytest.mark.parametrize("variational", [False, True])
@pytest.mark.parametrize("constant_inputs", [False, True])
@pytest.mark.parametrize("weights", [(.2, .5, .3), (.5, .5, 0.)])
def test_raw_training_and_validation_losses(variational, constant_inputs, weights):
    torch.set_num_threads(1)
    torch.manual_seed(721)
    rng = np.random.default_rng(20)
    X = 4 * rng.normal(size=(8, 4))
    M = .3 * rng.normal(size=(8, 3))
    if constant_inputs:
        X[:], M[:] = 2., -1.
    Xv, Mv = 7 * rng.normal(size=(6, 4)), 2 * rng.normal(size=(6, 3))
    A, Av = np.arange(8) % 2, np.arange(6) % 2
    fit = training.train_mediencoder_vae if variational else training.train_mediencoder
    kwargs = dict(beta_kl=.2) if variational else {}
    model, history, info = fit(
        X, M, A, latent_p=2, latent_q=2,
        X_val=Xv, M_val=Mv, A_val=Av, epochs=1, batch_size=8,
        hidden_dims_X=(4,), hidden_dims_M=(4,), hidden_dims_XM=(4,),
        lambda1=weights[0], lambda2=weights[1], lambda3=weights[2],
        allow_unbalanced_lambda=True, activation="tanh", scheduler_type="none", **kwargs)
    assert info["loss_normalization"] == "none"
    assert not {"var_X", "var_M", "var_align", "sd_X", "sd_M", "sd_align"} & info.keys()
    for key in ("loss_X", "loss_M", "loss_align", "weighted_loss"):
        np.testing.assert_allclose(history[key], history["raw_" + key])
        assert np.isfinite(history[key]).all()
    device = next(model.parameters()).device
    xv = torch.as_tensor(Xv, dtype=torch.float32, device=device)
    mv = torch.as_tensor(Mv, dtype=torch.float32, device=device)
    av = torch.as_tensor(Av, dtype=torch.float32, device=device)
    with torch.no_grad():
        if variational:
            mx, vx = model.encode_X(xv)
            mm, vm = model.encode_M(mv)
            xr, mr = model.decoder_X(mx), model.decoder_M(mm)
            penalty = .2 * (training.kl_divergence_standard_normal(mx, vx).item()
                            + training.kl_divergence_standard_normal(mm, vm).item())
        else:
            output = model(xv, mv, av)
            xr, mr, penalty = output["X_recon"], output["M_recon"], 0.
        expected = weights[0] * torch.mean((xr-xv)**2).item() + weights[1] * torch.mean((mr-mv)**2).item() + penalty
    assert info["best_weighted_loss"] == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("fit", [training.train_mediencoder, training.train_mediencoder_vae])
def test_raw_loss_trainers_still_reject_non_simplex_weights(fit):
    with pytest.raises(ValueError, match="lambda1.*lambda2.*lambda3 = 1"):
        fit(np.ones((8, 4)), np.ones((8, 3)), np.arange(8) % 2,
            latent_p=2, latent_q=2, lambda1=.5, lambda2=.5, lambda3=.25,
            allow_unbalanced_lambda=True, epochs=1)


def test_simulation_identity_records_raw_losses_with_unchanged_simplex_grid():
    from mediencoder.estimation import SHARED_TRAIN_CFG
    previous = dict(SHARED_TRAIN_CFG)
    try:
        config = runner.make_config(runner.parse_args(["--output-dir", "unused", "--pilot-epochs", "1"]))
    finally:
        SHARED_TRAIN_CFG.clear()
        SHARED_TRAIN_CFG.update(previous)
    assert config["loss_normalization"] == "none"
    assert config["loss_reduction"] == "mean_over_subjects_and_coordinates"
    assert (len(config["lambda_grid"]), len(config["lambda_grid_zero"])) == (36, 9)
    for weights in config["lambda_grid"] + config["lambda_grid_zero"]:
        assert sum(weights) == pytest.approx(1.)
