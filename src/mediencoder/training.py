# ============================================================
# MediEncoder_and_Train.py
#
# MediEncoder training module
#
# Main points:
#   1. Train with NORMALIZED losses:
#        loss_X     / sd(X)^2
#        loss_M     / sd(M)^2
#        loss_align / sd(M_train)^2
#      All scales use observed representation-training data only.
#   2. Validation checkpointing uses reconstruction (+ KL for MediVAE).
#   3. Rich history / metadata recording for later analysis
#   4. Lambda helpers support:
#        lambda1 + lambda2 + lambda3 = C
#        optional ordering constraint lambda2 > lambda1
# ============================================================

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

from mediencoder.nn_utils import build_mlp, device


# ============================================================
# 1) Model
# ============================================================

class MediEncoder(nn.Module):
    def __init__(
        self,
        p_dim,
        q_dim,
        latent_p,
        latent_q,
        hidden_dims_X=(300, 300),
        hidden_dims_M=(300, 300),
        hidden_dims_XM=(50, 50),
        activation="relu",
        dropout=0.0
    ):
        super().__init__()

        self.encoder_X = build_mlp(
            p_dim, latent_p, hidden_dims_X, activation, dropout
        )
        self.decoder_X = build_mlp(
            latent_p, p_dim, hidden_dims_X[::-1], activation, dropout
        )

        self.encoder_M = build_mlp(
            q_dim, latent_q, hidden_dims_M, activation, dropout
        )
        self.decoder_M = build_mlp(
            latent_q, q_dim, hidden_dims_M[::-1], activation, dropout
        )

        self.g_XM = build_mlp(
            latent_p + 1, latent_q, hidden_dims_XM, activation, dropout
        )

    def forward(self, X, M, A):
        z_X = self.encoder_X(X)
        z_M = self.encoder_M(M)

        X_recon = self.decoder_X(z_X)
        M_recon = self.decoder_M(z_M)

        A_col = A.unsqueeze(1)
        z_M_pred = self.g_XM(torch.cat([A_col, z_X], dim=1))

        return {
            "z_X": z_X,
            "z_M": z_M,
            "X_recon": X_recon,
            "M_recon": M_recon,
            "z_M_pred": z_M_pred
        }


# ============================================================
# 1b) MediEncoderVAE — VAE version with alignment term
# ============================================================

class MediEncoderVAE(nn.Module):
    """
    VAE variant of MediEncoder.

    Encoder_X and Encoder_M each output (mu, logvar).
    Latent z is sampled via reparameterization trick.
    The alignment term operates on the sampled z_M vs g_XM(A, z_X).

    Loss:
        lambda1 * recon_X + lambda2 * recon_M
        + lambda3 * alignment(z_M, g_XM(A, z_X))
        + beta_kl * (KL_X + KL_M)
    """

    def __init__(
        self,
        p_dim,
        q_dim,
        latent_p,
        latent_q,
        hidden_dims_X=(300, 300),
        hidden_dims_M=(300, 300),
        hidden_dims_XM=(50, 50),
        activation="relu",
        dropout=0.0
    ):
        super().__init__()

        self.latent_p = latent_p
        self.latent_q = latent_q

        # Encoder X: outputs features, then split into mu and logvar
        self.encoder_X_backbone = build_mlp(
            p_dim, hidden_dims_X[-1], hidden_dims_X[:-1], activation, dropout
        )
        self.mu_X = nn.Linear(hidden_dims_X[-1], latent_p)
        self.logvar_X = nn.Linear(hidden_dims_X[-1], latent_p)

        # Decoder X
        self.decoder_X = build_mlp(
            latent_p, p_dim, hidden_dims_X[::-1], activation, dropout
        )

        # Encoder M: outputs features, then split into mu and logvar
        self.encoder_M_backbone = build_mlp(
            q_dim, hidden_dims_M[-1], hidden_dims_M[:-1], activation, dropout
        )
        self.mu_M = nn.Linear(hidden_dims_M[-1], latent_q)
        self.logvar_M = nn.Linear(hidden_dims_M[-1], latent_q)

        # Decoder M
        self.decoder_M = build_mlp(
            latent_q, q_dim, hidden_dims_M[::-1], activation, dropout
        )

        # Cross-factor network (same as deterministic version)
        self.g_XM = build_mlp(
            latent_p + 1, latent_q, hidden_dims_XM, activation, dropout
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_X(self, X):
        h = self.encoder_X_backbone(X)
        return self.mu_X(h), self.logvar_X(h)

    def encode_M(self, M):
        h = self.encoder_M_backbone(M)
        return self.mu_M(h), self.logvar_M(h)

    def forward(self, X, M, A):
        mu_X, logvar_X = self.encode_X(X)
        mu_M, logvar_M = self.encode_M(M)

        z_X = self.reparameterize(mu_X, logvar_X)
        z_M = self.reparameterize(mu_M, logvar_M)

        X_recon = self.decoder_X(z_X)
        M_recon = self.decoder_M(z_M)

        A_col = A.unsqueeze(1)
        z_M_pred = self.g_XM(torch.cat([A_col, z_X], dim=1))

        return {
            "z_X": z_X,
            "z_M": z_M,
            "mu_X": mu_X,
            "logvar_X": logvar_X,
            "mu_M": mu_M,
            "logvar_M": logvar_M,
            "X_recon": X_recon,
            "M_recon": M_recon,
            "z_M_pred": z_M_pred
        }

    def encode_X_mean(self, X):
        """For inference: use mu directly (no sampling)."""
        h = self.encoder_X_backbone(X)
        return self.mu_X(h)

    def encode_M_mean(self, M):
        """For inference: use mu directly (no sampling)."""
        h = self.encoder_M_backbone(M)
        return self.mu_M(h)


def kl_divergence_standard_normal(mu, logvar):
    """KL(q(z|x) || N(0,I)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))"""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


# ============================================================
# 2) Lambda helpers
# ============================================================

def validate_lambdas(lambda1, lambda2, lambda3, C=1.0, tol=1e-10):
    if lambda1 < -tol or lambda2 < -tol or lambda3 < -tol:
        return False
    # strict positivity on the two reconstruction weights: lambda1 = 0 leaves
    # theta_X unconstrained (argmin set = whole parameter space, so the encoder
    # is its random init, not an estimator), lambda2 = 0 does the same to
    # theta_M. lambda3 = 0 is well posed and stays allowed.
    if lambda1 <= tol or lambda2 <= tol:
        return False
    if abs((lambda1 + lambda2 + lambda3) - C) > tol:
        return False
    if not (lambda2 > lambda1):
        return False
    return True


def validate_lambdas_unbalanced(lambda1, lambda2, lambda3, C=1.0, tol=1e-10):
    """
    Relaxed validation: the ordering lambda2 > lambda1 is dropped, but the two
    reconstruction weights must still be STRICTLY positive. At lambda1 = 0 the
    X-reconstruction block leaves the objective, so theta_X is unconstrained --
    the argmin set is the whole parameter space and the encoder returned is its
    random initialisation, not an estimator. Symmetrically for lambda2 = 0 and
    theta_M. lambda3 = 0 is well posed and accepted.
    """
    if lambda1 < -tol or lambda2 < -tol or lambda3 < -tol:
        return False
    # strict positivity on the two reconstruction weights: lambda1 = 0 leaves
    # theta_X unconstrained (argmin set = whole parameter space, so the encoder
    # is its random init, not an estimator), lambda2 = 0 does the same to
    # theta_M. lambda3 = 0 is well posed and stays allowed.
    if lambda1 <= tol or lambda2 <= tol:
        return False
    if abs((lambda1 + lambda2 + lambda3) - C) > tol:
        return False
    return True


def generate_lambda_grid(
    C=1.0, step=0.05, tol=1e-10, require_order=True,
    require_positive_recon=True
):
    """
    Generate lambda tuples satisfying:
        lambda1 + lambda2 + lambda3 = C
        lambda3 >= 0
        and optionally lambda2 > lambda1

    require_positive_recon : bool, default True
        Drop the degenerate corners lambda1 = 0 and lambda2 = 0. At lambda1 = 0
        the X-reconstruction block leaves the objective, theta_X is
        unconstrained, the argmin set is the whole parameter space and the
        returned encoder is its random initialisation rather than an estimator;
        symmetrically for lambda2 = 0 and theta_M. lambda3 = 0 is well posed and
        is always kept: the objective then separates into the two
        reconstruction blocks, each of which still has a unique argmin.
    """
    if C <= 0:
        raise ValueError("C must be positive.")
    if step <= 0:
        raise ValueError("step must be positive.")

    grid = []
    l1_values = np.arange(0.0, C + tol, step)

    for l1 in l1_values:
        l2_values = np.arange(0.0, C - l1 + tol, step)
        for l2 in l2_values:
            if require_order and not (l2 > l1):
                continue

            if require_positive_recon and (l1 <= tol or l2 <= tol):
                continue

            l3 = C - l1 - l2
            if l3 < -tol:
                continue

            l3 = max(0.0, float(l3))

            if require_order:
                ok = validate_lambdas(float(l1), float(l2), l3, C=C, tol=1e-8)
            else:
                ok = validate_lambdas_unbalanced(float(l1), float(l2), l3, C=C, tol=1e-8)

            if ok:
                grid.append((float(l1), float(l2), l3))

    return grid


def generate_log_lambda_candidates(C=1.0, num_points=20, min_exp=-4.0, max_exp=0.0):
    """
    Generate positive log-scale candidate values:
        C * 10^e,  e in [min_exp, max_exp]
    """
    if C <= 0:
        raise ValueError("C must be positive.")
    if num_points <= 0:
        raise ValueError("num_points must be positive.")

    exp_grid = np.linspace(min_exp, max_exp, num_points)
    base_grid = 10.0 ** exp_grid
    return C * base_grid


def generate_lambda_grid_from_candidates(
    lambda_candidates,
    C=1.0,
    tol=1e-10,
    require_order=False
):
    """
    Build lambda grid from a given candidate list, e.g. log-scale candidates.

    Parameters
    ----------
    lambda_candidates : iterable
        Candidate values for lambda1 and lambda2.
    C : float
        Sum constraint.
    require_order : bool
        If True, enforce lambda2 > lambda1.
    """
    if C <= 0:
        raise ValueError("C must be positive.")

    lambda_candidates = sorted(set(float(x) for x in lambda_candidates if x >= 0))
    grid = []

    for lam1 in lambda_candidates:
        for lam2 in lambda_candidates:
            lam3 = C - lam1 - lam2

            if lam3 < -tol:
                continue

            lam3 = max(0.0, float(lam3))

            if require_order:
                ok = validate_lambdas(lam1, lam2, lam3, C=C, tol=1e-8)
            else:
                ok = validate_lambdas_unbalanced(lam1, lam2, lam3, C=C, tol=1e-8)

            if ok:
                grid.append((float(lam1), float(lam2), float(lam3)))

    # remove duplicates while preserving order
    seen = set()
    unique_grid = []
    for triple in grid:
        key = tuple(round(v, 12) for v in triple)
        if key not in seen:
            seen.add(key)
            unique_grid.append(triple)

    return unique_grid


def generate_ablation_lambda_grid(
    fixed_lambda_idx=3,
    fixed_value=0.0,
    step=0.1,
    C=1.0,
    tol=1e-10,
    require_order=False
):
    """
    Generate lambda grid for ablation:
        lambda1 + lambda2 + lambda3 = C
        one lambda is fixed to fixed_value
        the remaining two are tuned over a 1D grid
    """
    if fixed_lambda_idx not in {1, 2, 3}:
        raise ValueError("fixed_lambda_idx must be one of {1,2,3}.")
    if fixed_value < -tol or fixed_value > C + tol:
        raise ValueError("fixed_value must lie in [0, C].")
    if step <= 0:
        raise ValueError("step must be positive.")
    if C <= 0:
        raise ValueError("C must be positive.")

    free_sum = C - fixed_value
    if free_sum < -tol:
        raise ValueError("fixed_value cannot exceed C.")

    grid = []
    # start at `step`, not 0, so the tuned lambda is strictly positive. The
    # derived partner `other = free_sum - v` needs the same lower bound, which
    # is why the loop stops one step short of free_sum: with v = free_sum the
    # partner would be 0.
    free_vals = np.arange(step, free_sum + tol, step)

    for v in free_vals:
        other = free_sum - v

        if other < -tol:
            continue

        other = max(0.0, float(other))

        if fixed_lambda_idx == 1:
            lam1 = float(fixed_value)
            lam2 = float(v)
            lam3 = float(other)
        elif fixed_lambda_idx == 2:
            lam1 = float(v)
            lam2 = float(fixed_value)
            lam3 = float(other)
        else:
            lam1 = float(v)
            lam2 = float(other)
            lam3 = float(fixed_value)

        if lam1 < -tol or lam2 < -tol or lam3 < -tol:
            continue
        if abs((lam1 + lam2 + lam3) - C) > 1e-8:
            continue
        if require_order and not (lam2 > lam1):
            continue

        # Reject the degenerate corners. With lambda1 = 0 the X-reconstruction
        # block leaves the objective entirely, so theta_X is unconstrained: the
        # argmin set is the whole parameter space and the returned encoder is
        # its random initialisation, not an estimator. Same for lambda2 = 0 and
        # theta_M. lambda3 = 0 IS well posed -- the objective then separates
        # into the two reconstruction blocks, each of which still has a unique
        # argmin -- so it is only excluded when it is one of the *tuned*
        # coordinates hitting the boundary, never when it is fixed_value.
        if fixed_lambda_idx != 1 and lam1 <= tol:
            continue
        if fixed_lambda_idx != 2 and lam2 <= tol:
            continue

        grid.append((lam1, lam2, lam3))

    if len(grid) == 0:
        raise ValueError(
            "generate_ablation_lambda_grid produced an empty grid after "
            "removing the degenerate lambda1 = 0 / lambda2 = 0 corners. "
            f"fixed_lambda_idx={fixed_lambda_idx}, fixed_value={fixed_value}, "
            f"step={step}, C={C}."
        )

    return grid


# ============================================================
# 3) Scale helpers
# ============================================================

def _safe_global_sd(arr, eps=1e-8):
    """
    Compute a single global SD over the whole array.
    This matches the idea of dividing each loss block by one scale.
    """
    arr = np.asarray(arr, dtype=float)
    sd = float(np.std(arr))
    return max(sd, eps)


def compute_loss_scales(X_train, M_train, eps=1e-8):
    """Freeze global observed-data scales on the representation-training fold.

    Each loss is a mean over subjects AND coordinates. Alignment uses the
    observed mediator variance in both simulation and real-data fits.
    """
    for name, values in (("X_train", X_train), ("M_train", M_train)):
        values = np.asarray(values)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain finite observed values")
    sd_X = _safe_global_sd(X_train, eps=eps)
    sd_M = _safe_global_sd(M_train, eps=eps)
    return {
        "sd_X": sd_X, "sd_M": sd_M, "sd_align": sd_M,
        "var_X": sd_X ** 2, "var_M": sd_M ** 2,
        "var_align": sd_M ** 2,
    }


# ============================================================
# 4) Training
# ============================================================

def _alignment_loss(z_M, z_M_pred):
    """Retained stop-gradient update: only the prediction branch receives gradients."""
    return nn.functional.mse_loss(z_M.detach(), z_M_pred)


def train_mediencoder(
    X_train,
    M_train,
    A_train,
    *,
    latent_p,
    latent_q,
    X_val=None,
    M_val=None,
    A_val=None,
    hidden_dims_X=(300, 300),
    hidden_dims_M=(300, 300),
    hidden_dims_XM=(50, 50),
    activation="relu",
    dropout=0.0,
    lambda1=0.2,
    lambda2=0.5,
    lambda3=0.3,
    epochs=150,   # run_and_eval.SHARED_TRAIN_CFG, identical for every method
    lr_init=1e-3,
    # 0.0 for every trainer. Adam's weight_decay is an L2 penalty on the
    # weights, i.e. a capacity control whose right value depends on the
    # architecture -- so any nonzero value favours whichever method happens to
    # suit it, and a per-method value is precisely the unequalised knob being
    # removed. Held-out early stopping is the capacity control that remains,
    # and that one is identical across methods.
    weight_decay=0.0,
    betas=(0.9, 0.999),
    adam_eps=1e-8,
    batch_size=512,
    scheduler_type="none",
    step_size=30,
    gamma=0.5,
    patience=25,
    early_stop=True,
    min_delta=1e-4,
    verbose=False,
    return_history=True,
    allow_unbalanced_lambda=False,
):
    """
    Train MediEncoder using NORMALIZED losses.

    Objective:
        lambda1 * mse(X_recon, X) / sd(X)^2
      + lambda2 * mse(M_recon, M) / sd(M)^2
      + lambda3 * mse(z_M, z_M_pred) / sd(M_train)^2

    Alignment uses the observed M_train scale for every dataset.

    If validation data are supplied, choose the checkpoint using weighted
    reconstruction loss only, reusing the frozen training-fold scales.
    Otherwise it falls back to the training weighted loss. Pass a validation set whenever one
    is available.
    """

    if allow_unbalanced_lambda:
        ok = validate_lambdas_unbalanced(
            lambda1, lambda2, lambda3, C=1.0, tol=1e-6
        )
        if not ok:
            raise ValueError(
                "Invalid lambda combination. Need lambda1 + lambda2 + lambda3 = 1, "
                "lambda1 > 0, lambda2 > 0, lambda3 >= 0. At lambda1 = 0 or "
                "lambda2 = 0 the corresponding reconstruction block leaves the "
                "objective, that encoder is unconstrained, and the fit returns "
                "its random initialisation."
            )
    else:
        ok = validate_lambdas(lambda1, lambda2, lambda3, C=1.0, tol=1e-6)
        if not ok:
            raise ValueError(
                "Invalid lambda combination. Need lambda1 + lambda2 + lambda3 = 1, "
                "lambda2 > lambda1, lambda1 > 0, lambda2 > 0, lambda3 >= 0."
            )

    if epochs < 1:
        raise ValueError("Training epochs must be positive")
    scales = compute_loss_scales(X_train, M_train, eps=1e-8)

    var_X = scales["var_X"]
    var_M = scales["var_M"]
    var_align = scales["var_align"]

    X_t = torch.tensor(X_train, dtype=torch.float32)
    M_t = torch.tensor(M_train, dtype=torch.float32)
    A_t = torch.tensor(A_train, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_t, M_t, A_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    model = MediEncoder(
        p_dim=X_train.shape[1],
        q_dim=M_train.shape[1],
        latent_p=latent_p,
        latent_q=latent_q,
        hidden_dims_X=hidden_dims_X,
        hidden_dims_M=hidden_dims_M,
        hidden_dims_XM=hidden_dims_XM,
        activation=activation,
        dropout=dropout
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr_init,
        weight_decay=weight_decay,
        betas=betas,
        eps=adam_eps
    )

    if scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    elif scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )
    elif scheduler_type == "none":
        scheduler = None
    else:
        raise ValueError("scheduler_type must be one of {'none','step','cosine'}")

    history = {
        "loss_X": [],
        "loss_M": [],
        "loss_align": [],
        "weighted_loss": [],
        "raw_loss_X": [],
        "raw_loss_M": [],
        "raw_loss_align": [],
        "raw_weighted_loss": [],
        "lr": [],
        "epoch": [],
        "val_weighted_loss": []
    }

    best_state = None
    best_epoch = -1
    best_weighted_loss = np.inf
    no_improve = 0

    mse_fn = nn.functional.mse_loss

    use_val = X_val is not None and M_val is not None and A_val is not None
    if use_val:
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        M_val_t = torch.tensor(M_val, dtype=torch.float32).to(device)
        A_val_t = torch.tensor(A_val, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        model.train()

        total_n = 0
        sum_loss_X = 0.0
        sum_loss_M = 0.0
        sum_loss_align = 0.0
        sum_weighted = 0.0

        sum_raw_X = 0.0
        sum_raw_M = 0.0
        sum_raw_align = 0.0
        sum_raw_weighted = 0.0

        for xb, mb, ab in loader:
            xb = xb.to(device)
            mb = mb.to(device)
            ab = ab.to(device)

            optimizer.zero_grad(set_to_none=True)

            out = model(xb, mb, ab)

            X_recon = out["X_recon"]
            M_recon = out["M_recon"]
            z_M = out["z_M"]
            z_M_pred = out["z_M_pred"]

            raw_loss_X = mse_fn(X_recon, xb)
            raw_loss_M = mse_fn(M_recon, mb)
            # Detach alignment TARGET z_M: lambda3 trains g_XM (and encoder_X via z_X)
            # to PREDICT the mediator code, not drag encoder_M's code down to be
            # predictable. Without detach the gradient collapses z_M's scale
            # (std ~0.2 vs ~2.7), destroying the arm-difference and the NIE.
            raw_loss_align = _alignment_loss(z_M, z_M_pred)

            loss_X = raw_loss_X / var_X
            loss_M = raw_loss_M / var_M
            loss_align = raw_loss_align / var_align

            weighted_loss = (
                lambda1 * loss_X +
                lambda2 * loss_M +
                lambda3 * loss_align
            )

            raw_weighted_loss = (
                lambda1 * raw_loss_X +
                lambda2 * raw_loss_M +
                lambda3 * raw_loss_align
            )

            if not torch.isfinite(weighted_loss):
                raise FloatingPointError("Nonfinite coupled-encoder training loss")
            weighted_loss.backward()
            optimizer.step()

            bs = xb.size(0)
            total_n += bs

            sum_loss_X += loss_X.item() * bs
            sum_loss_M += loss_M.item() * bs
            sum_loss_align += loss_align.item() * bs
            sum_weighted += weighted_loss.item() * bs

            sum_raw_X += raw_loss_X.item() * bs
            sum_raw_M += raw_loss_M.item() * bs
            sum_raw_align += raw_loss_align.item() * bs
            sum_raw_weighted += raw_weighted_loss.item() * bs

        if scheduler is not None:
            scheduler.step()

        avg_loss_X = sum_loss_X / total_n
        avg_loss_M = sum_loss_M / total_n
        avg_loss_align = sum_loss_align / total_n
        avg_weighted = sum_weighted / total_n

        avg_raw_X = sum_raw_X / total_n
        avg_raw_M = sum_raw_M / total_n
        avg_raw_align = sum_raw_align / total_n
        avg_raw_weighted = sum_raw_weighted / total_n

        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["loss_X"].append(avg_loss_X)
        history["loss_M"].append(avg_loss_M)
        history["loss_align"].append(avg_loss_align)
        history["weighted_loss"].append(avg_weighted)

        history["raw_loss_X"].append(avg_raw_X)
        history["raw_loss_M"].append(avg_raw_M)
        history["raw_loss_align"].append(avg_raw_align)
        history["raw_weighted_loss"].append(avg_raw_weighted)

        history["lr"].append(current_lr)

        # Validation monitors reconstruction only, using frozen training scales.
        # Without validation, retain the training-weighted-loss fallback.
        monitor = avg_weighted
        if use_val:
            model.eval()
            with torch.no_grad():
                out_v = model(X_val_t, M_val_t, A_val_t)
                v_X = mse_fn(out_v["X_recon"], X_val_t).item() / var_X
                v_M = mse_fn(out_v["M_recon"], M_val_t).item() / var_M
                # Checkpoint on RECONSTRUCTION only. The align term is
                # minimized by a near-init tiny-scale z_M, so including it
                # makes early-stopping pick a collapsed encoder_M (best_ep~0)
                # and destroys the arm-difference / NIE. Selection over lambda
                # is what weighs alignment; the checkpoint just avoids overfit.
                monitor = lambda1 * v_X + lambda2 * v_M
            history["val_weighted_loss"].append(monitor)

        if not np.isfinite(monitor):
            raise FloatingPointError("Nonfinite coupled-encoder checkpoint criterion")
        if monitor < best_weighted_loss - min_delta:
            best_weighted_loss = monitor
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            print(
                f"[Epoch {epoch:03d}] "
                f"weighted={avg_weighted:.6f} | "
                f"X={avg_loss_X:.6f}, "
                f"M={avg_loss_M:.6f}, "
                f"align={avg_loss_align:.6f} | "
                f"lr={current_lr:.3e}"
            )

        if early_stop and no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}. Best epoch = {best_epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    fit_info = {
        "alignment_stop_gradient": True,
        "scale_source": "observed_representation_training_fold",
        "checkpoint_criterion": "reconstruction" if use_val else "training_weighted_loss",
        "lambda1": float(lambda1),
        "lambda2": float(lambda2),
        "lambda3": float(lambda3),
        "best_epoch": int(best_epoch),
        "best_weighted_loss": float(best_weighted_loss),
        "final_epoch": int(history["epoch"][-1]) if len(history["epoch"]) > 0 else -1,
        "sd_X": float(scales["sd_X"]),
        "sd_M": float(scales["sd_M"]),
        "sd_align": float(scales["sd_align"]),
        "var_X": float(scales["var_X"]),
        "var_M": float(scales["var_M"]),
        "var_align": float(scales["var_align"]),
        "best_loss_X": float(history["loss_X"][best_epoch]) if best_epoch >= 0 else np.nan,
        "best_loss_M": float(history["loss_M"][best_epoch]) if best_epoch >= 0 else np.nan,
        "best_loss_align": float(history["loss_align"][best_epoch]) if best_epoch >= 0 else np.nan,
        "best_raw_loss_X": float(history["raw_loss_X"][best_epoch]) if best_epoch >= 0 else np.nan,
        "best_raw_loss_M": float(history["raw_loss_M"][best_epoch]) if best_epoch >= 0 else np.nan,
        "best_raw_loss_align": float(history["raw_loss_align"][best_epoch]) if best_epoch >= 0 else np.nan
    }

    if return_history:
        return model, history, fit_info
    else:
        return model, fit_info


# ============================================================
# 5) Encoding helpers
# ============================================================

@torch.no_grad()
def encode_with_mediencoder(
    model,
    X=None,
    M=None,
    A=None,
    part="X",
    batch_size=4096
):
    """
    Encode X or M using a trained MediEncoder.

    part="M" returns encoder_M(M). There used to be a `use_gxm` switch that made
    this return g_XM(A, encoder_X(X)) instead -- a path that never touches M, so
    f_M became a deterministic function of (A, X) and every part of the mediator
    not predictable from (A, f_X) was discarded before the nuisances were fit.
    g_XM belongs to the TRAINING objective (the lambda3 alignment term shapes
    encoder_M through it); it is not the representation to plug into the
    estimator. A is still accepted and ignored so that callers passing it keep
    working.
    """
    model.eval()

    if part not in {"X", "M"}:
        raise ValueError("part must be 'X' or 'M'")

    if part == "X":
        if X is None:
            raise ValueError("X must be provided when part='X'.")

        X_t = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(X_t),
            batch_size=batch_size,
            shuffle=False
        )

        outputs = []
        for (xb,) in loader:
            xb = xb.to(device)
            z_X = model.encoder_X(xb)
            outputs.append(z_X.cpu())

        return torch.cat(outputs, dim=0).numpy()

    if M is None:
        raise ValueError("M must be provided when part='M'.")

    M_t = torch.tensor(M, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(M_t),
        batch_size=batch_size,
        shuffle=False
    )

    outputs = []
    for (mb,) in loader:
        mb = mb.to(device)
        z_M = model.encoder_M(mb)
        outputs.append(z_M.cpu())

    return torch.cat(outputs, dim=0).numpy()


# ============================================================
# 6) VAE Training
# ============================================================

def train_mediencoder_vae(
    X_train, M_train, A_train, *,
    latent_p, latent_q,
    X_val=None, M_val=None, A_val=None,
    hidden_dims_X=(300, 300),
    hidden_dims_M=(300, 300),
    hidden_dims_XM=(50, 50),
    activation="relu",
    dropout=0.0,
    lambda1=0.2, lambda2=0.5, lambda3=0.3,
    beta_kl=1.0,
    epochs=150,   # run_and_eval.SHARED_TRAIN_CFG, identical for every method
    lr_init=1e-3,
    # 0.0 for every trainer. Adam's weight_decay is an L2 penalty on the
    # weights, i.e. a capacity control whose right value depends on the
    # architecture -- so any nonzero value favours whichever method happens to
    # suit it, and a per-method value is precisely the unequalised knob being
    # removed. Held-out early stopping is the capacity control that remains,
    # and that one is identical across methods.
    weight_decay=0.0,
    betas=(0.9, 0.999),
    adam_eps=1e-8,
    batch_size=512,
    scheduler_type="none",
    step_size=30, gamma=0.5,
    patience=25,
    early_stop=True,
    min_delta=1e-4,
    verbose=False,
    return_history=True,
    allow_unbalanced_lambda=False,
):
    """
    Train MediEncoderVAE using NORMALIZED losses + KL divergence.

    Objective:
        lambda1 * mse(X_recon, X) / sd(X)^2
      + lambda2 * mse(M_recon, M) / sd(M)^2
      + lambda3 * mse(z_M, z_M_pred) / sd(M_train)^2
      + beta_kl * (KL_X + KL_M)

    Alignment uses the observed M_train scale for every dataset.
    """

    if allow_unbalanced_lambda:
        ok = validate_lambdas_unbalanced(
            lambda1, lambda2, lambda3, C=1.0, tol=1e-6
        )
        if not ok:
            raise ValueError(
                "Invalid lambda combination. Need lambda1 + lambda2 + lambda3 = 1, "
                "lambda1 > 0, lambda2 > 0, lambda3 >= 0. At lambda1 = 0 or "
                "lambda2 = 0 the corresponding reconstruction block leaves the "
                "objective, that encoder is unconstrained, and the fit returns "
                "its random initialisation."
            )
    else:
        ok = validate_lambdas(lambda1, lambda2, lambda3, C=1.0, tol=1e-6)
        if not ok:
            raise ValueError(
                "Invalid lambda combination. Need lambda1 + lambda2 + lambda3 = 1, "
                "lambda2 > lambda1, lambda1 > 0, lambda2 > 0, lambda3 >= 0."
            )

    if epochs < 1:
        raise ValueError("Training epochs must be positive")
    scales = compute_loss_scales(X_train, M_train, eps=1e-8)

    var_X = scales["var_X"]
    var_M = scales["var_M"]
    var_align = scales["var_align"]

    X_t = torch.tensor(X_train, dtype=torch.float32)
    M_t = torch.tensor(M_train, dtype=torch.float32)
    A_t = torch.tensor(A_train, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_t, M_t, A_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    model = MediEncoderVAE(
        p_dim=X_train.shape[1],
        q_dim=M_train.shape[1],
        latent_p=latent_p,
        latent_q=latent_q,
        hidden_dims_X=hidden_dims_X,
        hidden_dims_M=hidden_dims_M,
        hidden_dims_XM=hidden_dims_XM,
        activation=activation,
        dropout=dropout
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr_init,
        weight_decay=weight_decay,
        betas=betas,
        eps=adam_eps
    )

    if scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    elif scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )
    elif scheduler_type == "none":
        scheduler = None
    else:
        raise ValueError("scheduler_type must be one of {'none','step','cosine'}")

    history = {
        "loss_X": [],
        "loss_M": [],
        "loss_align": [],
        "kl_X": [],
        "kl_M": [],
        "weighted_loss": [],
        "raw_loss_X": [],
        "raw_loss_M": [],
        "raw_loss_align": [],
        "raw_weighted_loss": [],
        "lr": [],
        "epoch": [],
        "val_weighted_loss": []
    }

    best_state = None
    best_epoch = -1
    best_weighted_loss = np.inf
    no_improve = 0

    mse_fn = nn.functional.mse_loss

    # Validation monitors reconstruction plus KL, using frozen training scales.
    # Without validation, retain the training-weighted-loss fallback.
    use_val = X_val is not None and M_val is not None and A_val is not None
    if use_val:
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        M_val_t = torch.tensor(M_val, dtype=torch.float32).to(device)
        A_val_t = torch.tensor(A_val, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        model.train()

        total_n = 0
        sum_loss_X = 0.0
        sum_loss_M = 0.0
        sum_loss_align = 0.0
        sum_kl_X = 0.0
        sum_kl_M = 0.0
        sum_weighted = 0.0

        sum_raw_X = 0.0
        sum_raw_M = 0.0
        sum_raw_align = 0.0
        sum_raw_weighted = 0.0

        for xb, mb, ab in loader:
            xb = xb.to(device)
            mb = mb.to(device)
            ab = ab.to(device)

            optimizer.zero_grad(set_to_none=True)

            out = model(xb, mb, ab)

            raw_loss_X = mse_fn(out["X_recon"], xb)
            raw_loss_M = mse_fn(out["M_recon"], mb)
            raw_loss_align = _alignment_loss(out["z_M"], out["z_M_pred"])

            kl_X = kl_divergence_standard_normal(out["mu_X"], out["logvar_X"])
            kl_M = kl_divergence_standard_normal(out["mu_M"], out["logvar_M"])

            loss_X = raw_loss_X / var_X
            loss_M = raw_loss_M / var_M
            loss_align = raw_loss_align / var_align

            weighted_loss = (
                lambda1 * loss_X +
                lambda2 * loss_M +
                lambda3 * loss_align +
                beta_kl * (kl_X + kl_M)
            )

            raw_weighted_loss = (
                lambda1 * raw_loss_X +
                lambda2 * raw_loss_M +
                lambda3 * raw_loss_align +
                beta_kl * (kl_X + kl_M)
            )

            if not torch.isfinite(weighted_loss):
                raise FloatingPointError("Nonfinite coupled-encoder training loss")
            weighted_loss.backward()
            optimizer.step()

            bs = xb.size(0)
            total_n += bs

            sum_loss_X += loss_X.item() * bs
            sum_loss_M += loss_M.item() * bs
            sum_loss_align += loss_align.item() * bs
            sum_kl_X += kl_X.item() * bs
            sum_kl_M += kl_M.item() * bs
            sum_weighted += weighted_loss.item() * bs

            sum_raw_X += raw_loss_X.item() * bs
            sum_raw_M += raw_loss_M.item() * bs
            sum_raw_align += raw_loss_align.item() * bs
            sum_raw_weighted += raw_weighted_loss.item() * bs

        if scheduler is not None:
            scheduler.step()

        avg_loss_X = sum_loss_X / total_n
        avg_loss_M = sum_loss_M / total_n
        avg_loss_align = sum_loss_align / total_n
        avg_kl_X = sum_kl_X / total_n
        avg_kl_M = sum_kl_M / total_n
        avg_weighted = sum_weighted / total_n

        avg_raw_X = sum_raw_X / total_n
        avg_raw_M = sum_raw_M / total_n
        avg_raw_align = sum_raw_align / total_n
        avg_raw_weighted = sum_raw_weighted / total_n

        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["loss_X"].append(avg_loss_X)
        history["loss_M"].append(avg_loss_M)
        history["loss_align"].append(avg_loss_align)
        history["kl_X"].append(avg_kl_X)
        history["kl_M"].append(avg_kl_M)
        history["weighted_loss"].append(avg_weighted)

        history["raw_loss_X"].append(avg_raw_X)
        history["raw_loss_M"].append(avg_raw_M)
        history["raw_loss_align"].append(avg_raw_align)
        history["raw_weighted_loss"].append(avg_raw_weighted)

        history["lr"].append(current_lr)

        # Held-out reconstruction plus KL; otherwise the training weighted loss.
        monitor = avg_weighted
        if use_val:
            model.eval()
            with torch.no_grad():
                # deterministic pass: use the posterior means, not samples, so
                # the monitor is not perturbed by reparameterisation noise
                mu_Xv = model.encode_X_mean(X_val_t)
                mu_Mv = model.encode_M_mean(M_val_t)
                Xv_recon = model.decoder_X(mu_Xv)
                Mv_recon = model.decoder_M(mu_Mv)
                v_X = mse_fn(Xv_recon, X_val_t).item() / var_X
                v_M = mse_fn(Mv_recon, M_val_t).item() / var_M
                # the KL block is part of the training objective, so keep it in
                # the monitor as well
                lv_Xv = model.logvar_X(model.encoder_X_backbone(X_val_t))
                lv_Mv = model.logvar_M(model.encoder_M_backbone(M_val_t))
                v_kl = (
                    kl_divergence_standard_normal(mu_Xv, lv_Xv).item() +
                    kl_divergence_standard_normal(mu_Mv, lv_Mv).item()
                )
                # Checkpoint on reconstruction (+KL) only; see train_mediencoder.
                monitor = (
                    lambda1 * v_X + lambda2 * v_M +
                    beta_kl * v_kl
                )
            history["val_weighted_loss"].append(monitor)

        if not np.isfinite(monitor):
            raise FloatingPointError("Nonfinite coupled-encoder checkpoint criterion")
        if monitor < best_weighted_loss - min_delta:
            best_weighted_loss = monitor
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            print(
                f"[Epoch {epoch:03d}] "
                f"weighted={avg_weighted:.6f} | "
                f"X={avg_loss_X:.6f}, "
                f"M={avg_loss_M:.6f}, "
                f"align={avg_loss_align:.6f}, "
                f"kl_X={avg_kl_X:.6f}, "
                f"kl_M={avg_kl_M:.6f} | "
                f"lr={current_lr:.3e}"
            )

        if early_stop and no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}. Best epoch = {best_epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    fit_info = {
        "alignment_stop_gradient": True,
        "scale_source": "observed_representation_training_fold",
        "checkpoint_criterion": "reconstruction_plus_kl" if use_val else "training_weighted_loss",
        "lambda1": float(lambda1),
        "lambda2": float(lambda2),
        "lambda3": float(lambda3),
        "beta_kl": float(beta_kl),
        "best_epoch": int(best_epoch),
        "best_weighted_loss": float(best_weighted_loss),
        "final_epoch": int(history["epoch"][-1]) if len(history["epoch"]) > 0 else -1,
        "sd_X": float(scales["sd_X"]),
        "sd_M": float(scales["sd_M"]),
        "sd_align": float(scales["sd_align"]),
        "var_X": float(scales["var_X"]),
        "var_M": float(scales["var_M"]),
        "var_align": float(scales["var_align"]),
        "best_loss_X": float(history["loss_X"][best_epoch]) if best_epoch >= 0 else np.nan,
        "best_loss_M": float(history["loss_M"][best_epoch]) if best_epoch >= 0 else np.nan,
        "best_loss_align": float(history["loss_align"][best_epoch]) if best_epoch >= 0 else np.nan,
        "best_kl_X": float(history["kl_X"][best_epoch]) if best_epoch >= 0 else np.nan,
        "best_kl_M": float(history["kl_M"][best_epoch]) if best_epoch >= 0 else np.nan,
        "best_raw_loss_X": float(history["raw_loss_X"][best_epoch]) if best_epoch >= 0 else np.nan,
        "best_raw_loss_M": float(history["raw_loss_M"][best_epoch]) if best_epoch >= 0 else np.nan,
        "best_raw_loss_align": float(history["raw_loss_align"][best_epoch]) if best_epoch >= 0 else np.nan
    }

    if return_history:
        return model, history, fit_info
    else:
        return model, fit_info


# ============================================================
# 7) VAE Encoding helpers
# ============================================================

@torch.no_grad()
def encode_with_mediencoder_vae(
    model,
    X=None,
    M=None,
    A=None,
    part="X",
    batch_size=4096
):
    """
    Encode X or M using a trained MediEncoderVAE.

    Uses the mean (mu) for encoding at inference time (no sampling).

    part="M" returns encode_M_mean(M); see encode_with_mediencoder for why the
    g_XM path is gone. A is accepted and ignored.
    """
    model.eval()

    if part not in {"X", "M"}:
        raise ValueError("part must be 'X' or 'M'")

    if part == "X":
        if X is None:
            raise ValueError("X must be provided when part='X'.")

        X_t = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(X_t),
            batch_size=batch_size,
            shuffle=False
        )

        outputs = []
        for (xb,) in loader:
            xb = xb.to(device)
            z_X = model.encode_X_mean(xb)
            outputs.append(z_X.cpu())

        return torch.cat(outputs, dim=0).numpy()

    if M is None:
        raise ValueError("M must be provided when part='M'.")

    M_t = torch.tensor(M, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(M_t),
        batch_size=batch_size,
        shuffle=False
    )

    outputs = []
    for (mb,) in loader:
        mb = mb.to(device)
        z_M = model.encode_M_mean(mb)
        outputs.append(z_M.cpu())

        return torch.cat(outputs, dim=0).numpy()
