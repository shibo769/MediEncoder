# ============================================================
# NNModel_and_Train.py (FULLY FLEXIBLE VERSION)
# ============================================================

import copy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from mediencoder.nn_utils import build_mlp, device

# ============================================================
# 1) Generic NN for nuisance
# ============================================================

class DeepNN(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim=1,
        hidden_dims=(300, 300, 300),
        activation="relu",
        dropout=0.0
    ):
        super().__init__()
        self.net = build_mlp(
            input_dim,
            output_dim,
            hidden_dims,
            activation,
            dropout
        )

    def forward(self, x):
        return self.net(x)


def train_nuisance_nn(
    X_train, Y_train,
    X_val, Y_val,
    *,
    binary=False,
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
    eps=1e-8,
    hidden_dims=(300, 300, 300),
    activation="relu",
    dropout=0.0,
    batch_size=512,
    scheduler_type="none",   # none / step / cosine
    step_size=30,
    gamma=0.5,
    patience=25,
    early_stop=True,
    verbose=False
):
    arrays = [np.asarray(v) for v in (X_train, Y_train, X_val, Y_val)]
    if any(v.size == 0 or not np.isfinite(v).all() for v in arrays):
        raise ValueError("Nuisance training/validation subsets must be nonempty and finite")
    if (arrays[0].ndim != 2 or arrays[2].ndim != 2
            or arrays[1].shape != (len(arrays[0]),)
            or arrays[3].shape != (len(arrays[2]),)
            or arrays[0].shape[1] != arrays[2].shape[1]):
        raise ValueError("Nuisance input dimensions do not match")
    if epochs < 1:
        raise ValueError("Nuisance epochs must be positive")
    if binary and any(not np.isin(v, [0, 1]).all() for v in (arrays[1], arrays[3])):
        raise ValueError("Binary nuisance labels must be zero or one")
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32).unsqueeze(1).to(device)

    model = DeepNN(
        X_train.shape[1],
        hidden_dims=hidden_dims,
        activation=activation,
        dropout=dropout
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr_init,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps
    )

    if scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
    else:
        scheduler = None

    criterion = nn.BCEWithLogitsLoss() if binary else nn.MSELoss()

    loader = DataLoader(
        TensorDataset(X_train_t, Y_train_t),
        batch_size=batch_size,
        shuffle=True
    )

    best_val = np.inf
    best_state = None
    no_improve = 0

    for epoch in range(epochs):

        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite nuisance training loss")
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), Y_val_t).item()
        if not np.isfinite(val_loss):
            raise FloatingPointError("Nonfinite nuisance validation loss")

        if scheduler is not None:
            scheduler.step()

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            # deep copy: state_dict() returns references to the live parameters,
            # which optimizer.step() overwrites in place
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if early_stop and no_improve >= patience:
                break

        if verbose and epoch % 20 == 0:
            print(f"[NN] Epoch {epoch} | Val {val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, None, None


@torch.no_grad()
def predict_nn(model, X, binary=False):

    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    # squeeze(-1), NOT a bare squeeze(): DeepNN emits (n, 1), and squeeze()
    # collapses a single-row batch all the way to a 0-d scalar instead of (1,).
    # Every caller treats the result as a 1-d vector, so the scalar propagates
    # as a shape error -- e.g. the mu10 pseudo-outcome mu1_val_u, whose control
    # subset of an early-stopping slice holds exactly one observation once the
    # folds get small (n = 100 in the wavelet table: folds of 25, a
    # ~6-observation slice, sometimes 1 control in it).
    preds = model(X_t).squeeze(-1)

    if binary:
        preds = torch.sigmoid(preds)

    return preds.cpu().numpy()

# ============================================================
# 2) AutoEncoder / VAE
# ============================================================

class AutoEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        latent_dim,
        hidden_dims=(300, 300),
        activation="relu",
        dropout=0.0
    ):
        super().__init__()

        self.encoder = build_mlp(
            input_dim,
            latent_dim,
            hidden_dims,
            activation,
            dropout
        )

        self.decoder = build_mlp(
            latent_dim,
            input_dim,
            hidden_dims[::-1],
            activation,
            dropout
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class VAE(nn.Module):

    def __init__(
        self,
        input_dim,
        latent_dim,
        hidden_dims=(300, 300),
        activation="relu",
        dropout=0.0
    ):
        super().__init__()

        self.backbone = build_mlp(
            input_dim,
            hidden_dims[-1],
            hidden_dims[:-1],
            activation,
            dropout
        )

        last_dim = hidden_dims[-1]

        self.mu_layer = nn.Linear(last_dim, latent_dim)
        self.logvar_layer = nn.Linear(last_dim, latent_dim)

        self.decoder = build_mlp(
            latent_dim,
            input_dim,
            hidden_dims[::-1],
            activation,
            dropout
        )

    def encode(self, x):
        h = self.backbone(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def train_autoencoder(
    X_train,
    *,
    latent_dim,
    X_val=None,
    model_type="AE",
    hidden_dims=(300, 300),
    activation="relu",
    dropout=0.0,
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
    eps=1e-8,
    batch_size=512,
    scheduler_type="none",
    step_size=30,
    gamma=0.5,
    patience=25,
    early_stop=True,
    verbose=False
):
    for values in (X_train,) if X_val is None else (X_train, X_val):
        values = np.asarray(values)
        if values.ndim != 2 or not values.size or not np.isfinite(values).all():
            raise ValueError("Autoencoder subsets must be nonempty finite matrices")
    if epochs < 1 or latent_dim < 1:
        raise ValueError("Autoencoder epochs and latent dimension must be positive")
    if model_type.upper() not in {"AE", "VAE"}:
        raise ValueError("model_type must be AE or VAE")
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    if X_val is not None:
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    if model_type.upper() == "VAE":
        model = VAE(
            X_train.shape[1],
            latent_dim,
            hidden_dims,
            activation,
            dropout
        ).to(device)
    else:
        model = AutoEncoder(
            X_train.shape[1],
            latent_dim,
            hidden_dims,
            activation,
            dropout
        ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr_init,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps
    )

    if scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
    else:
        scheduler = None

    loader = DataLoader(
        TensorDataset(X_train_t),
        batch_size=batch_size,
        shuffle=True
    )

    best_val = np.inf
    best_state = None
    no_improve = 0

    for epoch in range(epochs):

        model.train()
        total_loss = 0.0

        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad(set_to_none=True)

            if model_type.upper() == "VAE":
                recon, mu, logvar = model(xb)
                recon_loss = nn.functional.mse_loss(recon, xb)
                kl = -0.5 * torch.mean(
                    1 + logvar - mu.pow(2) - logvar.exp()
                )
                loss = recon_loss + beta_kl * kl
            else:
                recon = model(xb)
                loss = nn.functional.mse_loss(recon, xb)

            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite autoencoder training loss")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        val_loss = total_loss / len(loader.dataset)

        if X_val is not None:
            model.eval()
            with torch.no_grad():
                if model_type.upper() == "VAE":
                    recon, mu, logvar = model(X_val_t)
                    recon_loss = nn.functional.mse_loss(recon, X_val_t)
                    kl = -0.5 * torch.mean(
                        1 + logvar - mu.pow(2) - logvar.exp()
                    )
                    val_loss = (recon_loss + beta_kl * kl).item()
                else:
                    val_loss = nn.functional.mse_loss(
                        model(X_val_t), X_val_t
                    ).item()

        if not np.isfinite(val_loss):
            raise FloatingPointError("Nonfinite autoencoder checkpoint criterion")
        if scheduler is not None:
            scheduler.step()

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            # deep copy: state_dict() returns references to the live parameters,
            # which optimizer.step() overwrites in place
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if early_stop and no_improve >= patience:
                break

        if verbose and epoch % 20 == 0:
            print(f"[AE/VAE] Epoch {epoch} | Val {val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ============================================================
# 2b) Two autoencoders trained JOINTLY (no alignment term)
# ============================================================

class JointAutoEncoder(nn.Module):
    """
    Two independent autoencoders, one for X and one for M, held in a single
    module so they can be optimised together.

    Objective:
        lambda1 * mse(X_recon, X) / var(X) + lambda2 * mse(M_recon, M) / var(M)

    This is exactly the MediEncoder objective with lambda3 = 0, but built out of
    plain autoencoders: no g_XM, no alignment. The two blocks share no
    parameters, so the minimiser of each is independent of lambda1, lambda2 --
    the only thing joint training changes is that both blocks now share one
    optimizer, one loss to monitor, and one early-stopping checkpoint.
    """

    def __init__(
        self,
        p_dim,
        q_dim,
        latent_p,
        latent_q,
        hidden_dims_X=(300, 300),
        hidden_dims_M=(300, 300),
        activation="relu",
        dropout=0.0
    ):
        super().__init__()
        self.ae_X = AutoEncoder(
            p_dim, latent_p, hidden_dims_X, activation, dropout
        )
        self.ae_M = AutoEncoder(
            q_dim, latent_q, hidden_dims_M, activation, dropout
        )

    def forward(self, X, M):
        return {
            "z_X": self.ae_X.encoder(X),
            "z_M": self.ae_M.encoder(M),
            "X_recon": self.ae_X(X),
            "M_recon": self.ae_M(M),
        }


# ============================================================
# 3) MediEncoder (FULL FLEX)
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
        hidden_dims_XM=(200,),
        activation="relu",
        dropout=0.0
    ):
        super().__init__()

        self.encoder_X = build_mlp(
            p_dim, latent_p,
            hidden_dims_X,
            activation,
            dropout
        )

        self.decoder_X = build_mlp(
            latent_p, p_dim,
            hidden_dims_X[::-1],
            activation,
            dropout
        )

        self.encoder_M = build_mlp(
            q_dim, latent_q,
            hidden_dims_M,
            activation,
            dropout
        )

        self.decoder_M = build_mlp(
            latent_q, q_dim,
            hidden_dims_M[::-1],
            activation,
            dropout
        )

        self.g_XM = build_mlp(
            latent_p + 1,
            latent_q,
            hidden_dims_XM,
            activation,
            dropout
        )

    def forward(self, X, M, A):

        z_X = self.encoder_X(X)
        z_M = self.encoder_M(M)

        X_recon = self.decoder_X(z_X)
        M_recon = self.decoder_M(z_M)

        A = A.unsqueeze(1)
        z_M_pred = self.g_XM(torch.cat([A, z_X], dim=1))

        return X_recon, M_recon, z_M, z_M_pred



# ============================================================
# Encode with AutoEncoder / VAE
# ============================================================

@torch.no_grad()
def encode_with_autoencoder(
    model,
    X,
    *,
    model_type="AE",
    batch_size=4096
):
    """
    Encode full dataset using trained AE or VAE.

    Parameters
    ----------
    model : trained AutoEncoder or VAE
    X : np.ndarray (n,p)
    model_type : "AE" or "VAE"
    batch_size : safe encoding batch size

    Returns
    -------
    np.ndarray (n, latent_dim)
    """

    model.eval()

    X_t = torch.tensor(X, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t),
        batch_size=batch_size,
        shuffle=False
    )

    outputs = []

    for (xb,) in loader:
        xb = xb.to(device)

        if model_type.upper() == "VAE":
            mu, _ = model.encode(xb)   # use mean as latent representation
            z = mu
        else:
            z = model.encoder(xb)

        outputs.append(z.cpu())

    return torch.cat(outputs, dim=0).numpy()
