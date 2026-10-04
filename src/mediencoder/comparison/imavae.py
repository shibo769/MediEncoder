"""Project-specific conditional IMAVAE adaptation; see docs/comparison.md.

This is the author's existing model implementation, with canonical imports and
strict nonfinite-loss checks. It is not a vendored upstream implementation.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from mediencoder.nn_utils import build_mlp, device

def _safe_global_sd(arr, eps=1e-08):
    arr = np.asarray(arr, dtype=float)
    sd = float(np.std(arr))
    return max(sd, eps)

def compute_imavae_loss_scales(M_train, Y_train, eps=1e-08):
    sd_M = _safe_global_sd(M_train, eps=eps)
    sd_Y = _safe_global_sd(Y_train, eps=eps)
    scales = {'sd_M': sd_M, 'sd_Y': sd_Y, 'var_M': sd_M ** 2, 'var_Y': sd_Y ** 2}
    return scales

def _reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def _gaussian_kl(mu_q, logvar_q, mu_p, logvar_p):
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0)
    kl = kl.sum(dim=1).mean()
    return kl

class IMAVAE(nn.Module):

    def __init__(self, x_dim, m_dim, latent_q, hidden_dims_enc=(300, 300), hidden_dims_dec=(300, 300), hidden_dims_prior=(150, 150), hidden_dims_y=(150, 150), activation='relu', dropout=0.0):
        super().__init__()
        enc_in_dim = m_dim + x_dim + 1
        self.encoder_mu = build_mlp(enc_in_dim, latent_q, hidden_dims_enc, activation, dropout)
        self.encoder_logvar = build_mlp(enc_in_dim, latent_q, hidden_dims_enc, activation, dropout)
        self.decoder_M = build_mlp(latent_q, m_dim, hidden_dims_dec[::-1], activation, dropout)
        prior_in_dim = x_dim + 1
        self.prior_mu = build_mlp(prior_in_dim, latent_q, hidden_dims_prior, activation, dropout)
        self.prior_logvar = build_mlp(prior_in_dim, latent_q, hidden_dims_prior, activation, dropout)
        y_in_dim = latent_q + x_dim + 1
        self.predictor_Y = build_mlp(y_in_dim, 1, hidden_dims_y, activation, dropout)

    def forward(self, X, M, A, sample_latent=True):
        A_col = A.unsqueeze(1)
        enc_input = torch.cat([M, X, A_col], dim=1)
        mu_q = self.encoder_mu(enc_input)
        logvar_q = self.encoder_logvar(enc_input)
        prior_input = torch.cat([X, A_col], dim=1)
        mu_p = self.prior_mu(prior_input)
        logvar_p = self.prior_logvar(prior_input)
        if sample_latent:
            z_M = _reparameterize(mu_q, logvar_q)
        else:
            z_M = mu_q
        M_recon = self.decoder_M(z_M)
        pred_input = torch.cat([z_M, X, A_col], dim=1)
        Y_hat = self.predictor_Y(pred_input).squeeze(1)
        return {'mu_q': mu_q, 'logvar_q': logvar_q, 'mu_p': mu_p, 'logvar_p': logvar_p, 'z_M': z_M, 'M_recon': M_recon, 'Y_hat': Y_hat}

def train_imavae(X_train, M_train, A_train, Y_train, *, latent_q, X_val=None, M_val=None, A_val=None, Y_val=None, hidden_dims_enc=(300, 300), hidden_dims_dec=(300, 300), hidden_dims_prior=(150, 150), hidden_dims_y=(150, 150), activation='relu', dropout=0.0, alpha=1.0, beta=1.0, epochs=150, lr_init=0.001, weight_decay=0.0, betas=(0.9, 0.999), adam_eps=1e-08, batch_size=512, scheduler_type='none', step_size=30, gamma=0.5, patience=25, early_stop=True, min_delta=0.0001, verbose=False, return_history=True, use_kl_annealing=True, kl_anneal_start=0.0, kl_anneal_end=1.0, kl_anneal_epochs=50):
    if alpha < 0:
        raise ValueError('alpha must be >= 0.')
    if beta < 0:
        raise ValueError('beta must be >= 0.')
    if latent_q <= 0:
        raise ValueError('latent_q must be positive.')
    scales = compute_imavae_loss_scales(M_train, Y_train, eps=1e-08)
    var_M = scales['var_M']
    var_Y = scales['var_Y']
    X_t = torch.tensor(X_train, dtype=torch.float32)
    M_t = torch.tensor(M_train, dtype=torch.float32)
    A_t = torch.tensor(A_train, dtype=torch.float32)
    Y_t = torch.tensor(Y_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t, M_t, A_t, Y_t), batch_size=batch_size, shuffle=True, drop_last=False)
    model = IMAVAE(x_dim=X_train.shape[1], m_dim=M_train.shape[1], latent_q=latent_q, hidden_dims_enc=hidden_dims_enc, hidden_dims_dec=hidden_dims_dec, hidden_dims_prior=hidden_dims_prior, hidden_dims_y=hidden_dims_y, activation=activation, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr_init, weight_decay=weight_decay, betas=betas, eps=adam_eps)
    if scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == 'none':
        scheduler = None
    else:
        raise ValueError("scheduler_type must be one of {'none','step','cosine'}")
    history = {'loss_recon_M': [], 'loss_kl': [], 'loss_pred_Y': [], 'weighted_loss': [], 'raw_loss_recon_M': [], 'raw_loss_pred_Y': [], 'beta_effective': [], 'lr': [], 'epoch': [], 'val_weighted_loss': []}
    best_state = None
    best_epoch = -1
    best_weighted_loss = np.inf
    no_improve = 0
    mse_fn = nn.functional.mse_loss
    use_val = X_val is not None and M_val is not None and (A_val is not None) and (Y_val is not None)
    if use_val:
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        M_val_t = torch.tensor(M_val, dtype=torch.float32).to(device)
        A_val_t = torch.tensor(A_val, dtype=torch.float32).to(device)
        Y_val_t = torch.tensor(Y_val, dtype=torch.float32).to(device)
    for epoch in range(epochs):
        model.train()
        if use_kl_annealing:
            if kl_anneal_epochs <= 1:
                anneal_mult = kl_anneal_end
            else:
                progress = min(epoch / float(kl_anneal_epochs - 1), 1.0)
                anneal_mult = kl_anneal_start + (kl_anneal_end - kl_anneal_start) * progress
            beta_effective = beta * anneal_mult
        else:
            beta_effective = beta
        total_n = 0
        sum_recon_M = 0.0
        sum_kl = 0.0
        sum_pred_Y = 0.0
        sum_weighted = 0.0
        sum_raw_recon_M = 0.0
        sum_raw_pred_Y = 0.0
        for xb, mb, ab, yb in loader:
            xb = xb.to(device)
            mb = mb.to(device)
            ab = ab.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(xb, mb, ab, sample_latent=True)
            mu_q = out['mu_q']
            logvar_q = out['logvar_q']
            mu_p = out['mu_p']
            logvar_p = out['logvar_p']
            M_recon = out['M_recon']
            Y_hat = out['Y_hat']
            raw_loss_recon_M = mse_fn(M_recon, mb)
            raw_loss_pred_Y = mse_fn(Y_hat, yb)
            loss_recon_M = raw_loss_recon_M / var_M
            loss_pred_Y = raw_loss_pred_Y / var_Y
            loss_kl = _gaussian_kl(mu_q, logvar_q, mu_p, logvar_p)
            weighted_loss = alpha * loss_recon_M + beta_effective * loss_kl + loss_pred_Y
            if not torch.isfinite(weighted_loss):
                raise FloatingPointError('Nonfinite IMAVAE training loss')
            weighted_loss.backward()
            optimizer.step()
            bs = xb.size(0)
            total_n += bs
            sum_recon_M += loss_recon_M.item() * bs
            sum_kl += loss_kl.item() * bs
            sum_pred_Y += loss_pred_Y.item() * bs
            sum_weighted += weighted_loss.item() * bs
            sum_raw_recon_M += raw_loss_recon_M.item() * bs
            sum_raw_pred_Y += raw_loss_pred_Y.item() * bs
        if scheduler is not None:
            scheduler.step()
        avg_recon_M = sum_recon_M / total_n
        avg_kl = sum_kl / total_n
        avg_pred_Y = sum_pred_Y / total_n
        avg_weighted = sum_weighted / total_n
        avg_raw_recon_M = sum_raw_recon_M / total_n
        avg_raw_pred_Y = sum_raw_pred_Y / total_n
        current_lr = optimizer.param_groups[0]['lr']
        history['epoch'].append(epoch)
        history['loss_recon_M'].append(avg_recon_M)
        history['loss_kl'].append(avg_kl)
        history['loss_pred_Y'].append(avg_pred_Y)
        history['weighted_loss'].append(avg_weighted)
        history['raw_loss_recon_M'].append(avg_raw_recon_M)
        history['raw_loss_pred_Y'].append(avg_raw_pred_Y)
        history['beta_effective'].append(float(beta_effective))
        history['lr'].append(current_lr)
        monitor = avg_weighted
        if use_val:
            model.eval()
            with torch.no_grad():
                out_v = model(X_val_t, M_val_t, A_val_t, sample_latent=False)
                v_recon = mse_fn(out_v['M_recon'], M_val_t).item() / var_M
                v_pred = mse_fn(out_v['Y_hat'], Y_val_t).item() / var_Y
                v_kl = _gaussian_kl(out_v['mu_q'], out_v['logvar_q'], out_v['mu_p'], out_v['logvar_p']).item()
                monitor = alpha * v_recon + beta_effective * v_kl + v_pred
            history['val_weighted_loss'].append(monitor)
        if not np.isfinite(monitor):
            raise FloatingPointError('Nonfinite IMAVAE validation loss')
        if monitor < best_weighted_loss - min_delta:
            best_weighted_loss = monitor
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            print(f'[Epoch {epoch:03d}] weighted={avg_weighted:.6f} | recon_M={avg_recon_M:.6f}, KL={avg_kl:.6f}, pred_Y={avg_pred_Y:.6f} | beta_eff={beta_effective:.4f} | lr={current_lr:.3e}')
        if early_stop and no_improve >= patience:
            if verbose:
                print(f'Early stopping at epoch {epoch}. Best epoch = {best_epoch}.')
            break
    if best_state is None:
        raise RuntimeError('No finite IMAVAE checkpoint; epochs must be positive')
    if best_state is not None:
        model.load_state_dict(best_state)
    fit_info = {'alpha': float(alpha), 'beta': float(beta), 'best_epoch': int(best_epoch), 'best_weighted_loss': float(best_weighted_loss), 'final_epoch': int(history['epoch'][-1]) if len(history['epoch']) > 0 else -1, 'sd_M': float(scales['sd_M']), 'sd_Y': float(scales['sd_Y']), 'var_M': float(scales['var_M']), 'var_Y': float(scales['var_Y']), 'best_loss_recon_M': float(history['loss_recon_M'][best_epoch]) if best_epoch >= 0 else np.nan, 'best_loss_kl': float(history['loss_kl'][best_epoch]) if best_epoch >= 0 else np.nan, 'best_loss_pred_Y': float(history['loss_pred_Y'][best_epoch]) if best_epoch >= 0 else np.nan, 'best_raw_loss_recon_M': float(history['raw_loss_recon_M'][best_epoch]) if best_epoch >= 0 else np.nan, 'best_raw_loss_pred_Y': float(history['raw_loss_pred_Y'][best_epoch]) if best_epoch >= 0 else np.nan}
    if return_history:
        return (model, history, fit_info)
    else:
        return (model, fit_info)

@torch.no_grad()
def encode_with_imavae(model, X, M, A, batch_size=4096, use_posterior_mean=True):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    M_t = torch.tensor(M, dtype=torch.float32)
    A_t = torch.tensor(A, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t, M_t, A_t), batch_size=batch_size, shuffle=False)
    outputs = []
    for xb, mb, ab in loader:
        xb = xb.to(device)
        mb = mb.to(device)
        ab = ab.to(device)
        out = model(xb, mb, ab, sample_latent=not use_posterior_mean)
        if use_posterior_mean:
            z = out['mu_q']
        else:
            z = out['z_M']
        outputs.append(z.cpu())
    return torch.cat(outputs, dim=0).numpy()

@torch.no_grad()
def reconstruct_M_with_imavae(model, X, M, A, batch_size=4096, use_posterior_mean=True):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    M_t = torch.tensor(M, dtype=torch.float32)
    A_t = torch.tensor(A, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t, M_t, A_t), batch_size=batch_size, shuffle=False)
    outputs = []
    for xb, mb, ab in loader:
        xb = xb.to(device)
        mb = mb.to(device)
        ab = ab.to(device)
        out = model(xb, mb, ab, sample_latent=not use_posterior_mean)
        outputs.append(out['M_recon'].cpu())
    return torch.cat(outputs, dim=0).numpy()

@torch.no_grad()
def predict_y_with_imavae(model, X, M, A, batch_size=4096, use_posterior_mean=True):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    M_t = torch.tensor(M, dtype=torch.float32)
    A_t = torch.tensor(A, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t, M_t, A_t), batch_size=batch_size, shuffle=False)
    outputs = []
    for xb, mb, ab in loader:
        xb = xb.to(device)
        mb = mb.to(device)
        ab = ab.to(device)
        out = model(xb, mb, ab, sample_latent=not use_posterior_mean)
        outputs.append(out['Y_hat'].cpu())
    return torch.cat(outputs, dim=0).numpy()

@torch.no_grad()
def sample_zm_from_prior_imavae(model, X, A, n_draws=1, batch_size=4096, use_prior_mean=False):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    A_t = torch.tensor(A, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t, A_t), batch_size=batch_size, shuffle=False)
    all_draws = []
    for _ in range(n_draws):
        outputs = []
        for xb, ab in loader:
            xb = xb.to(device)
            ab = ab.to(device)
            A_col = ab.unsqueeze(1)
            prior_input = torch.cat([xb, A_col], dim=1)
            mu_p = model.prior_mu(prior_input)
            logvar_p = model.prior_logvar(prior_input)
            if use_prior_mean:
                z = mu_p
            else:
                z = _reparameterize(mu_p, logvar_p)
            outputs.append(z.cpu())
        one_draw = torch.cat(outputs, dim=0).numpy()
        all_draws.append(one_draw)
    return np.stack(all_draws, axis=0)

@torch.no_grad()
def predict_y_from_prior_imavae(model, X, A, n_draws=1, batch_size=4096, use_prior_mean=False):
    model.eval()
    z_draws = sample_zm_from_prior_imavae(model=model, X=X, A=A, n_draws=n_draws, batch_size=batch_size, use_prior_mean=use_prior_mean)
    X_t = torch.tensor(X, dtype=torch.float32)
    A_t = torch.tensor(A, dtype=torch.float32)
    y_draws = []
    for d in range(n_draws):
        z_t = torch.tensor(z_draws[d], dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_t, A_t, z_t), batch_size=batch_size, shuffle=False)
        outputs = []
        for xb, ab, zb in loader:
            xb = xb.to(device)
            ab = ab.to(device)
            zb = zb.to(device)
            A_col = ab.unsqueeze(1)
            pred_input = torch.cat([zb, xb, A_col], dim=1)
            y_hat = model.predictor_Y(pred_input).squeeze(1)
            outputs.append(y_hat.cpu())
        one_draw = torch.cat(outputs, dim=0).numpy()
        y_draws.append(one_draw)
    return np.stack(y_draws, axis=0)
