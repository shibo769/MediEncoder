"""Project-specific baseline adaptations, not official upstream implementations.

Adapted from the author's local Baselines_and_Train.py. No upstream repository
is vendored. The public adapters label these methods explicitly as adaptations;
no inference is fabricated for these point-estimate implementations.
"""
import copy
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from mediencoder.nn_utils import build_mlp, device

_SHARED_HIDDEN = (300, 300)
_SHARED_LR = 0.001
_SHARED_WEIGHT_DECAY = 0.0
_SHARED_EPOCHS = 150
_SHARED_BATCH_SIZE = 512
_SHARED_PATIENCE = 25
_SHARED_ACTIVATION = 'relu'
_SHARED_DROPOUT = 0.0

def _as2d(a):
    a = np.asarray(a, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else a

def _standardize_fit(Z):
    Z = _as2d(Z)
    mu = Z.mean(axis=0)
    sd = Z.std(axis=0)
    sd = np.where(sd < 1e-08, 1.0, sd)
    return ((Z - mu) / sd, mu, sd)

def _standardize_apply(Z, mu, sd):
    return (_as2d(Z) - mu) / sd

def _ridge_fit(Z, y, lam=0.001):
    Z = _as2d(Z)
    y = np.asarray(y, dtype=float).reshape(-1)
    n, d = Z.shape
    Zc = np.hstack([np.ones((n, 1)), Z])
    G = Zc.T @ Zc
    pen = lam * n * np.eye(d + 1)
    pen[0, 0] = 0.0
    coef = np.linalg.solve(G + pen, Zc.T @ y)
    return (float(coef[0]), coef[1:])

def _pca_fit(Z, k):
    Z = _as2d(Z)
    mu = Z.mean(axis=0)
    Zc = Z - mu
    n, d = Zc.shape
    if k >= min(n, d):
        k = max(1, min(n, d) - 1)
    if d <= n:
        C = Zc.T @ Zc / max(n - 1, 1)
        w, V = np.linalg.eigh(C)
        V = V[:, np.argsort(w)[::-1][:k]]
    else:
        G = Zc @ Zc.T / max(n - 1, 1)
        w, U = np.linalg.eigh(G)
        order = np.argsort(w)[::-1][:k]
        w = np.clip(w[order], 1e-12, None)
        U = U[:, order]
        V = Zc.T @ U / np.sqrt(w * max(n - 1, 1))
        V /= np.maximum(np.linalg.norm(V, axis=0, keepdims=True), 1e-12)
    return (mu, V)

def _pca_apply(Z, mu, V):
    return (_as2d(Z) - mu) @ V

def effects_from_means(y11, y10, y00):
    return {'Y11': float(y11), 'Y10': float(y10), 'Y00': float(y00), 'NIE': float(y11 - y10), 'NDE': float(y10 - y00), 'TE': float(y11 - y00)}

class _PhiNet(nn.Module):

    def __init__(self, m_dim, hidden_dims=_SHARED_HIDDEN, activation=_SHARED_ACTIVATION, dropout=_SHARED_DROPOUT, d_z=1):
        super().__init__()
        self.d_z = int(d_z)
        self.net = build_mlp(m_dim, self.d_z, hidden_dims, activation, dropout)

    def forward(self, M):
        out = self.net(M)
        return out.squeeze(1) if self.d_z == 1 else out
_ES_VAL_FRAC = 0.2

def _fit_phi_to_target(phi_model, M, target, *, epochs, lr, weight_decay, batch_size, patience, min_delta=1e-05, val_frac=_ES_VAL_FRAC, val_seed=0):
    M_t = torch.tensor(M, dtype=torch.float32)
    tgt = np.asarray(target, dtype=float)
    if tgt.ndim == 2 and tgt.shape[1] == 1:
        tgt = tgt.reshape(-1)
    d_t = torch.tensor(tgt, dtype=torch.float32)
    n_all = M.shape[0]
    n_val = int(round(val_frac * n_all)) if val_frac and val_frac > 0 else 0
    use_val = n_val >= 2 and n_all - n_val >= 2
    if not use_val:
        raise ValueError('Insufficient observations for a disjoint validation split')
    if use_val:
        perm = np.random.default_rng(val_seed).permutation(n_all)
        va_i = torch.as_tensor(perm[:n_val].copy(), dtype=torch.long)
        tr_i = torch.as_tensor(perm[n_val:].copy(), dtype=torch.long)
        M_va, d_va = (M_t[va_i].to(device), d_t[va_i].to(device))
        M_tr, d_tr = (M_t[tr_i], d_t[tr_i])
    else:
        M_tr, d_tr = (M_t, d_t)
    loader = DataLoader(TensorDataset(M_tr, d_tr), batch_size=min(batch_size, M_tr.shape[0]), shuffle=True, drop_last=False)
    opt = optim.Adam(phi_model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.functional.mse_loss
    best = np.inf
    best_state = None
    no_improve = 0
    for _ in range(epochs):
        phi_model.train()
        tot = 0.0
        n_tot = 0
        for mb, db in loader:
            mb = mb.to(device)
            db = db.to(device)
            opt.zero_grad(set_to_none=True)
            loss = mse(phi_model(mb), db)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite baseline training loss')
            loss.backward()
            opt.step()
            bs = mb.size(0)
            tot += loss.item() * bs
            n_tot += bs
        avg = tot / max(n_tot, 1)
        monitor = avg
        if use_val:
            phi_model.eval()
            with torch.no_grad():
                monitor = mse(phi_model(M_va), d_va).item()
        if not np.isfinite(monitor):
            raise FloatingPointError('Nonfinite baseline validation loss')
        if monitor < best - min_delta:
            best = monitor
            best_state = copy.deepcopy(phi_model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state is None:
        raise RuntimeError('No finite fitted baseline checkpoint')
    if best_state is not None:
        phi_model.load_state_dict(best_state)
    return phi_model

@torch.no_grad()
def _phi_predict(phi_model, M, batch_size=4096):
    phi_model.eval()
    M_t = torch.tensor(M, dtype=torch.float32)
    out = []
    for mb, in DataLoader(TensorDataset(M_t), batch_size=batch_size, shuffle=False):
        out.append(phi_model(mb.to(device)).cpu().numpy())
    return np.concatenate(out, axis=0)

def fit_nath_deep_mediation(X, M, A, Y, *, covar=True, covar_mode='learned', n_covar_pc=10, d_z=1, hidden_dims=_SHARED_HIDDEN, activation=_SHARED_ACTIVATION, dropout=_SHARED_DROPOUT, iterations=20, inner_epochs=_SHARED_EPOCHS, lr=_SHARED_LR, weight_decay=_SHARED_WEIGHT_DECAY, batch_size=_SHARED_BATCH_SIZE, patience=_SHARED_PATIENCE, ridge_lam=0.0001, seed=None):
    if iterations < 1 or inner_epochs < 1 or patience < 1 or d_z < 1:
        raise ValueError('Nath iterations, epochs, patience and summary dimension must be positive')
    rng = np.random.default_rng(seed)
    if seed is not None:
        torch.manual_seed(int(rng.integers(0, 2 ** 31 - 1)))
    M = _as2d(M)
    X = _as2d(X)
    A = np.asarray(A, dtype=float).reshape(-1)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    n = M.shape[0]
    d_z = int(d_z)
    Ms, _, _ = _standardize_fit(M)
    Xs, _, _ = _standardize_fit(X)
    phi_x = None
    if not covar:
        C = np.zeros((n, 0))
    elif covar_mode == 'pca':
        mu_x, V_x = _pca_fit(X, n_covar_pc)
        C = _pca_apply(X, mu_x, V_x)
        C, _, _ = _standardize_fit(C)
    elif covar_mode == 'learned':
        phi_x = _PhiNet(Xs.shape[1], hidden_dims, activation, dropout, d_z=n_covar_pc).to(device)
        C = _as2d(_phi_predict(phi_x, Xs))
        C, _, _ = _standardize_fit(C)
    else:
        raise ValueError(f'unknown covar_mode {covar_mode!r}')
    n_c = C.shape[1]
    phi = _PhiNet(Ms.shape[1], hidden_dims, activation, dropout, d_z=d_z).to(device)
    z = _as2d(_phi_predict(phi, Ms))
    alpha = np.zeros(d_z)
    beta = np.zeros(d_z)
    gamma = 0.0
    alpha_0 = np.zeros(d_z)
    beta_0 = 0.0
    alpha_C = np.zeros((n_c, d_z))
    gamma_C = np.zeros(n_c)
    for _ in range(int(iterations)):
        for j in range(d_z):
            if np.std(z[:, j]) < 1e-10:
                raise FloatingPointError('Collapsed mediator summary; no artificial noise is added')
            if np.corrcoef(z[:, j], Y)[0, 1] < 0:
                z[:, j] = -z[:, j]
        z = (z - z.mean(0)) / np.maximum(z.std(0), 1e-10)
        D_med = np.hstack([A.reshape(-1, 1), C])
        for j in range(d_z):
            a0_j, coef_med = _ridge_fit(D_med, z[:, j], lam=ridge_lam)
            alpha_0[j] = a0_j
            alpha[j] = float(coef_med[0])
            if n_c:
                alpha_C[:, j] = coef_med[1:]
        D_out = np.hstack([z, A.reshape(-1, 1), C])
        beta_0, coef_out = _ridge_fit(D_out, Y, lam=ridge_lam)
        beta = np.asarray(coef_out[:d_z], dtype=float)
        gamma = float(coef_out[d_z])
        gamma_C = coef_out[d_z + 1:]
        e = Y - beta_0 - A * gamma - (C @ gamma_C if n_c else 0.0)
        h = alpha_0[None, :] + np.outer(A, alpha)
        if n_c:
            h = h + C @ alpha_C
        if d_z == 1:
            b = float(beta[0])
            d = (b * e + h[:, 0]) / (b ** 2 + 1.0)
        else:
            Gram = np.outer(beta, beta) + np.eye(d_z)
            rhs = np.outer(e, beta) + h
            d = np.linalg.solve(Gram, rhs.T).T
        phi = _fit_phi_to_target(phi, Ms, d, epochs=inner_epochs, lr=lr, weight_decay=weight_decay, batch_size=batch_size, patience=patience, val_seed=(0 if seed is None else int(seed)) + 101)
        z = _as2d(_phi_predict(phi, Ms))
        if phi_x is not None:
            r_out = Y - beta_0 - A * gamma - z @ beta
            r_med = z - alpha_0[None, :] - np.outer(A, alpha)
            Gram_c = np.outer(gamma_C, gamma_C) + alpha_C @ alpha_C.T + np.eye(n_c)
            rhs_c = np.outer(r_out, gamma_C) + r_med @ alpha_C.T
            c_tgt = np.linalg.solve(Gram_c, rhs_c.T).T
            phi_x = _fit_phi_to_target(phi_x, Xs, c_tgt, epochs=inner_epochs, lr=lr, weight_decay=weight_decay, batch_size=batch_size, patience=patience, val_seed=(0 if seed is None else int(seed)) + 202)
            C = _as2d(_phi_predict(phi_x, Xs))
            C, _, _ = _standardize_fit(C)
    # Refit path coefficients on the final returned representation. The legacy
    # code returned coefficients from the previous alternating iteration.
    D_med = np.hstack([A.reshape(-1, 1), C])
    for j in range(d_z):
        alpha[j] = _ridge_fit(D_med, z[:, j], lam=ridge_lam)[1][0]
    _, final_coef = _ridge_fit(np.hstack([z, A.reshape(-1, 1), C]), Y, lam=ridge_lam)
    beta, gamma = np.asarray(final_coef[:d_z]), float(final_coef[d_z])
    indirect = float(beta @ alpha)
    direct = float(gamma)
    if not covar:
        tag = 'nath_nocovar'
    elif d_z > 1:
        tag = f'nath_covar_dz{d_z}'
    elif covar_mode == 'pca':
        tag = 'nath_covar_pca'
    else:
        tag = 'nath_covar'
    return {'method': tag, 'alpha': alpha.tolist() if d_z > 1 else float(alpha[0]), 'beta': beta.tolist() if d_z > 1 else float(beta[0]), 'gamma': direct, 'd_z': d_z, 'NIE': indirect, 'NDE': direct, 'TE': direct + indirect, 'Y11': np.nan, 'Y10': np.nan, 'Y00': np.nan, 'estimand': "path-product (beta'alpha, gamma)"}

class _DP2LMNet(nn.Module):

    def __init__(self, x_dim, m_dim, z_dim, hidden_dims=_SHARED_HIDDEN, activation=_SHARED_ACTIVATION, dropout=_SHARED_DROPOUT):
        super().__init__()
        self.lin_x = nn.Linear(x_dim, 1, bias=True)
        self.lin_m = nn.Linear(m_dim, 1, bias=False) if m_dim > 0 else None
        self.f_z = build_mlp(z_dim, 1, hidden_dims, activation, dropout)

    def forward(self, x, m, z):
        out = self.lin_x(x).squeeze(1) + self.f_z(z).squeeze(1)
        if self.lin_m is not None:
            out = out + self.lin_m(m).squeeze(1)
        return out

def _scad_penalty(w, lam, a=3.7):
    aw = w.abs()
    p1 = lam * aw
    p2 = (2.0 * a * lam * aw - aw ** 2 - lam ** 2) / (2.0 * (a - 1.0))
    p3 = torch.full_like(aw, lam ** 2 * (a + 1.0) / 2.0)
    out = torch.where(aw <= lam, p1, torch.where(aw <= a * lam, p2, p3))
    return out.sum()

def _train_dp2lm_stage(x, m, z, y, *, lam, hidden_dims, activation, dropout, epochs, lr, weight_decay, batch_size, patience, scad_a=3.7, min_delta=1e-06, val_frac=_ES_VAL_FRAC, val_seed=0):
    x_t = torch.tensor(_as2d(x), dtype=torch.float32)
    m_t = torch.tensor(_as2d(m) if _as2d(m).shape[1] else np.zeros((len(y), 0)), dtype=torch.float32)
    z_t = torch.tensor(_as2d(z), dtype=torch.float32)
    y_t = torch.tensor(np.asarray(y, dtype=float).reshape(-1), dtype=torch.float32)
    model = _DP2LMNet(x_t.shape[1], m_t.shape[1], z_t.shape[1], hidden_dims=hidden_dims, activation=activation, dropout=dropout).to(device)
    n_all = len(y)
    n_val = int(round(val_frac * n_all)) if val_frac and val_frac > 0 else 0
    use_val = n_val >= 2 and n_all - n_val >= 2
    if not use_val:
        raise ValueError('Insufficient observations for a disjoint validation split')
    if use_val:
        perm = np.random.default_rng(val_seed).permutation(n_all)
        va_i = torch.as_tensor(perm[:n_val].copy(), dtype=torch.long)
        tr_i = torch.as_tensor(perm[n_val:].copy(), dtype=torch.long)
        xv, mv, zv, yv = (t[va_i].to(device) for t in (x_t, m_t, z_t, y_t))
        x_tr, m_tr, z_tr, y_tr = (t[tr_i] for t in (x_t, m_t, z_t, y_t))
    else:
        x_tr, m_tr, z_tr, y_tr = (x_t, m_t, z_t, y_t)
    loader = DataLoader(TensorDataset(x_tr, m_tr, z_tr, y_tr), batch_size=min(batch_size, x_tr.shape[0]), shuffle=True, drop_last=False)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.functional.mse_loss
    best = np.inf
    best_state = None
    no_improve = 0
    for _ in range(epochs):
        model.train()
        tot = 0.0
        n_tot = 0
        for xb, mb, zb, yb in loader:
            xb, mb, zb, yb = (t.to(device) for t in (xb, mb, zb, yb))
            opt.zero_grad(set_to_none=True)
            pred = model(xb, mb, zb)
            loss = mse(pred, yb)
            if model.lin_m is not None and lam > 0:
                loss = loss + _scad_penalty(model.lin_m.weight, lam, a=scad_a)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite baseline training loss')
            loss.backward()
            opt.step()
            bs = xb.size(0)
            tot += loss.item() * bs
            n_tot += bs
        avg = tot / max(n_tot, 1)
        monitor = avg
        if use_val:
            model.eval()
            with torch.no_grad():
                monitor = mse(model(xv, mv, zv), yv).item()
                if model.lin_m is not None and lam > 0:
                    monitor = monitor + _scad_penalty(model.lin_m.weight, lam, a=scad_a).item()
        if not np.isfinite(monitor):
            raise FloatingPointError('Nonfinite baseline validation loss')
        if monitor < best - min_delta:
            best = monitor
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state is None:
        raise RuntimeError('No finite fitted baseline checkpoint')
    if best_state is not None:
        model.load_state_dict(best_state)
    coef_x = model.lin_x.weight.detach().cpu().numpy().reshape(-1)
    coef_m = model.lin_m.weight.detach().cpu().numpy().reshape(-1) if model.lin_m is not None else np.zeros(0)
    return (coef_x, coef_m, model)

def fit_dp2lm(X, M, A, Y, *, n_covar_pc=None, n_mediator_pc=None, lam=None, lam_grid=(0.0, 0.01, 0.05, 0.1), hidden_dims=_SHARED_HIDDEN, activation=_SHARED_ACTIVATION, dropout=_SHARED_DROPOUT, epochs=_SHARED_EPOCHS, lr=_SHARED_LR, weight_decay=_SHARED_WEIGHT_DECAY, batch_size=None, patience=_SHARED_PATIENCE, scad_a=3.7, active_threshold=0.01, seed=None):
    if seed is not None:
        torch.manual_seed(int(seed) % (2 ** 31 - 1))
    A = np.asarray(A, dtype=float).reshape(-1)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    n = len(Y)
    if batch_size is None:
        batch_size = min(512, 2 ** int(np.floor(np.log2(max(n, 2)))))
    if n_covar_pc is not None:
        mu_x, V_x = _pca_fit(X, n_covar_pc)
        Z = _pca_apply(X, mu_x, V_x)
    else:
        Z = _as2d(X)
    Z, _, _ = _standardize_fit(Z)
    if n_mediator_pc is not None:
        mu_m, V_m = _pca_fit(M, n_mediator_pc)
        Mw = _pca_apply(M, mu_m, V_m)
    else:
        Mw = _as2d(M)
    Mw, _, _ = _standardize_fit(Mw)
    x = A.reshape(-1, 1)
    common = dict(hidden_dims=hidden_dims, activation=activation, dropout=dropout, epochs=epochs, lr=lr, weight_decay=weight_decay, batch_size=batch_size, patience=patience, scad_a=scad_a, val_seed=(0 if seed is None else int(seed)) + 303)
    gamma_e = np.zeros(Mw.shape[1])
    for k in range(Mw.shape[1]):
        c_x, _, _ = _train_dp2lm_stage(x, np.zeros((n, 0)), Z, Mw[:, k], lam=0.0, **common)
        gamma_e[k] = c_x[0]

    def _hbic(coef_m, resid):
        rss = float(np.mean(resid ** 2))
        df = int(np.sum(np.abs(coef_m) > active_threshold))
        return n * np.log(max(rss, 1e-12)) + df * np.log(np.log(n)) * np.log(Mw.shape[1])
    if lam is None:
        best_score = np.inf
        best = None
        for lam_try in lam_grid:
            c_x, c_m, mdl = _train_dp2lm_stage(x, Mw, Z, Y, lam=lam_try, **common)
            with torch.no_grad():
                pred = mdl(torch.tensor(x, dtype=torch.float32).to(device), torch.tensor(Mw, dtype=torch.float32).to(device), torch.tensor(Z, dtype=torch.float32).to(device)).cpu().numpy()
            score = _hbic(c_m, Y - pred)
            if score < best_score:
                best_score = score
                best = (c_x, c_m, lam_try)
        if best is None:
            raise FloatingPointError('All SCAD candidates produced nonfinite selection criteria')
        alpha_e_vec, alpha_m, lam_sel = best
    else:
        alpha_e_vec, alpha_m, _ = _train_dp2lm_stage(x, Mw, Z, Y, lam=lam, **common)
        lam_sel = lam
    alpha_e = float(alpha_e_vec[0])
    theta_e_vec, _, _ = _train_dp2lm_stage(x, np.zeros((n, 0)), Z, Y, lam=0.0, **common)
    theta_e = float(theta_e_vec[0])
    nie_diff = theta_e - alpha_e
    nie_prod = float(gamma_e @ alpha_m)
    return {'method': 'dp2lm', 'alpha_e': alpha_e, 'theta_e': theta_e, 'lambda': float(lam_sel), 'n_active_mediators': int(np.sum(np.abs(alpha_m) > active_threshold)), 'NDE': alpha_e, 'NIE': float(nie_diff), 'NIE_product': nie_prod, 'TE': theta_e, 'Y11': np.nan, 'Y10': np.nan, 'Y00': np.nan, 'estimand': 'path-product (alpha_e, theta_e - alpha_e)'}

@torch.no_grad()
def imavae_counterfactual_mean(imavae_model, X, *, a_outcome, a_mediator, target_idx=None, n_mc=200, batch_size=4096, use_prior_mean=False):
    from .imavae import sample_zm_from_prior_imavae
    X = np.asarray(X, dtype=float)
    if target_idx is not None:
        X = X[target_idx]
    n_tar = X.shape[0]
    A_med = np.full(n_tar, float(a_mediator))
    A_out = np.full(n_tar, float(a_outcome))
    z_draws = sample_zm_from_prior_imavae(imavae_model, X, A_med, n_draws=n_mc, batch_size=batch_size, use_prior_mean=use_prior_mean)
    X_t = torch.tensor(X, dtype=torch.float32)
    A_t = torch.tensor(A_out, dtype=torch.float32)
    imavae_model.eval()
    means = []
    for d in range(n_mc):
        z_t = torch.tensor(z_draws[d], dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_t, A_t, z_t), batch_size=batch_size, shuffle=False)
        preds = []
        for xb, ab, zb in loader:
            xb, ab, zb = (xb.to(device), ab.to(device), zb.to(device))
            pred_input = torch.cat([zb, xb, ab.unsqueeze(1)], dim=1)
            preds.append(imavae_model.predictor_Y(pred_input).squeeze(1).cpu().numpy())
        means.append(np.concatenate(preds, axis=0))
    return float(np.mean(np.asarray(means)))

def imavae_effects(imavae_model, X, *, target_idx=None, n_mc=200, batch_size=4096, use_prior_mean=False):
    kw = dict(target_idx=target_idx, n_mc=n_mc, batch_size=batch_size, use_prior_mean=use_prior_mean)
    y11 = imavae_counterfactual_mean(imavae_model, X, a_outcome=1, a_mediator=1, **kw)
    y10 = imavae_counterfactual_mean(imavae_model, X, a_outcome=1, a_mediator=0, **kw)
    y00 = imavae_counterfactual_mean(imavae_model, X, a_outcome=0, a_mediator=0, **kw)
    out = effects_from_means(y11, y10, y00)
    out['method'] = 'imavae'
    out['estimand'] = 'cross-world (MC plug-in, no SE)'
    return out

def fit_lsem_ridge(X, M, A, Y, *, n_covar_pc=10, n_mediator_pc=10, ridge_lam=0.01, seed=None):
    A = np.asarray(A, dtype=float).reshape(-1)
    Y = np.asarray(Y, dtype=float).reshape(-1)
    mu_w, V_w = _pca_fit(X, n_covar_pc)
    W, _, _ = _standardize_fit(_pca_apply(X, mu_w, V_w))
    mu_m, V_m = _pca_fit(M, n_mediator_pc)
    Mr, _, _ = _standardize_fit(_pca_apply(M, mu_m, V_m))
    D = np.hstack([A.reshape(-1, 1), W])
    gamma_e = np.array([_ridge_fit(D, Mr[:, k], ridge_lam)[1][0] for k in range(Mr.shape[1])])
    D2 = np.hstack([Mr, A.reshape(-1, 1), W])
    _, coef = _ridge_fit(D2, Y, ridge_lam)
    alpha_m = coef[:Mr.shape[1]]
    alpha_e = float(coef[Mr.shape[1]])
    _, coef_t = _ridge_fit(D, Y, ridge_lam)
    theta_e = float(coef_t[0])
    return {'method': 'lsem_ridge', 'NDE': alpha_e, 'NIE': float(theta_e - alpha_e), 'NIE_product': float(gamma_e @ alpha_m), 'TE': theta_e, 'Y11': np.nan, 'Y10': np.nan, 'Y00': np.nan, 'estimand': 'path-product (linear)'}
