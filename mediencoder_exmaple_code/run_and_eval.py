# ============================================================
# run_and_eval.py
#
# ONE estimator, ONE selection rule, no modes.
#
#   * Estimation: Algorithm 1, cross-fitted over four folds. estimate_triply_IF
#     is the only entry point; the single-split path (0.4 / 0.2 / 0.4, no
#     cross-fitting) is deleted.
#   * lambda: always selected, never fixed, by Algorithm 2 on each
#     representation half's own (I_tr, I_val). A lambda-bearing method called
#     without a grid is an error, not a fixed-lambda run.
#   * Selection score: held-out prediction error on the TREATED, and only that
#     -- see _check_selection_rule().
# ============================================================

import copy
import warnings

import numpy as np

from sklearn.model_selection import train_test_split
from scipy.linalg import svd
from scipy.stats import norm

from DGP_and_estimate import get_f_hat
from NNModel_and_Train import (
    train_nuisance_nn,
    predict_nn,
    train_autoencoder,
    encode_with_autoencoder
)

from MediEncoder_and_Train import (
    train_mediencoder,
    encode_with_mediencoder,
    train_mediencoder_vae,
    encode_with_mediencoder_vae
)

import random
import torch
import numpy as np

def lambda_to_seed(base_seed, lam1, lam2, lam3):
    a = int(round(lam1 * 1000))
    b = int(round(lam2 * 1000))
    c = int(round(lam3 * 1000))
    return int(base_seed + 1000000 * a + 1000 * b + c)

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
# ============================================================
# 0) Projection helpers
# ============================================================

def _fit_projection(Z_train: np.ndarray, tilde: int):
    p = Z_train.shape[1]
    _, _, Vt = svd(Z_train, full_matrices=False)
    V = Vt.T[:, :tilde]
    W = np.sqrt(p) * V
    return W


def _apply_projection(Z: np.ndarray, W: np.ndarray):
    p = Z.shape[1]
    return (Z @ W) / p


# ============================================================
# 1) Split helper
#
# _split_indices_4fold is the ONLY data split of the estimator: Algorithm 1,
# step 1. The old three-way _split_indices (0.4 / 0.2 / 0.4) belonged to the
# deleted single-split path and is gone.
# ============================================================

def _split_indices_4fold(n, *, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return np.array_split(idx, 4)


def _split_nuisance_fold(nuisance_idx, A, *, val_ratio=0.25, seed=42):
    nuisance_idx = np.asarray(nuisance_idx)

    stratify = None
    vals, counts = np.unique(A[nuisance_idx], return_counts=True)
    if len(vals) == 2 and np.min(counts) >= 2:
        stratify = A[nuisance_idx]

    nuis_train_idx, nuis_val_idx = train_test_split(
        nuisance_idx,
        test_size=val_ratio,
        random_state=seed,
        stratify=stratify
    )

    return nuis_train_idx, nuis_val_idx
# ============================================================
# 2) Small helpers
# ============================================================

def _safe_clip_prob(x, eps=1e-2):
    return np.clip(x, eps, 1 - eps)


# The methods whose objective carries the (lambda1, lambda2, lambda3) weights.
# A lambda grid is REQUIRED for them: Algorithm 1 step 4 selects it per fold via
# Algorithm 2, and there is no fixed-lambda alternative.
#
# medivae is back, and it is the deliberate one. It is MediEncoder's objective plus
# a variational bottleneck, so the triple VAE / MediVAE / MediEncoder separates the
# two regularizers that were otherwise confounded: bottleneck alone, bottleneck plus
# alignment, alignment alone. Plain VAE is the hardest baseline in every regime tried
# -- its KL term shrinks the latent toward N(0, I), which overlaps what lambda3 does
# but is UNCONDITIONAL -- so without MediVAE a reviewer can ask whether MediEncoder's
# gain is just regularization by another name and the table cannot answer.
#
# imavae was dropped from the comparison set (author's call). Note for the record that
# it was NOT the binding baseline: MediEncoder beats it wherever A explains a
# non-trivial share of f_M (delta1_spread = 2, n = 1200: 0.468 vs 0.507). Its branch
# also fed RAW X as f_X rather than a learned representation, so it was never a
# like-for-like representation-learning comparator. VAE is the one to beat.
_LAMBDA_METHODS = frozenset({"mediencoder", "medivae"})

# The methods that read an me_cfg block at all -- here, exactly the lambda-bearing
# ones, since both share the coupled-network config shape.
_ME_CFG_METHODS = frozenset(_LAMBDA_METHODS)


def _check_selection_rule(selection_rule):
    """
    predictionError is the only lambda-selection rule.

    The task tuples keep the slot so existing drivers do not have to change
    shape, but anything other than predictionError is an error rather than a
    silently different estimator.
    """
    if selection_rule != "predictionError":
        raise ValueError(
            "selection_rule must be 'predictionError'; got %r."
            % (selection_rule,)
        )


def _abs_error(theta_hat, theta_true):
    return float(abs(theta_hat - theta_true))


def _mse_error(theta_hat, theta_true):
    diff = float(theta_hat - theta_true)
    return diff * diff

# ============================================================
# 3) Nuisance estimation + IF evaluation
# ============================================================

def _fit_nuisances_and_eval_theta(
    f_X_all,
    f_M_all,
    A,
    Y,
    *,
    subtrain_idx,
    val_idx,
    target_idx,
    nn_cfg=None
):
    nn_cfg = {} if nn_cfg is None else dict(nn_cfg)
    # train_nuisance_nn has no min_delta argument (its improvement threshold is
    # hard-coded to 1e-4, the shared value), so drop the key if the caller's cfg
    # carries it -- drivers build their cfg from _shared_cfg(), which includes
    # min_delta because the other trainers do take it.
    assert nn_cfg.pop("min_delta", 1e-4) == 1e-4

    # e(X)
    e_cfg = dict(nn_cfg)
    e_cfg["binary"] = True

    e_model, *_ = train_nuisance_nn(
        f_X_all[subtrain_idx], A[subtrain_idx],
        f_X_all[val_idx], A[val_idx],
        **e_cfg
    )

    e_hat = predict_nn(e_model, f_X_all[target_idx], binary=True)
    e_hat = _safe_clip_prob(e_hat, eps=1e-2)

    # mu1(X,M) on treated
    treated_sub = (A[subtrain_idx] == 1)
    treated_val = (A[val_idx] == 1)

    mu1_cfg = dict(nn_cfg)
    mu1_cfg["binary"] = False

    mu1_model, *_ = train_nuisance_nn(
        np.hstack([
            f_X_all[subtrain_idx][treated_sub],
            f_M_all[subtrain_idx][treated_sub]
        ]),
        Y[subtrain_idx][treated_sub],
        np.hstack([
            f_X_all[val_idx][treated_val],
            f_M_all[val_idx][treated_val]
        ]),
        Y[val_idx][treated_val],
        **mu1_cfg
    )

    mu1_hat = predict_nn(
        mu1_model,
        np.hstack([f_X_all[target_idx], f_M_all[target_idx]]),
        binary=False
    )

    # mu10(X)
    untreated_sub = (A[subtrain_idx] == 0)
    untreated_val = (A[val_idx] == 0)

    Xm_sub_u = np.hstack([
        f_X_all[subtrain_idx][untreated_sub],
        f_M_all[subtrain_idx][untreated_sub]
    ])
    Xm_val_u = np.hstack([
        f_X_all[val_idx][untreated_val],
        f_M_all[val_idx][untreated_val]
    ])

    mu1_sub_u = predict_nn(mu1_model, Xm_sub_u, binary=False)
    mu1_val_u = predict_nn(mu1_model, Xm_val_u, binary=False)

    mu10_cfg = dict(nn_cfg)
    mu10_cfg["binary"] = False

    mu10_model, *_ = train_nuisance_nn(
        f_X_all[subtrain_idx][untreated_sub], mu1_sub_u,
        f_X_all[val_idx][untreated_val], mu1_val_u,
        **mu10_cfg
    )

    mu10_hat = predict_nn(
        mu10_model,
        f_X_all[target_idx],
        binary=False
    )

    # pi2(X,M)
    pi2_cfg = dict(nn_cfg)
    pi2_cfg["binary"] = True

    pi2_model, *_ = train_nuisance_nn(
        np.hstack([f_X_all[subtrain_idx], f_M_all[subtrain_idx]]),
        A[subtrain_idx],
        np.hstack([f_X_all[val_idx], f_M_all[val_idx]]),
        A[val_idx],
        **pi2_cfg
    )

    post = predict_nn(
        pi2_model,
        np.hstack([f_X_all[target_idx], f_M_all[target_idx]]),
        binary=True
    )
    post = _safe_clip_prob(post, eps=1e-2)
    prior = _safe_clip_prob(e_hat, eps=1e-2)

    pi2_hat = ((1 - post) / post) * (prior / (1 - prior))

    # IF
    A_tar = A[target_idx]
    Y_tar = Y[target_idx]

    phi = (
        mu10_hat
        + (A_tar / e_hat) * pi2_hat * (Y_tar - mu1_hat)
        + ((1 - A_tar) / (1 - e_hat)) * (mu1_hat - mu10_hat)
    )

    theta_hat_IF = float(np.mean(phi))
    theta_hat_mu10 = float(np.mean(mu10_hat))

    return {
        "theta_hat_IF": theta_hat_IF,
        "theta_hat_mu10": theta_hat_mu10,
        "mu1_model": mu1_model,
        "phi": np.asarray(phi, dtype=float),
    }


# ============================================================
# 4) Outcome prediction helper for predictionError
# ============================================================

def _fit_outcome_predictor_and_eval(
    f_X_all,
    f_M_all,
    A,
    Y,
    *,
    subtrain_idx,
    val_idx,
    nn_cfg=None
):
    """
    Algorithm 2, steps 1-6: the auxiliary TREATED-outcome regression whose
    held-out error scores a candidate lambda.

        m1_hat = argmin_g  (1/|I_tr,1|) sum_{i in I_tr,1} [Y_i - g(f_X_i, f_M_i)]^2
        PE_hat = (1/|I_val,1|) sum_{i in I_val,1} [Y_i - m1_hat(f_X_i, f_M_i)]^2

    with I_tr,1 = {i in I_tr : A_i = 1} and I_val,1 = {i in I_val : A_i = 1}.

    This used to fit g_Y(A, f_X, f_M) on ALL of subtrain and score it on ALL of
    val, i.e. a pooled model with A as an extra input feature. That is a
    different estimand from the one the algorithm specifies and from the mu1
    the estimator actually plugs in: mu1 = E[Y | A = 1, f_X, f_M] is a
    treated-arm regression, so the representation should be selected by how
    well it supports THAT regression. Pooling in the controls let a lambda win
    by making f_M good for the control arm, which the estimator never uses
    f_M for on its own, and made the score partly a measure of how separable
    the two arms are.
    """
    nn_cfg = {} if nn_cfg is None else dict(nn_cfg)
    # train_nuisance_nn has no min_delta argument (its improvement threshold is
    # hard-coded to 1e-4, the shared value), so drop the key if the caller's cfg
    # carries it -- drivers build their cfg from _shared_cfg(), which includes
    # min_delta because the other trainers do take it.
    assert nn_cfg.pop("min_delta", 1e-4) == 1e-4

    y_cfg = dict(nn_cfg)
    y_cfg["binary"] = False

    subtrain_idx = np.asarray(subtrain_idx)
    val_idx = np.asarray(val_idx)

    # Algorithm 2, step 1: restrict to the treated.
    tr1 = subtrain_idx[np.asarray(A)[subtrain_idx] == 1]
    va1 = val_idx[np.asarray(A)[val_idx] == 1]

    # Algorithm 2, step 2: "Assume |I_tr,1| > 0 and |I_val,1| > 0; otherwise,
    # redraw the sample split." A caller cannot redraw from in here, so an
    # unusable split is reported as an infinite score: that lambda simply loses,
    # and the split failure does not silently become a good score.
    if len(tr1) < 4 or len(va1) < 1:
        return {
            "yhat_val": np.array([]),
            "prediction_mse": float("inf"),
            "prediction_rmse": float("inf"),
            "n_treated_train": int(len(tr1)),
            "n_treated_val": int(len(va1)),
        }

    Xtr = np.hstack([f_X_all[tr1], f_M_all[tr1]])
    Xva = np.hstack([f_X_all[va1], f_M_all[va1]])

    y_model, *_ = train_nuisance_nn(
        Xtr, Y[tr1],
        Xva, Y[va1],
        **y_cfg
    )

    yhat_val = predict_nn(y_model, Xva, binary=False)

    mse = float(np.mean((Y[va1] - yhat_val) ** 2))
    rmse = float(np.sqrt(mse))

    return {
        "yhat_val": yhat_val,
        "prediction_mse": mse,
        "prediction_rmse": rmse,
        "n_treated_train": int(len(tr1)),
        "n_treated_val": int(len(va1)),
    }

# ============================================================
# 5) Representation learning
# ============================================================

# ------------------------------------------------------------
# ONE shared optimisation block for every representation learner
# ------------------------------------------------------------
# Every representation branch below -- autoencoder, vae, mediencoder, medivae --
# takes its optimisation knobs from
# this single dict. Anything a table compares must not differ in a knob the
# table is not about, and these had drifted apart: the autoencoder branch ran
# StepLR(30, 0.5) while every other branch ran a constant learning rate, and
# epochs were 200 for some branches and 150 for others. A shared source makes
# such a divergence impossible to reintroduce by editing one branch.
#
# weight_decay is 0.0 for ALL methods, on the author's instruction: the paper's
# Sec 5.1 text ("weight decay 1e-3") is wrong, and 0.0 is the intended setting.
# The paper text needs correcting, not this block.
#
# The LR schedule and epoch cap DO follow Sec 5.1 -- step(30, 0.5) and at most 300
# epochs -- because those were plain drift, not a disputed value. On the paper's
# main-table configuration (p = 2000, q = 1000, n = 1200) the four-way deviation
# cost MediEncoder RMSE 0.556 vs 0.364; how much of that was the schedule and
# widths rather than the L2 is what the wd = 0 rerun measures.
#
# Method-specific structure (lambda weights, latent widths, g_XM width, KL
# weight, annealing) is NOT in here: those are what distinguish the methods and
# are set per branch.
SHARED_TRAIN_CFG = dict(
    epochs=300,
    lr_init=1e-3,
    weight_decay=0.0,
    betas=(0.9, 0.999),
    batch_size=512,
    activation="relu",
    dropout=0.0,
    scheduler_type="step",
    step_size=30,
    gamma=0.5,
    patience=25,
    early_stop=True,
    min_delta=1e-4,
    verbose=False,
)

# Encoder / decoder widths, also shared. The trainers spell the same knob three
# different ways (eps vs adam_eps; hidden_dims vs hidden_dims_X/M vs
# hidden_dims_enc/dec), so the aliases are listed here rather than open-coded
# per branch.
SHARED_HIDDEN = (300, 200)   # paper Sec 5.1: encoders use widths 300 and 200
SHARED_ADAM_EPS = 1e-8


def _shared_cfg(**overrides):
    """
    SHARED_TRAIN_CFG plus the requested method-specific keys.

    Callers pass only what genuinely distinguishes their method. Passing a key
    that is already shared is an error, so a branch cannot quietly re-specify
    epochs or a scheduler and drift away from the others again.
    """
    clash = sorted(set(overrides) & set(SHARED_TRAIN_CFG))
    if clash:
        raise ValueError(
            "these knobs are shared across all representation learners and "
            "must not be overridden per method: %s. Change SHARED_TRAIN_CFG "
            "if the value should change for everyone." % (clash,)
        )
    cfg = dict(SHARED_TRAIN_CFG)
    cfg.update(overrides)
    return cfg


# Keys naming the WIDTH of a representation network. Equalised for the same
# reason as the optimiser: capacity is not what distinguishes one method's
# hypothesis from another's here, so leaving it per-method makes the table
# partly a comparison of architectures. Excludes hidden_dims_XM /
# hidden_dims_prior / hidden_dims_y, which are auxiliary heads that only some
# methods have at all -- those are topology, i.e. part of the method.
SHARED_WIDTH_KEYS = (
    "hidden_dims",
    "hidden_dims_X",
    "hidden_dims_M",
    "hidden_dims_enc",
    "hidden_dims_dec",
)

_UNEQUAL_OPT_OUT = "allow_unequal_train_cfg"


def _merge_method_cfg(cfg, defaults, *, method):
    """
    Merge a branch's defaults into a caller-supplied cfg, with the SHARED knobs
    winning over the caller.

    This is deliberately not setdefault. The drivers in this repo each carry
    their own copy of an optimisation block and those copies disagree with one
    another, so under setdefault a table built from two drivers would be comparing
    optimisers as much as estimators. Forcing them here fixes every driver at once
    instead of relying on ~50 config blocks staying in step.

    Note that epochs=300 / scheduler_type="step" / patience=30, which the older
    drivers carry, are the PAPER'S values (Sec 5.1) and are now the shared values
    too -- they were drift, not staleness. weight_decay is the exception: the
    drivers and the paper both say 1e-3, but the author's instruction is 0.0, so
    the shared block forces 0.0 over any driver that asks for 1e-3.

    Everything that genuinely defines a method -- its loss weights (lambda1,
    lambda2, lambda3, beta_kl, alpha, beta), the topology of its auxiliary heads,
    its flags -- is still setdefault, so the caller and the lambda tuner control
    it as before.

    A driver that really means to vary a shared knob (a weight-decay ablation,
    say) sets cfg["allow_unequal_train_cfg"] = True and takes responsibility for
    the fact that its numbers are then not comparable to the others.
    """
    cfg = dict(cfg)
    opt_out = bool(cfg.pop(_UNEQUAL_OPT_OUT, False))

    forced = set(SHARED_TRAIN_CFG) | (set(SHARED_WIDTH_KEYS) & set(defaults))
    overridden = []
    for k, v in defaults.items():
        if k in forced and not opt_out:
            if k in cfg and cfg[k] != v:
                overridden.append((k, cfg[k], v))
            cfg[k] = v
        else:
            # opt-out, or a knob that defines the method rather than the
            # optimisation: the caller wins.
            cfg.setdefault(k, v)

    if overridden:
        # Loud, because a silently discarded hyperparameter is exactly the
        # failure mode this function exists to prevent.
        msg = "; ".join(
            "%s: %r -> %r" % (k, was, now) for k, was, now in sorted(overridden)
        )
        warnings.warn(
            "%s: caller-supplied values for shared knobs were replaced by the "
            "shared block (%s). Change SHARED_TRAIN_CFG / SHARED_HIDDEN if the "
            "value should change for every method, or pass "
            "allow_unequal_train_cfg=True in the cfg if this run is "
            "deliberately not comparable." % (method, msg),
            RuntimeWarning,
            stacklevel=2,
        )
    return cfg


def _learn_representations_fixed_split(
    X,
    M,
    A,
    Y=None,
    *,
    tilde_p,
    tilde_q,
    factor_method,
    subtrain_idx,
    val_idx,
    ae_cfg=None,
    me_cfg=None,
    encode_cfg=None,
    f_M_true_all=None,
    nn_cfg=None
):
    factor_method = factor_method.lower()
    ae_cfg = {} if ae_cfg is None else dict(ae_cfg)
    me_cfg = {} if me_cfg is None else dict(me_cfg)
    encode_cfg = {} if encode_cfg is None else dict(encode_cfg)

    encode_cfg.setdefault("batch_size", 4096)

    if factor_method == "projection":
        trainval_idx = np.concatenate([subtrain_idx, val_idx])

        W_X = _fit_projection(X[trainval_idx], tilde_p)
        W_M = _fit_projection(M[trainval_idx], tilde_q)

        f_X_all = _apply_projection(X, W_X)
        f_M_all = _apply_projection(M, W_M)

        rep_fit_info = {"method": "projection"}
        rep_history = None
        rep_model = None

    elif factor_method in {"autoencoder", "vae"}:
        model_type = "VAE" if factor_method == "vae" else "AE"

        ae_defaults = _shared_cfg(
            model_type=model_type,
            beta_kl=1.0,
            eps=SHARED_ADAM_EPS,          # train_autoencoder spells it "eps"
            hidden_dims=SHARED_HIDDEN,
            # the two per-block widths this branch actually reads below; they
            # have to appear here to be equalised, since _merge_method_cfg only
            # forces width keys it is given
            hidden_dims_X=SHARED_HIDDEN,
            hidden_dims_M=SHARED_HIDDEN,
        )
        # train_autoencoder has no min_delta argument: its improvement
        # threshold is hard-coded to 1e-4, which is the shared value. Popped
        # from the caller's cfg as well, so a driver that builds its cfg with
        # _shared_cfg() (which includes min_delta, since the other trainers do
        # take it) does not hit a TypeError here.
        assert ae_defaults.pop("min_delta") == 1e-4
        ae_cfg = _merge_method_cfg(ae_cfg, ae_defaults, method=factor_method)
        assert ae_cfg.pop("min_delta", 1e-4) == 1e-4

        # beta_kl_grid: if present, SELECT beta_kl per fold by the SAME criterion
        # MediEncoder uses to select lambda -- held-out prediction error on val_idx
        # -- so the VAE is tuned, not fixed, and the comparison is fair. Scoring by
        # prediction error (not the VAE's own recon+KL, whose magnitude changes with
        # beta itself) is what puts VAE and MediEncoder on the identical selection
        # rule. VAE-only knob; ignored by the AE branch (model_type == "AE").
        beta_grid = ae_cfg.pop("beta_kl_grid", None)
        if beta_grid is not None and model_type == "VAE":
            beta_grid = list(beta_grid)
            best_beta, best_beta_score = None, np.inf
            for cand_beta in beta_grid:
                try:
                    trial_cfg = dict(ae_cfg)
                    trial_cfg["beta_kl"] = float(cand_beta)
                    tcx = dict(trial_cfg); tcx["hidden_dims"] = trial_cfg.get("hidden_dims_X", (300, 300))
                    tcm = dict(trial_cfg); tcm["hidden_dims"] = trial_cfg.get("hidden_dims_M", (300, 300))
                    for _c in (tcx, tcm):
                        _c.pop("hidden_dims_X", None); _c.pop("hidden_dims_M", None)
                    t_X = train_autoencoder(X[subtrain_idx], latent_dim=tilde_p, X_val=X[val_idx],
                                            model_type=model_type,
                                            **{k: v for k, v in tcx.items() if k != "model_type"})
                    t_M = train_autoencoder(M[subtrain_idx], latent_dim=tilde_q, X_val=M[val_idx],
                                            model_type=model_type,
                                            **{k: v for k, v in tcm.items() if k != "model_type"})
                    fx = encode_with_autoencoder(t_X, X, model_type=model_type, **encode_cfg)
                    fm = encode_with_autoencoder(t_M, M, model_type=model_type, **encode_cfg)
                    pe = _fit_outcome_predictor_and_eval(
                        fx, fm, A, Y, subtrain_idx=subtrain_idx, val_idx=val_idx,
                        nn_cfg=dict({} if nn_cfg is None else nn_cfg)
                    ) if Y is not None else {"prediction_mse": np.inf}
                    score = float(pe["prediction_mse"])
                except Exception:
                    score = np.inf
                if np.isfinite(score) and score < best_beta_score:
                    best_beta_score, best_beta = score, float(cand_beta)
            if best_beta is None:
                best_beta = float(beta_grid[0])
            ae_cfg["beta_kl"] = best_beta
            rep_selected_beta = best_beta
        else:
            ae_cfg.pop("beta_kl_grid", None)
            rep_selected_beta = ae_cfg.get("beta_kl", None)

        ae_cfg_X = dict(ae_cfg)
        ae_cfg_M = dict(ae_cfg)

        ae_cfg_X["hidden_dims"] = ae_cfg.get("hidden_dims_X", (300, 300))
        ae_cfg_M["hidden_dims"] = ae_cfg.get("hidden_dims_M", (300, 300))

        # remove split-only keys so train_autoencoder won't receive unknown args
        ae_cfg_X.pop("hidden_dims_X", None)
        ae_cfg_X.pop("hidden_dims_M", None)

        ae_cfg_M.pop("hidden_dims_X", None)
        ae_cfg_M.pop("hidden_dims_M", None)

        ae_X = train_autoencoder(
            X[subtrain_idx],
            latent_dim=tilde_p,
            X_val=X[val_idx],
            model_type=model_type,
            **{k: v for k, v in ae_cfg_X.items() if k != "model_type"}
        )

        ae_M = train_autoencoder(
            M[subtrain_idx],
            latent_dim=tilde_q,
            X_val=M[val_idx],
            model_type=model_type,
            **{k: v for k, v in ae_cfg_M.items() if k != "model_type"}
        )

        f_X_all = encode_with_autoencoder(
            ae_X, X, model_type=model_type, **encode_cfg
        )
        f_M_all = encode_with_autoencoder(
            ae_M, M, model_type=model_type, **encode_cfg
        )

        rep_fit_info = {"method": factor_method, "selected_beta_kl": rep_selected_beta}
        rep_history = None
        rep_model = (ae_X, ae_M)

    elif factor_method == "mediencoder":
        me_defaults = _shared_cfg(
            hidden_dims_X=SHARED_HIDDEN,
            hidden_dims_M=SHARED_HIDDEN,
            hidden_dims_XM=(50, 50),
            adam_eps=SHARED_ADAM_EPS,
            lambda1=0.2,
            lambda2=0.5,
            lambda3=0.3,
            return_history=True,
            # early stopping on the held-out weighted loss, as the autoencoder
            # branch does. Set False to checkpoint on the training loss instead.
            use_val=True,
        )
        
        me_cfg = _merge_method_cfg(me_cfg, me_defaults, method="mediencoder")

        f_M_train = None
        f_M_val = None
        if f_M_true_all is not None:
            f_M_train = f_M_true_all[subtrain_idx]
            f_M_val = f_M_true_all[val_idx]

        me_use_val = bool(me_cfg["use_val"])

        me_model, me_history, me_fit_info = train_mediencoder(
            X[subtrain_idx],
            M[subtrain_idx],
            A[subtrain_idx],
            latent_p=tilde_p,
            latent_q=tilde_q,
            f_M_train=f_M_train,
            X_val=X[val_idx] if me_use_val else None,
            M_val=M[val_idx] if me_use_val else None,
            A_val=A[val_idx] if me_use_val else None,
            f_M_val=f_M_val if me_use_val else None,
            hidden_dims_X=me_cfg["hidden_dims_X"],
            hidden_dims_M=me_cfg["hidden_dims_M"],
            hidden_dims_XM=me_cfg["hidden_dims_XM"],
            activation=me_cfg["activation"],
            dropout=me_cfg["dropout"],
            lambda1=me_cfg["lambda1"],
            lambda2=me_cfg["lambda2"],
            lambda3=me_cfg["lambda3"],
            epochs=me_cfg["epochs"],
            lr_init=me_cfg["lr_init"],
            weight_decay=me_cfg["weight_decay"],
            betas=me_cfg["betas"],
            adam_eps=me_cfg["adam_eps"],
            batch_size=me_cfg["batch_size"],
            scheduler_type=me_cfg["scheduler_type"],
            step_size=me_cfg["step_size"],
            gamma=me_cfg["gamma"],
            patience=me_cfg["patience"],
            early_stop=me_cfg["early_stop"],
            min_delta=me_cfg["min_delta"],
            verbose=me_cfg["verbose"],
            return_history=me_cfg["return_history"],
            allow_unbalanced_lambda=me_cfg.get("allow_unbalanced_lambda", False),
        )

        f_X_all = encode_with_mediencoder(
            me_model, X=X, part="X", **encode_cfg
        )
        f_M_all = encode_with_mediencoder(
            me_model,
            X=X, M=M, A=A,
            part="M",
            **encode_cfg
        )

        rep_fit_info = dict(me_fit_info)
        rep_fit_info["method"] = "mediencoder"
        rep_history = me_history
        rep_model = me_model

    elif factor_method == "medivae":
        # MediVAE: the VAE analogue of MediEncoder that KEEPS the alignment term.
        #   lambda1*recon_X + lambda2*recon_M + lambda3*align + beta_kl*(KL_X + KL_M)
        #
        # This is the baseline that isolates what lambda3 contributes, as opposed to
        # what a variational bottleneck contributes. Plain VAE has been the hardest
        # baseline throughout (it beats MediEncoder wherever A explains little of f_M),
        # and the reason is that its KL term shrinks the latent toward N(0, I) -- a
        # regularizer functionally overlapping lambda3's, but UNCONDITIONAL. MediVAE
        # has BOTH, so the VAE / MediVAE / MediEncoder triple separates the two
        # mechanisms: bottleneck only, bottleneck + alignment, alignment only.
        # Without it, a reviewer can ask whether MediEncoder's gain is just
        # regularization by another name, and the table cannot answer.
        mv_defaults = _shared_cfg(
            hidden_dims_X=SHARED_HIDDEN,
            hidden_dims_M=SHARED_HIDDEN,
            hidden_dims_XM=(50, 50),   # Remark 3.1: g_XM stays lower-capacity
            adam_eps=SHARED_ADAM_EPS,
            lambda1=0.2,
            lambda2=0.5,
            lambda3=0.3,
            beta_kl=1.0,
            return_history=True,
            use_val=True,
        )
        me_cfg = _merge_method_cfg(me_cfg, mv_defaults, method="medivae")

        f_M_train = None
        f_M_val = None
        if f_M_true_all is not None:
            f_M_train = f_M_true_all[subtrain_idx]
            f_M_val = f_M_true_all[val_idx]

        mv_use_val = bool(me_cfg["use_val"])

        mv_model, mv_history, mv_fit_info = train_mediencoder_vae(
            X[subtrain_idx],
            M[subtrain_idx],
            A[subtrain_idx],
            latent_p=tilde_p,
            latent_q=tilde_q,
            f_M_train=f_M_train,
            X_val=X[val_idx] if mv_use_val else None,
            M_val=M[val_idx] if mv_use_val else None,
            A_val=A[val_idx] if mv_use_val else None,
            f_M_val=f_M_val if mv_use_val else None,
            hidden_dims_X=me_cfg["hidden_dims_X"],
            hidden_dims_M=me_cfg["hidden_dims_M"],
            hidden_dims_XM=me_cfg["hidden_dims_XM"],
            activation=me_cfg["activation"],
            dropout=me_cfg["dropout"],
            lambda1=me_cfg["lambda1"],
            lambda2=me_cfg["lambda2"],
            lambda3=me_cfg["lambda3"],
            beta_kl=me_cfg["beta_kl"],
            epochs=me_cfg["epochs"],
            lr_init=me_cfg["lr_init"],
            weight_decay=me_cfg["weight_decay"],
            betas=me_cfg["betas"],
            adam_eps=me_cfg["adam_eps"],
            batch_size=me_cfg["batch_size"],
            scheduler_type=me_cfg["scheduler_type"],
            step_size=me_cfg["step_size"],
            gamma=me_cfg["gamma"],
            patience=me_cfg["patience"],
            early_stop=me_cfg["early_stop"],
            min_delta=me_cfg["min_delta"],
            verbose=me_cfg["verbose"],
            return_history=me_cfg["return_history"],
            allow_unbalanced_lambda=me_cfg.get("allow_unbalanced_lambda", False),
        )

        # Posterior MEAN for both blocks, matching how the vae branch encodes.
        f_X_all = encode_with_mediencoder_vae(
            mv_model, X=X, part="X", **encode_cfg
        )
        f_M_all = encode_with_mediencoder_vae(
            mv_model, X=X, M=M, A=A, part="M", **encode_cfg
        )

        rep_fit_info = dict(mv_fit_info)
        rep_fit_info["method"] = "medivae"
        rep_history = mv_history
        rep_model = mv_model

    else:
        raise ValueError("Unknown factor_method.")

    return {
        "f_X_all": f_X_all,
        "f_M_all": f_M_all,
        "rep_fit_info": rep_fit_info,
        "rep_history": rep_history,
        "rep_model": rep_model
    }

def _merge_rep_fit_info(info_a, info_b):
    """
    Merge representation fitting information from the two representation trainings.
    Numeric quantities are averaged; non-numeric quantities are taken from info_a.
    """
    info_a = {} if info_a is None else dict(info_a)
    info_b = {} if info_b is None else dict(info_b)

    merged = {}
    keys = set(info_a.keys()).union(set(info_b.keys()))

    for key in keys:
        va = info_a.get(key, np.nan)
        vb = info_b.get(key, np.nan)

        vals = []
        for v in [va, vb]:
            if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v):
                vals.append(float(v))

        if len(vals) > 0:
            merged[key] = float(np.mean(vals))
        else:
            merged[key] = va if key in info_a else vb

    if "method" in info_a:
        merged["method"] = info_a["method"]

    return merged

def _check_lambda_grid_wellposed(lambda_grid):
    """
    Reject degenerate candidates before any fitting. lambda1 = 0 removes the
    X-reconstruction block from the objective, so theta_X is unconstrained --
    the argmin set is the whole parameter space and the "selected" encoder is a
    random initialisation. lambda2 = 0 does the same to theta_M. Such a point
    can win the selection rule by chance, which would put an untrained encoder
    in the reported table. lambda3 = 0 is well posed and allowed (it is the
    ablation arm).
    """
    if len(lambda_grid) == 0:
        raise ValueError("lambda_grid is empty.")
    bad = [
        tuple(float(v) for v in t) for t in lambda_grid
        if float(t[0]) <= 1e-6 or float(t[1]) <= 1e-6
    ]
    if len(bad) > 0:
        raise ValueError(
            "lambda_grid contains ill-posed candidates with lambda1 = 0 or "
            f"lambda2 = 0: {bad}. Build the grid with "
            "generate_lambda_grid(..., require_positive_recon=True) (the "
            "default) or generate_ablation_lambda_grid(), both of which drop "
            "these corners."
        )


def _select_lambda_for_fold(
    X, M, A, Y,
    *,
    tilde_p,
    tilde_q,
    factor_method,
    lambda_grid,
    subtrain_idx,
    val_idx,
    nn_cfg=None,
    ae_cfg=None,
    me_cfg=None,
    encode_cfg=None,
    f_M_true_all=None,
    seed=42
):
    """
    Algorithm 2 for ONE fold: select lambda using only (I_tr, I_val).

    Returns (lambda_tuple, rows). Each candidate is fit on subtrain_idx alone
    and scored by the held-out treated prediction error on val_idx, so nothing
    from I_nu or I_est enters the selection -- which is what makes the
    cross-fitting in Algorithm 1 honest.
    """
    _check_lambda_grid_wellposed(lambda_grid)
    best, best_score, rows = None, np.inf, []

    for (l1, l2, l3) in lambda_grid:
        lam = (float(l1), float(l2), float(l3))
        lam_seed = lambda_to_seed(seed, *lam)
        set_all_seeds(lam_seed)

        cand_cfg = dict({} if me_cfg is None else me_cfg)
        cand_cfg["lambda1"], cand_cfg["lambda2"], cand_cfg["lambda3"] = lam
        cand_cfg.setdefault("allow_unbalanced_lambda", True)

        try:
            rep = _learn_representations_fixed_split(
                X, M, A,
                Y=Y,
                tilde_p=tilde_p,
                tilde_q=tilde_q,
                factor_method=factor_method,
                subtrain_idx=subtrain_idx,
                val_idx=val_idx,
                ae_cfg=ae_cfg,
                me_cfg=cand_cfg,
                encode_cfg=encode_cfg,
                                f_M_true_all=f_M_true_all,
                nn_cfg=nn_cfg
            )
            pe = _fit_outcome_predictor_and_eval(
                rep["f_X_all"], rep["f_M_all"], A, Y,
                subtrain_idx=subtrain_idx,
                val_idx=val_idx,
                nn_cfg=dict({} if nn_cfg is None else nn_cfg)
            )
            score = float(pe["prediction_mse"])
        except Exception:
            # A candidate that cannot be fit loses; it must not abort the fold.
            score = np.inf

        rows.append({"lambda1": lam[0], "lambda2": lam[1], "lambda3": lam[2],
                     "prediction_mse": score})

        if np.isfinite(score) and score < best_score:
            best_score, best = score, lam

    if best is None:
        raise RuntimeError(
            "every lambda candidate failed on this fold; no selection possible."
        )
    return best, rows


def estimate_triply_IF(
    X, M, A, Y,
    *,
    tilde_p,
    tilde_q,
    factor_method="projection",
    nn_cfg=None,
    ae_cfg=None,
    me_cfg=None,
    encode_cfg=None,
    seed=42,
    f_M_true_all=None,
    mu10_true_vals=None,
    lambda_grid=None
):
    """
    Algorithm 1: cross-fitted estimation of theta_0 = E[Y(1, M(0))].

    Four disjoint folds I1..I4 and four fold assignments

        k   I_tr   I_val   I_nu   I_est
        1   I1     I2      I3     I4
        2   I1     I2      I4     I3
        3   I3     I4      I1     I2
        4   I3     I4      I2     I1

    so k = 1, 2 share the representation trained on (I1, I2) and k = 3, 4 share
    the one trained on (I3, I4); each representation is therefore fit twice, not
    four times, which is what "reuse_rep" refers to.

    lambda_grid (Algorithm 1, step 4): when given, lambda is selected SEPARATELY
    for each representation half from its own (I_tr, I_val) via Algorithm 2. Pass
    None for methods that have no lambda (projection, oracle, autoencoder, vae).
    """
    n = X.shape[0]
    I1, I2, I3, I4 = _split_indices_4fold(n, seed=seed + 999)

    tuning = {}

    # =========================
    # Representation A (I1,I2)
    # =========================
    set_all_seeds(seed + 1000)

    me_cfg_A = me_cfg
    if lambda_grid is not None:
        lam_A, rows_A = _select_lambda_for_fold(
            X, M, A, Y,
            tilde_p=tilde_p, tilde_q=tilde_q, factor_method=factor_method,
            lambda_grid=lambda_grid, subtrain_idx=I1, val_idx=I2,
            nn_cfg=nn_cfg, ae_cfg=ae_cfg, me_cfg=me_cfg,
            encode_cfg=encode_cfg,
            f_M_true_all=f_M_true_all, seed=seed + 1000
        )
        me_cfg_A = dict({} if me_cfg is None else me_cfg)
        me_cfg_A["lambda1"], me_cfg_A["lambda2"], me_cfg_A["lambda3"] = lam_A
        me_cfg_A.setdefault("allow_unbalanced_lambda", True)
        tuning["lambda_A"] = lam_A
        tuning["candidate_rows_A"] = rows_A
        set_all_seeds(seed + 1000)

    rep_A = _learn_representations_fixed_split(
        X, M, A,
        Y=Y,
        tilde_p=tilde_p,
        tilde_q=tilde_q,
        factor_method=factor_method,
        subtrain_idx=I1,
        val_idx=I2,
        ae_cfg=ae_cfg,
        me_cfg=me_cfg_A,
        encode_cfg=encode_cfg,
                f_M_true_all=f_M_true_all,
        nn_cfg=nn_cfg
    )

    fX_A = rep_A["f_X_all"]
    fM_A = rep_A["f_M_all"]

    # Fold 1
    nuis_tr, nuis_val = _split_nuisance_fold(I3, A, seed=seed)
    out_1 = _fit_nuisances_and_eval_theta(
        fX_A, fM_A, A, Y,
        subtrain_idx=nuis_tr,
        val_idx=nuis_val,
        target_idx=I4,
        nn_cfg=nn_cfg
    )

    # Fold 2
    nuis_tr, nuis_val = _split_nuisance_fold(I4, A, seed=seed+1)
    out_2 = _fit_nuisances_and_eval_theta(
        fX_A, fM_A, A, Y,
        subtrain_idx=nuis_tr,
        val_idx=nuis_val,
        target_idx=I3,
        nn_cfg=nn_cfg
    )

    # =========================
    # Representation B (I3,I4)
    # =========================
    set_all_seeds(seed + 2000)

    me_cfg_B = me_cfg
    if lambda_grid is not None:
        lam_B, rows_B = _select_lambda_for_fold(
            X, M, A, Y,
            tilde_p=tilde_p, tilde_q=tilde_q, factor_method=factor_method,
            lambda_grid=lambda_grid, subtrain_idx=I3, val_idx=I4,
            nn_cfg=nn_cfg, ae_cfg=ae_cfg, me_cfg=me_cfg,
            encode_cfg=encode_cfg,
            f_M_true_all=f_M_true_all, seed=seed + 2000
        )
        me_cfg_B = dict({} if me_cfg is None else me_cfg)
        me_cfg_B["lambda1"], me_cfg_B["lambda2"], me_cfg_B["lambda3"] = lam_B
        me_cfg_B.setdefault("allow_unbalanced_lambda", True)
        tuning["lambda_B"] = lam_B
        tuning["candidate_rows_B"] = rows_B
        set_all_seeds(seed + 2000)

    rep_B = _learn_representations_fixed_split(
        X, M, A,
        Y=Y,
        tilde_p=tilde_p,
        tilde_q=tilde_q,
        factor_method=factor_method,
        subtrain_idx=I3,
        val_idx=I4,
        ae_cfg=ae_cfg,
        me_cfg=me_cfg_B,
        encode_cfg=encode_cfg,
                f_M_true_all=f_M_true_all,
        nn_cfg=nn_cfg
    )

    fX_B = rep_B["f_X_all"]
    fM_B = rep_B["f_M_all"]

    # Fold 3
    nuis_tr, nuis_val = _split_nuisance_fold(I1, A, seed=seed+2)
    out_3 = _fit_nuisances_and_eval_theta(
        fX_B, fM_B, A, Y,
        subtrain_idx=nuis_tr,
        val_idx=nuis_val,
        target_idx=I2,
        nn_cfg=nn_cfg
    )

    # Fold 4
    nuis_tr, nuis_val = _split_nuisance_fold(I2, A, seed=seed+3)
    out_4 = _fit_nuisances_and_eval_theta(
        fX_B, fM_B, A, Y,
        subtrain_idx=nuis_tr,
        val_idx=nuis_val,
        target_idx=I1,
        nn_cfg=nn_cfg
    )
    
    pred_1 = _fit_outcome_predictor_and_eval(
        fX_A, fM_A, A, Y,
        subtrain_idx=I1,
        val_idx=I2,
        nn_cfg=nn_cfg
    )
    
    pred_3 = _fit_outcome_predictor_and_eval(
        fX_B, fM_B, A, Y,
        subtrain_idx=I3,
        val_idx=I4,
        nn_cfg=nn_cfg
    )
    
    prediction_mse = np.mean([
        pred_1["prediction_mse"],
        pred_3["prediction_mse"]
    ])
    
    prediction_rmse = float(np.sqrt(prediction_mse))

    theta_list = [
        out_1["theta_hat_IF"],
        out_2["theta_hat_IF"],
        out_3["theta_hat_IF"],
        out_4["theta_hat_IF"]
    ]

    # Algorithm 1, final step: theta_hat = sum_k (|I_est^(k)| / n) * theta_hat_k.
    # np.array_split gives folds differing by at most one observation, so the
    # unweighted mean this used to take is only equal to the weighted one when
    # n % 4 == 0. The weights are the estimation-fold sizes, in the k = 1..4
    # order of the table: I4, I3, I2, I1.
    est_sizes = np.array([len(I4), len(I3), len(I2), len(I1)], dtype=float)
    assert est_sizes.sum() == n, (est_sizes, n)
    theta_hat = float(np.dot(est_sizes / n, np.asarray(theta_list, dtype=float)))

    # Cross-fitted EIF-based standard error and 95% CI.
    # Pool the per-sample influence values across the four folds; the
    # variance of the mediation functional estimate is Var(phi)/n_total.
    phi_pooled = np.concatenate([
        out_1.get("phi", np.array([])),
        out_2.get("phi", np.array([])),
        out_3.get("phi", np.array([])),
        out_4.get("phi", np.array([])),
    ])
    phi_pooled = phi_pooled[np.isfinite(phi_pooled)]
    n_phi = phi_pooled.size
    if n_phi > 1:
        se_IF = float(np.std(phi_pooled, ddof=1) / np.sqrt(n_phi))
        ci_lower = float(theta_hat - 1.959963984540054 * se_IF)
        ci_upper = float(theta_hat + 1.959963984540054 * se_IF)
    else:
        se_IF = np.nan
        ci_lower = np.nan
        ci_upper = np.nan

    if mu10_true_vals is not None:
        theta_true = float(np.mean(mu10_true_vals))
    else:
        theta_true = np.nan

    rep_fit_info = _merge_rep_fit_info(
        rep_A.get("rep_fit_info", None),
        rep_B.get("rep_fit_info", None)
        )

    out = {
        "theta_hat_IF": float(theta_hat),
        "theta_true": float(theta_true),
        "fold_thetas": theta_list,
        "fold_est_sizes": [int(v) for v in est_sizes],
        "se_IF": se_IF,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "rep_fit_info": rep_fit_info,
        "prediction_mse": float(prediction_mse),
        "prediction_rmse": float(prediction_rmse)
    }
    if isinstance(rep_fit_info, dict) and "selected_beta_kl" in rep_fit_info:
        # VAE's per-fold-selected beta_kl (averaged over the two representation
        # halves by _merge_rep_fit_info), surfaced so the driver can record which
        # KL weight the TUNED VAE actually chose in each replicate.
        out["selected_beta_kl"] = rep_fit_info["selected_beta_kl"]
    if tuning:
        # Per-half selections. There is no single "the" lambda under Algorithm 1;
        # selected_lambda* below is half A's, reported so the existing table
        # columns still have a value, with half B's kept alongside it.
        out["fold_tuning"] = tuning
        lam_A = tuning["lambda_A"]
        out["selected_lambda1"], out["selected_lambda2"], out["selected_lambda3"] = lam_A
        out["selected_lambda_A"] = lam_A
        out["selected_lambda_B"] = tuning["lambda_B"]
    return out
# ============================================================
# 10) Worker
# ============================================================

def _check_task_len(args, expected, fmt):
    """
    Guard the POSITIONAL task tuple against callers written before
    `use_gxm_for_fm` was removed.

    Without this, a stale tuple unpacks one element too long and raises a bare
    "too many values to unpack" from inside a worker process, or -- worse, if a
    driver is later edited to drop a different field -- unpacks successfully with
    every argument after the removed slot shifted by one (dgp_cfg receiving the
    old flag, mu receiving dgp_cfg). That silently produces a run on the wrong
    DGP, which is the failure mode this whole exercise exists to avoid.
    """
    if len(args) > expected:
        raise ValueError(
            "Format %s task tuple has %d elements, expected %d. Fields have been "
            "REMOVED from this contract: use_gxm_for_fm (the flag that followed "
            "selection_rule -- f_M is always encoder_M(M) now) and, in Format A, "
            "Fx_oracle / Fm_oracle (the oracle branch is gone). Drop them."
            % (fmt, len(args), expected)
        )
    if len(args) != expected:
        raise ValueError(
            "Format %s task tuple has %d elements, expected %d."
            % (fmt, len(args), expected)
        )
    return args


def simulate_one_run(args):
    """
    Supports TWO task formats.

    NOTE: both formats used to carry a `use_gxm_for_fm` flag immediately after
    `selection_rule`. It is gone -- MediEncoder's f_M is always encoder_M(M).
    The flag's True branch returned g_XM(A, encoder_X(X)) instead, which never
    touches M and so threw away every part of the mediator not predictable from
    (A, f_X). Because these tuples are POSITIONAL, a stale caller that still
    passes the flag would silently shift dgp_cfg into it and mu into dgp_cfg, so
    the length is asserted below rather than left to fail somewhere downstream.

    ------------------------------------------------------------
    Format A: sensitivity mode (data already generated outside)
    ------------------------------------------------------------
    (
        X, M, A, Y, mu10_true_vals,
        tilde_p, tilde_q,
        seed,
        factor_method,
        split_cfg, nn_cfg, ae_cfg, factor_cfg, encode_cfg,
        extra_obj,
        selection_rule,          # must be "predictionError" (only rule left)
        f_M_true_all
    )

    ------------------------------------------------------------
    Format B: main Monte Carlo mode (generate data inside)
    ------------------------------------------------------------
    (
        n, p, q,
        bar_p, bar_q,
        tilde_p, tilde_q,
        seed,
        factor_method,
        split_cfg, nn_cfg, ae_cfg, factor_cfg, encode_cfg,
        extra_obj,
        selection_rule,          # must be "predictionError" (only rule left)
        dgp_cfg,
        mu,
        sigma_eps_X,
        sigma_eps_M,
        sigma_y
    )
    """

    import numpy as np
    from DGP_and_estimate import generate_dgp

    # ============================================================
    # Detect task format
    # ============================================================
    if isinstance(args[0], np.ndarray):
        # --------------------------------------------------------
        # Format A: sensitivity mode
        # --------------------------------------------------------
        (
            X, M, A, Y, mu10_true_vals,
            tilde_p, tilde_q,
            seed,
            factor_method,
            split_cfg, nn_cfg, ae_cfg, factor_cfg, encode_cfg,
            extra_obj,
            selection_rule,
            f_M_true_all
        ) = _check_task_len(args, 17, "A")

        _check_selection_rule(selection_rule)

    else:
        # --------------------------------------------------------
        # Format B: main Monte Carlo mode
        # --------------------------------------------------------
        (
            n, p, q,
            bar_p, bar_q,
            tilde_p, tilde_q,
            seed,
            factor_method,
            split_cfg, nn_cfg, ae_cfg, factor_cfg, encode_cfg,
            extra_obj,
            selection_rule,
            dgp_cfg,
            mu,
            sigma_eps_X,
            sigma_eps_M,
            sigma_y
        ) = _check_task_len(args, 21, "B")

        _check_selection_rule(selection_rule)

        set_all_seeds(seed)

        data = generate_dgp(
            n=n,
            p=p,
            q=q,
            bar_p=bar_p,
            bar_q=bar_q,
            mu=mu,
            sigma_eps_X=sigma_eps_X,
            sigma_eps_M=sigma_eps_M,
            sigma_y=sigma_y,
            seed=seed,
            **dgp_cfg
        )

        X = data["X"]
        M = data["M"]
        A = data["A"]
        Y = data["Y"]
        f_M_true_all = data["f_M"]
        mu10_true_vals = data["mu10_true_vals"]

    # ============================================================
    # Standardize configs
    # ============================================================
    split_cfg = {} if split_cfg is None else dict(split_cfg)
    nn_cfg = {} if nn_cfg is None else dict(nn_cfg)
    ae_cfg = {} if ae_cfg is None else dict(ae_cfg)
    factor_cfg = {} if factor_cfg is None else dict(factor_cfg)
    encode_cfg = {} if encode_cfg is None else dict(encode_cfg)

    factor_method = factor_method.lower()

    # ============================================================
    # Algorithm 1, the ONLY path. For the lambda-bearing methods extra_obj must
    # be the tuning grid Lambda of step 4, selected per representation half from
    # that half's own (I_tr, I_val) by Algorithm 2. There is no other mode.
    #
    # Two other modes existed and are deleted:
    #
    #   * "Mode A" -- taken whenever extra_obj was a lambda grid, i.e. for every
    #     tuned MediEncoder run -- called tune_lambda_one_split, which returned
    #     estimate_triply_IF_fixed_split: a SINGLE 0.4/0.2/0.4 split with
    #     nuisances on the 0.4 train fold and theta on the 0.4 test fold. No
    #     cross-fitting, 40% of the sample in the final average instead of 100%,
    #     one train fold shared by every nuisance, and lambda selected on a split
    #     overlapping the fold theta was computed from.
    #
    #   * "Mode B" -- fixed lambda. lambda is chosen from the data, so it is part
    #     of the estimator; pinning it turns the method into an oracle.
    # ============================================================
    if isinstance(extra_obj, (list, tuple)) and len(extra_obj) > 0:
        if factor_method not in _LAMBDA_METHODS:
            raise ValueError(
                "a lambda grid was supplied but factor_method=%r has no lambda; "
                "pass extra_obj=None for it." % (factor_method,)
            )
        lambda_grid = list(extra_obj)
        _check_lambda_grid_wellposed(lambda_grid)
    else:
        # Fixed lambda is not a mode. lambda is chosen from the data, so it is
        # part of the estimator; pinning it (e.g. at the value most often
        # selected in earlier tuned runs) is an oracle the method does not have,
        # and it makes the reported estimator different from the one the
        # algorithm defines. Algorithm 1 step 4 is not optional.
        if factor_method in _LAMBDA_METHODS:
            raise ValueError(
                "factor_method=%r requires a lambda grid: pass the grid as "
                "extra_obj. There is no fixed-lambda mode -- lambda must be "
                "selected per fold by Algorithm 2." % (factor_method,)
            )
        lambda_grid = None

    out = estimate_triply_IF(
        X, M, A, Y,
        tilde_p=tilde_p,
        tilde_q=tilde_q,
        factor_method=factor_method,
        nn_cfg=nn_cfg,
        ae_cfg=ae_cfg,
        me_cfg=factor_cfg if factor_method in _ME_CFG_METHODS else None,
        encode_cfg=encode_cfg,
        seed=seed,
        f_M_true_all=f_M_true_all,
        mu10_true_vals=mu10_true_vals,
        lambda_grid=lambda_grid
    )

    theta_true = float(out["theta_true"])

    if lambda_grid is not None:
        # No single "the" lambda under Algorithm 1: half A's is reported in the
        # legacy columns, with both halves kept alongside.
        sel1, sel2, sel3 = (out["selected_lambda1"], out["selected_lambda2"],
                            out["selected_lambda3"])
        cand_rows = out.get("fold_tuning", {}).get("candidate_rows_A")
    else:
        # a baseline with no lambda at all (projection / oracle / AE / VAE / IMAVAE)
        sel1 = sel2 = sel3 = np.nan
        cand_rows = None

    return {
            "factor_method": factor_method,
            "mode": "predictionError",
            "seed": int(seed),

            "selected_lambda1": sel1,
            "selected_lambda2": sel2,
            "selected_lambda3": sel3,
            "selected_lambda_A": out.get("selected_lambda_A"),
            "selected_lambda_B": out.get("selected_lambda_B"),
            "selected_beta_kl": out.get("selected_beta_kl", np.nan),
            "fold_thetas": out.get("fold_thetas"),
            "fold_est_sizes": out.get("fold_est_sizes"),

            "theta_hat": float(out["theta_hat_IF"]),
            "theta_true": theta_true,
            "mu10_hat": np.nan,

            "se_IF": out.get("se_IF", np.nan),
            "ci_lower": out.get("ci_lower", np.nan),
            "ci_upper": out.get("ci_upper", np.nan),
        
            "val_abs_error": np.nan,
            "test_abs_error": float(abs(out["theta_hat_IF"] - theta_true)),
        
            "prediction_mse": float(out["prediction_mse"]),
            "prediction_rmse": float(out["prediction_rmse"]),
            "selection_metric_name": "prediction_mse",
            "selection_metric_value": float(out["prediction_mse"]),
        
            "best_epoch": out["rep_fit_info"].get("best_epoch", np.nan) if out["rep_fit_info"] is not None else np.nan,
            "best_weighted_loss": out["rep_fit_info"].get("best_weighted_loss", np.nan) if out["rep_fit_info"] is not None else np.nan,
            "best_loss_X": out["rep_fit_info"].get("best_loss_X", np.nan) if out["rep_fit_info"] is not None else np.nan,
            "best_loss_M": out["rep_fit_info"].get("best_loss_M", np.nan) if out["rep_fit_info"] is not None else np.nan,
            "best_loss_align": out["rep_fit_info"].get("best_loss_align", np.nan) if out["rep_fit_info"] is not None else np.nan,
        
            "best_raw_loss_X": out["rep_fit_info"].get("best_raw_loss_X", np.nan) if out["rep_fit_info"] is not None else np.nan,
            "best_raw_loss_M": out["rep_fit_info"].get("best_raw_loss_M", np.nan) if out["rep_fit_info"] is not None else np.nan,
            "best_raw_loss_align": out["rep_fit_info"].get("best_raw_loss_align", np.nan) if out["rep_fit_info"] is not None else np.nan,
        
            "sd_X": out["rep_fit_info"].get("sd_X", np.nan) if out["rep_fit_info"] is not None else np.nan,
            "sd_M": out["rep_fit_info"].get("sd_M", np.nan) if out["rep_fit_info"] is not None else np.nan,
            "sd_fM": out["rep_fit_info"].get("sd_fM", np.nan) if out["rep_fit_info"] is not None else np.nan,
        
            "subtrain_n": np.nan,
            "val_n": np.nan,
            "test_n": X.shape[0],
        
            "candidate_rows": cand_rows
        }


# ============================================================
# 11) Evaluation summary
# ============================================================

def evaluate_estimator_performance(theta_hats, theta_trues, *, alpha=0.05):
    theta_hats = np.asarray(theta_hats, dtype=float)
    theta_trues = np.asarray(theta_trues, dtype=float)

    valid = np.isfinite(theta_hats) & np.isfinite(theta_trues)
    theta_hats = theta_hats[valid]
    theta_trues = theta_trues[valid]

    if len(theta_hats) == 0:
        return {
            "Bias": np.nan,
            "SD": np.nan,
            "RMSE": np.nan,
            "CI_low": np.nan,
            "CI_high": np.nan,
            "CI_Length": np.nan,
            "Coverage": np.nan
        }

    diffs = theta_hats - theta_trues

    bias = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    rmse = float(np.sqrt(np.mean(diffs ** 2)))

    z = float(norm.ppf(1 - alpha / 2))

    ci_low = theta_hats - z * sd
    ci_high = theta_hats + z * sd
    ci_len = float(np.mean(ci_high - ci_low))

    coverage = float(np.mean(
        (theta_trues >= ci_low) &
        (theta_trues <= ci_high)
    ))

    return {
        "Bias": round(bias, 6),
        "SD": round(sd, 6),
        "RMSE": round(rmse, 6),
        "CI_low": round(float(np.mean(ci_low)), 6),
        "CI_high": round(float(np.mean(ci_high)), 6),
        "CI_Length": round(ci_len, 6),
        "Coverage": round(coverage, 6)
    }