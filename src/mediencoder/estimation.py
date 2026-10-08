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
from contextlib import contextmanager

import numpy as np

from sklearn.model_selection import train_test_split
from scipy.linalg import svd
from scipy.stats import norm

from mediencoder.models import (
    train_nuisance_nn,
    predict_nn,
    train_autoencoder,
    encode_with_autoencoder
)

from mediencoder.training import (
    train_mediencoder,
    encode_with_mediencoder,
    train_mediencoder_vae,
    encode_with_mediencoder_vae,
    validate_lambdas_unbalanced,
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
    if V.shape[1] != tilde:
        raise ValueError("Projection dimension exceeds the available training rank/dimension")
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


def _clip_eps():
    # Overlap floor for the two fitted propensities. 1e-2 is the historical default;
    # MEDIENC_CLIP_EPS raises it, which is the standard way to keep the inverse-
    # probability weights from being driven by a handful of near-deterministic units.
    return _os_env_float("MEDIENC_CLIP_EPS", 1e-2)


def _os_env_float(name, default):
    import os as _os
    try:
        return float(_os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _resolve_numerical_safeguards(settings=None):
    """Resolve once per analysis; explicit settings never inherit environment."""
    if settings is None:
        settings = {"clip_eps": _clip_eps(),
                    "pi2_soft": _os_env_float("MEDIENC_PI2_SOFT", 0.0),
                    "pi2_cap": _os_env_float("MEDIENC_PI2_CAP", 0.0)}
    else:
        if not isinstance(settings, dict) or set(settings) - {"clip_eps", "pi2_soft", "pi2_cap"}:
            raise ValueError("Unknown numerical safeguard settings")
        settings = {"clip_eps": 1e-2, "pi2_soft": 0.0, "pi2_cap": 0.0, **settings}
    settings = {key: float(value) for key, value in settings.items()}
    if not all(np.isfinite(value) for value in settings.values()):
        raise ValueError("Numerical safeguards must be finite")
    if not 0 < settings["clip_eps"] < .5:
        raise ValueError("clip_eps must be strictly between zero and one half")
    if settings["pi2_soft"] < 0 or settings["pi2_cap"] < 0:
        raise ValueError("Density-ratio safeguards cannot be negative")
    if settings["pi2_soft"] > 0 and settings["pi2_cap"] > 0:
        raise ValueError("Specify at most one density-ratio safeguard")
    return settings


def _is_resource_exhaustion(error):
    """A hardware failure must not shrink the scientific tuning grid."""
    if isinstance(error, (MemoryError, torch.cuda.OutOfMemoryError)):
        return True
    message = str(error).lower()
    return isinstance(error, RuntimeError) and any(term in message for term in (
        "out of memory", "cannot allocate memory", "can't allocate memory",
        "not enough memory", "cublas_status_alloc_failed", "cudnn_status_alloc_failed"))


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
    nn_cfg=None,
    numerical_safeguards=None,
):
    safeguards = _resolve_numerical_safeguards(numerical_safeguards)
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
    e_hat = _safe_clip_prob(e_hat, eps=safeguards["clip_eps"])

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
    post = _safe_clip_prob(post, eps=safeguards["clip_eps"])
    prior = _safe_clip_prob(e_hat, eps=safeguards["clip_eps"])

    pi2_hat = ((1 - post) / post) * (prior / (1 - prior))

    # Overlap truncation on the cross-world density ratio. pi2 is a ratio of two
    # fitted odds, so a single near-deterministic mediator propensity sends it to
    # O(1e3) and one such unit dominates the whole EIF average (measured: median
    # max pi2 = 166, worst rep 1191, and the untruncated per-replication SD is 5x
    # the truncated one). Capping it is the standard overlap-trimming step; the
    # cap is a quantile of the within-fold ratio when MEDIENC_PI2_CAP is given as
    # a value in (0, 1] (read as a percentile), and a hard constant otherwise.
    # MEDIENC_PI2_SOFT = c applies the smooth cap pi2 / (1 + pi2 / c), which is
    # monotone, tends to c, and leaves the bulk of the ratio almost untouched, so it
    # buys the same variance reduction as a hard cut at a fraction of the truncation
    # bias (hard cut at 10 leaves bias -0.27 at n = 1500; the soft version does not).
    _soft = safeguards["pi2_soft"]
    _cap = safeguards["pi2_cap"]
    if _soft > 0.0:
        pi2_hat = pi2_hat / (1.0 + pi2_hat / _soft)
    elif _cap > 0.0:
        thr = np.percentile(pi2_hat, 100.0 * _cap) if _cap <= 1.0 else _cap
        pi2_hat = np.minimum(pi2_hat, thr)

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
        "propensity": np.asarray(e_hat, dtype=float),
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
# The current manuscript and code use weight_decay=0.0 for all methods,
# including representation, auxiliary, and final nuisance networks.
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

    The shared defaults use epochs=300, scheduler_type="step", patience=25,
    and weight_decay=0.0. They override stale per-driver values unless the
    explicit opt-out below is used. A prespecified shared override is recorded
    in the simulation manifest and is not the default experiment.

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


def _preprocess_representation_inputs(X, M, train_idx, *, mode="none"):
    """Fit input transformations exclusively on representation-training rows."""
    if mode == "none":
        return X, M, {"mode": "none"}
    if mode != "standardize":
        raise ValueError("Unknown preprocessing mode")
    transformed, metadata = [], {"mode": mode, "fit_rows": np.asarray(train_idx).copy()}
    for name, data in (("X", X), ("M", M)):
        data = np.asarray(data, dtype=float)
        train = data[train_idx]
        mean, scale = train.mean(axis=0), train.std(axis=0)
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError("Nonfinite preprocessing statistics")
        # Constant features remain zero on the training fold; do not divide by zero.
        constant = scale <= 1e-8
        scale = np.where(constant, 1., scale)
        transformed.append((data - mean) / scale)
        metadata[name] = {"mean": mean, "scale": scale, "constant_columns": np.flatnonzero(constant)}
    return transformed[0], transformed[1], metadata


@contextmanager
def _isolated_random_state(seed):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            set_all_seeds(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _fit_marginal_outcome_scores(f_X, A, Y, train_idx, val_idx, target_idx, propensity, *, nn_cfg=None):
    """AIPW scores for E[Y(1)] and E[Y(0)], conditioning only on pretreatment factors.

    Using mu_a(f_X, f_M) at the observed post-treatment mediator here would not
    integrate the mediator under intervention a and would target another quantity.
    """
    config = {} if nn_cfg is None else dict(nn_cfg)
    if config.pop("min_delta", 1e-4) != 1e-4:
        raise ValueError("Nuisance improvement threshold must be 1e-4")
    config["binary"] = False
    propensity = np.asarray(propensity, dtype=float)
    if propensity.shape != (len(target_idx),) or not np.all(np.isfinite(propensity)) or np.any((propensity <= 0) | (propensity >= 1)):
        raise ValueError("Marginal scores require finite overlap propensities")
    scores = {}
    for arm, key in ((1, "theta11"), (0, "theta00")):
        train_arm = train_idx[A[train_idx] == arm]
        val_arm = val_idx[A[val_idx] == arm]
        if len(train_arm) == 0 or len(val_arm) == 0:
            raise ValueError(f"Nuisance training/validation fold lacks treatment arm {arm}")
        model, *_ = train_nuisance_nn(f_X[train_arm], Y[train_arm],
                                     f_X[val_arm], Y[val_arm], **config)
        mean = np.asarray(predict_nn(model, f_X[target_idx], binary=False), dtype=float)
        if mean.shape != (len(target_idx),) or not np.all(np.isfinite(mean)):
            raise ValueError("Nonfinite or invalid marginal-outcome predictions")
        probability = propensity if arm == 1 else 1. - propensity
        scores[key] = mean + (A[target_idx] == arm) / probability * (Y[target_idx] - mean)
        if not np.all(np.isfinite(scores[key])):
            raise ValueError("Nonfinite marginal-outcome score")
    return scores


def _foldwise_score_covariance(scores, estimation_indices):
    """Estimate covariance of the size-weighted cross-fitted score mean.

    Center within each evaluation fold: sum_k n_k * S_k / n**2, where
    S_k is the sample covariance with denominator n_k - 1. This is an
    asymptotic cross-fitting variance estimator, not a claim that the fitted
    fold estimates are independent in finite samples.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or len(scores) < 2 or scores.shape[1] < 1 or not np.isfinite(scores).all():
        raise ValueError("Scores must be a finite subject-by-component matrix")
    n, dimension = scores.shape
    seen = np.zeros(n, dtype=int)
    covariance = np.zeros((dimension, dimension), dtype=float)
    fold_covariances = []
    for indices in estimation_indices:
        indices = np.asarray(indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("Estimation indices must be one-dimensional integers")
        if len(indices) < 2 or np.any(indices < 0) or np.any(indices >= n):
            raise ValueError("Each estimation fold needs at least two in-range subjects")
        np.add.at(seen, indices, 1)
        centered = scores[indices] - scores[indices].mean(axis=0)
        fold_covariance = centered.T @ centered / (len(indices) - 1)
        covariance += (len(indices) / n**2) * fold_covariance
        fold_covariances.append(fold_covariance)
    if not np.all(seen == 1):
        raise ValueError("Estimation folds must cover each subject exactly once")
    if not np.isfinite(covariance).all():
        raise ValueError("Nonfinite within-fold score covariance")
    return covariance, fold_covariances


def summarize_effect_scores(theta11, theta10, theta00, *, estimation_indices=None):
    """Use matched-subject contrasts and within-fold covariance of their mean.

    Cross-fitted callers must supply the evaluation folds. Without folds,
    treat the input as one sample; this preserves the standalone score API.
    """
    arrays = [np.asarray(v, dtype=float) for v in (theta11, theta10, theta00)]
    if any(v.ndim != 1 for v in arrays) or len(arrays[0]) < 2 or any(v.shape != arrays[0].shape for v in arrays):
        raise ValueError("Component scores must be matching vectors with at least two subjects")
    if not all(np.all(np.isfinite(v)) for v in arrays):
        raise ValueError("All component scores must be finite; no subjects are dropped")
    s11, s10, s00 = arrays
    order = ["NIE", "NDE", "TE"]
    matrix = np.column_stack((s11-s10, s10-s00, s11-s00))
    if estimation_indices is None:
        estimation_indices = [np.arange(len(s11))]
    covariance, fold_covariances = _foldwise_score_covariance(matrix, estimation_indices)
    means, errors = matrix.mean(axis=0), np.sqrt(np.diag(covariance))
    if not np.all(np.isfinite(covariance)):
        raise ValueError("Nonfinite effect covariance")
    z = 1.959963984540054
    return {
        "effects": dict(zip(order, map(float, means))),
        "effect_se": dict(zip(order, map(float, errors))),
        "effect_ci": {key: [float(means[j]-z*errors[j]), float(means[j]+z*errors[j])] for j, key in enumerate(order)},
        "effect_scores": {key: matrix[:, j].copy() for j, key in enumerate(order)},
        "effect_covariance": covariance, "effect_order": order,
        "effect_fold_score_covariances": fold_covariances,
        "effect_variance_estimator": "within_fold_size_weighted",
        "component_scores": dict(zip(["theta11", "theta10", "theta00"], arrays)),
        "component_means": dict(zip(["theta11", "theta10", "theta00"], [float(v.mean()) for v in arrays])),
    }


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
    nn_cfg=None,
    preprocessing="none",
):
    factor_method = factor_method.lower()
    ae_cfg = {} if ae_cfg is None else dict(ae_cfg)
    me_cfg = {} if me_cfg is None else dict(me_cfg)
    encode_cfg = {} if encode_cfg is None else dict(encode_cfg)

    encode_cfg.setdefault("batch_size", 4096)
    X, M, preprocessing_info = _preprocess_representation_inputs(
        X, M, subtrain_idx, mode=preprocessing)

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
        beta_rows = None
        if beta_grid is not None and model_type == "VAE":
            beta_grid = [float(value) for value in beta_grid]
            if not beta_grid or any(not np.isfinite(value) or value < 0 for value in beta_grid):
                raise ValueError("beta_kl_grid must contain finite nonnegative candidates")
            beta_rows = []
            best_beta, best_beta_score = None, np.inf
            for cand_beta in beta_grid:
                failure = None
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
                    if not np.isfinite(score):
                        failure = {"type": "NonfinitePredictionError", "message": "Candidate prediction MSE is not finite"}
                except Exception as exc:
                    score = np.inf
                    failure = {"type": type(exc).__name__, "message": str(exc)}
                    if _is_resource_exhaustion(exc):
                        exc.tuning_rows = beta_rows + [{"beta_kl": cand_beta, "prediction_mse": score, "failure": failure}]
                        raise
                beta_rows.append({"beta_kl": cand_beta, "prediction_mse": score, "failure": failure})
                if np.isfinite(score) and score < best_beta_score:
                    best_beta_score, best_beta = score, float(cand_beta)
            if best_beta is None:
                examples = [row["failure"] for row in beta_rows if row["failure"] is not None][:3]
                error = RuntimeError(f"All {len(beta_rows)} beta candidates failed on this fold; examples: {examples}")
                error.tuning_rows = beta_rows
                raise error
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
        if beta_rows is not None:
            rep_fit_info["beta_tuning_rows"] = beta_rows
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
            # Held-out reconstruction checkpointing. Set False to monitor the
            # training-weighted loss instead.
            use_val=True,
        )
        
        me_cfg = _merge_method_cfg(me_cfg, me_defaults, method="mediencoder")

        me_use_val = bool(me_cfg["use_val"])

        me_model, me_history, me_fit_info = train_mediencoder(
            X[subtrain_idx],
            M[subtrain_idx],
            A[subtrain_idx],
            latent_p=tilde_p,
            latent_q=tilde_q,
            X_val=X[val_idx] if me_use_val else None,
            M_val=M[val_idx] if me_use_val else None,
            A_val=A[val_idx] if me_use_val else None,
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

        mv_use_val = bool(me_cfg["use_val"])

        mv_model, mv_history, mv_fit_info = train_mediencoder_vae(
            X[subtrain_idx],
            M[subtrain_idx],
            A[subtrain_idx],
            latent_p=tilde_p,
            latent_q=tilde_q,
            X_val=X[val_idx] if mv_use_val else None,
            M_val=M[val_idx] if mv_use_val else None,
            A_val=A[val_idx] if mv_use_val else None,
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
        "rep_model": rep_model,
        "resolved_config": dict(me_cfg if factor_method in _LAMBDA_METHODS else ae_cfg),
        "preprocessing": preprocessing_info,
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
    try:
        size = len(lambda_grid)
    except TypeError as exc:
        raise ValueError("lambda_grid must be a nonempty sequence of three-number candidates") from exc
    if size == 0:
        raise ValueError("lambda_grid is empty.")
    for index, candidate in enumerate(lambda_grid):
        try:
            values = np.asarray(candidate)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"lambda_grid candidate {index} must contain exactly three finite numbers") from exc
        if values.shape != (3,):
            raise ValueError(f"lambda_grid candidate {index} must contain exactly three finite numbers")
        if not validate_lambdas_unbalanced(*values, C=1.0, tol=1e-6):
            raise ValueError(
                f"Invalid lambda_grid candidate {index}: {candidate!r}. "
                "Need three finite nonnegative numbers, lambda1 > 1e-6, "
                "lambda2 > 1e-6, and lambda1 + lambda2 + lambda3 = 1 "
                "(absolute tolerance 1e-6). Candidates are not automatically normalized."
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
    seed=42,
    preprocessing="none",
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

        failure = None
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
                nn_cfg=nn_cfg,
                preprocessing=preprocessing,
            )
            pe = _fit_outcome_predictor_and_eval(
                rep["f_X_all"], rep["f_M_all"], A, Y,
                subtrain_idx=subtrain_idx,
                val_idx=val_idx,
                nn_cfg=dict({} if nn_cfg is None else nn_cfg)
            )
            score = float(pe["prediction_mse"])
            if not np.isfinite(score):
                failure = {"type": "NonfinitePredictionError", "message": "Candidate prediction MSE is not finite"}
        except Exception as exc:
            # Failed candidates lose, and their causes are saved in tuning rows.
            score = np.inf
            failure = {"type": type(exc).__name__, "message": str(exc)}
            if _is_resource_exhaustion(exc):
                exc.tuning_rows = rows + [{"lambda1": lam[0], "lambda2": lam[1], "lambda3": lam[2],
                                          "prediction_mse": score, "failure": failure}]
                raise

        rows.append({"lambda1": lam[0], "lambda2": lam[1], "lambda3": lam[2],
                     "prediction_mse": score, "failure": failure})

        if np.isfinite(score) and score < best_score:
            best_score, best = score, lam

    if best is None:
        examples = [row["failure"] for row in rows if row["failure"] is not None][:3]
        error = RuntimeError(
            f"All {len(rows)} lambda candidates failed on this fold; examples: {examples}"
        )
        error.tuning_rows = rows
        raise error
    return best, rows


def _aggregate_crossfit_scores(n, estimation_indices, fold_outputs):
    """Validate once-only cross-fitting and compute dataset-specific inference.

    Scores are uncentered, and are returned in original subject order. No
    observation or nonfinite score is dropped from the inferential sample.
    """
    if n < 2 or len(estimation_indices) != len(fold_outputs):
        raise ValueError("Invalid cross-fit aggregation inputs")
    scores = np.empty(n, dtype=float)
    seen = np.zeros(n, dtype=int)
    weighted_theta = 0.0
    for indices, result in zip(estimation_indices, fold_outputs):
        indices = np.asarray(indices)
        phi = np.asarray(result["phi"], dtype=float)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("Estimation indices must be one-dimensional integers")
        if not len(indices) or np.any(indices < 0) or np.any(indices >= n):
            raise ValueError("Empty or out-of-range estimation fold")
        if phi.ndim != 1 or len(phi) != len(indices):
            raise ValueError("Each estimation observation must have one score")
        if not np.all(np.isfinite(phi)):
            raise ValueError("Nonfinite cross-fit score; replication must fail")
        theta = float(result["theta_hat_IF"])
        if not np.isfinite(theta) or not np.isclose(theta, phi.mean(), rtol=1e-12, atol=1e-12):
            raise ValueError("Fold estimate does not equal its score mean")
        np.add.at(seen, indices, 1)
        scores[indices] = phi
        weighted_theta += len(indices) * theta / n
    if not np.all(seen == 1):
        raise ValueError("Estimation folds must cover each subject exactly once")
    theta = float(scores.mean())
    if not np.isclose(theta, weighted_theta, rtol=1e-12, atol=1e-12):
        raise ValueError("Pooled score mean differs from size-weighted estimate")
    covariance, fold_covariances = _foldwise_score_covariance(scores[:, None], estimation_indices)
    se = float(np.sqrt(covariance[0, 0]))
    if not np.isfinite(se):
        raise ValueError("Nonfinite cross-fit standard error")
    z = 1.959963984540054
    return dict(theta_hat_IF=theta, crossfit_scores=scores, n_scores=n,
                variance_estimator="within_fold_size_weighted",
                fold_score_variances=[float(value[0, 0]) for value in fold_covariances],
                se_IF=se, ci_lower=float(theta-z*se), ci_upper=float(theta+z*se))


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
    lambda_grid=None,
    return_effects=False,
    preprocessing="none",
    numerical_safeguards=None,
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
    None for methods that have no lambda (projection, autoencoder, vae).
    """
    X, M, A, Y = map(np.asarray, (X, M, A, Y))
    if X.ndim != 2 or M.ndim != 2 or A.ndim != 1 or Y.ndim != 1:
        raise ValueError("X/M must be matrices and A/Y must be vectors")
    if X.shape[1] == 0 or M.shape[1] == 0:
        raise ValueError("X and M must each contain at least one feature")
    for name, dimension in (("tilde_p", tilde_p), ("tilde_q", tilde_q)):
        if isinstance(dimension, (bool, np.bool_)) or not isinstance(dimension, (int, np.integer)) or dimension <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if preprocessing not in {"none", "standardize"}:
        raise ValueError("preprocessing must be 'none' or 'standardize'")
    safeguards = _resolve_numerical_safeguards(numerical_safeguards)
    n = X.shape[0]
    if n < 8 or any(len(v) != n for v in (M, A, Y)):
        raise ValueError("Observed arrays must have the same length, at least eight for within-fold variance")
    if not all(np.all(np.isfinite(v)) for v in (X, M, A, Y)):
        raise ValueError("Observed inputs must all be finite")
    if not np.all(np.isin(A, [0, 1])):
        raise ValueError("Treatment A must be binary")
    factor_method = factor_method.lower()
    if factor_method in _LAMBDA_METHODS:
        if lambda_grid is None:
            raise ValueError("MediEncoder/MediVAE requires a tuning grid")
        _check_lambda_grid_wellposed(lambda_grid)
    elif lambda_grid is not None:
        raise ValueError("This representation method does not use a lambda grid")
    I1, I2, I3, I4 = _split_indices_4fold(n, seed=seed + 999)
    fold_indices = [
        dict(representation_train=I1, representation_validation=I2, nuisance=I3, estimation=I4),
        dict(representation_train=I1, representation_validation=I2, nuisance=I4, estimation=I3),
        dict(representation_train=I3, representation_validation=I4, nuisance=I1, estimation=I2),
        dict(representation_train=I3, representation_validation=I4, nuisance=I2, estimation=I1),
    ]
    for roles in fold_indices:
        joined = np.concatenate(list(roles.values()))
        if len(joined) != n or not np.array_equal(np.sort(joined), np.arange(n)):
            raise ValueError("Cross-fit fold roles must partition all subjects")

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
            seed=seed + 1000, preprocessing=preprocessing,
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
        nn_cfg=nn_cfg,
        preprocessing=preprocessing,
    )

    fX_A = rep_A["f_X_all"]
    fM_A = rep_A["f_M_all"]

    # Fold 1
    nuis_tr, nuis_val = _split_nuisance_fold(I3, A, seed=seed)
    fold_indices[0]["nuisance_train"] = nuis_tr
    fold_indices[0]["nuisance_validation"] = nuis_val
    out_1 = _fit_nuisances_and_eval_theta(
        fX_A, fM_A, A, Y,
        subtrain_idx=nuis_tr,
        val_idx=nuis_val,
        target_idx=I4,
        nn_cfg=nn_cfg,
        numerical_safeguards=safeguards,
    )

    # Fold 2
    nuis_tr, nuis_val = _split_nuisance_fold(I4, A, seed=seed+1)
    fold_indices[1]["nuisance_train"] = nuis_tr
    fold_indices[1]["nuisance_validation"] = nuis_val
    out_2 = _fit_nuisances_and_eval_theta(
        fX_A, fM_A, A, Y,
        subtrain_idx=nuis_tr,
        val_idx=nuis_val,
        target_idx=I3,
        nn_cfg=nn_cfg,
        numerical_safeguards=safeguards,
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
            seed=seed + 2000, preprocessing=preprocessing,
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
        nn_cfg=nn_cfg,
        preprocessing=preprocessing,
    )

    fX_B = rep_B["f_X_all"]
    fM_B = rep_B["f_M_all"]

    # Fold 3
    nuis_tr, nuis_val = _split_nuisance_fold(I1, A, seed=seed+2)
    fold_indices[2]["nuisance_train"] = nuis_tr
    fold_indices[2]["nuisance_validation"] = nuis_val
    out_3 = _fit_nuisances_and_eval_theta(
        fX_B, fM_B, A, Y,
        subtrain_idx=nuis_tr,
        val_idx=nuis_val,
        target_idx=I2,
        nn_cfg=nn_cfg,
        numerical_safeguards=safeguards,
    )

    # Fold 4
    nuis_tr, nuis_val = _split_nuisance_fold(I2, A, seed=seed+3)
    fold_indices[3]["nuisance_train"] = nuis_tr
    fold_indices[3]["nuisance_validation"] = nuis_val
    out_4 = _fit_nuisances_and_eval_theta(
        fX_B, fM_B, A, Y,
        subtrain_idx=nuis_tr,
        val_idx=nuis_val,
        target_idx=I1,
        nn_cfg=nn_cfg,
        numerical_safeguards=safeguards,
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

    inference = _aggregate_crossfit_scores(
        n, [I4, I3, I2, I1], [out_1, out_2, out_3, out_4])
    est_sizes = [len(I4), len(I3), len(I2), len(I1)]

    rep_fit_info = _merge_rep_fit_info(
        rep_A.get("rep_fit_info", None),
        rep_B.get("rep_fit_info", None)
        )

    out = {
        **inference,
        "fold_thetas": theta_list,
        "fold_est_sizes": [int(v) for v in est_sizes],
        "rep_fit_info": rep_fit_info,
        "rep_fit_info_by_half": {"A": rep_A.get("rep_fit_info"), "B": rep_B.get("rep_fit_info")},
        "fold_indices": fold_indices,
        "resolved_config": {
            "representation_A": rep_A.get("resolved_config", {}),
            "representation_B": rep_B.get("resolved_config", {}),
            "preprocessing_A": rep_A.get("preprocessing", {"mode": preprocessing}),
            "preprocessing_B": rep_B.get("preprocessing", {"mode": preprocessing}),
            "nuisance": {} if nn_cfg is None else dict(nn_cfg),
            "encode": {} if encode_cfg is None else dict(encode_cfg),
            **safeguards,
        },
        "estimator_contract": {
            "observed_data_only": True,
            "alignment_stop_gradient": True if factor_method in _LAMBDA_METHODS else None,
            "coupled_loss_scales": "none" if factor_method in _LAMBDA_METHODS else None,
            "coupled_loss_reduction": "mean_over_subjects_and_coordinates" if factor_method in _LAMBDA_METHODS else None,
            "scores": "uncentered_crossfit_scores_in_original_subject_order",
            "standard_error": "sqrt(sum_k(n_k * var(scores_in_fold_k, ddof=1))/n^2)",
        },
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
    if return_effects:
        # Add marginal-outcome fits only after the original theta10 path has
        # finished; isolate their random state from caller and theta10 fitting.
        score11, score00 = np.empty(n), np.empty(n)
        base_outputs = [out_1, out_2, out_3, out_4]
        for k, (roles, base) in enumerate(zip(fold_indices, base_outputs)):
            features = fX_A if k < 2 else fX_B
            with _isolated_random_state(seed + 3000 + k):
                marginal = _fit_marginal_outcome_scores(
                    features, A, Y, roles["nuisance_train"],
                    roles["nuisance_validation"], roles["estimation"],
                    base["propensity"], nn_cfg=nn_cfg)
            score11[roles["estimation"]] = marginal["theta11"]
            score00[roles["estimation"]] = marginal["theta00"]
        out.update(summarize_effect_scores(
            score11, inference["crossfit_scores"], score00,
            estimation_indices=[roles["estimation"] for roles in fold_indices]))
    return out
# ============================================================
# 10) Worker
# ============================================================


def simulate_one_run(*args, **kwargs):
    """The legacy worker mixed estimation and sample-based simulation truth."""
    raise RuntimeError("Legacy simulation worker is disabled; use run_corrected_tables.py")


def evaluate_estimator_performance(*args, **kwargs):
    """The legacy Monte Carlo-SD interval summarizer is intentionally disabled."""
    raise RuntimeError("Use replicate-specific intervals in run_corrected_tables.py")
