"""Observed-data-only adapters. Baseline adaptations return point estimates only."""
import importlib
from contextlib import contextmanager
import numpy as np

from .reporting import EFFECTS, validate_observed, validate_result

CORE_METHODS = ("mediencoder", "projection", "autoencoder", "vae")
ADAPTED_METHODS = ("nath-adapted", "dp2lm-adapted", "imavae-adapted", "lsem-ridge")
METHODS = CORE_METHODS + ADAPTED_METHODS


@contextmanager
def _canonical_budget(epochs, patience, hidden):
    """Apply this serial run's declared budget and restore core defaults afterwards."""
    import mediencoder.estimation as estimation
    previous = dict(estimation.SHARED_TRAIN_CFG)
    previous_hidden = estimation.SHARED_HIDDEN
    try:
        estimation.SHARED_TRAIN_CFG.update(epochs=epochs, patience=patience)
        estimation.SHARED_HIDDEN = hidden
        yield
    finally:
        estimation.SHARED_TRAIN_CFG.clear()
        estimation.SHARED_TRAIN_CFG.update(previous)
        estimation.SHARED_HIDDEN = previous_hidden


def inference_available(method):
    return method in CORE_METHODS


def _point_result(raw, description):
    return {"effects": {effect: float(raw[effect]) for effect in EFFECTS},
            "effect_se": {effect: None for effect in EFFECTS},
            "effect_ci": {effect: None for effect in EFFECTS},
            "inference_available": False, "implementation": description}


def fit_method(method, X, M, A, Y, *, seed, options=None, external_adapter=None):
    """Fit using observed arrays only; no truth or latent factors enter this API.

    Each method receives private copies, so an external implementation cannot
    mutate another method's paired dataset. External adapters must follow the
    result schema described in docs/comparison.md.
    """
    X, M, A, Y = (v.copy() for v in validate_observed(X, M, A, Y))
    options = dict(options or {})
    from mediencoder.estimation import set_all_seeds
    set_all_seeds(int(seed))
    if external_adapter is not None:
        module_name, function_name = external_adapter.split(":", 1)
        function = getattr(importlib.import_module(module_name), function_name)
        result = function(X, M, A, Y, seed=int(seed), options=options)
        if "inference_available" not in result or "implementation" not in result:
            raise ValueError("External adapter must declare inference_available and implementation provenance")
        return validate_result(result)
    epochs = int(options.pop("epochs", 150))
    hidden = tuple(options.pop("hidden_dims", (300, 300)))
    tilde_p = int(options.pop("tilde_p", 7))
    tilde_q = int(options.pop("tilde_q", 7))
    patience = int(options.pop("patience", 25))
    if epochs < 1 or patience < 1 or not hidden or min(hidden) < 1:
        raise ValueError("Training budget and hidden dimensions must be positive")
    if method in CORE_METHODS:
        from mediencoder.estimation import estimate_triply_IF
        from mediencoder.training import generate_lambda_grid
        grid = options.pop("lambda_grid", None)
        if grid is None and method == "mediencoder":
            grid = generate_lambda_grid(C=1.0, step=0.1, require_order=False)
        common = {"epochs": epochs, "patience": patience, "lr_init": 1e-3,
                  "weight_decay": 0.0, "batch_size": 512}
        nn_cfg = {**common, "hidden_dims": hidden}
        ae_cfg = {**common, "hidden_dims_X": hidden, "hidden_dims_M": hidden}
        me_cfg = {**common, "hidden_dims_X": hidden, "hidden_dims_M": hidden,
                  "hidden_dims_XM": tuple(options.pop("hidden_dims_XM", (50, 50))),
                  "allow_unbalanced_lambda": True}
        numerical_safeguards = options.pop("numerical_safeguards", {})
        if options:
            raise ValueError(f"Unknown options for {method}: {sorted(options)}")
        with _canonical_budget(epochs, patience, hidden):
            raw = estimate_triply_IF(X, M, A, Y, tilde_p=tilde_p, tilde_q=tilde_q,
                                    factor_method=method, nn_cfg=nn_cfg, ae_cfg=ae_cfg,
                                    me_cfg=me_cfg, seed=int(seed), lambda_grid=grid,
                                    return_effects=True, numerical_safeguards=numerical_safeguards)
        result = {k: raw[k] for k in ("effects", "effect_se", "effect_ci", "effect_scores")}
        result.update(inference_available=True, implementation=f"canonical cross-fitted {method} representation / EIF")
        for key in ("effect_covariance", "effect_order", "effect_fold_score_covariances",
                    "effect_variance_estimator", "fold_indices"):
            if key in raw:
                result[key] = raw[key]
        return validate_result(result)
    from . import baselines
    if method == "nath-adapted":
        iterations = int(options.pop("iterations", 20))
        if iterations < 1:
            raise ValueError("Nath iterations must be positive")
        raw = baselines.fit_nath_deep_mediation(
            X, M, A, Y, covar=True, covar_mode="learned", n_covar_pc=tilde_p,
            hidden_dims=hidden, inner_epochs=epochs, patience=patience,
            iterations=iterations, seed=int(seed), **options)
    elif method == "dp2lm-adapted":
        raw = baselines.fit_dp2lm(X, M, A, Y, hidden_dims=hidden, epochs=epochs,
                                  patience=patience, seed=int(seed), **options)
    elif method == "lsem-ridge":
        raw = baselines.fit_lsem_ridge(X, M, A, Y, n_covar_pc=tilde_p,
                                       n_mediator_pc=tilde_q, seed=int(seed), **options)
    elif method == "imavae-adapted":
        from .imavae import train_imavae
        n_mc = int(options.pop("n_mc", 200))
        if n_mc < 1:
            raise ValueError("n_mc must be positive")
        idx = np.random.default_rng(seed).permutation(len(Y))
        n_val = max(2, int(round(0.2 * len(Y))))
        train, val = idx[n_val:], idx[:n_val]
        if len(train) < 2:
            raise ValueError("Insufficient disjoint IMAVAE training/validation observations")
        model, _, _ = train_imavae(
            X[train], M[train], A[train], Y[train], latent_q=tilde_q,
            X_val=X[val], M_val=M[val], A_val=A[val], Y_val=Y[val],
            hidden_dims_enc=hidden, hidden_dims_dec=hidden, hidden_dims_prior=hidden,
            hidden_dims_y=hidden, epochs=epochs, patience=patience, **options)
        raw = baselines.imavae_effects(model, X, n_mc=n_mc)
    else:
        raise ValueError(f"Unknown comparison method: {method}")
    return validate_result(_point_result(raw, f"project-local {method}; no validated native standard error"))
