"""Observed-data input and cross-fitted natural-effect analysis."""

from .io import load_npz, load_rds, validate_observed_data

__all__ = ["load_npz", "load_rds", "validate_observed_data", "analyze"]


def analyze(X, M, A, Y, *, tilde_p, tilde_q, seed, factor_method="mediencoder",
            lambda_grid=None, preprocessing="standardize", nn_cfg=None,
            ae_cfg=None, me_cfg=None, encode_cfg=None, numerical_safeguards=None):
    """Run one prespecified partition seed; report score-based sampling uncertainty.

    No bootstrap, latent factors, simulation truth, or across-seed standard error
    enters this observed-data procedure. Dimensions must be chosen explicitly.
    """
    from mediencoder.estimation import estimate_triply_IF
    from mediencoder.training import generate_lambda_grid

    data = validate_observed_data(X, M, A, Y)
    if factor_method in {"mediencoder", "medivae"} and lambda_grid is None:
        lambda_grid = [v for v in generate_lambda_grid(require_order=False)
                       if v[2] > 1e-6]
    return estimate_triply_IF(
        **data, tilde_p=tilde_p, tilde_q=tilde_q, seed=seed,
        factor_method=factor_method, lambda_grid=lambda_grid,
        preprocessing=preprocessing, return_effects=True,
        numerical_safeguards={} if numerical_safeguards is None else numerical_safeguards,
        nn_cfg=nn_cfg, ae_cfg=ae_cfg, me_cfg=me_cfg, encode_cfg=encode_cfg,
    )
