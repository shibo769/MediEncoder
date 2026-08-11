# ============================================================
# DGP_and_estimate.py
#
# Contents:
#   1) Diversified projection (SVD-based) factor extractor
#   2) DGP with NONLINEAR factor loading for observed X and M
#
# Nonlinear loading options (defined on ℝ):
#   (A) Wavelet (Haar atoms): complexity via L (#atoms)
#   (B) Polynomial basis:    complexity via poly_degree
#   (C) B-spline basis:      complexity via spline_K (#basis funcs)
#
# All three use the same *additive* structure across latent coordinates:
#   Z_mean[:, j] = sum_{m=1}^d  < basis(F[:,m]),  Lambda_coef[j,m,:] >
# ============================================================

import numpy as np
from scipy.linalg import svd
from scipy.special import expit
from scipy.interpolate import BSpline


# ============================================================
# 1) Diversified Projection
# ============================================================

def get_f_hat(Z: np.ndarray, tilde: int) -> np.ndarray:
    p = Z.shape[1]
    _, _, Vt = svd(Z, full_matrices=False)
    V = Vt.T[:, :tilde]
    W = np.sqrt(p) * V
    return (Z @ W) / p


# ============================================================
# 2A) Haar Wavelet on ℝ
# ============================================================

def haar_mother_wavelet(t: np.ndarray) -> np.ndarray:
    """
    Standard Haar mother wavelet defined on ℝ:

        ψ(t) =  1   if 0 ≤ t < 1
              = -1   if 1 ≤ t < 2
              =  0   otherwise
    """
    t = np.asarray(t)
    out = np.zeros_like(t, dtype=float)
    out[(t >= 0.0) & (t < 1.0)] = 1.0
    out[(t >= 1.0) & (t < 2.0)] = -1.0
    return out


def haar_atom(t: np.ndarray, r: int, s: int) -> np.ndarray:
    """
    Haar wavelet atom on ℝ:

        ψ_{r,s}(t) = 2^{r/2} ψ(2^r t - s)

    where r ∈ ℤ, s ∈ ℤ.
    """
    return (2.0 ** (r / 2.0)) * haar_mother_wavelet((2.0 ** r) * t - s)

def sample_wavelet_atoms(L, r_min, r_max, x_min, x_max, rng):

    if L <= 0:
        raise ValueError("L must be positive.")

    r_list = rng.integers(r_min, r_max + 1, size=L)
    s_list = np.empty(L, dtype=int)

    for i, r in enumerate(r_list):

        scale = 2.0 ** r

        # randomly pick a latent location
        t0 = rng.uniform(x_min, x_max)

        # choose translation so the atom is centered near t0
        s_center = scale * t0

        s = int(np.round(s_center + rng.uniform(-1, 1)))

        s_list[i] = s

    return r_list.astype(int), s_list.astype(int)


def evaluate_wavelet_basis_1d(x: np.ndarray,
                              r_list: np.ndarray,
                              s_list: np.ndarray) -> np.ndarray:
    """
    Evaluate L Haar atoms on 1D input x.
    Returns shape (n, L)
    """
    n = x.shape[0]
    L = len(r_list)
    Phi = np.empty((n, L))
    for ell in range(L):
        Phi[:, ell] = haar_atom(x, int(r_list[ell]), int(s_list[ell]))
    return Phi


def additive_wavelet_loading(F: np.ndarray,
                             Lambda_coef: np.ndarray,
                             r_list: np.ndarray,
                             s_list: np.ndarray) -> np.ndarray:
    """
    Additive nonlinear loading with Haar atoms:

        Z_mean[:, j] =
            sum_{m=1}^d sum_{ℓ=1}^L
            Lambda_coef[j,m,ℓ] ψ_{r_ℓ,s_ℓ}(F[:,m])

    Parameters
    ----------
    F : (n,d)
    Lambda_coef : (p,d,L)
    """
    n, d = F.shape
    p, d_check, L = Lambda_coef.shape
    assert d == d_check

    Z_mean = np.zeros((n, p))
    for m in range(d):
        Phi_m = evaluate_wavelet_basis_1d(F[:, m], r_list, s_list)  # (n,L)
        Z_mean += Phi_m @ Lambda_coef[:, m, :].T                    # (n,p)
    return Z_mean


# ============================================================
# 2B) Polynomial basis on ℝ
# ============================================================

def evaluate_polynomial_basis_1d(x: np.ndarray, degree: int,
                                 include_intercept: bool = False) -> np.ndarray:
    """
    Polynomial basis:
      [x, x^2, ..., x^degree]   (optionally prepend 1)

    Returns shape:
      (n, K) where K = degree (+1 if include_intercept)
    """
    if degree <= 0:
        raise ValueError("poly_degree must be positive.")
    x = np.asarray(x).reshape(-1)

    cols = []
    if include_intercept:
        cols.append(np.ones_like(x))
    for k in range(1, degree + 1):
        cols.append(x ** k)

    return np.column_stack(cols)


def additive_polynomial_loading(F: np.ndarray,
                                Lambda_coef: np.ndarray,
                                poly_degree: int,
                                include_intercept: bool = False) -> np.ndarray:
    """
    Additive nonlinear loading with polynomial basis:

      Z_mean[:, j] =
        sum_{m=1}^d  sum_{k=1}^D  Lambda_coef[j,m,k] * (F[:,m]^k)

    If include_intercept=True, then k=0 term is included too.

    Parameters
    ----------
    F : (n,d)
    Lambda_coef : (p,d,K) where K = D (+1 if include_intercept)
    """
    n, d = F.shape
    p, d_check, K = Lambda_coef.shape
    assert d == d_check

    expected_K = poly_degree + (1 if include_intercept else 0)
    if K != expected_K:
        raise ValueError(f"Lambda_coef last dim K={K} but expected {expected_K} "
                         f"from poly_degree={poly_degree}, include_intercept={include_intercept}")

    Z_mean = np.zeros((n, p))
    for m in range(d):
        Phi_m = evaluate_polynomial_basis_1d(F[:, m], poly_degree, include_intercept)  # (n,K)
        Z_mean += Phi_m @ Lambda_coef[:, m, :].T                                       # (n,p)
    return Z_mean


# ============================================================
# 2C) B-spline basis on ℝ
# ============================================================

def sample_bspline_knots(*,
                         K: int,
                         degree: int,
                         x_min: float,
                         x_max: float,
                         rng: np.random.Generator,
                         jitter: float = 0.0) -> np.ndarray:
    """
    Build a clamped (open-uniform-ish) knot vector for B-splines.

    For B-splines:
      n_basis = len(t) - degree - 1
    We want n_basis = K, so len(t) = K + degree + 1.

    With clamped endpoints:
      t = [a]*(degree+1) + internal_knots + [b]*(degree+1)
    If #internal = m, then len(t) = m + 2*(degree+1),
    so K + degree + 1 = m + 2*(degree+1)  =>  m = K - degree - 1.

    Requirements: K >= degree + 1, and internal_knots_count = K - degree - 1 >= 0.
    """
    if K < degree + 1:
        raise ValueError(f"spline_K must be >= spline_degree+1; got K={K}, degree={degree}")

    a, b = float(x_min), float(x_max)
    if not (a < b):
        raise ValueError("x_min must be < x_max for spline knots.")

    m_internal = K - degree - 1  # can be 0
    if m_internal > 0:
        internal = rng.uniform(a, b, size=m_internal)
        internal.sort()
        if jitter > 0:
            internal = np.clip(internal + rng.normal(0.0, jitter, size=m_internal), a, b)
            internal.sort()
    else:
        internal = np.array([], dtype=float)

    t = np.concatenate([
        np.full(degree + 1, a),
        internal,
        np.full(degree + 1, b)
    ])
    return t


def evaluate_bspline_basis_1d(x: np.ndarray, knot_vector: np.ndarray, degree: int) -> np.ndarray:
    """
    Evaluate the full set of B-spline basis functions implied by (knot_vector, degree).

    Returns shape (n, K) where K = len(t) - degree - 1.
    """
    x = np.asarray(x).reshape(-1)
    t = np.asarray(knot_vector, dtype=float)
    K = len(t) - degree - 1
    if K <= 0:
        raise ValueError("Invalid knot vector / degree: computed K<=0 basis functions.")

    Phi = np.empty((x.shape[0], K), dtype=float)
    # Build each basis function by one-hot coefficient vector
    for j in range(K):
        c = np.zeros(K, dtype=float)
        c[j] = 1.0
        Phi[:, j] = BSpline(t, c, degree, extrapolate=False)(x)
    # Replace NaNs from extrapolate=False (outside [a,b]) with 0
    Phi = np.nan_to_num(Phi, nan=0.0, posinf=0.0, neginf=0.0)
    return Phi


def additive_bspline_loading(F: np.ndarray,
                             Lambda_coef: np.ndarray,
                             knot_vector: np.ndarray,
                             degree: int) -> np.ndarray:
    """
    Additive nonlinear loading with B-spline basis:

      Z_mean[:, j] =
        sum_{m=1}^d  < B(F[:,m]), Lambda_coef[j,m,:] >

    Parameters
    ----------
    F : (n,d)
    Lambda_coef : (p,d,K)
    knot_vector : (K + degree + 1,)
    """
    n, d = F.shape
    p, d_check, K = Lambda_coef.shape
    assert d == d_check

    implied_K = len(knot_vector) - degree - 1
    if implied_K != K:
        raise ValueError(f"Lambda_coef last dim K={K}, but knot_vector implies K={implied_K}.")

    Z_mean = np.zeros((n, p))
    for m in range(d):
        Phi_m = evaluate_bspline_basis_1d(F[:, m], knot_vector, degree)  # (n,K)
        Z_mean += Phi_m @ Lambda_coef[:, m, :].T                         # (n,p)
    return Z_mean


# ============================================================
# 3) Data Generating Process
# ============================================================

def generate_dgp(
    n: int,
    p: int,
    q: int,
    bar_p: int,
    bar_q: int,
    mu: float = 0.0,
    sigma_eps_X: float = 1.0,
    sigma_eps_M: float = 1.0,
    sigma_y: float = 1.0,
    *,
    loading_method: str = "wavelet",
    L: int = 30,
    r_min: int = -1,
    r_max: int = 1,
    poly_degree: int = 5,
    poly_include_intercept: bool = False,
    spline_K: int = 10,
    spline_degree: int = 3,
    spline_range_pad: float = 0.05,
    spline_knot_jitter: float = 0.0,
    coef_scale_X: float = 1.0,
    coef_scale_M: float = 1.0,
    sigma_U_scale: float = 1.0,
    delta1_shift: float = 0.0,
    delta1_spread: float = 1.0,
    delta1_contrast: float = 0.0,
    seed: int | None = None,
    concatenation: bool = True,
    treatment_interaction: bool = True,
    mediator_interaction: bool = True,
    effect_scale: float = 1.0
):
    """
    Two switches degenerate the structure that path-coefficient methods
    cannot represent.  Both default to True, which reproduces the DGP of
    the paper bit-for-bit.

    ``treatment_interaction=False``
        A enters each stage additively with a single coefficient instead
        of selecting between two separate equations:

            f_M = mean_0 + A * alpha_M + eps_M
            Y   = base(f_M) + A * alpha_Y + eps_Y.

        This makes the natural direct effect exactly constant across
        units, and makes the outcome's mediator slope the same under
        both arms.

    ``mediator_interaction=False``
        The mediator enters the outcome linearly, ``f_M @ gamma``,
        instead of through ``(f_M * f_X^2) @ gamma``.  The outcome's
        slope in the mediator then no longer varies with f_X, which is
        the condition under which a product of path coefficients equals
        the natural indirect effect.

    ``effect_scale``
        Multiplies the two treatment coefficients ``alpha_M`` and
        ``alpha_Y`` that the degenerate form introduces, and nothing
        else.  It therefore shrinks the natural indirect and direct
        effects without touching the confounding, the propensity, the
        outcome's main effects, the measurement noise, or any of the
        nonlinearity.  It exists because the degenerate form makes the
        indirect effect a constant that no longer partially cancels
        across units, which inflates it relative to the DGP of the paper;
        setting ``effect_scale`` below one restores a signal-to-noise
        ratio comparable to the paper's.  It has no effect when
        ``treatment_interaction=True``, since ``alpha_M`` and ``alpha_Y``
        are not drawn on that branch.

    Turning both switches off leaves every other feature intact: the quadratic
    mediator mean in f_X, the expit propensity, the sin/softplus terms in
    the outcome, and above all the nonparametric wavelet/spline/poly
    loadings that make X and M noisy high-dimensional measurements of the
    latent factors.  Those are difficulties of *estimation* shared by all
    methods, not disagreements about the *target parameter*.
    """

    # =========================================================
    # GLOBAL STORAGE (per seed)
    # =========================================================
    if not hasattr(generate_dgp, "_storage"):
        generate_dgp._storage = {}

    # sigma_U_scale belongs in the key: it changes the data, so two scales under
    # one seed must not share a cache entry (the second call would silently get
    # the first one's f_M).
    key = (seed, p, q, bar_p, bar_q, loading_method, float(sigma_U_scale), float(delta1_shift), float(delta1_spread), float(delta1_contrast))

    if (not concatenation) or (key not in generate_dgp._storage):
        generate_dgp._storage[key] = {
            "n_generated": 0,
            "data": None,
            "rng": np.random.default_rng(seed)
        }

    storage = generate_dgp._storage[key]
    rng = storage["rng"]

    n_existing = storage["n_generated"]

    # =========================================================
    # if already enough
    # =========================================================
    if n_existing >= n:
        return {k: v[:n] for k, v in storage["data"].items()}

    # =========================================================
    # need new data
    # =========================================================
    n_new = n - n_existing

    # ---------------- Step 1: latent X ----------------
    f_X = rng.uniform(-1, 1, size=(n_new, bar_p)) + mu #  rng.uniform(-1, 1, size=(n_new, bar_p)) + mu
    f_X_sq = f_X ** 2

    # ---------------- Step 2: treatment ----------------
    alpha = rng.uniform(0, 2.0, size=bar_p)
    logits = f_X @ alpha
    e_true_vals = expit(logits)
    A = rng.binomial(1, e_true_vals)

    # ---------------- Step 3: latent M ----------------
    # delta1_shift separates the two arms' mediator equations. At the paper's value
    # 0.0 both loadings are drawn U(0.5, 1.5), so the arms are statistically
    # exchangeable and the treatment moves only ~3.6% of the variance of f_M's mean
    # structure. That is what starves MediEncoder's third loss term
    # (lambda3 * mse(z_M, g_XM(A, z_X))): it fits E[f_M | A, f_X], so if A barely
    # enters f_M there is next to no A-dependence for it to capture, and the term is
    # a constraint that buys nothing while the competing methods lose nothing by
    # ignoring A. shift = 2.0 raises that share to ~9.1%.
    #
    # The shift is added AFTER the draw, so the rng stream is untouched and the
    # default reproduces the paper's DGP bit-for-bit.
    #
    # BUT delta1_shift IS THE WRONG KNOB, measured: it adds a CONSTANT to the treated
    # arm's loading, so E[f_M | A=1] - E[f_M | A=0] grows and the two arms' mediator
    # distributions translate apart. theta = E[Y(1, M(0))] is cross-world -- mu10 has
    # to evaluate mu1 (fit on TREATED f_M) on CONTROL-arm f_M values -- so pulling the
    # supports apart destroys the region both arms share. At shift = 0.3 (SMD 0.54)
    # every method already breaks: coverage 0.32 / 0.92 / 0.06 / 0.30 for projection /
    # AE / VAE / IMAVAE with bias ~1.0, against 0.94-0.98 at shift = 0. A DGP where all
    # five estimators are invalid cannot demonstrate anything about any of them.
    #
    # delta1_spread is the knob that raises A's influence WITHOUT translating the arms.
    # It scales each arm's loading away from their common mean:
    #     delta_0 -> dbar - spread * (dbar - delta_0_raw)
    #     delta_1 -> dbar + spread * (delta_1_raw - dbar)
    # where dbar = 1.0 is the U(0.5, 1.5) mean. E[delta_1 - delta_0] stays 0, so the
    # marginal f_M means stay together and overlap survives, while var(delta_1 -
    # delta_0) grows as spread^2 -- and it is that VARIANCE, not the mean gap, that
    # makes f_M depend on A given f_X. This is the quantity the third term fits.
    # spread = 1.0 reproduces the paper exactly.
    _d0_raw = rng.uniform(0.5, 1.5, size=(bar_q, bar_p))
    _d1_raw = rng.uniform(0.5, 1.5, size=(bar_q, bar_p))
    _sp = float(delta1_spread)
    if _sp != 1.0:
        _dbar = 1.0
        _d0_raw = _dbar - _sp * (_dbar - _d0_raw)
        _d1_raw = _dbar + _sp * (_d1_raw - _dbar)
    delta_0 = _d0_raw
    delta_1 = _d1_raw + float(delta1_shift)

    # delta1_contrast is the knob that raises A's share of f_M (R2_A, the channel the
    # third term fits) WITHOUT touching cross-world overlap (SMD, which spread and
    # shift both wreck). It REPLACES the treated loading by delta_0 plus a contrast
    # whose rows are centered across covariates:
    #     delta_1 = delta_0 + contrast * V,   V ~ N(0,1)^(bar_q,bar_p),  V[k,:] -= mean_k
    # Row-centering makes sum_j V[k,j] = 0, and since E[f_X_j^2] = 1/3 for every j the
    # arm-mean gap E[mean_1 - mean_0] = (1/3) contrast * sum_j V[k,j] = 0 exactly --
    # so the marginals of f_M coincide (SMD ~ 0, measured 0.007) and the cross-world
    # mu10 never extrapolates. But per unit mean_1 - mean_0 = f_X^2 @ (contrast V)^T
    # varies with f_X, so A given f_X moves real variance of f_M: measured R2_A rises
    # 0.06 (c=0.5) -> 0.25 (c=1.5) -> 0.31 (c=2.0), matching spread=2's R2_A at SMD 0.
    # This is the structure VAE's UNCONDITIONAL KL cannot represent and the third term's
    # A-CONDITIONAL alignment can. Drawn only when nonzero, so the default is bit-for-bit
    # the paper's DGP. Composes after spread/shift, overwriting delta_1.
    _ct = float(delta1_contrast)
    if _ct != 0.0:
        _V = rng.normal(size=(bar_q, bar_p))
        _V = _V - _V.mean(axis=1, keepdims=True)
        delta_1 = delta_0 + _ct * _V

    Q, _ = np.linalg.qr(rng.normal(size=(bar_q, bar_q)))
    # sigma_U_scale multiplies the eigenvalues of Sigma_U, i.e. it scales the
    # mediator equation's residual variance while leaving its mean structure
    # (delta_0, delta_1, f_X^2) alone. It therefore controls the signal-to-noise
    # ratio of the relation f_M = h(A, f_X) + u_XM, which is exactly what
    # MediEncoder's third loss term (lambda3 * mse(z_M, g_XM(A, z_X))) models
    # and what no competing method represents at all.
    #
    # At the paper's value 1.0 the eigenvalues are U(1, 2) against a signal
    # variance of ~0.56 per coordinate, so R^2 of (A, f_X) on f_M is only 0.26 --
    # the third term has little to capture. 0.1 raises it to ~0.80.
    #
    # Default 1.0 reproduces the paper's DGP bit-for-bit: the draw is
    # rng.uniform(1, 2) either way, so the rng stream is untouched.
    Lambda_vals = rng.uniform(1.0, 2.0, size=bar_q) * float(sigma_U_scale)
    Sigma_U = Q @ np.diag(Lambda_vals) @ Q.T

    eps_M = rng.multivariate_normal(np.zeros(bar_q), Sigma_U, size=n_new)
    eps_M_prime = rng.multivariate_normal(np.zeros(bar_q), Sigma_U, size=n_new)

    mean_0 = f_X_sq @ delta_0.T
    mean_1 = f_X_sq @ delta_1.T

    if treatment_interaction:
        f_M = (1 - A[:, None]) * (mean_0 + eps_M) + A[:, None] * (mean_1 + eps_M_prime)
    else:
        # Degenerate form: A enters the mediator equation additively with a
        # coefficient, rather than selecting between two separate mediator
        # equations.  This is the structure the path-coefficient methods
        # posit, so their target parameter can coincide with the natural
        # indirect effect.  The rng draw happens only on this branch, so
        # the default DGP is bit-for-bit unchanged.
        alpha_M = float(effect_scale) * rng.uniform(0.5, 1.5, size=bar_q)
        f_M = mean_0 + A[:, None] * alpha_M[None, :] + eps_M

    # ---------------- Step 4: outcome ----------------
    beta_0 = rng.uniform(0.5, 1.5, size=bar_p)
    beta_1 = rng.uniform(0.5, 1.5, size=bar_p)
    gamma_0 = rng.uniform(0.5, 1.5, size=bar_q)
    gamma_1 = rng.uniform(0.5, 1.5, size=bar_q)
    kappa_0 = rng.uniform(0.5, 1.5, size=bar_p)
    kappa_1 = rng.uniform(0.5, 1.5, size=bar_p)

    sin_5fx_beta0 = np.sin(5 * f_X) @ beta_0
    sin_5fx_beta1 = np.sin(5 * f_X) @ beta_1

    softplus_0 = np.logaddexp(0.0, f_X @ kappa_0)
    softplus_1 = np.logaddexp(0.0, f_X @ kappa_1)

    # The mediator-by-covariate product is what makes the outcome's slope in
    # the mediator depend on f_X, and hence what breaks the equality between
    # a product of path coefficients and the natural indirect effect.
    interaction = f_M * f_X_sq if mediator_interaction else f_M

    eps_Y = rng.normal(0, sigma_y, size=n_new)
    eps_Y_prime = rng.normal(0, sigma_y, size=n_new)

    mu1_true_vals = sin_5fx_beta1 + softplus_1 + (interaction @ gamma_1)
    mu0_true_vals = sin_5fx_beta0 + softplus_0 + (interaction @ gamma_0)

    if treatment_interaction:
        Y = A * (mu1_true_vals + eps_Y) + (1 - A) * (mu0_true_vals + eps_Y_prime)
    else:
        # Same degeneration on the outcome side: A carries a single
        # coefficient instead of switching between two outcome surfaces.
        # The mediator still enters through f_M * f_X_sq, so the outcome
        # remains nonlinear in the latent factors -- only the treatment
        # interaction is removed.
        alpha_Y = float(effect_scale) * float(rng.uniform(0.5, 1.5))
        mu_base = sin_5fx_beta0 + softplus_0 + (interaction @ gamma_0)
        mu0_true_vals = mu_base
        mu1_true_vals = mu_base + alpha_Y
        Y = mu_base + A * alpha_Y + eps_Y

    # =========================================================
    # Step 5: loading
    # =========================================================

    method = loading_method.lower()
    extras = {}

    if method == "wavelet":
        f_all = np.concatenate([f_X.reshape(-1), f_M.reshape(-1)])
        x_min, x_max = float(f_all.min()), float(f_all.max())

        r_list, s_list = sample_wavelet_atoms(L, r_min, r_max, x_min, x_max, rng)

        Lambda_X_coef = rng.normal(0, coef_scale_X / np.sqrt(L), size=(p, bar_p, L))
        Lambda_M_coef = rng.normal(0, coef_scale_M / np.sqrt(L), size=(q, bar_q, L))

        X_mean = additive_wavelet_loading(f_X, Lambda_X_coef, r_list, s_list)
        M_mean = additive_wavelet_loading(f_M, Lambda_M_coef, r_list, s_list)

        extras.update({"r_list": r_list, "s_list": s_list})

    elif method == "poly":

        K = poly_degree + (1 if poly_include_intercept else 0)

        Lambda_X_coef = rng.normal(0, coef_scale_X / np.sqrt(K), size=(p, bar_p, K))
        Lambda_M_coef = rng.normal(0, coef_scale_M / np.sqrt(K), size=(q, bar_q, K))

        X_mean = additive_polynomial_loading(f_X, Lambda_X_coef, poly_degree, poly_include_intercept)
        M_mean = additive_polynomial_loading(f_M, Lambda_M_coef, poly_degree, poly_include_intercept)

        extras.update({"poly_degree": poly_degree})

    elif method == "spline":

        f_all = np.concatenate([f_X.reshape(-1), f_M.reshape(-1)])
        x_min, x_max = float(f_all.min()), float(f_all.max())

        span = x_max - x_min
        x_min -= spline_range_pad * span
        x_max += spline_range_pad * span

        knot_vector = sample_bspline_knots(
            K=spline_K,
            degree=spline_degree,
            x_min=x_min,
            x_max=x_max,
            rng=rng,
            jitter=spline_knot_jitter
        )

        Lambda_X_coef = rng.normal(0, coef_scale_X / np.sqrt(spline_K), size=(p, bar_p, spline_K))
        Lambda_M_coef = rng.normal(0, coef_scale_M / np.sqrt(spline_K), size=(q, bar_q, spline_K))

        X_mean = additive_bspline_loading(f_X, Lambda_X_coef, knot_vector, spline_degree)
        M_mean = additive_bspline_loading(f_M, Lambda_M_coef, knot_vector, spline_degree)

        extras.update({"spline_K": spline_K})

    else:
        raise ValueError("Invalid loading_method")

    # ---------------- noise ----------------
    X = X_mean + rng.normal(0, sigma_eps_X, size=(n_new, p))
    M = M_mean + rng.normal(0, sigma_eps_M, size=(n_new, q))

    # ---------------- cross-world and pure counterfactual truths ----------------
    # E[Y(a, M(a'))] integrates the outcome model over M(a'), whose conditional
    # mean given f_X is mean_{a'}.  The eps_M term drops out of the mediated
    # contribution because the outcome is linear in f_M and E[eps_M] = 0.
    #
    #   mu11 = E[Y(1, M(1)) | f_X],  mu00 = E[Y(0, M(0)) | f_X],
    #   mu10 = E[Y(1, M(0)) | f_X].
    #
    # The mediated term is (E[f_M(a')] * f_X^2) @ gamma_a when the mediator
    # interacts with the covariates and E[f_M(a')] @ gamma_a when it does not.
    def _mediated(mean_a, gamma_a):
        return ((mean_a * f_X_sq) if mediator_interaction else mean_a) @ gamma_a

    mu11_true_vals = sin_5fx_beta1 + softplus_1 + _mediated(mean_1, gamma_1)
    mu00_true_vals = sin_5fx_beta0 + softplus_0 + _mediated(mean_0, gamma_0)
    mu10_true_vals = sin_5fx_beta1 + softplus_1 + _mediated(mean_0, gamma_1)

    if not treatment_interaction:
        # With A additive in both stages, E[f_M(a) | f_X] = mean_0 + a * alpha_M
        # and the outcome surface no longer depends on a except through the
        # single coefficient alpha_Y, so the three means are
        #
        #   mu11 = base(mean_0 + alpha_M) + alpha_Y
        #   mu10 = base(mean_0)           + alpha_Y
        #   mu00 = base(mean_0)
        #
        # where base(m) = sin(5 f_X) beta_0 + softplus_0 + _mediated(m, gamma_0).
        # NDE = alpha_Y exactly.  With mediator_interaction off, base is linear
        # in m and NIE = alpha_M @ gamma_0 is a constant as well -- precisely
        # the form the path-coefficient methods target.
        base_0 = sin_5fx_beta0 + softplus_0 + _mediated(mean_0, gamma_0)
        base_1 = sin_5fx_beta0 + softplus_0 + _mediated(mean_0 + alpha_M[None, :], gamma_0)

        mu11_true_vals = base_1 + alpha_Y
        mu10_true_vals = base_0 + alpha_Y
        mu00_true_vals = base_0

    new_data = {
        "f_X": f_X,
        "f_M": f_M,
        "X": X,
        "M": M,
        "A": A,
        "Y": Y,
        "Lambda_X_coef": Lambda_X_coef,
        "Lambda_M_coef": Lambda_M_coef,
        "e_true_vals": e_true_vals,
        "mu1_true_vals": mu1_true_vals,
        "mu0_true_vals": mu0_true_vals,
        "mu11_true_vals": mu11_true_vals,
        "mu00_true_vals": mu00_true_vals,
        "mu10_true_vals": mu10_true_vals,
        "loading_method": method,
        **extras
    }

    # =========================================================
    # CONCAT
    # =========================================================
    if storage["data"] is None:
        storage["data"] = new_data
    else:
        for k in storage["data"]:
            if k in new_data:
                old_v = storage["data"][k]
                new_v = new_data[k]
    
                if isinstance(old_v, np.ndarray) and isinstance(new_v, np.ndarray):
                    storage["data"][k] = np.concatenate([old_v, new_v], axis=0)
                else:
                    # metadata like loading_method / spline_K / poly_degree / r_list / s_list
                    # keep the old one
                    storage["data"][k] = old_v
    

    storage["n_generated"] = n

    return storage["data"]

# def generate_dgp(
#     n: int,
#     p: int,
#     q: int,
#     bar_p: int,
#     bar_q: int,
#     mu: float = 0.0,
#     # sigma_eps: float = 1.0,
#     sigma_eps_X: float = 1.0,   # NEW
#     sigma_eps_M: float = 1.0,   # NEW
#     sigma_y: float = 1.0,
#     *,
#     # ---------------- Nonlinear loading selection -------------
#     loading_method: str = "wavelet",  # {"wavelet","poly","spline"}
#     # wavelet complexity
#     L: int = 30,
#     r_min: int = -1, # -2
#     r_max: int = 1, # 3
#     # polynomial complexity
#     poly_degree: int = 5,
#     poly_include_intercept: bool = False,
#     # spline complexity
#     spline_K: int = 10,
#     spline_degree: int = 3,
#     spline_range_pad: float = 0.05,   # widen knot range a bit beyond observed
#     spline_knot_jitter: float = 0.0,  # optional random jitter to internal knots
#     # coefficient scales for observed loadings
#     coef_scale_X: float = 1.0,
#     coef_scale_M: float = 1.0,
#     seed: int | None = None
# ):

#     rng = np.random.default_rng(seed)

#     # ---------------- Step 1: latent covariates ----------------
#     f_X = rng.uniform(-1, 1, size=(n, bar_p)) + mu
#     f_X_sq = f_X ** 2

#     # ---------------- Step 2: treatment ------------------------
#     alpha = rng.uniform(0, 2.0, size=bar_p)
#     logits = f_X @ alpha
#     e_true_vals = expit(logits)
#     A = rng.binomial(1, e_true_vals)

#     # ---------------- Step 3: latent mediators -----------------
#     delta_0 = rng.uniform(0.5, 1.5, size=(bar_q, bar_p))
#     delta_1 = rng.uniform(0.5, 1.5, size=(bar_q, bar_p))

#     # correlated Gaussian noise for f_M
#     Q, _ = np.linalg.qr(rng.normal(size=(bar_q, bar_q)))
#     R_U = Q
#     Lambda_vals = rng.uniform(1.0, 2.0, size=bar_q)
#     Lambda_U = np.diag(Lambda_vals)
#     Sigma_U = R_U @ Lambda_U @ R_U.T

#     eps_M = rng.multivariate_normal(mean=np.zeros(bar_q), cov=Sigma_U, size=n)
#     eps_M_prime = rng.multivariate_normal(mean=np.zeros(bar_q), cov=Sigma_U, size=n)

#     mean_0 = f_X_sq @ delta_0.T
#     mean_1 = f_X_sq @ delta_1.T

#     f_M = (1 - A[:, None]) * (mean_0 + eps_M) + A[:, None] * (mean_1 + eps_M_prime)

#     # ---------------- Step 4: outcome --------------------------
#     beta_0 = rng.uniform(0.5, 1.5, size=bar_p)
#     beta_1 = rng.uniform(0.5, 1.5, size=bar_p)
#     gamma_0 = rng.uniform(0.5, 1.5, size=bar_q)
#     gamma_1 = rng.uniform(0.5, 1.5, size=bar_q)

#     kappa_0 = rng.uniform(0.5, 1.5, size=bar_p)
#     kappa_1 = rng.uniform(0.5, 1.5, size=bar_p)

#     sin_5fx_beta0 = np.sin(5 * f_X) @ beta_0
#     sin_5fx_beta1 = np.sin(5 * f_X) @ beta_1

#     # Softplus nonlinear terms (instead of ReLU)
#     lin0 = f_X @ kappa_0
#     lin1 = f_X @ kappa_1
#     softplus_0 = np.logaddexp(0.0, lin0)
#     softplus_1 = np.logaddexp(0.0, lin1)

#     interaction = f_M * f_X_sq

#     eps_Y = rng.normal(0, sigma_y, size=n)
#     eps_Y_prime = rng.normal(0, sigma_y, size=n)

#     mu1_true_vals = sin_5fx_beta1 + softplus_1 + (interaction @ gamma_1)
#     mu0_true_vals = sin_5fx_beta0 + softplus_0 + (interaction @ gamma_0)

#     Y = A * (mu1_true_vals + eps_Y) + (1 - A) * (mu0_true_vals + eps_Y_prime)

#     # ---------------- Step 5: observed X, M via nonlinear loading --------
#     method = loading_method.lower().strip()
#     extras = {}

#     if method == "wavelet":
#         # complexity = L atoms
#         # r_list, s_list = sample_wavelet_atoms(L, r_min, r_max, rng)
        
#         f_all = np.concatenate([f_X.reshape(-1), f_M.reshape(-1)])

#         x_min = float(np.min(f_all))
#         x_max = float(np.max(f_all))
        
#         r_list, s_list = sample_wavelet_atoms(
#             L,
#             r_min,
#             r_max,
#             x_min,
#             x_max,
#             rng
#         )
# ##################
#         Lambda_X_coef = rng.normal(
#             0.0, coef_scale_X / np.sqrt(L),
#             size=(p, bar_p, L)
#         )
#         Lambda_M_coef = rng.normal(
#             0.0, coef_scale_M / np.sqrt(L),
#             size=(q, bar_q, L)
#         )

#         X_mean = additive_wavelet_loading(f_X, Lambda_X_coef, r_list, s_list)
#         M_mean = additive_wavelet_loading(f_M, Lambda_M_coef, r_list, s_list)

#         extras.update({
#             "r_list": r_list,
#             "s_list": s_list,
#         })

#     elif method == "poly":
#         # complexity = poly_degree (+ intercept if chosen)
#         Kx = poly_degree + (1 if poly_include_intercept else 0)

#         Lambda_X_coef = rng.normal(
#             0.0, coef_scale_X / np.sqrt(max(1, Kx)),
#             size=(p, bar_p, Kx)
#         )
#         Lambda_M_coef = rng.normal(
#             0.0, coef_scale_M / np.sqrt(max(1, Kx)),
#             size=(q, bar_q, Kx)
#         )

#         X_mean = additive_polynomial_loading(f_X, Lambda_X_coef, poly_degree, poly_include_intercept)
#         M_mean = additive_polynomial_loading(f_M, Lambda_M_coef, poly_degree, poly_include_intercept)

#         extras.update({
#             "poly_degree": poly_degree,
#             "poly_include_intercept": poly_include_intercept,
#         })

#     elif method == "spline":
#         # complexity = spline_K basis functions
#         # Build a shared knot vector (same for all latent coordinates, for comparability)
#         # Use a padded range based on latent support.
#         f_all = np.concatenate([f_X.reshape(-1), f_M.reshape(-1)])
#         x_min = float(np.min(f_all))
#         x_max = float(np.max(f_all))
#         span = x_max - x_min
#         x_min -= spline_range_pad * span
#         x_max += spline_range_pad * span

#         knot_vector = sample_bspline_knots(
#             K=spline_K,
#             degree=spline_degree,
#             x_min=x_min,
#             x_max=x_max,
#             rng=rng,
#             jitter=spline_knot_jitter
#         )

#         Lambda_X_coef = rng.normal(
#             0.0, coef_scale_X / np.sqrt(max(1, spline_K)),
#             size=(p, bar_p, spline_K)
#         )
#         Lambda_M_coef = rng.normal(
#             0.0, coef_scale_M / np.sqrt(max(1, spline_K)),
#             size=(q, bar_q, spline_K)
#         )

#         X_mean = additive_bspline_loading(f_X, Lambda_X_coef, knot_vector, spline_degree)
#         M_mean = additive_bspline_loading(f_M, Lambda_M_coef, knot_vector, spline_degree)

#         extras.update({
#             "spline_K": spline_K,
#             "spline_degree": spline_degree,
#             "spline_knot_vector": knot_vector,
#             "spline_x_min": x_min,
#             "spline_x_max": x_max,
#         })

#     else:
#         raise ValueError("loading_method must be one of {'wavelet','poly','spline'}; "
#                          f"got {loading_method!r}")

#     # observed noise
#     u_X = rng.normal(0, sigma_eps_X, size=(n, p))
#     u_M = rng.normal(0, sigma_eps_M, size=(n, q))

#     X = X_mean + u_X
#     M = M_mean + u_M

#     # ---------------- True mediation functional ----------------
#     E_fM_given_A0 = mean_0
#     mu10_true_vals = (
#         sin_5fx_beta1
#         + np.log1p(np.exp(f_X @ kappa_1))     # softplus
#         + (E_fM_given_A0 * f_X_sq) @ gamma_1
#     )

#     out = {
#         "f_X": f_X,
#         "f_M": f_M,
#         "X": X,
#         "M": M,
#         "A": A,
#         "Y": Y,
#         "Lambda_X_coef": Lambda_X_coef,
#         "Lambda_M_coef": Lambda_M_coef,
#         "e_true_vals": e_true_vals,
#         "mu1_true_vals": mu1_true_vals,
#         "mu10_true_vals": mu10_true_vals,
#         "loading_method": method,
#         **extras,
#     }
#     return out

# ============================================================
# 4) Sanity Check
# ============================================================

# if __name__ == "__main__":

#     import matplotlib.pyplot as plt

#     print("Running DGP sanity check...\n")

#     # --------------------------------------------------------
#     # generate one dataset
#     # --------------------------------------------------------

#     data = generate_dgp(
#         n=3000,
#         p=500,
#         q=500,
#         bar_p=5,
#         bar_q=5,
#         loading_method="wavelet",   # try: wavelet / spline / poly
#         L=30,
#         seed=123
#     )

#     f_X = data["f_X"]
#     f_M = data["f_M"]
#     X = data["X"]
#     M = data["M"]
#     A = data["A"]
#     e = data["e_true_vals"]

#     print("========== LATENT RANGE ==========")
#     print("f_X min/max:", f_X.min(), f_X.max())
#     print("f_M min/max:", f_M.min(), f_M.max())

#     print("\n========== OBSERVED RANGE ==========")
#     print("X min/max:", X.min(), X.max())
#     print("M min/max:", M.min(), M.max())

#     print("\n========== TREATMENT BALANCE ==========")
#     p1 = A.mean()
#     print("P(A=1):", p1)
#     print("P(A=0):", 1 - p1)

#     # --------------------------------------------------------
#     # Propensity score overlap
#     # --------------------------------------------------------

#     plt.figure(figsize=(6,4))

#     plt.hist(e[A==0], bins=40, alpha=0.5, density=True, label="A=0")
#     plt.hist(e[A==1], bins=40, alpha=0.5, density=True, label="A=1")

#     plt.xlabel("True Propensity Score")
#     plt.ylabel("Density")
#     plt.title("Overlap Check for Treatment Assignment")
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

#     # --------------------------------------------------------
#     # Check wavelet activation
#     # --------------------------------------------------------

#     if data["loading_method"] == "wavelet":

#         print("\n========== WAVELET ACTIVATION ==========")

#         r_list = data["r_list"]
#         s_list = data["s_list"]

#         # evaluate basis on one latent dimension
#         Phi = evaluate_wavelet_basis_1d(f_M[:,0], r_list, s_list)

#         activation_rate = np.mean(Phi != 0, axis=0)

#         print("activation rate min:", activation_rate.min())
#         print("activation rate median:", np.median(activation_rate))
#         print("activation rate max:", activation_rate.max())

#         dead_atoms = np.mean(activation_rate < 0.01)

#         print("fraction of nearly-dead atoms (<1% active):", dead_atoms)

#         plt.figure(figsize=(6,4))
#         plt.hist(activation_rate, bins=20)
#         plt.xlabel("Activation Rate")
#         plt.ylabel("Count")
#         plt.title("Wavelet Atom Activation Rates")
#         plt.tight_layout()
#         plt.show()

#     print("\nSanity check finished.")