"""Fixed-mechanism main simulation with additive cubic measurement loadings.

New mechanisms use powers one through three with independent fixed Gaussian
coefficients. Structural equations, parameter distributions and population truth
are unchanged by this measurement-map choice. Haar loadings remain an explicit
legacy option; schema-1 artifacts retain their original family and content hash.
No estimation observations are used to draw either loading family.

The population target is E[Y(1,M(0))].  All polynomial moments are analytic; only
the smooth softplus expectation uses deterministic, successively refined tensor
Gauss--Legendre quadrature.  Truth is cached in each serialized mechanism.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit


SCHEMA_VERSION = 2
SUPPORT_RULE = "independent-fixed-latent-pilot-minmax"
LOADING_FAMILIES = ("polynomial", "haar")


@dataclass(frozen=True)
class DGPConfig:
    p: int = 2000
    q: int = 1000
    bar_p: int = 5
    bar_q: int = 5
    mu: float = 0.0
    sigma_eps_X: float = 2.0
    sigma_eps_M: float = 1.0
    sigma_y: float = 1.0
    sigma_U_scale: float = 0.05
    delta1_shift: float = 0.0
    delta1_spread: float = 1.0
    delta1_contrast: float = 2.0
    L: int = 5
    r_min: int = 1
    r_max: int = 3
    coef_scale_X: float = 1.0
    coef_scale_M: float = 1.0
    pilot_size: int = 4096
    loading_family: str = "polynomial"

    def validate(self) -> None:
        if self.loading_family not in LOADING_FAMILIES:
            raise ValueError(f"loading_family must be one of {LOADING_FAMILIES}")
        for name in ("p", "q", "bar_p", "bar_q", "L", "pilot_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.bar_p != self.bar_q:
            raise ValueError("The elementwise mediator-by-covariate interaction requires bar_p == bar_q")
        if self.bar_p > 5:
            raise ValueError("This main-simulation quadrature implementation supports at most 5 latent dimensions")
        if not isinstance(self.r_min, int) or not isinstance(self.r_max, int) or self.r_min > self.r_max:
            raise ValueError("r_min and r_max must be ordered integers")
        for name in ("mu", "sigma_eps_X", "sigma_eps_M", "sigma_y", "sigma_U_scale",
                     "delta1_shift", "delta1_spread", "delta1_contrast", "coef_scale_X", "coef_scale_M"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in ("sigma_eps_X", "sigma_eps_M", "sigma_y", "coef_scale_X", "coef_scale_M"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.sigma_U_scale <= 0:
            raise ValueError("sigma_U_scale must be positive")


@dataclass(frozen=True)
class TruthResult:
    value: float
    sine_term: float
    polynomial_term: float
    softplus_expectation: float
    orders: tuple[int, ...]
    quadrature_values: tuple[float, ...]
    successive_differences: tuple[float, ...]
    atol: float
    rtol: float
    converged: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Mechanism:
    config: DGPConfig
    parameter_seed: int
    arrays: dict[str, np.ndarray] = field(repr=False)
    metadata: dict[str, Any]
    truth: TruthResult | None = None
    schema_version: int = SCHEMA_VERSION

    def serialized_config(self) -> dict[str, Any]:
        config = asdict(self.config)
        if self.schema_version == 1:
            # Schema 1 predates the family field. Keep its exact hashed payload.
            config.pop("loading_family")
        return config

    @property
    def mechanism_hash(self) -> str:
        """Hash the complete numerical mechanism and cached truth, not file bytes."""
        h = sha256()
        payload = {"schema_version": self.schema_version, "config": self.serialized_config(),
                   "parameter_seed": self.parameter_seed, "metadata": self.metadata,
                   "truth": None if self.truth is None else self.truth.to_dict()}
        h.update(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        for name in sorted(self.arrays):
            array = np.ascontiguousarray(self.arrays[name])
            h.update(name.encode())
            h.update(array.dtype.str.encode())
            h.update(json.dumps(array.shape).encode())
            h.update(array.tobytes())
        return h.hexdigest()


def _seed(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _stream_seeds(seed: int, domain: int, count: int) -> list[int]:
    return [int(s.generate_state(1, dtype=np.uint64)[0])
            for s in np.random.SeedSequence([seed, domain]).spawn(count)]


def _sample_latent(params: Mechanism, n: int, seed: int) -> dict[str, np.ndarray]:
    """Separate streams make latent draws stable under measurement dimensions."""
    c, a = params.config, params.arrays
    rng = [np.random.default_rng(s) for s in _stream_seeds(seed, 0x4C415445, 6)]
    f_X = rng[0].uniform(-1, 1, size=(n, c.bar_p)) + c.mu
    e = expit(f_X @ a["alpha"])
    A = (rng[1].random(n) < e).astype(np.int64)
    mean_0 = (f_X ** 2) @ a["delta_0"].T
    mean_1 = (f_X ** 2) @ a["delta_1"].T
    chol = np.linalg.cholesky(a["Sigma_U"])
    eps_M0 = rng[2].standard_normal((n, c.bar_q)) @ chol.T
    eps_M1 = rng[3].standard_normal((n, c.bar_q)) @ chol.T
    f_M = np.where(A[:, None] == 1, mean_1 + eps_M1, mean_0 + eps_M0)
    interaction = f_M * f_X ** 2
    sin_X = np.sin(5 * f_X)
    mu0 = sin_X @ a["beta_0"] + np.logaddexp(0, f_X @ a["kappa_0"]) + interaction @ a["gamma_0"]
    mu1 = sin_X @ a["beta_1"] + np.logaddexp(0, f_X @ a["kappa_1"]) + interaction @ a["gamma_1"]
    eps_Y0 = rng[4].normal(0, c.sigma_y, n)
    eps_Y1 = rng[5].normal(0, c.sigma_y, n)
    Y = np.where(A == 1, mu1 + eps_Y1, mu0 + eps_Y0)
    return {"f_X": f_X, "f_M": f_M, "A": A, "Y": Y, "e_true_vals": e,
            "mu0_true_vals": mu0, "mu1_true_vals": mu1,
            **conditional_counterfactual_means(params, f_X)}


def conditional_counterfactual_means(params: Mechanism, f_X: np.ndarray) -> dict[str, np.ndarray]:
    """Oracle-only conditional means, used for truth checks, never fitting."""
    x = np.asarray(f_X, dtype=float)
    if x.ndim != 2 or x.shape[1] != params.config.bar_p or not np.isfinite(x).all():
        raise ValueError("f_X must be a finite n by bar_p matrix")
    a = params.arrays
    x2 = x ** 2
    m0, m1 = x2 @ a["delta_0"].T, x2 @ a["delta_1"].T
    b0 = np.sin(5 * x) @ a["beta_0"] + np.logaddexp(0, x @ a["kappa_0"])
    b1 = np.sin(5 * x) @ a["beta_1"] + np.logaddexp(0, x @ a["kappa_1"])
    return {"mu00_true_vals": b0 + (m0 * x2) @ a["gamma_0"],
            "mu10_true_vals": b1 + (m0 * x2) @ a["gamma_1"],
            "mu11_true_vals": b1 + (m1 * x2) @ a["gamma_1"]}


def _freeze(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {}
    for name, value in arrays.items():
        array = np.array(value, copy=True)
        array.flags.writeable = False
        result[name] = array
    return result


def draw_parameters(config: DGPConfig | dict[str, Any] | None = None,
                    parameter_seed: int = 910000) -> Mechanism:
    """Draw one prespecified mechanism and cache its population truth once."""
    c = DGPConfig() if config is None else (DGPConfig(**config) if isinstance(config, dict) else config)
    if not isinstance(c, DGPConfig):
        raise TypeError("config must be DGPConfig, a dictionary, or None")
    c.validate()
    parameter_seed = _seed(parameter_seed, "parameter_seed")
    structural_seed, pilot_seed, loading_seed = _stream_seeds(parameter_seed, 0x50415241, 3)
    rng = np.random.default_rng(structural_seed)
    d = c.bar_p
    arrays = {"alpha": rng.uniform(0, 2, size=d)}
    delta_0 = rng.uniform(0.5, 1.5, size=(d, d))
    delta_1 = rng.uniform(0.5, 1.5, size=(d, d))
    delta_0 = 1 + c.delta1_spread * (delta_0 - 1)
    delta_1 = 1 + c.delta1_spread * (delta_1 - 1) + c.delta1_shift
    if c.delta1_contrast != 0:
        v = rng.normal(size=(d, d))
        v -= v.mean(axis=1, keepdims=True)
        delta_1 = delta_0 + c.delta1_contrast * v
    arrays.update(delta_0=delta_0, delta_1=delta_1)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    eigenvalues = rng.uniform(1, 2, size=d) * c.sigma_U_scale
    arrays["Sigma_U"] = Q @ np.diag(eigenvalues) @ Q.T
    for name in ("beta_0", "beta_1", "gamma_0", "gamma_1", "kappa_0", "kappa_1"):
        arrays[name] = rng.uniform(0.5, 1.5, size=d)
    metadata = {"loading_family": c.loading_family,
                "structural_seed": structural_seed, "loading_seed": loading_seed,
                "generator": "numpy.PCG64", "numpy_version_at_creation": np.__version__,
                "design_note": "Fixed independent parameter and loading streams; no estimation-data calibration."}
    if c.loading_family == "polynomial":
        loading_X_seed, loading_M_seed = _stream_seeds(loading_seed, 0x504F4C59, 2)
        arrays.update(
            polynomial_X=np.random.default_rng(loading_X_seed).normal(
                0, c.coef_scale_X / np.sqrt(3), size=(c.p, d, 3)),
            polynomial_M=np.random.default_rng(loading_M_seed).normal(
                0, c.coef_scale_M / np.sqrt(3), size=(c.q, d, 3)))
        metadata.update(polynomial_degree=3,
                        polynomial_formula="sum_j sum_{r=1}^3 C[k,j,r-1]*f[j]^r",
                        loading_X_seed=loading_X_seed, loading_M_seed=loading_M_seed,
                        inactive_config_fields=["L", "r_min", "r_max", "pilot_size"])
    else:
        provisional = Mechanism(c, parameter_seed, arrays, {})
        pilot = _sample_latent(provisional, c.pilot_size, pilot_seed)
        low = float(min(pilot["f_X"].min(), pilot["f_M"].min()))
        high = float(max(pilot["f_X"].max(), pilot["f_M"].max()))
        rng = np.random.default_rng(loading_seed)
        r_list = rng.integers(c.r_min, c.r_max + 1, size=c.L, dtype=np.int64)
        s_list = np.array([int(np.round((2.0 ** r) * rng.uniform(low, high) + rng.uniform(-1, 1)))
                           for r in r_list], dtype=np.int64)
        arrays.update(r_list=r_list, s_list=s_list,
                      Lambda_X_coef=rng.normal(0, c.coef_scale_X / np.sqrt(c.L), size=(c.p, d, c.L)),
                      Lambda_M_coef=rng.normal(0, c.coef_scale_M / np.sqrt(c.L), size=(c.q, d, c.L)))
        metadata.update(support_rule=SUPPORT_RULE, pilot_size=c.pilot_size,
                        pilot_seed=pilot_seed, support_bounds=[low, high])
    params = Mechanism(c, parameter_seed, _freeze(arrays), metadata)
    truth = population_truth(params)
    return Mechanism(c, parameter_seed, params.arrays, metadata, truth)


def _softplus_expectation(kappa: np.ndarray, mu: float, order: int,
                          chunk_size: int = 65536) -> float:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes = nodes + mu
    weights = weights / 2
    d, total = len(kappa), order ** len(kappa)
    result = 0.0
    for start in range(0, total, chunk_size):
        indices = np.unravel_index(np.arange(start, min(start + chunk_size, total)), (order,) * d)
        linear = np.zeros(len(indices[0]))
        joint_weight = np.ones(len(indices[0]))
        for j, ix in enumerate(indices):
            linear += kappa[j] * nodes[ix]
            joint_weight *= weights[ix]
        result += float(np.dot(joint_weight, np.logaddexp(0, linear)))
    return result


def population_truth(params: Mechanism, orders: tuple[int, ...] = (8, 12, 16, 24, 32),
                     atol: float = 1e-9, rtol: float = 1e-9) -> TruthResult:
    """Compute the population target without using any estimation observations.

    Two consecutive refinements must meet the tolerance.  Reported differences
    are convergence diagnostics, not a mathematical bound on integration error.
    Raises if the requested sequence does not establish numerical convergence.
    """
    if len(orders) < 3 or any(not isinstance(o, int) or o < 2 for o in orders) or any(
            a >= b for a, b in zip(orders, orders[1:])):
        raise ValueError("orders must contain at least three increasing integers >= 2")
    if not np.isfinite([atol, rtol]).all() or atol < 0 or rtol < 0 or atol + rtol <= 0:
        raise ValueError("quadrature tolerances must be finite, nonnegative, and not both zero")
    c, a = params.config, params.arrays
    c.validate()
    mu = c.mu
    m2 = mu ** 2 + 1 / 3
    m4 = mu ** 4 + 2 * mu ** 2 + 1 / 5
    cross = np.full((c.bar_p, c.bar_p), m2 ** 2)
    np.fill_diagonal(cross, m4)
    poly = float(np.sum(a["gamma_1"][:, None] * a["delta_0"] * cross))
    sine = float(np.sum(a["beta_1"]) * np.sin(5 * mu) * np.sin(5) / 5)
    values, differences, used, consecutive = [], [], [], 0
    for order in orders:
        current = _softplus_expectation(a["kappa_1"], mu, order)
        values.append(current)
        used.append(order)
        if len(values) > 1:
            difference = abs(current - values[-2])
            differences.append(difference)
            consecutive = consecutive + 1 if difference <= atol + rtol * abs(current) else 0
        if consecutive >= 2:
            return TruthResult(sine + poly + current, sine, poly, current, tuple(used),
                               tuple(values), tuple(differences), atol, rtol, True)
    raise RuntimeError(f"Population truth quadrature did not converge: orders={used}, differences={differences}")


def _loading(factors: np.ndarray, coefficients: np.ndarray,
             r_list: np.ndarray, s_list: np.ndarray) -> np.ndarray:
    n, d = factors.shape
    result = np.zeros((n, coefficients.shape[0]))
    scales = np.exp2(r_list.astype(float))
    amplitudes = np.sqrt(scales)
    for j in range(d):
        t = factors[:, j, None] * scales - s_list
        basis = ((t >= 0) & (t < 1)).astype(float) - ((t >= 1) & (t < 2)).astype(float)
        result += (basis * amplitudes) @ coefficients[:, j, :].T
    return result


def _polynomial_loading(factors: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Additive cubic map: sum_j sum_{r=1}^3 C[k,j,r-1] * factors[j]**r."""
    if coefficients.ndim != 3 or coefficients.shape[1:] != (factors.shape[1], 3):
        raise ValueError("Cubic coefficients must have shape (observed_dim, latent_dim, 3)")
    result = np.zeros((len(factors), coefficients.shape[0]))
    for j in range(factors.shape[1]):
        basis = np.column_stack([factors[:, j] ** power for power in (1, 2, 3)])
        result += basis @ coefficients[:, j, :].T
    return result


def sample_data(params: Mechanism, n: int, data_seed: int) -> dict[str, Any]:
    """Sample one dataset; oracle information is deliberately nested separately.

    No quadrature is performed here.  The caller must use draw_parameters or
    load_mechanism so the population truth has already been cached.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    data_seed = _seed(data_seed, "data_seed")
    if params.truth is None or not params.truth.converged:
        raise ValueError("Mechanism requires a cached, converged population truth")
    c, a = params.config, params.arrays
    latent = _sample_latent(params, int(n), data_seed)
    noise_seeds = _stream_seeds(data_seed, 0x4D454153, 2)
    if c.loading_family == "polynomial":
        X = _polynomial_loading(latent["f_X"], a["polynomial_X"])
        M = _polynomial_loading(latent["f_M"], a["polynomial_M"])
    elif c.loading_family == "haar":
        X = _loading(latent["f_X"], a["Lambda_X_coef"], a["r_list"], a["s_list"])
        M = _loading(latent["f_M"], a["Lambda_M_coef"], a["r_list"], a["s_list"])
    else:
        raise ValueError(f"Unsupported loading family: {c.loading_family}")
    X += np.random.default_rng(noise_seeds[0]).normal(0, c.sigma_eps_X, (n, c.p))
    M += np.random.default_rng(noise_seeds[1]).normal(0, c.sigma_eps_M, (n, c.q))
    oracle = {k: v for k, v in latent.items() if k not in ("A", "Y")}
    return {"X": X, "M": M, "A": latent["A"], "Y": latent["Y"],
            "theta_population": params.truth.value, "oracle": oracle}


def _artifact_paths(path_prefix: str | Path) -> tuple[Path, Path]:
    prefix = Path(path_prefix)
    if prefix.suffix in (".json", ".npz"):
        prefix = prefix.with_suffix("")
    return Path(str(prefix) + ".json"), Path(str(prefix) + ".npz")


def save_mechanism(params: Mechanism, path_prefix: str | Path) -> dict[str, str]:
    """Write JSON metadata and non-pickle NPZ arrays for use by independent jobs."""
    if params.truth is None or not params.truth.converged:
        raise ValueError("Only a mechanism with converged cached truth can be saved")
    json_path, npz_path = _artifact_paths(path_prefix)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **params.arrays)
    payload = {"schema_version": params.schema_version, "config": params.serialized_config(),
               "parameter_seed": params.parameter_seed, "metadata": params.metadata,
               "truth": params.truth.to_dict(), "mechanism_hash": params.mechanism_hash,
               "array_names": sorted(params.arrays), "arrays_file": npz_path.name}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {"json": str(json_path), "npz": str(npz_path), "mechanism_hash": params.mechanism_hash}


def load_mechanism(path_prefix: str | Path) -> Mechanism:
    """Load and verify artifact content, without recalculating quadrature."""
    json_path, npz_path = _artifact_paths(path_prefix)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    schema_version = payload.get("schema_version")
    if schema_version not in (1, SCHEMA_VERSION):
        raise ValueError("Unsupported mechanism artifact schema")
    if payload.get("arrays_file") != npz_path.name:
        raise ValueError("Mechanism JSON/NPZ file names do not match")
    config_data = dict(payload["config"])
    if schema_version == 1:
        if "loading_family" in config_data:
            raise ValueError("Schema-1 artifacts cannot contain a loading_family field")
        legacy_family = payload["metadata"].get("exploration_measurement", "wavelet")
        families = {"wavelet": "haar", "polynomial_degree3": "polynomial"}
        if legacy_family not in families:
            raise ValueError(f"Unsupported legacy measurement family: {legacy_family}")
        config_data["loading_family"] = families[legacy_family]
    elif "loading_family" not in config_data:
        raise ValueError("Schema-2 artifacts must specify loading_family")
    config = DGPConfig(**config_data)
    config.validate()
    with np.load(npz_path, allow_pickle=False) as archive:
        if sorted(archive.files) != payload["array_names"]:
            raise ValueError("Mechanism array inventory mismatch")
        arrays = _freeze({name: archive[name] for name in archive.files})
    truth_data = payload["truth"]
    for name in ("orders", "quadrature_values", "successive_differences"):
        truth_data[name] = tuple(truth_data[name])
    truth = TruthResult(**truth_data)
    if not truth.converged or not np.isfinite(truth.value):
        raise ValueError("Mechanism truth must be converged and finite")
    params = Mechanism(config, _seed(payload["parameter_seed"], "parameter_seed"), arrays,
                       payload["metadata"], truth, schema_version)
    if params.mechanism_hash != payload["mechanism_hash"]:
        raise ValueError("Mechanism content hash mismatch")
    return params
