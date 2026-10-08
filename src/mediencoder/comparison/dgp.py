"""Fixed polynomial comparison design from Supplementary Section S7.3.

The main interaction DGP is deliberately separate. Here treatment enters additively,
the outcome is linear in latent mediators, and all natural effects have exact
population truths. All structural and measurement coefficients are fixed once.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit


@dataclass(frozen=True)
class ComparisonConfig:
    p: int = 500
    q: int = 500
    bar_p: int = 3
    bar_q: int = 3
    degree: int = 3
    mu: float = 0.0
    sigma_X: float = 1.0
    sigma_M: float = 1.0
    sigma_Y: float = 1.0
    sigma_U_scale: float = 1.0
    effect_scale: float = 0.5

    def validate(self):
        for name in ("p", "q", "bar_p", "bar_q", "degree"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite([self.mu, self.sigma_X, self.sigma_M, self.sigma_Y, self.sigma_U_scale, self.effect_scale]).all():
            raise ValueError("DGP parameters must be finite")
        if min(self.sigma_X, self.sigma_M, self.sigma_Y, self.effect_scale) < 0 or self.sigma_U_scale <= 0:
            raise ValueError("Noise/effect scales must be nonnegative and sigma_U_scale positive")


@dataclass(frozen=True)
class ComparisonMechanism:
    config: ComparisonConfig
    parameter_seed: int
    arrays: dict

    @property
    def truth(self):
        nie = float(self.arrays["alpha_M"] @ self.arrays["gamma"])
        nde = float(self.arrays["alpha_Y"][0])
        return {"NIE": nie, "NDE": nde, "TE": nie + nde}

    @property
    def mechanism_hash(self):
        h = hashlib.sha256(json.dumps({"config": asdict(self.config), "parameter_seed": self.parameter_seed}, sort_keys=True).encode())
        for name, a in sorted(self.arrays.items()):
            h.update(name.encode())
            h.update(str(a.shape).encode())
            h.update(a.dtype.str.encode())
            h.update(a.tobytes())
        return h.hexdigest()


def _seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(seed)


def _freeze(arrays):
    result = {k: np.array(v, dtype=float, copy=True) for k, v in arrays.items()}
    for a in result.values():
        a.flags.writeable = False
    return result


def draw_mechanism(config=None, parameter_seed=910000):
    c = ComparisonConfig() if config is None else (ComparisonConfig(**config) if isinstance(config, dict) else config)
    c.validate()
    seed = _seed(parameter_seed)
    structural, loading = [np.random.default_rng(s) for s in np.random.SeedSequence([seed, 1001]).spawn(2)]
    Q, _ = np.linalg.qr(structural.normal(size=(c.bar_q, c.bar_q)))
    arrays = {
        "alpha": structural.uniform(0, 2, c.bar_p),
        "delta": structural.uniform(0.5, 1.5, (c.bar_q, c.bar_p)),
        "beta": structural.uniform(0.5, 1.5, c.bar_p),
        "kappa": structural.uniform(0.5, 1.5, c.bar_p),
        "gamma": structural.uniform(0.5, 1.5, c.bar_q),
        "alpha_M": c.effect_scale * structural.uniform(0.5, 1.5, c.bar_q),
        "alpha_Y": c.effect_scale * structural.uniform(0.5, 1.5, 1),
        "Sigma_U": Q @ np.diag(structural.uniform(1, 2, c.bar_q) * c.sigma_U_scale) @ Q.T,
        "loading_X": loading.normal(0, 1 / np.sqrt(c.degree), (c.p, c.bar_p, c.degree)),
        "loading_M": loading.normal(0, 1 / np.sqrt(c.degree), (c.q, c.bar_q, c.degree)),
    }
    return ComparisonMechanism(c, seed, _freeze(arrays))


def _load(f, coef):
    result = np.zeros((len(f), coef.shape[0]))
    for j in range(f.shape[1]):
        basis = np.column_stack([f[:, j] ** power for power in range(1, coef.shape[2] + 1)])
        result += basis @ coef[:, j, :].T
    return result


def sample_data(mechanism, n, data_seed):
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    c, a = mechanism.config, mechanism.arrays
    rng = [np.random.default_rng(s) for s in np.random.SeedSequence([_seed(data_seed), 2002]).spawn(6)]
    fX = rng[0].uniform(-1, 1, (n, c.bar_p)) + c.mu
    propensity = expit(fX @ a["alpha"])
    A = (rng[1].random(n) < propensity).astype(int)
    mean0 = fX ** 2 @ a["delta"].T
    u = rng[2].standard_normal((n, c.bar_q)) @ np.linalg.cholesky(a["Sigma_U"]).T
    fM = mean0 + A[:, None] * a["alpha_M"] + u
    baseline = np.sin(5 * fX) @ a["beta"] + np.logaddexp(0, fX @ a["kappa"])
    Y = baseline + fM @ a["gamma"] + A * a["alpha_Y"][0] + rng[3].normal(0, c.sigma_Y, n)
    X = _load(fX, a["loading_X"]) + rng[4].normal(0, c.sigma_X, (n, c.p))
    M = _load(fM, a["loading_M"]) + rng[5].normal(0, c.sigma_M, (n, c.q))
    mu00 = baseline + mean0 @ a["gamma"]
    mu10 = mu00 + a["alpha_Y"][0]
    mu11 = mu10 + a["alpha_M"] @ a["gamma"]
    return {"X": X, "M": M, "A": A, "Y": Y, "truth": mechanism.truth,
            "oracle": {"f_X": fX, "f_M": fM, "e": propensity, "mu00": mu00, "mu10": mu10, "mu11": mu11}}


def save_mechanism(mechanism, prefix):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(prefix) + ".npz", **mechanism.arrays)
    payload = {"schema_version": 1, "config": asdict(mechanism.config), "parameter_seed": mechanism.parameter_seed,
               "truth": mechanism.truth, "mechanism_hash": mechanism.mechanism_hash,
               "design": "fixed-parameter additive treatment; linear latent mediator outcome; polynomial measurements"}
    Path(str(prefix) + ".json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_mechanism(prefix):
    prefix = str(prefix)
    payload = json.loads(Path(prefix + ".json").read_text(encoding="utf-8"))
    if payload["schema_version"] != 1:
        raise ValueError("Unsupported comparison mechanism schema")
    c = ComparisonConfig(**payload["config"])
    c.validate()
    with np.load(prefix + ".npz", allow_pickle=False) as f:
        result = ComparisonMechanism(c, _seed(payload["parameter_seed"]), _freeze({k: f[k] for k in f.files}))
    if result.mechanism_hash != payload["mechanism_hash"] or result.truth != payload["truth"]:
        raise ValueError("Comparison mechanism integrity check failed")
    return result
