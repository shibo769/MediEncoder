"""Small actual neural fits across shared estimation and the observed-data CLI."""
import json

import numpy as np
import torch

from mediencoder import estimation
from mediencoder.real_data import analyze
from mediencoder.real_data.cli import main as real_data_main


def test_actual_effect_pipeline_preserves_theta10_and_covariance(monkeypatch, tmp_path):
    torch.set_num_threads(1)
    monkeypatch.setitem(estimation.SHARED_TRAIN_CFG, "epochs", 1)
    monkeypatch.setattr(estimation, "SHARED_HIDDEN", (4,))
    rng = np.random.default_rng(112)
    n = 160
    X, M = rng.normal(size=(n, 8)), rng.normal(size=(n, 6))
    A = (np.arange(n) % 2).astype(float)
    Y = X[:, 0] + M[:, 0] + A + rng.normal(size=n)
    data = dict(X=X, M=M, A=A, Y=Y)
    common = dict(tilde_p=2, tilde_q=2, seed=13,
                  factor_method="mediencoder", lambda_grid=[(.3, .3, .4)],
                  preprocessing="standardize",
                  nn_cfg=estimation._shared_cfg(hidden_dims=(4,), eps=1e-8),
                  numerical_safeguards={})
    base = estimation.estimate_triply_IF(**data, **common)
    effects = estimation.estimate_triply_IF(**data, **common, return_effects=True)
    np.testing.assert_array_equal(base["crossfit_scores"], effects["component_scores"]["theta10"])
    for key in ("theta_hat_IF", "se_IF", "ci_lower", "ci_upper"):
        assert base[key] == effects[key]
    for name, scores in effects["effect_scores"].items():
        assert scores.shape == (n,) and np.isfinite(scores).all()
        np.testing.assert_allclose(effects["effect_se"][name], scores.std(ddof=1) / np.sqrt(n))
    np.testing.assert_allclose(effects["effects"]["TE"], effects["effects"]["NIE"] + effects["effects"]["NDE"])

    # Execute the real CLI through input loading, fitting and saved-score output.
    source, output = tmp_path / "observed.npz", tmp_path / "analysis"
    np.savez_compressed(source, **data)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"nn_cfg": estimation._shared_cfg(hidden_dims=(4,), eps=1e-8)}))
    code = real_data_main(["--input", str(source), "--output", str(output),
                           "--method", "projection", "--tilde-p", "2", "--tilde-q", "2",
                           "--seed", "13", "--device", "cpu", "--config", str(config)])
    assert code == 0
    summary = json.loads((output / "summary.json").read_text())
    with np.load(output / "scores.npz", allow_pickle=False) as saved:
        for name in ("NIE", "NDE", "TE"):
            np.testing.assert_allclose(summary["effects"][name], saved[name].mean())
            np.testing.assert_allclose(summary["effect_se"][name], saved[name].std(ddof=1) / np.sqrt(n))
