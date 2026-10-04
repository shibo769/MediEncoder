"""Canonical real-data CLI; all inputs and outputs remain local."""
import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np

from .io import load_npz, load_rds


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"filename": Path(path).name, "sha256": digest.hexdigest()}


def _code_fingerprints():
    package = Path(__file__).resolve().parents[1]
    return {path.relative_to(package).as_posix(): _fingerprint(path)["sha256"]
            for path in sorted(package.rglob("*.py"))}


def _package_versions():
    versions = {}
    for name in ("mediencoder", "numpy", "scipy", "scikit-learn", "pandas", "torch", "pyreadr", "filelock", "tqdm"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def _resolved_device_metadata(requested):
    import torch
    from mediencoder.nn_utils import device

    if requested != "auto" and device.type != requested:
        raise RuntimeError(f"Requested device {requested!r}, but trainers already use {str(device)!r}; restart the process with the requested device")
    metadata = {"resolved_device": str(device)}
    if device.type == "cuda":
        metadata.update(gpu_name=torch.cuda.get_device_name(device),
                        gpu_compute_capability=list(torch.cuda.get_device_capability(device)))
    return metadata


def parser():
    result = argparse.ArgumentParser(description="Cross-fitted real-data NIE/NDE/TE with score-based confidence intervals")
    result.add_argument("--input", type=Path, help="NPZ with exactly X,M,A,Y")
    result.add_argument("--covariates-rds", type=Path)
    result.add_argument("--mediators-rds", type=Path)
    result.add_argument("--outcome-rds", type=Path)
    result.add_argument("--treatment-rds", type=Path)
    result.add_argument("--treatment-column", default="GDTOTAL")
    result.add_argument("--threshold", type=float, default=6.)
    result.add_argument("--assume-row-aligned", action="store_true")
    result.add_argument("--tilde-p", type=int, required=True)
    result.add_argument("--tilde-q", type=int, required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--method", choices=["mediencoder", "autoencoder", "vae", "projection"], default="mediencoder")
    result.add_argument("--preprocessing", choices=["standardize", "none"], default="standardize")
    result.add_argument("--grid-step", type=float, default=.1)
    result.add_argument("--clip-eps", type=float, default=.01, help="Explicit probability clipping floor")
    result.add_argument("--pi2-soft", type=float, default=0., help="Optional smooth density-ratio cap; zero disables")
    result.add_argument("--pi2-cap", type=float, default=0., help="Optional hard ratio cap (>1) or fold quantile (0,1]; zero disables")
    result.add_argument("--config", type=Path, help="JSON with nn_cfg, ae_cfg, me_cfg, encode_cfg overrides")
    result.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    result.add_argument("--output", type=Path, required=True, help="New, empty local output directory")
    return result


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    rds_paths = [args.covariates_rds, args.mediators_rds, args.outcome_rds, args.treatment_rds]
    if bool(args.input) == any(p is not None for p in rds_paths) or (not args.input and not all(rds_paths)):
        argument_parser.error("Supply either --input NPZ or all four RDS paths")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        argument_parser.error("Output directory must be empty; existing analysis is not overwritten")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["MEDIENC_DEVICE"] = args.device
    safeguards = {"clip_eps": args.clip_eps, "pi2_soft": args.pi2_soft, "pi2_cap": args.pi2_cap}
    treatment_definition = ({"source": "supplied_binary_A", "values": [0, 1]}
                            if args.input else
                            {"source": "RDS_column", "column": args.treatment_column,
                             "comparison": ">=", "threshold": args.threshold,
                             "assume_row_aligned": args.assume_row_aligned})
    manifest = {"status": "started", "seed": args.seed, "method": args.method,
                "tilde_p": args.tilde_p, "tilde_q": args.tilde_q,
                "preprocessing": args.preprocessing, "device": args.device,
                "python": platform.python_version(), "inference": "matched cross-fitted score covariance / n",
                "bootstrap": False, "across_seed_SE": False,
                "code_sha256": _code_fingerprints(), "outcome_scaling": "none",
                "package_versions": _package_versions(), "treatment_definition": treatment_definition,
                "numerical_safeguards": safeguards}
    _write_json(args.output / "manifest.json", manifest)
    try:
        from . import analyze
        from mediencoder.estimation import _shared_cfg, SHARED_ADAM_EPS, _resolve_numerical_safeguards
        from mediencoder.training import generate_lambda_grid
        import torch

        torch.set_num_threads(1)
        manifest.update(_resolved_device_metadata(args.device))
        safeguards = _resolve_numerical_safeguards(safeguards)
        if args.input:
            data = load_npz(args.input)
            input_paths = [args.input]
        else:
            data = load_rds(covariates=args.covariates_rds, mediators=args.mediators_rds,
                            outcome=args.outcome_rds, treatment_table=args.treatment_rds,
                            treatment_column=args.treatment_column, threshold=args.threshold,
                            assume_row_aligned=args.assume_row_aligned)
            input_paths = rds_paths
        manifest.update(inputs=[_fingerprint(p) for p in input_paths], n=len(data["Y"]),
                        p=data["X"].shape[1], q=data["M"].shape[1], torch=torch.__version__,
                        torch_cuda=torch.version.cuda, torch_threads=torch.get_num_threads())
        config = {} if args.config is None else json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or set(config) - {"nn_cfg", "ae_cfg", "me_cfg", "encode_cfg"}:
            raise ValueError("Config accepts only nn_cfg, ae_cfg, me_cfg, encode_cfg")
        config.setdefault("nn_cfg", _shared_cfg(eps=SHARED_ADAM_EPS, hidden_dims=(300, 300, 300)))
        grid = None
        if args.method == "mediencoder":
            grid = [v for v in generate_lambda_grid(step=args.grid_step, require_order=False) if v[2] > 1e-6]
            if not grid:
                raise ValueError("Tuning grid has no positive-alignment candidates")
        manifest.update(config=config, lambda_grid=grid)
        _write_json(args.output / "manifest.json", manifest)
        result = analyze(**data, tilde_p=args.tilde_p, tilde_q=args.tilde_q,
                         seed=args.seed, factor_method=args.method,
                         preprocessing=args.preprocessing, lambda_grid=grid,
                         numerical_safeguards=safeguards, **config)
        arrays = {**result["component_scores"], **result["effect_scores"]}
        arrays["subject_index"] = np.arange(len(data["Y"]))
        for k, fold in enumerate(result["fold_indices"]):
            for role, indices in fold.items():
                arrays[f"fold_{k+1}_{role}"] = indices
        # Per-subject scores are sensitive analysis output; save locally, not X/M/A/Y.
        np.savez_compressed(args.output / "scores.npz", **arrays)
        summary = {key: value for key, value in result.items()
                   if key not in {"component_scores", "effect_scores", "crossfit_scores", "fold_indices"}}
        _write_json(args.output / "summary.json", summary)
        manifest["status"] = "complete"
        _write_json(args.output / "manifest.json", manifest)
        for effect in result["effect_order"]:
            lo, hi = result["effect_ci"][effect]
            print(f"{effect}: {result['effects'][effect]:.6g}; SE {result['effect_se'][effect]:.6g}; 95% CI [{lo:.6g}, {hi:.6g}]")
        return 0
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                        tuning_rows=getattr(exc, "tuning_rows", None))
        _write_json(args.output / "manifest.json", manifest)
        print(f"Analysis failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
