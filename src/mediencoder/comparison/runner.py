"""Reproducible serial comparison CLI; one dataset is paired across all methods."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from .dgp import ComparisonConfig, draw_mechanism, load_mechanism, sample_data, save_mechanism
from .reporting import summarize, markdown_table, validate_result


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def dataset_hash(data):
    h = hashlib.sha256()
    for key in ("X", "M", "A", "Y"):
        arr = np.ascontiguousarray(data[key])
        h.update(key.encode())
        h.update(arr.dtype.str.encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def method_seed(data_seed, method):
    marker = int.from_bytes(hashlib.sha256(method.encode()).digest()[:4], "little")
    return int(np.random.SeedSequence([data_seed, marker]).generate_state(1)[0] % (2 ** 31 - 1))


def scientific_fingerprints(manifest, method):
    """Exclude shard bounds, retain implementation, runtime and method settings."""
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    common = {key: manifest[key] for key in (
        "mode", "config", "mechanism_hash", "device", "threads", "python", "numpy", "torch", "source_sha256")}
    common["packages"] = manifest.get("packages", {})
    study = digest(common)
    estimator = digest({"study": study, "method": method, "options": manifest["options"][method],
                        "external_adapter": manifest.get("external_adapters", {}).get(method),
                        "external_source": manifest.get("external_source_sha256", {}).get(method)})
    return study, estimator


def run_replicate(mechanism, n, replicate, data_seed, methods, options=None, external=None,
                  fit=None, score_dir=None):
    """Every requested method produces a success/failure record; no retries."""
    from .adapters import fit_method, inference_available
    fit = fit or fit_method
    options, external = options or {}, external or {}
    data = sample_data(mechanism, n, data_seed)
    paired_hash = dataset_hash(data)
    rows = []
    for method in methods:
        seed = method_seed(data_seed, method)
        row = {"method": method, "n": n, "replicate": replicate, "data_seed": data_seed,
               "training_seed": seed, "mechanism_hash": mechanism.mechanism_hash,
               "dataset_hash": paired_hash, "truth": mechanism.truth,
               "inference_available": inference_available(method), "status": "failed"}
        start = time.perf_counter()
        try:
            result = fit(method, *(data[k].copy() for k in ("X", "M", "A", "Y")), seed=seed,
                         options=options.get(method, {}), external_adapter=external.get(method))
            validate_result(result)
            if score_dir is not None and "effect_scores" in result:
                scores = result.pop("effect_scores")
                for effect, score in scores.items():
                    if np.asarray(score).shape != (n,) or not np.isfinite(score).all():
                        raise ValueError(f"Invalid {effect} score coverage; require exactly n finite scores")
                Path(score_dir).mkdir(parents=True, exist_ok=True)
                score_path = Path(score_dir) / f"n{n}-rep{replicate}-{method}.npz"
                np.savez_compressed(score_path, **scores)
                result["score_file"] = str(score_path.name)
            else:
                result.pop("effect_scores", None)
            row.update(_jsonable(result), status="ok", error="")
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["elapsed_seconds"] = time.perf_counter() - start
        rows.append(row)
    return rows


def merge_records(paths):
    """Merge explicit JSONL shards with identical scientific/runtime provenance."""
    rows = [json.loads(line) for path in paths for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    summarize(rows)  # validates duplicate keys and nonfinite successful results
    paired = {}
    studies, estimators = {}, {}
    for row in rows:
        if not row.get("study_fingerprint") or not row.get("estimator_fingerprint"):
            raise ValueError("Shard lacks scientific fingerprints; historical/handmade rows cannot be pooled automatically")
        study_key = row["mechanism_hash"], row["n"]
        estimator_key = study_key + (row["method"],)
        for identities, key, fingerprint in (
                (studies, study_key, row["study_fingerprint"]),
                (estimators, estimator_key, row["estimator_fingerprint"])):
            if key in identities and identities[key] != fingerprint:
                raise ValueError("Attempted merge with different scientific configuration, source or runtime fingerprints")
            identities[key] = fingerprint
        key = row["mechanism_hash"], row["n"], row["replicate"]
        identity = row["data_seed"], row["dataset_hash"]
        if key in paired and paired[key] != identity:
            raise ValueError("Attempted merge of unpaired method datasets")
        paired[key] = identity
    return rows


def _write_reports(output, rows):
    summary = summarize(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "table.md").write_text(markdown_table(summary), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-grid", default="500,800,1500,2000")
    parser.add_argument("--replications", type=int, default=200)
    parser.add_argument("--parameter-seed", type=int, default=910000)
    parser.add_argument("--data-seed", type=int, default=2000000)
    parser.add_argument("--p", type=int, default=500)
    parser.add_argument("--q", type=int, default=500)
    parser.add_argument("--latent-dim", type=int, default=3)
    parser.add_argument("--tilde", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--methods", default="mediencoder,nath-adapted,dp2lm-adapted,imavae-adapted")
    parser.add_argument("--options", type=Path, help="JSON object keyed by method; all options saved in manifest")
    parser.add_argument("--external-adapter", action="append", default=[], metavar="NAME=MODULE:FUNCTION")
    parser.add_argument("--mechanism", type=Path, help="Load an existing fixed comparison mechanism prefix")
    parser.add_argument("--smoke", action="store_true", help="Tiny CPU integration run; never publication results")
    args = parser.parse_args(argv)
    if args.replications < 1 or args.threads < 1 or args.data_seed < 0:
        parser.error("replications/threads must be positive and data-seed nonnegative")
    os.environ["MEDIENC_DEVICE"] = "cpu" if args.smoke else args.device
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    import torch
    torch.set_num_threads(args.threads)
    from .adapters import METHODS
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    external = dict(item.split("=", 1) for item in args.external_adapter)
    if len(methods) != len(set(methods)) or not methods or any(m not in METHODS and m not in external for m in methods):
        parser.error("Choose unique built-in methods or explicitly supplied external adapters")
    if any(not name.replace("-", "").replace("_", "").isalnum() for name in methods):
        parser.error("Method names must be alphanumeric with optional hyphens/underscores")
    n_grid = [int(n) for n in args.n_grid.split(",")]
    cfg = ComparisonConfig(p=args.p, q=args.q, bar_p=args.latent_dim, bar_q=args.latent_dim)
    options = json.loads(args.options.read_text(encoding="utf-8")) if args.options else {}
    if args.smoke:
        cfg = ComparisonConfig(p=8, q=6, bar_p=2, bar_q=2)
        n_grid, args.replications, args.tilde, args.epochs = [160], 1, 2, 1
    if len(set(n_grid)) != len(n_grid) or min(n_grid) < 12:
        parser.error("n-grid must contain unique sample sizes >=12")
    effective = {}
    for method in methods:
        effective[method] = {"epochs": args.epochs, "tilde_p": args.tilde, "tilde_q": args.tilde, **options.get(method, {})}
        if method in ("mediencoder", "projection", "autoencoder", "vae"):
            effective[method].setdefault("numerical_safeguards", {"clip_eps": 0.01, "pi2_soft": 0.0, "pi2_cap": 0.0})
        if args.smoke:
            effective[method].update(hidden_dims=[4], patience=1)
            if method == "mediencoder":
                effective[method].update(lambda_grid=[[0.3, 0.3, 0.4]], hidden_dims_XM=[4])
            elif method == "nath-adapted":
                effective[method]["iterations"] = 1
            elif method == "dp2lm-adapted":
                effective[method]["lam_grid"] = [0.0]
            elif method == "imavae-adapted":
                effective[method]["n_mc"] = 2
    mechanism = load_mechanism(args.mechanism) if args.mechanism else draw_mechanism(cfg, args.parameter_seed)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty; use a new path to preserve existing results")
    args.output.mkdir(parents=True, exist_ok=True)
    save_mechanism(mechanism, args.output / "mechanism")
    package_root = Path(__file__).parents[1]
    sources = sorted(package_root.rglob("*.py"))
    external_sources = {}
    for name, target in external.items():
        specification = importlib.util.find_spec(target.split(":", 1)[0])
        origin = specification.origin if specification is not None else None
        if origin is None or not Path(origin).is_file():
            parser.error(f"External adapter {name} requires an inspectable installed source module")
        external_sources[name] = {"file": Path(origin).name, "sha256": hashlib.sha256(Path(origin).read_bytes()).hexdigest()}
    manifest = {"mode": "smoke" if args.smoke else "pilot", "publication_ready": False,
                "config": asdict(mechanism.config),
                "mechanism_hash": mechanism.mechanism_hash, "truth": mechanism.truth, "n_grid": n_grid,
                "replications_requested": args.replications, "data_seed_start": args.data_seed,
                "methods": methods, "options": effective, "external_adapters": external,
                "device": os.environ["MEDIENC_DEVICE"], "threads": args.threads,
                "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                "packages": {name: importlib.metadata.version(name) for name in ("scipy", "scikit-learn", "pandas")},
                "source_sha256": {str(p.relative_to(package_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                "external_source_sha256": external_sources}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    all_rows = []
    for n in n_grid:
        for replicate in range(args.replications):
            rows = run_replicate(mechanism, n, replicate, args.data_seed + replicate, methods,
                                 effective, external, score_dir=args.output / "scores")
            for row in rows:
                row["B_requested"] = args.replications
                row["study_fingerprint"], row["estimator_fingerprint"] = scientific_fingerprints(manifest, row["method"])
            with (args.output / "replications.jsonl").open("a", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            all_rows.extend(rows)
            _write_reports(args.output, all_rows)
            print(f"n={n} replicate={replicate + 1}/{args.replications}: " + ", ".join(f"{r['method']}={r['status']}" for r in rows), flush=True)
    return 1 if any(row["status"] != "ok" for row in all_rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
