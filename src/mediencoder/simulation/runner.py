#!/usr/bin/env python3
r"""Restartable, paired main-table and alignment-ablation experiment.

Examples (PowerShell):
  python -m mediencoder.simulation.runner --output-dir results/formal --workers 1 --device cuda
  python -m mediencoder.simulation.runner --output-dir results/pilot --workers 1 --device cpu --n 100 --reps 1 --pilot-epochs 2

Re-running the same command resumes completed checkpoints. Scientific settings and
source hashes must match; worker count may change. Pilot results are explicitly
marked and must never be merged into formal results. No manuscript is edited.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("projection", "autoencoder", "vae", "mediencoder", "mediencoder_l3zero")
MAIN_METHODS = METHODS[:4]
LABELS = dict(zip(METHODS, ("Projection", "Autoencoder", "VAE", "MediEncoder", r"$\lambda_3=0$")))
SOURCE_NAMES = ("simulation/runner.py", "simulation/dgp.py", "estimation.py",
                "training.py", "models.py", "nn_utils.py")
_WORKER = {}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if hasattr(value, "item"):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json(value):
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def configure_environment(device):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["MEDIENC_DEVICE"] = device
    os.environ["MEDIENCODER_QUIET_TQDM"] = "1"
    # Preserve the original numeric safeguards and disable local exploratory caps.
    os.environ["MEDIENC_CLIP_EPS"] = "0.01"
    os.environ["MEDIENC_PI2_CAP"] = "0"
    os.environ["MEDIENC_PI2_SOFT"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def lambda_grids():
    from mediencoder.training import generate_lambda_grid
    full = generate_lambda_grid(C=1.0, step=0.1, require_order=False)
    tune = [list(map(float, t)) for t in full if t[2] > 1e-6]
    zero = [list(map(float, t)) for t in full if t[2] < 1e-6]
    if len(tune) != 36 or len(zero) != 9:
        raise ValueError(f"Unexpected lambda grids: {len(tune)}, {len(zero)}")
    return tune, zero


def training_configuration(epochs):
    from mediencoder.estimation import SHARED_ADAM_EPS, SHARED_HIDDEN, SHARED_TRAIN_CFG, _shared_cfg
    # The existing implementation intentionally forbids per-method overrides.
    # Set the one shared cap before resolving ALL learners, including nuisance fits.
    SHARED_TRAIN_CFG["epochs"] = int(epochs)
    return dict(
        nn_cfg=_shared_cfg(eps=SHARED_ADAM_EPS, hidden_dims=(300, 300, 300)),
        ae_cfg=_shared_cfg(eps=SHARED_ADAM_EPS, hidden_dims_X=SHARED_HIDDEN,
                           hidden_dims_M=SHARED_HIDDEN, beta_kl=0.5),
        me_cfg=_shared_cfg(hidden_dims_X=SHARED_HIDDEN, hidden_dims_M=SHARED_HIDDEN,
                           hidden_dims_XM=(50, 50), adam_eps=SHARED_ADAM_EPS,
                           lambda1=0.2, lambda2=0.5, lambda3=0.3, beta_kl=0.5,
                           return_history=True, allow_unbalanced_lambda=True),
        encode_cfg=dict(batch_size=4096),
    )


def make_config(args):
    from mediencoder.simulation.dgp import DGPConfig
    from dataclasses import asdict
    tune, zero = lambda_grids()
    pilot = args.pilot_epochs is not None
    epochs = args.pilot_epochs if pilot else 300
    return jsonable(dict(
        schema_version=1, run_kind=("PILOT_NOT_FOR_PAPER" if pilot else
                                   "BENCHMARK_NOT_FOR_PAPER" if args.max_tasks is not None else "FORMAL"),
        estimand="population E[mu10(f_X)] conditional on fixed mechanism",
        mechanism_seed=args.mechanism_seed, seed_base=args.seed_base,
        n_values=args.n, B_requested=args.reps, methods=args.arms,
        dgp=asdict(DGPConfig()), tilde_p=10, tilde_q=10,
        training=training_configuration(epochs),
        lambda_grid=tune, lambda_grid_zero=zero,
        stop_gradient=True, loss_normalization="observed representation-training fold only; validation reuses training scales",
        interval="theta_hat +/- 1.959963984540054 * sd(crossfit_scores, ddof=1)/sqrt(n)",
        numeric_safeguards=dict(propensity_clip_eps=0.01, density_ratio_cap=0.0, density_ratio_soft_cap=0.0),
        device=args.device, torch_threads=1, deterministic_algorithms=True,
    ))


def task_seeds(seed_base, n, rep):
    import numpy as np
    state = np.random.SeedSequence([int(seed_base), int(n), int(rep)]).generate_state(2)
    # Existing lambda-derived offsets stay safely below NumPy's uint32 seed limit.
    return int(state[0]), int(state[1] % 1_000_000_000)


def runtime_training_configuration(training):
    # JSON stores tuples as lists. Restore trainer sequence types so equality
    # checks do not falsely report that shared hyperparameters were overridden.
    return {name: {key:tuple(value) if isinstance(value, list)
                        and (key == "betas" or key.startswith("hidden_dims")) else value
                   for key, value in settings.items()}
            for name, settings in training.items()}


def build_tasks(config):
    # Finish small paired groups across sample sizes early, instead of one entire arm first.
    for rep in range(config["B_requested"]):
        for n in config["n_values"]:
            data_seed, training_seed = task_seeds(config["seed_base"], n, rep)
            for method in config["methods"]:
                yield dict(task_id=f"n{n:05d}_r{rep:04d}_{method}", n=n, rep=rep,
                           method=method, data_seed=data_seed, training_seed=training_seed)


def execution_config(config, target_reps=None):
    """Select a replication prefix without changing the reserved run identity."""
    reserved = config["B_requested"]
    target = reserved if target_reps is None else target_reps
    if not isinstance(target, int) or not 1 <= target <= reserved:
        raise ValueError("target replications must be between 1 and the reserved --reps")
    return dict(config, B_requested=target, B_reserved=reserved)


def phase_records(records, config):
    # A later request to inspect an earlier phase never deletes later checkpoints.
    return {key: row for key, row in records.items()
            if row.get("rep", 0) < config["B_requested"]}


def collect_code_hashes():
    return {name: file_hash(ROOT / name) for name in SOURCE_NAMES}


def validate_manifest(existing, candidate):
    for field in ("config", "code_hashes", "mechanism_hash", "environment"):
        if canonical_json(existing.get(field)) != canonical_json(candidate.get(field)):
            raise ValueError(f"Cannot resume: {field} changed. Use a new output directory; do not mix runs.")
    if existing["run_hash"] != candidate["run_hash"]:
        raise ValueError("Cannot resume: run fingerprint changed")


def validate_checkpoint(record, manifest, output_dir):
    if record.get("run_hash") != manifest["run_hash"]:
        raise ValueError(f"Checkpoint belongs to a different run: {record.get('task_id')}")
    if record.get("status") == "complete":
        artifact = Path(output_dir) / record["score_artifact"]
        if not artifact.is_file() or file_hash(artifact) != record["score_artifact_sha256"]:
            raise ValueError(f"Missing or damaged score artifact: {artifact}")
    elif record.get("status") != "failed":
        raise ValueError(f"Unsupported checkpoint status: {record.get('status')}")


def _worker_init(config, output_dir, mechanism_prefix, run_hash):
    configure_environment(config["device"])
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    from mediencoder.estimation import SHARED_TRAIN_CFG
    epoch_caps = {config["training"][name]["epochs"] for name in ("nn_cfg", "ae_cfg", "me_cfg")}
    if len(epoch_caps) != 1:
        raise ValueError("All learners must use the same shared epoch cap")
    SHARED_TRAIN_CFG["epochs"] = int(epoch_caps.pop())
    from mediencoder.simulation.dgp import load_mechanism
    _WORKER.update(config=config, output_dir=Path(output_dir),
                   mechanism=load_mechanism(mechanism_prefix), run_hash=run_hash,
                   training=runtime_training_configuration(config["training"]))


def observed_data_hash(data):
    import numpy as np
    h = hashlib.sha256()
    for name in ("X", "M", "A", "Y"):
        a = np.ascontiguousarray(data[name])
        h.update(name.encode()); h.update(str(a.shape).encode()); h.update(a.dtype.str.encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _run_task(task):
    import numpy as np
    from mediencoder.simulation.dgp import sample_data
    from mediencoder.estimation import estimate_triply_IF, set_all_seeds
    config = _WORKER["config"]
    output_dir = _WORKER["output_dir"]
    started = time.monotonic()
    record = dict(task, run_hash=_WORKER["run_hash"], started_at=utc_now(),
                  mechanism_hash=_WORKER["mechanism"].mechanism_hash,
                  theta_population=float(_WORKER["mechanism"].truth.value), worker_pid=os.getpid())
    running_path = output_dir / "running" / (task["task_id"] + ".json")
    atomic_json(running_path, record)
    print(f"[{record['started_at']}] START {task['task_id']} worker={os.getpid()}", flush=True)
    try:
        data = sample_data(_WORKER["mechanism"], task["n"], task["data_seed"])
        record["observed_data_sha256"] = observed_data_hash(data)
        truth = float(_WORKER["mechanism"].truth.value)
        # True factors/nuisances are intentionally unavailable to the estimator call.
        data.pop("oracle", None)
        method = "mediencoder" if task["method"] == "mediencoder_l3zero" else task["method"]
        grid = None
        if method == "mediencoder":
            grid = config["lambda_grid_zero"] if task["method"].endswith("l3zero") else config["lambda_grid"]
        set_all_seeds(task["training_seed"])
        result = estimate_triply_IF(data["X"], data["M"], data["A"], data["Y"],
                                   tilde_p=config["tilde_p"], tilde_q=config["tilde_q"],
                                   factor_method=method, seed=task["training_seed"],
                                   lambda_grid=grid, **_WORKER["training"])
        scores = np.asarray(result["crossfit_scores"], dtype=np.float64).reshape(-1)
        if len(scores) != task["n"] or not np.isfinite(scores).all():
            raise ValueError("Every input subject must have exactly one finite cross-fitted score")
        theta = float(result["theta_hat_IF"])
        se = float(result["se_IF"])
        lower, upper = float(result["ci_lower"]), float(result["ci_upper"])
        if not all(map(math.isfinite, (truth, theta, se, lower, upper))) or se < 0:
            raise ValueError("Nonfinite point estimate/truth/interval or negative standard error")
        expected_se = float(scores.std(ddof=1) / math.sqrt(len(scores)))
        if not math.isclose(theta, float(scores.mean()), rel_tol=1e-9, abs_tol=1e-10):
            raise ValueError("Point estimate is not the mean of saved scores")
        if not math.isclose(se, expected_se, rel_tol=1e-9, abs_tol=1e-10):
            raise ValueError("Standard error differs from the saved per-subject scores")
        if not (math.isclose(lower, theta - 1.959963984540054 * se, rel_tol=1e-9, abs_tol=1e-10)
                and math.isclose(upper, theta + 1.959963984540054 * se, rel_tol=1e-9, abs_tol=1e-10)):
            raise ValueError("Saved confidence interval does not match per-dataset standard error")
        arrays = {"crossfit_scores": scores, "subject_index": np.arange(task["n"])}
        for fold, roles in enumerate(result["fold_indices"]):
            for role, indices in roles.items():
                arrays[f"fold{fold}_{role}"] = np.asarray(indices, dtype=np.int64)
        artifact = output_dir / "scores" / (task["task_id"] + ".npz")
        artifact.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact.with_name(artifact.name + f".{os.getpid()}.tmp")
        with temporary.open("wb") as f:
            np.savez_compressed(f, **arrays)
            f.flush(); os.fsync(f.fileno())
        os.replace(temporary, artifact)
        record.update(status="complete", theta_hat=theta, theta_population=truth,
                      error=theta - truth, se_IF=se, ci_lower=lower, ci_upper=upper,
                      ci_length=upper-lower, covered=bool(lower <= truth <= upper),
                      score_artifact=str(artifact.relative_to(output_dir)),
                      score_artifact_sha256=file_hash(artifact))
        keys = ("selected_lambda_A", "selected_lambda_B", "rep_fit_info", "rep_fit_info_by_half",
                "fold_tuning", "resolved_config", "selected_beta_kl",
                "estimator_contract", "fold_thetas", "fold_est_sizes")
        record["estimator_metadata"] = {key: result[key] for key in keys if key in result}
    except Exception as exc:
        record.update(status="failed", error_type=type(exc).__name__, error_message=str(exc),
                      traceback=traceback.format_exc())
        if getattr(exc, "tuning_rows", None) is not None:
            record["failed_tuning_candidates"] = exc.tuning_rows
    record.update(finished_at=utc_now(), elapsed_seconds=time.monotonic()-started)
    atomic_json(output_dir / "tasks" / (task["task_id"] + ".json"), record)
    running_path.unlink(missing_ok=True)
    return jsonable(record)


def summarize_records(records, config):
    rows = []
    for n in config["n_values"]:
        for method in config["methods"]:
            group = [r for r in records if r["n"] == n and r["method"] == method]
            valid = [r for r in group if r["status"] == "complete"]
            failed = sum(r["status"] == "failed" for r in group)
            count = len(valid)
            row = dict(n=n, method=method, B_requested=config["B_requested"], B_completed=count,
                       B_failed=failed, B_pending=config["B_requested"]-count-failed,
                       Bias=None, SD=None, RMSE=None, Mean_SE=None, CI_Length=None,
                       Coverage=None, Coverage_MCSE=None, run_kind=config["run_kind"])
            if count:
                errors = [r["theta_hat"]-r["theta_population"] for r in valid]
                coverage = statistics.mean(float(r["covered"]) for r in valid)
                row.update(Bias=statistics.mean(errors),
                           SD=statistics.stdev([r["theta_hat"] for r in valid]) if count > 1 else None,
                           RMSE=math.sqrt(statistics.mean(e*e for e in errors)),
                           Mean_SE=statistics.mean(r["se_IF"] for r in valid),
                           CI_Length=statistics.mean(r["ci_upper"]-r["ci_lower"] for r in valid),
                           Coverage=coverage, Coverage_MCSE=math.sqrt(coverage*(1-coverage)/count))
            rows.append(row)
    return rows


def csv_text(rows):
    if not rows:
        return ""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue()


def fmt(value):
    return "--" if value is None else f"{value:.3f}"


def table_texts(rows, config):
    by = {(row["n"], row["method"]): row for row in rows}
    def result(n, method):
        return by.get((n, method), {key:None for key in ("SD", "RMSE", "CI_Length", "Coverage")})
    caveat = ("% Automatically generated; coverage is conditional on completed, valid replications.\n"
              "% See summary.csv for requested/completed/failed counts and Monte Carlo precision.\n"
              f"% Run kind: {config['run_kind']}. Partial rows are not final paper results.\n")
    main = [caveat, r"\begin{tabular}{rlrrrr}", r"\toprule",
            r"$n$ & Estimator & SD & RMSE & CI Length & Coverage \\", r"\midrule"]
    abl = [caveat, r"\begin{tabular}{rrrrrrr}", r"\toprule",
           r"& \multicolumn{2}{c}{SD} & \multicolumn{2}{c}{RMSE} & \multicolumn{2}{c}{CI Length} \\",
           r"$n$ & $\lambda_3=0$ & Tuning & $\lambda_3=0$ & Tuning & $\lambda_3=0$ & Tuning \\", r"\midrule"]
    for n in config["n_values"]:
        for i, method in enumerate(MAIN_METHODS):
            r = result(n, method)
            main.append(f"{n if i == 0 else ''} & {LABELS[method]} & " + " & ".join(fmt(r[k]) for k in ("SD", "RMSE", "CI_Length", "Coverage")) + r" \\")
        main.append(r"\midrule")
        cells = [fmt(result(n, method)[metric]) for metric in ("SD", "RMSE", "CI_Length")
                 for method in ("mediencoder_l3zero", "mediencoder")]
        abl.append(str(n) + " & " + " & ".join(cells) + r" \\")
    main[-1] = r"\bottomrule"
    main.append(r"\end{tabular}")
    abl.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(main)+"\n", "\n".join(abl)+"\n"


def write_reports(output_dir, records, config, started, phase, workers, active_tasks=None):
    records = phase_records(records, config)
    rows = summarize_records(list(records.values()), config)
    atomic_text(output_dir / "summary.csv", csv_text(rows))
    main, ablation = table_texts(rows, config)
    atomic_text(output_dir / "main_table.tex", main)
    atomic_text(output_dir / "ablation_table.tex", ablation)
    completed = sum(r["B_completed"] for r in rows)
    failed = sum(r["B_failed"] for r in rows)
    requested = sum(r["B_requested"] for r in rows)
    active = []
    for task in active_tasks or []:
        progress_path = output_dir / "running" / (task["task_id"] + ".json")
        progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
        active.append(dict(task, state="running" if progress else "queued",
                           started_at=progress.get("started_at"), worker_pid=progress.get("worker_pid")))
    atomic_json(output_dir / "status.json", dict(
        phase=phase, updated_at=utc_now(), started_at=started, parent_pid=os.getpid(),
        requested=requested, completed=completed, failed=failed,
        pending=requested-completed-failed, workers=workers, run_kind=config["run_kind"],
        execution_target_reps=config["B_requested"], reserved_reps=config.get("B_reserved", config["B_requested"]),
        reserved_fits=config.get("B_reserved", config["B_requested"])*len(config["n_values"])*len(config["methods"]),
        active_tasks=active,
        device=config["device"], coverage_denominator="completed valid replications only",
        failures=[{k:r.get(k) for k in ("task_id", "error_type", "error_message")}
                  for r in records.values() if r["status"] == "failed"]))
    return completed, failed, requested


def environment_identity(device):
    import torch
    versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "torch", "pandas", "filelock"):
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = None
    identity = dict(python=sys.version, platform=platform.platform(), packages=versions,
                    cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    if device == "cuda":
        identity["gpu"] = torch.cuda.get_device_name(0)
        identity["gpu_compute_capability"] = list(torch.cuda.get_device_capability(0))
    return identity


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tasks-per-worker", type=int, default=1,
                        help="Recycle worker processes to release CPU/CUDA allocations (default 1)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--n", type=lambda s: [int(x) for x in s.split(",")], default=[100,300,800,1200,2000,3000])
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--target-reps", type=int, default=None,
                        help="Run/report only this replication prefix; later raise it up to reserved --reps without changing run identity")
    parser.add_argument("--arms", type=lambda s:s.split(","), default=list(METHODS),
                        help="Comma-separated arm names; default all five arms")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Run at most this many pending tasks for benchmarking; marks run as benchmark")
    parser.add_argument("--mechanism-seed", type=int, default=910000)
    parser.add_argument("--seed-base", type=int, default=880000)
    parser.add_argument("--pilot-epochs", type=int, default=None)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    if args.tasks_per_worker < 1:
        parser.error("--tasks-per-worker must be positive")
    if args.target_reps is not None and not 1 <= args.target_reps <= args.reps:
        parser.error("--target-reps must be between 1 and --reps")
    if args.workers < 1 or args.reps < 1 or min(args.n) < 40 or len(args.n) != len(set(args.n)):
        parser.error("workers/reps must be positive; unique sample sizes must be at least 40")
    if args.pilot_epochs is not None and not 1 <= args.pilot_epochs <= 300:
        parser.error("--pilot-epochs must be in [1,300]")
    if not args.arms or len(set(args.arms)) != len(args.arms) or any(m not in METHODS for m in args.arms):
        parser.error("--arms must contain unique valid method names: " + ",".join(METHODS))
    if args.max_tasks is not None and args.max_tasks < 1:
        parser.error("--max-tasks must be positive")
    return args


def _main_locked(args):
    configure_environment(args.device)
    config = make_config(args)
    report_config = execution_config(config, args.target_reps)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    code_hashes = collect_code_hashes()
    environment = environment_identity(args.device)
    if existing and (canonical_json(existing["config"]) != canonical_json(config)
                     or existing["code_hashes"] != code_hashes
                     or canonical_json(existing.get("environment")) != canonical_json(environment)):
        raise ValueError("Cannot resume: configuration, source files, or runtime environment changed. Use a new output directory.")
    from mediencoder.simulation.dgp import draw_parameters, load_mechanism, save_mechanism
    prefix = output_dir / "mechanism"
    if existing:
        mechanism = load_mechanism(prefix)
    else:
        if any(output_dir.glob("tasks/*.json")):
            raise ValueError("Existing checkpoints have no manifest; refusing to overwrite provenance")
        mechanism = draw_parameters(config["dgp"], parameter_seed=config["mechanism_seed"])
        save_mechanism(mechanism, prefix)
    manifest = dict(config=config, code_hashes=code_hashes, mechanism_hash=mechanism.mechanism_hash,
                    environment=environment)
    manifest["run_hash"] = digest(manifest)
    if existing:
        validate_manifest(existing, manifest)
        manifest = existing
    else:
        manifest.update(created_at=utc_now(), truth=mechanism.truth.to_dict(),
                        implementation_note="Paper/theory untouched. Fixed-mechanism population truth; observed-only train-fold scales; retained stop-gradient; dataset-specific IF intervals.")
        atomic_json(manifest_path, manifest)
    tasks = list(build_tasks(config))
    expected = {t["task_id"]: t for t in tasks}
    records = {}
    for path in sorted((output_dir / "tasks").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["task_id"] not in expected:
            raise ValueError(f"Unexpected checkpoint: {path}")
        validate_checkpoint(record, manifest, output_dir)
        for key, value in expected[record["task_id"]].items():
            if record.get(key) != value:
                raise ValueError(f"Checkpoint task metadata mismatch: {path}: {key}")
        records[record["task_id"]] = record
    started = utc_now()
    pending = [t for t in tasks if t["task_id"] not in records
               or (args.retry_failed and records[t["task_id"]]["status"] == "failed")]
    pending = [task for task in pending if task["rep"] < report_config["B_requested"]]
    if args.retry_failed:
        # A retry keeps the original seed and preserves every failed attempt.
        # Never turn discarded failures into invisible successful replications.
        for task in pending:
            previous = records.get(task["task_id"])
            if previous and previous["status"] == "failed":
                attempt_hash = digest(previous)[:16]
                atomic_json(output_dir / "attempts" / task["task_id"] / (attempt_hash + ".json"), previous)
    if args.max_tasks is not None:
        pending = pending[:args.max_tasks]
    atomic_json(output_dir / "execution_plan.json", dict(
        run_hash=manifest["run_hash"], reserved_reps=config["B_requested"],
        execution_target_reps=report_config["B_requested"], requested_at=started,
        replication_indices=[0, report_config["B_requested"]-1]))
    counts = write_reports(output_dir, records, report_config, started, "summarized" if args.summarize_only else "running", args.workers)
    print(f"{config['run_kind']} {manifest['run_hash'][:12]}: complete={counts[0]} failed={counts[1]} requested={counts[2]} to_run={len(pending)}", flush=True)
    print(f"Population truth={mechanism.truth.value:.12g}; output={output_dir}", flush=True)
    if args.summarize_only:
        return 0
    iterator = iter(pending)
    phase = "benchmark_batch_finished" if args.max_tasks is not None else "finished"
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                 max_tasks_per_child=args.tasks_per_worker,
                                 initializer=_worker_init,
                                 initargs=(config, str(output_dir), str(prefix), manifest["run_hash"])) as executor:
            futures = {}
            def submit_next():
                task = next(iterator, None)
                if task is not None:
                    futures[executor.submit(_run_task, task)] = task
            for _ in range(min(2*args.workers, len(pending))):
                submit_next()
            while futures:
                done, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
                if not done:
                    write_reports(output_dir, records, report_config, started, "running", args.workers, list(futures.values()))
                    continue
                for future in done:
                    task = futures.pop(future)
                    result = future.result()
                    records[task["task_id"]] = result
                    counts = write_reports(output_dir, records, report_config, started, "running", args.workers, list(futures.values()))
                    print(f"[{utc_now()}] {task['task_id']} {result['status']} {result['elapsed_seconds']:.1f}s; complete={counts[0]} failed={counts[1]}/{counts[2]}", flush=True)
                    if result["status"] == "failed":
                        print(f"  {result['error_type']}: {result['error_message']}", flush=True)
                    submit_next()
    except BaseException:
        phase = "interrupted_or_worker_error"
        raise
    finally:
        counts = write_reports(output_dir, records, report_config, started, phase, args.workers)
    return 2 if counts[1] else 0


def acquire_run_lock(output_dir):
    from filelock import FileLock, Timeout
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = FileLock(output_dir / ".runner.lock", timeout=0)
    try:
        lock.acquire()
    except Timeout as exc:
        raise RuntimeError(f"Another runner already owns {output_dir}. Inspect its status.json; refusing duplicate jobs.") from exc
    return lock


def main(argv=None):
    args = parse_args(argv)
    try:
        lock = acquire_run_lock(args.output_dir.resolve())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 3
    try:
        return _main_locked(args)
    finally:
        lock.release()


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
