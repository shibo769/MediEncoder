#!/usr/bin/env python3
r"""Restartable, paired main-table and alignment-ablation experiment.

Examples (PowerShell):
  python -m mediencoder.simulation.runner --output-dir results/formal --workers 1 --device cuda
  python -m mediencoder.simulation.runner --output-dir results/pilot --workers 1 --device cpu --n 100 --reps 1 --pilot-epochs 2

Re-running the same command resumes completed checkpoints. Scientific settings and
source hashes must match; worker count may change. Pilot results are explicitly
marked and must never be merged into formal results. No manuscript is edited.

Distributed execution uses --shard-index K --num-shards N. Every shard must
resume the same prepared manifest and mechanism; replication r belongs to r % N.
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
import signal
import statistics
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("projection", "autoencoder", "vae", "mediencoder", "mediencoder_l3zero")
MAIN_METHODS = METHODS[:4]
LABELS = dict(zip(METHODS, ("Projection", "Autoencoder", "VAE", "MediEncoder", r"$\lambda_3=0$")))
SOURCE_NAMES = ("simulation/runner.py", "simulation/dgp.py", "estimation.py",
                "training.py", "models.py", "nn_utils.py")
_WORKER = {}
PARTIAL_EXIT_CODE = 4


class ExecutionInterrupted(BaseException):
    """An execution interruption never changes a statistical task's outcome."""

    def __init__(self, phase):
        self.phase = phase
        super().__init__(phase)


def install_execution_signal_handlers():
    """Interrupt once, then allow bounded cleanup to finish before restoring."""
    previous = {}
    interrupted = False

    def interrupt(signum, frame):
        nonlocal interrupted
        if interrupted:
            return
        interrupted = True
        raise ExecutionInterrupted("interrupted_" + signal.Signals(signum).name)

    for name in ("SIGTERM", "SIGUSR2"):
        signum = getattr(signal, name, None)
        if signum is not None:
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, interrupt)
    return previous


def restore_signal_handlers(previous):
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def stop_executor_workers(executor, futures, grace_seconds=5.0, propagate_interruption=True):
    """Cancel queued work and stop only the given executors' process objects.

    Python 3.11/3.12 do not expose terminate_workers(). Prevent the manager from
    replacing recycled workers while shutdown starts, and retain its handles
    before shutdown(wait=False) clears the executor's references.
    """
    executors = executor if isinstance(executor, list) else [executor]
    previous, deferred = {}, []
    def defer(signum, frame):
        deferred.append(signum)
    # A second scheduler signal must not interrupt cleanup halfway through and
    # strand a captured process after shutdown has cleared executor references.
    for name in ("SIGTERM", "SIGUSR2"):
        signum = getattr(signal, name, None)
        if signum is not None:
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, defer)
    try:
        for future in futures:
            future.cancel()
        processes, managers = [], []
        for pool in executors:
            with pool._shutdown_lock:
                pool._max_workers = 0
                processes.extend((pool._processes or {}).values())
                if pool._executor_manager_thread is not None:
                    managers.append(pool._executor_manager_thread)
            pool.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            if process.is_alive():
                process.terminate()
        until = time.monotonic() + grace_seconds
        for process in processes:
            process.join(timeout=max(0.0, until-time.monotonic()))
        for process in processes:
            if process.is_alive():
                process.kill()
        until = time.monotonic() + grace_seconds
        for process in processes:
            process.join(timeout=max(0.0, until-time.monotonic()))
        until = time.monotonic() + grace_seconds
        for manager in managers:
            manager.join(timeout=max(0.0, until-time.monotonic()))
    finally:
        restore_signal_handlers(previous)
    if deferred and propagate_interruption:
        raise ExecutionInterrupted("interrupted_" + signal.Signals(deferred[0]).name)


def load_task_records(output_dir, manifest, expected):
    """Read durable records, including work finished just before interruption."""
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
    return records


def clear_running_markers(output_dir, tasks, grace_seconds=1.0):
    """The parent owns marker deletion; Windows readers may briefly hold a file.

    A diagnostic marker must never turn a durably saved successful fit into a
    failed replication. Retry transient sharing violations for a bounded time.
    """
    for task in tasks:
        path = output_dir / "running" / (task["task_id"] + ".json")
        until = time.monotonic() + grace_seconds
        while True:
            try:
                path.unlink(missing_ok=True)
                break
            except PermissionError:
                if time.monotonic() >= until:
                    print(f"Progress marker still busy: {path}. Durable task outcome is retained.",
                          file=sys.stderr, flush=True)
                    break
                time.sleep(.02)


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


def validate_weight_decay(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("weight decay must be finite and nonnegative")
    return value


def training_configuration(epochs, weight_decay=0.0):
    from mediencoder.estimation import (SHARED_ADAM_EPS, SHARED_HIDDEN, SHARED_NUISANCE_HIDDEN,
                                       SHARED_TRAIN_CFG, _shared_cfg)
    # The existing implementation intentionally forbids per-method overrides.
    # Set the one shared cap before resolving ALL learners, including nuisance fits.
    SHARED_TRAIN_CFG["epochs"] = int(epochs)
    SHARED_TRAIN_CFG["weight_decay"] = validate_weight_decay(weight_decay)
    return dict(
        nn_cfg=_shared_cfg(eps=SHARED_ADAM_EPS, hidden_dims=SHARED_NUISANCE_HIDDEN),
        ae_cfg=_shared_cfg(eps=SHARED_ADAM_EPS, hidden_dims_X=SHARED_HIDDEN,
                           hidden_dims_M=SHARED_HIDDEN, beta_kl=0.5),
        me_cfg=_shared_cfg(hidden_dims_X=SHARED_HIDDEN, hidden_dims_M=SHARED_HIDDEN,
                           hidden_dims_XM=(50, 50), adam_eps=SHARED_ADAM_EPS,
                           lambda1=0.45, lambda2=0.45, lambda3=0.10, beta_kl=0.5,
                           return_history=True, allow_unbalanced_lambda=True),
        encode_cfg=dict(batch_size=4096),
    )


def make_config(args):
    from mediencoder.simulation.dgp import DGPConfig
    from mediencoder.training import ALIGNMENT_NORMALIZATION, ALIGNMENT_NORM_EPS
    mode = getattr(args, "lambda_mode", "fixed")
    tune, zero = (lambda_grids() if mode == "grid" else
                  ([list(args.lambda_fixed)], [list(args.lambda_fixed_zero)]))
    pilot = args.pilot_epochs is not None
    epochs = args.pilot_epochs if pilot else 300
    training = training_configuration(epochs, getattr(args, "weight_decay", 0.0))
    nuisance_standardize = getattr(args, "nuisance_standardize", False)
    if nuisance_standardize:
        # This option belongs only to nuisance models. Do not put it in the
        # shared optimizer settings inherited by the representation trainers.
        training["nn_cfg"]["standardize_inputs"] = True
    if mode == "fixed" and training:
        for key, value in zip(("lambda1", "lambda2", "lambda3"), tune[0]):
            training["me_cfg"][key] = value
    config = dict(
        schema_version=2, run_kind=("PILOT_NOT_FOR_PAPER" if pilot else
                                   "BENCHMARK_NOT_FOR_PAPER" if args.max_tasks is not None else "FORMAL"),
        estimand="population E[mu10(f_X)] conditional on fixed mechanism",
        mechanism_seed=args.mechanism_seed, seed_base=args.seed_base,
        n_values=args.n, B_requested=args.reps, methods=args.arms,
        dgp=DGPConfig(p=args.p, q=args.q,
                            alpha_upper=args.alpha_upper,
                            treatment_link=getattr(args, "treatment_link", "linear"),
                            sigma_U_scale=getattr(args, "sigma_u_scale", 0.05),
                            delta1_spread=args.delta1_spread,
                            delta1_shift=args.delta1_shift, delta1_contrast=args.delta1_contrast,
                            loading_family=getattr(args, "loading_family", "polynomial"),
                            loading_strength=getattr(args, "loading_strength", 4.0),
                            loading_frequency=getattr(args, "loading_frequency", 1.0)).to_dict(),
        tilde_p=getattr(args, "latent_p", 10), tilde_q=getattr(args, "latent_q", 10),
        preprocessing=getattr(args, "preprocessing", "none"),
        training=training,
        lambda_mode=mode, lambda_grid=tune, lambda_grid_zero=zero,
        stop_gradient=False, loss_normalization="alignment_row_l2",
        reconstruction_loss_normalization="none", alignment_normalization=ALIGNMENT_NORMALIZATION,
        alignment_norm_eps=ALIGNMENT_NORM_EPS,
        loss_reduction="mean_over_subjects_and_coordinates",
        variance_estimator="within_fold_size_weighted",
        interval="theta_hat +/- 1.959963984540054 * sqrt(sum_k(n_k * var(scores_in_fold_k, ddof=1))/n^2)",
        numeric_safeguards=dict(propensity_clip_eps=0.01, density_ratio_cap=0.0, density_ratio_soft_cap=0.0),
        device=args.device, torch_threads=1, deterministic_algorithms=True,
    )
    # Omit disabled opt-in metadata so the default canonical configuration
    # remains compatible with the previous simulation settings.
    if nuisance_standardize:
        config["nuisance_standardize"] = True
    return jsonable(config)


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


def configure_shared_training(training):
    """Apply the recorded common budget in each spawned worker, including tuning."""
    from mediencoder.estimation import SHARED_TRAIN_CFG
    learners = ("nn_cfg", "ae_cfg", "me_cfg")
    epoch_caps = {training[name]["epochs"] for name in learners}
    if len(epoch_caps) != 1:
        raise ValueError("All learners must use the same shared epoch cap")
    weight_decays = {validate_weight_decay(training[name]["weight_decay"]) for name in learners}
    if len(weight_decays) != 1:
        raise ValueError("All learners must use the same shared weight decay")
    SHARED_TRAIN_CFG["epochs"] = int(epoch_caps.pop())
    SHARED_TRAIN_CFG["weight_decay"] = weight_decays.pop()


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


def select_shard_tasks(tasks, shard_index=0, num_shards=1):
    """Partition replications without changing seeds or splitting paired arms."""
    if (not isinstance(num_shards, int) or isinstance(num_shards, bool) or num_shards < 1
            or not isinstance(shard_index, int) or isinstance(shard_index, bool)
            or not 0 <= shard_index < num_shards):
        raise ValueError("num_shards must be positive and shard_index must be in [0, num_shards)")
    return [task for task in tasks if task["rep"] % num_shards == shard_index]


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


def mechanism_prefix(path):
    """Accept a saved bundle directory, prefix, or either artifact filename."""
    path = Path(path).resolve()
    if path.is_dir():
        return path / "mechanism"
    return path.with_suffix("") if path.suffix in (".json", ".npz") else path


def validate_mechanism_config(mechanism, config):
    if canonical_json(mechanism.config.to_dict()) != canonical_json(config["dgp"]):
        raise ValueError("Imported/saved mechanism DGP configuration differs from this run")
    if mechanism.parameter_seed != config["mechanism_seed"]:
        raise ValueError("Imported/saved mechanism parameter seed differs from --mechanism-seed")


def mechanism_artifact_hashes(prefix):
    return {suffix: file_hash(Path(str(prefix) + "." + suffix)) for suffix in ("json", "npz")}


def prepare_mechanism(output_dir, config, existing=None, mechanism_from=None):
    """Import or draw once; never overwrite unmanifested scientific output.

    Import locations are provenance, not scientific settings. The verified
    mechanism content hash is included in the run identity by the caller.
    Resume needs only the saved bundle; an explicitly supplied import is checked
    again and cannot silently replace the run's original mechanism.
    """
    from mediencoder.simulation.dgp import draw_parameters, load_mechanism, save_mechanism
    output_dir = Path(output_dir)
    prefix = output_dir / "mechanism"
    if existing is None:
        # The monitor writes only these non-scientific files before launching
        # the runner. Its two atomic-write temporaries can exist during startup.
        monitor_files = {".runner.lock", "run.log", "supervisor.json", "progress.html",
                         "supervisor.json.tmp", "progress.html.tmp"}
        occupied = []
        for path in output_dir.iterdir():
            if path.name in monitor_files and not path.is_symlink():
                if path.is_file() or not path.exists():
                    continue  # An atomic-write temporary may already be renamed.
            occupied.append(path.name)
        if occupied:
            raise ValueError("New runs require an empty output directory apart from monitor logs; refusing to overwrite existing files: "
                             + ", ".join(sorted(occupied)))
    imported = None
    source_hashes = None
    if mechanism_from is not None:
        source_prefix = mechanism_prefix(mechanism_from)
        imported = load_mechanism(source_prefix)
        validate_mechanism_config(imported, config)
        source_hashes = mechanism_artifact_hashes(source_prefix)
    if existing is not None:
        mechanism = load_mechanism(prefix)
        validate_mechanism_config(mechanism, config)
        if mechanism.mechanism_hash != existing["mechanism_hash"]:
            raise ValueError("Cannot resume: saved mechanism differs from the original manifest")
        if imported is not None and imported.mechanism_hash != mechanism.mechanism_hash:
            raise ValueError("Cannot resume: --mechanism-from differs from the original imported mechanism")
        provenance = existing.get("mechanism_provenance")
        if provenance and mechanism_artifact_hashes(prefix) != provenance["saved_artifact_sha256"]:
            raise ValueError("Cannot resume: saved mechanism artifacts changed since preparation")
        return mechanism, provenance
    mechanism = imported if imported is not None else draw_parameters(
        config["dgp"], parameter_seed=config["mechanism_seed"])
    validate_mechanism_config(mechanism, config)
    if imported is not None and source_prefix.name == "mechanism":
        # Preserve the canonical archived bundle byte for byte, including its
        # NPZ compression and JSON line endings; never regenerate its parameters.
        import shutil
        for suffix in ("json", "npz"):
            shutil.copyfile(Path(str(source_prefix) + "." + suffix),
                            Path(str(prefix) + "." + suffix))
        if (mechanism_artifact_hashes(prefix) != source_hashes
                or load_mechanism(prefix).mechanism_hash != mechanism.mechanism_hash):
            raise ValueError("Imported mechanism artifacts changed during preparation")
    else:
        # Custom prefix names need a canonical arrays_file field in the JSON.
        save_mechanism(mechanism, prefix)
    provenance = dict(kind="imported" if imported is not None else "generated",
                      mechanism_hash=mechanism.mechanism_hash,
                      saved_artifact_sha256=mechanism_artifact_hashes(prefix))
    if source_hashes is not None:
        provenance["source_artifact_sha256"] = source_hashes
    return mechanism, provenance


def validate_checkpoint(record, manifest, output_dir):
    if record.get("run_hash") != manifest["run_hash"]:
        raise ValueError(f"Checkpoint belongs to a different run: {record.get('task_id')}")
    if record.get("status") == "complete":
        artifact = Path(output_dir) / record["score_artifact"]
        if not artifact.is_file() or file_hash(artifact) != record["score_artifact_sha256"]:
            raise ValueError(f"Missing or damaged score artifact: {artifact}")
    elif record.get("status") != "failed":
        raise ValueError(f"Unsupported checkpoint status: {record.get('status')}")


def _worker_init(config, output_dir, mechanism_prefix, run_hash, observed_data_from=None):
    configure_environment(config["device"])
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    configure_shared_training(config["training"])
    from mediencoder.simulation.dgp import load_mechanism
    _WORKER.update(config=config, output_dir=Path(output_dir),
                   mechanism=load_mechanism(mechanism_prefix), run_hash=run_hash,
                   training=runtime_training_configuration(config["training"]),
                   observed_data_from=None if observed_data_from is None else Path(observed_data_from))


def observed_data_hash(data):
    import numpy as np
    h = hashlib.sha256()
    for name in ("X", "M", "A", "Y"):
        a = np.ascontiguousarray(data[name])
        h.update(name.encode()); h.update(str(a.shape).encode()); h.update(a.dtype.str.encode())
        h.update(a.tobytes())
    return h.hexdigest()


def observed_dataset_identity(mechanism, task, config):
    """Raw inputs are shared across training settings and CPU/GPU execution."""
    dgp = config["dgp"]
    return dict(schema_version=1, mechanism_hash=mechanism.mechanism_hash,
                n=int(task["n"]), rep=int(task["rep"]), data_seed=int(task["data_seed"]),
                p=int(dgp["p"]), q=int(dgp["q"]),
                dgp_source_sha256=file_hash(ROOT / "simulation/dgp.py"))


def observed_dataset_paths(directory, identity):
    stem = (f"n{identity['n']:05d}_r{identity['rep']:04d}_"
            + digest(identity))
    directory = Path(directory)
    return directory / (stem + ".npz"), directory / (stem + ".json"), directory / (stem + ".pending.json")


def validate_observed_arrays(data, identity):
    import numpy as np
    if set(data) != {"X", "M", "A", "Y"}:
        raise ValueError("Observed dataset must contain only X, M, A, Y")
    shapes = {"X": (identity["n"], identity["p"]), "M": (identity["n"], identity["q"]),
              "A": (identity["n"],), "Y": (identity["n"],)}
    for name, shape in shapes.items():
        array = data[name]
        if (array.shape != shape or not np.issubdtype(array.dtype, np.number)
                or np.issubdtype(array.dtype, np.complexfloating) or not np.isfinite(array).all()):
            raise ValueError(f"Invalid observed dataset array: {name}")
    if not np.isin(data["A"], (0, 1)).all():
        raise ValueError("Observed treatment must be binary")


def _read_observed_npz(path, identity):
    import numpy as np
    with np.load(path, allow_pickle=False) as saved:
        data = {name: np.ascontiguousarray(saved[name]) for name in saved.files}
    validate_observed_arrays(data, identity)
    return data


def read_observed_dataset(directory, identity):
    npz, receipt, _ = observed_dataset_paths(directory, identity)
    if not receipt.is_file() or not npz.is_file():
        raise ValueError(f"Observed dataset cache is missing or incomplete: {receipt}")
    metadata = json.loads(receipt.read_text(encoding="utf-8"))
    if (metadata.get("identity") != identity or metadata.get("npz_file") != npz.name
            or metadata.get("npz_sha256") != file_hash(npz)):
        raise ValueError(f"Observed dataset identity/checksum mismatch: {receipt}")
    data = _read_observed_npz(npz, identity)
    array_metadata = {name: dict(shape=list(array.shape), dtype=array.dtype.str)
                      for name, array in data.items()}
    if (metadata.get("arrays") != array_metadata
            or metadata.get("observed_data_sha256") != observed_data_hash(data)):
        raise ValueError(f"Observed dataset array metadata/hash mismatch: {receipt}")
    return data


def _pair_has_returned_fit(output_dir, identity):
    pattern = f"n{identity['n']:05d}_r{identity['rep']:04d}_*.json"
    for path in (Path(output_dir) / "tasks").glob(pattern):
        record = json.loads(path.read_text(encoding="utf-8"))
        if all(record.get(key) == identity[key] for key in ("n", "rep", "data_seed")):
            return True
    return False


def bind_observed_dataset(output_dir, identity, data):
    """A run may never switch an already selected pair to different bytes."""
    from filelock import FileLock
    binding = Path(output_dir) / "dataset_bindings" / (digest(identity) + ".json")
    expected = dict(identity=identity, observed_data_sha256=observed_data_hash(data))
    def check():
        if json.loads(binding.read_text(encoding="utf-8")) != expected:
            raise ValueError("Observed dataset differs from this run's immutable pair binding")
    if binding.exists():
        check()
        return data
    binding.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(binding) + ".lock", timeout=600):
        if binding.exists():
            check()
        else:
            # Also protect resumed runs whose earlier records predate a binding,
            # or whose binding was removed while task/score artifacts survived.
            pattern = f"n{identity['n']:05d}_r{identity['rep']:04d}_*.json"
            for path in (Path(output_dir) / "tasks").glob(pattern):
                record = json.loads(path.read_text(encoding="utf-8"))
                if (all(record.get(key) == identity[key] for key in ("n", "rep", "data_seed"))
                        and record.get("observed_data_sha256") is not None
                        and record["observed_data_sha256"] != expected["observed_data_sha256"]):
                    raise ValueError("Observed dataset differs from a previously returned fit for this pair")
            atomic_json(binding, expected)
    return data


def cached_observed_data(mechanism, task, config, output_dir, observed_data_from=None):
    """Produce a paired dataset once and load immutable bytes in every arm.

    Final receipts and NPZ files are never overwritten. An interrupted producer
    can be recovered only with its matching identity intent and before any fit
    for the pair has returned. External caches are strictly read-only.
    """
    import numpy as np
    from filelock import FileLock
    identity = observed_dataset_identity(mechanism, task, config)
    directory = Path(observed_data_from) if observed_data_from is not None else Path(output_dir) / "datasets"
    if observed_data_from is not None:
        return bind_observed_dataset(output_dir, identity, read_observed_dataset(directory, identity))
    directory.mkdir(parents=True, exist_ok=True)
    npz, receipt, intent = observed_dataset_paths(directory, identity)
    if not receipt.exists():
        with FileLock(str(receipt) + ".lock", timeout=600):
            if not receipt.exists():
                if _pair_has_returned_fit(output_dir, identity):
                    raise ValueError("Cannot recreate observed data after a fit for this pair has returned")
                if intent.exists():
                    if json.loads(intent.read_text(encoding="utf-8")) != identity:
                        raise ValueError("Observed dataset producer intent differs from its identity")
                elif npz.exists():
                    raise ValueError("Unidentified orphan observed dataset; refusing replacement")
                else:
                    atomic_json(intent, identity)
                if npz.exists():
                    # A matching producer wrote the complete atomic NPZ before
                    # interruption. Adopt its bytes; never redraw/overwrite it.
                    data = _read_observed_npz(npz, identity)
                else:
                    from mediencoder.simulation.dgp import sample_data
                    generated = sample_data(mechanism, task["n"], task["data_seed"])
                    data = {name: np.ascontiguousarray(generated[name]) for name in ("X", "M", "A", "Y")}
                    validate_observed_arrays(data, identity)
                    temporary = npz.with_name(npz.name + f".{os.getpid()}.tmp")
                    with temporary.open("wb") as stream:
                        np.savez(stream, **data)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, npz)
                atomic_json(receipt, dict(identity=identity, npz_file=npz.name,
                            npz_sha256=file_hash(npz), observed_data_sha256=observed_data_hash(data),
                            arrays={name: dict(shape=list(array.shape), dtype=array.dtype.str)
                                    for name, array in data.items()}))
                try:
                    intent.unlink(missing_ok=True)
                except PermissionError:
                    pass  # The completed receipt remains authoritative.
    # Final data are immutable; reads/checksums do not serialize on the writer lock.
    return bind_observed_dataset(output_dir, identity, read_observed_dataset(directory, identity))


def task_estimation_options(config, task, training):
    """Resolve an arm's fixed weights or tuning grid without mutating the manifest."""
    options = {name: dict(settings) for name, settings in training.items()}
    options["preprocessing"] = config.get("preprocessing", "none")
    method = "mediencoder" if task["method"] == "mediencoder_l3zero" else task["method"]
    grid = None
    if method == "mediencoder":
        candidates = config["lambda_grid_zero"] if task["method"].endswith("l3zero") else config["lambda_grid"]
        if config.get("lambda_mode", "grid") == "fixed":
            if len(candidates) != 1:
                raise ValueError("Fixed lambda mode requires exactly one recorded weight triple per arm")
            for key, value in zip(("lambda1", "lambda2", "lambda3"), candidates[0]):
                options["me_cfg"][key] = value
            options["fixed_lambda"] = tuple(candidates[0])
        else:
            grid = candidates
    return method, grid, options


def _run_task(task):
    import numpy as np
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
        data = cached_observed_data(_WORKER["mechanism"], task, config, output_dir,
                                    _WORKER.get("observed_data_from"))
        record["observed_data_sha256"] = observed_data_hash(data)
        truth = float(_WORKER["mechanism"].truth.value)
        # True factors/nuisances are intentionally unavailable to the estimator call.
        data.pop("oracle", None)
        method, grid, options = task_estimation_options(config, task, _WORKER["training"])
        set_all_seeds(task["training_seed"])
        result = estimate_triply_IF(data["X"], data["M"], data["A"], data["Y"],
                                   tilde_p=config["tilde_p"], tilde_q=config["tilde_q"],
                                   factor_method=method, seed=task["training_seed"],
                                   lambda_grid=grid, **options)
        scores = np.asarray(result["crossfit_scores"], dtype=np.float64).reshape(-1)
        if len(scores) != task["n"] or not np.isfinite(scores).all():
            raise ValueError("Every input subject must have exactly one finite cross-fitted score")
        theta = float(result["theta_hat_IF"])
        se = float(result["se_IF"])
        lower, upper = float(result["ci_lower"]), float(result["ci_upper"])
        if not all(map(math.isfinite, (truth, theta, se, lower, upper))) or se < 0:
            raise ValueError("Nonfinite point estimate/truth/interval or negative standard error")
        estimation_indices = [np.asarray(roles["estimation"]) for roles in result["fold_indices"]]
        if len(estimation_indices) != 4 or any(
                indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                or len(indices) < 2 for indices in estimation_indices):
            raise ValueError("Four estimation folds with at least two subjects each are required")
        if not np.array_equal(np.sort(np.concatenate(estimation_indices)), np.arange(len(scores))):
            raise ValueError("Estimation folds must cover each saved score exactly once")
        expected_se = float(np.sqrt(sum(len(indices) * np.var(scores[indices], ddof=1)
                                        for indices in estimation_indices) / len(scores)**2))
        if not math.isclose(theta, float(scores.mean()), rel_tol=1e-9, abs_tol=1e-10):
            raise ValueError("Point estimate is not the mean of saved scores")
        if not math.isclose(se, expected_se, rel_tol=1e-9, abs_tol=1e-10):
            raise ValueError("Standard error differs from the saved within-fold score variances")
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
                 "estimator_contract", "fold_thetas", "fold_est_sizes",
                 "variance_estimator", "fold_score_variances")
        record["estimator_metadata"] = {key: result[key] for key in keys if key in result}
    except Exception as exc:
        record.update(status="failed", error_type=type(exc).__name__, error_message=str(exc),
                      traceback=traceback.format_exc())
        if getattr(exc, "tuning_rows", None) is not None:
            record["failed_tuning_candidates"] = exc.tuning_rows
    record.update(finished_at=utc_now(), elapsed_seconds=time.monotonic()-started)
    atomic_json(output_dir / "tasks" / (task["task_id"] + ".json"), record)
    # The parent reads progress markers and owns their deletion after receiving
    # this durable outcome. Concurrent read/unlink otherwise races on Windows.
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
    me_label = "Fixed ME" if config.get("lambda_mode") == "fixed" else "Tuning"
    abl = [caveat, r"\begin{tabular}{rrrrrrr}", r"\toprule",
           r"& \multicolumn{2}{c}{SD} & \multicolumn{2}{c}{RMSE} & \multicolumn{2}{c}{CI Length} \\",
           "$n$ & $\\lambda_3=0$ & " + me_label + " & $\\lambda_3=0$ & " + me_label
           + " & $\\lambda_3=0$ & " + me_label + r" \\", r"\midrule"]
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
    shard_index, num_shards = config.get("shard_index", 0), config.get("num_shards", 1)
    shard_reps = list(range(shard_index, config["B_requested"], num_shards))
    shard_requested = len(shard_reps) * len(config["n_values"]) * len(config["methods"])
    shard_records = select_shard_tasks(records.values(), shard_index, num_shards)
    shard_completed = sum(row["status"] == "complete" for row in shard_records)
    shard_failed = sum(row["status"] == "failed" for row in shard_records)
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
        report_scope="shard_partial" if num_shards > 1 else "execution_phase",
        shard_index=shard_index, num_shards=num_shards, shard_replication_indices=shard_reps,
        shard_requested=shard_requested, shard_completed=shard_completed, shard_failed=shard_failed,
        shard_pending=shard_requested-shard_completed-shard_failed,
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
                        help="Replace a worker's single-process pool after this many fits to release CPU/CUDA allocations (default 1)")
    parser.add_argument("--time-limit-seconds", type=float, default=None,
                        help="Execution-only deadline, including preparation; stop owned workers and return 4, preserving completed tasks for resumption")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--n", type=lambda s: [int(x) for x in s.split(",")], default=[100,300,800,1200,2000,3000])
    parser.add_argument("--p", type=int, default=500, help="Number of observed covariates (default 500)")
    parser.add_argument("--q", type=int, default=500, help="Number of observed mediators (default 500)")
    parser.add_argument("--latent-p", type=int, default=10,
                        help="Learned covariate representation dimension, shared by every method (default 10)")
    parser.add_argument("--latent-q", type=int, default=10,
                        help="Learned mediator representation dimension, shared by every method (default 10)")
    from mediencoder.simulation.dgp import LOADING_FAMILIES
    parser.add_argument("--loading-family", choices=LOADING_FAMILIES, default="polynomial",
                        help="Fixed measurement family; polynomial is the retained default")
    parser.add_argument("--loading-strength", type=float, default=4.0,
                        help="Nonlinearity strength for mixed_tanh/deep_tanh; ignored by legacy families")
    parser.add_argument("--loading-frequency", type=float, default=1.0,
                        help="Positive tanh input frequency for nonlinear measurement families")
    parser.add_argument("--preprocessing", choices=("none", "standardize"), default="none",
                        help="Shared representation-input preprocessing, fitted only on representation-training rows")
    parser.add_argument("--nuisance-standardize", action="store_true",
                        help="Standardize each nuisance model's input features using only its training subset; leave Y unchanged (default disabled)")
    parser.add_argument("--alpha-upper", type=float, default=1.0,
                        help="Treatment coefficient upper bound for Uniform(0, bound), default 1")
    parser.add_argument("--treatment-link", choices=("linear", "quadratic_centered"), default="linear",
                        help="Treatment assignment logit: linear (default) or sum_j alpha_j * (f_Xj^2 - (mu^2 + 1/3))")
    parser.add_argument("--sigma-u-scale", type=float, default=0.05,
                        help="Positive scale of Sigma_U covariance eigenvalues, Uniform[1,2] times scale; not a mediator SD (default .05)")
    parser.add_argument("--delta1-spread", type=float, default=0.2,
                        help="Scale independent Uniform(.5,1.5) delta draws about 1; default .2 gives [.9,1.1]")
    parser.add_argument("--delta1-shift", type=float, default=0.0,
                        help="Optional shift for delta_1; default 0")
    parser.add_argument("--delta1-contrast", type=float, default=0.0,
                        help="Explicit legacy contrast branch; default 0 keeps independent delta matrices")
    parser.add_argument("--lambda-mode", choices=("fixed", "grid"), default="fixed",
                        help="Prespecified weights (default) or separate held-out tuning in each representation half")
    parser.add_argument("--lambda-fixed", type=lambda s: [float(x) for x in s.split(",")],
                        default=[0.45, 0.45, 0.10], help="Fixed ME weight triple, default .45,.45,.10")
    parser.add_argument("--lambda-fixed-zero", type=lambda s: [float(x) for x in s.split(",")],
                        default=[0.5, 0.5, 0.0], help="Fixed zero-alignment weight triple, default .5,.5,0")
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--target-reps", type=int, default=None,
                        help="Run/report only this replication prefix; later raise it up to reserved --reps without changing run identity")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Zero-based execution shard; requires the shared prepared manifest and mechanism")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Assign replication r to shard r %% num_shards; execution setting, not part of run identity")
    parser.add_argument("--arms", type=lambda s:s.split(","), default=list(METHODS),
                        help="Comma-separated arm names; default all five arms")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Run at most this many pending tasks for benchmarking; marks run as benchmark")
    parser.add_argument("--mechanism-seed", type=int, default=910000)
    parser.add_argument("--mechanism-from", type=Path, default=None,
                        help="Import the exact saved mechanism from a bundle directory, prefix, JSON or NPZ; DGP settings and parameter seed must match")
    parser.add_argument("--observed-data-from", type=Path, default=None,
                        help="Read paired X/M/A/Y bytes from this immutable datasets cache directory; missing or damaged data are errors (execution-only)")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Shared Adam weight decay for representation, nuisance and tuning fits; recorded in scientific run identity (default 0)")
    parser.add_argument("--seed-base", type=int, default=20261019)
    parser.add_argument("--pilot-epochs", type=int, default=None)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not math.isfinite(args.sigma_u_scale) or args.sigma_u_scale <= 0:
            raise ValueError("--sigma-u-scale must be finite and strictly positive")
        from mediencoder.simulation.dgp import DGPConfig
        DGPConfig(p=args.p, q=args.q, alpha_upper=args.alpha_upper,
                  treatment_link=args.treatment_link, sigma_U_scale=args.sigma_u_scale,
                  delta1_spread=args.delta1_spread, delta1_shift=args.delta1_shift,
                  delta1_contrast=args.delta1_contrast, loading_family=args.loading_family,
                  loading_strength=args.loading_strength, loading_frequency=args.loading_frequency).validate()
        validate_weight_decay(args.weight_decay)
        for name in ("lambda_fixed", "lambda_fixed_zero"):
            value = getattr(args, name)
            if (len(value) != 3 or not all(math.isfinite(x) and x >= 0 for x in value)
                    or value[0] <= 0 or value[1] <= 0 or not math.isclose(sum(value), 1.0, abs_tol=1e-9)):
                raise ValueError(f"--{name.replace('_', '-')} must contain three nonnegative weights summing to 1, with positive reconstruction weights")
        if args.lambda_fixed[2] <= 0 or args.lambda_fixed_zero[2] != 0:
            raise ValueError("--lambda-fixed requires positive alignment weight; --lambda-fixed-zero requires zero alignment weight")
    except ValueError as exc:
        parser.error(str(exc))
    if args.tasks_per_worker < 1:
        parser.error("--tasks-per-worker must be positive")
    if args.time_limit_seconds is not None and (
            not math.isfinite(args.time_limit_seconds) or args.time_limit_seconds <= 0):
        parser.error("--time-limit-seconds must be finite and positive")
    if args.p < 1 or args.q < 1:
        parser.error("--p and --q must be positive")
    if args.latent_p < 1 or args.latent_q < 1:
        parser.error("--latent-p and --latent-q must be positive")
    if args.target_reps is not None and not 1 <= args.target_reps <= args.reps:
        parser.error("--target-reps must be between 1 and --reps")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("--num-shards must be positive and --shard-index must be in [0, num-shards)")
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
    time_limit = getattr(args, "time_limit_seconds", None)
    deadline = None if time_limit is None else time.monotonic() + time_limit
    configure_environment(args.device)
    config = make_config(args)
    report_config = execution_config(config, args.target_reps)
    report_config.update(shard_index=args.shard_index, num_shards=args.num_shards)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    observed_data_from = args.observed_data_from.resolve() if args.observed_data_from is not None else None
    if observed_data_from is not None and not observed_data_from.is_dir():
        raise ValueError("--observed-data-from must name an existing datasets cache directory")
    manifest_path = output_dir / "manifest.json"
    existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if args.num_shards > 1 and existing is None:
        raise ValueError("Distributed shards require a shared prepared manifest and mechanism. Prepare once with --summarize-only and no sharding, then copy that directory to each shard.")
    code_hashes = collect_code_hashes()
    environment = environment_identity(args.device)
    if existing and (canonical_json(existing["config"]) != canonical_json(config)
                     or existing["code_hashes"] != code_hashes
                     or canonical_json(existing.get("environment")) != canonical_json(environment)):
        raise ValueError("Cannot resume: configuration, source files, or runtime environment changed. Use a new output directory.")
    prefix = output_dir / "mechanism"
    mechanism, mechanism_provenance = prepare_mechanism(
        output_dir, config, existing=existing, mechanism_from=args.mechanism_from)
    manifest = dict(config=config, code_hashes=code_hashes, mechanism_hash=mechanism.mechanism_hash,
                    environment=environment)
    manifest["run_hash"] = digest(manifest)
    if existing:
        validate_manifest(existing, manifest)
        manifest = existing
    else:
        manifest.update(created_at=utc_now(), truth=mechanism.truth.to_dict(),
                        mechanism_provenance=mechanism_provenance,
                        implementation_note="Paper/theory untouched. Fixed-mechanism population truth; raw reconstruction MSE; row-L2-normalized mediator alignment with gradients through both encoders; recorded fixed or tuned simplex lambda weights; dataset-specific within-fold IF intervals.")
        atomic_json(manifest_path, manifest)
    tasks = list(build_tasks(config))
    expected = {t["task_id"]: t for t in tasks}
    records = load_task_records(output_dir, manifest, expected)
    started = utc_now()
    pending = [t for t in tasks if t["task_id"] not in records
               or (args.retry_failed and records[t["task_id"]]["status"] == "failed")]
    pending = [task for task in pending if task["rep"] < report_config["B_requested"]]
    pending = select_shard_tasks(pending, args.shard_index, args.num_shards)
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
        replication_indices=[0, report_config["B_requested"]-1],
        shard_index=args.shard_index, num_shards=args.num_shards,
        shard_replication_indices=list(range(args.shard_index, report_config["B_requested"], args.num_shards)),
        shard_pending_tasks=len(pending), time_limit_seconds=time_limit,
        observed_data_from=observed_data_from))
    counts = write_reports(output_dir, records, report_config, started, "summarized" if args.summarize_only else "running", args.workers)
    print(f"{config['run_kind']} {manifest['run_hash'][:12]}: complete={counts[0]} failed={counts[1]} requested={counts[2]} to_run={len(pending)}", flush=True)
    print(f"Population truth={mechanism.truth.value:.12g}; output={output_dir}", flush=True)
    if args.num_shards > 1:
        print(f"Execution shard {args.shard_index}/{args.num_shards}: replication r % {args.num_shards} = {args.shard_index}. Reports retain the full {report_config['B_requested']}-replication target; shard outputs are partial.", flush=True)
    if args.summarize_only:
        return 0
    iterator = iter(pending)
    phase = ("benchmark_batch_finished" if args.max_tasks is not None else
             "shard_finished" if args.num_shards > 1 else "finished")
    # Independent slots avoid ProcessPoolExecutor's max_tasks_per_child
    # replacement bookkeeping. A completed slot can recycle immediately while
    # other slots keep fitting, without a batch-wide barrier.
    slots = [dict(executor=None, completed=0) for _ in range(args.workers)]
    futures = {}
    submitted = {}
    interrupted = False

    def check_deadline():
        if deadline is not None and time.monotonic() >= deadline:
            raise ExecutionInterrupted("time_limit_reached")

    try:
        if pending:
            check_deadline()
            def submit_next(slot):
                check_deadline()
                task = next(iterator, None)
                if task is not None:
                    if slot["executor"] is None:
                        slot["executor"] = ProcessPoolExecutor(
                            max_workers=1, mp_context=mp.get_context("spawn"),
                            initializer=_worker_init,
                            initargs=(config, str(output_dir), str(prefix), manifest["run_hash"], observed_data_from))
                        slot["completed"] = 0
                    submitted[task["task_id"]] = task
                    futures[slot["executor"].submit(_run_task, task)] = (task, slot)
            for slot in slots[:len(pending)]:
                submit_next(slot)
            while futures:
                check_deadline()
                wait_seconds = 30 if deadline is None else min(30, max(0, deadline-time.monotonic()))
                done, _ = wait(futures, timeout=wait_seconds, return_when=FIRST_COMPLETED)
                check_deadline()
                if not done:
                    write_reports(output_dir, records, report_config, started, "running", args.workers,
                                  [task for task, slot in futures.values()])
                    continue
                for future in done:
                    task, slot = futures.pop(future)
                    result = future.result()
                    records[task["task_id"]] = result
                    clear_running_markers(output_dir, [task])
                    counts = write_reports(output_dir, records, report_config, started, "running", args.workers,
                                           [task for task, active_slot in futures.values()])
                    print(f"[{utc_now()}] {task['task_id']} {result['status']} {result['elapsed_seconds']:.1f}s; complete={counts[0]} failed={counts[1]}/{counts[2]}", flush=True)
                    if result["status"] == "failed":
                        print(f"  {result['error_type']}: {result['error_message']}", flush=True)
                    slot["completed"] += 1
                    if slot["completed"] >= args.tasks_per_worker:
                        # The fit is durably saved and this slot is idle. Retire
                        # only its process, while other slots continue training.
                        stop_executor_workers(slot["executor"], [])
                        slot["executor"] = None
                    submit_next(slot)
    except ExecutionInterrupted as exc:
        interrupted = True
        phase = exc.phase
        print(f"Execution stopped: {phase}. Completed tasks are retained; unfinished tasks keep their original seeds.", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        phase = "interrupted_SIGINT"
        print("Execution stopped: interrupted_SIGINT. Completed tasks are retained.", flush=True)
    except BrokenProcessPool:
        interrupted = True
        phase = "interrupted_or_worker_error"
        traceback.print_exc()
    except BaseException:
        interrupted = True
        phase = "interrupted_or_worker_error"
        raise
    finally:
        pools = [slot["executor"] for slot in slots if slot["executor"] is not None]
        if pools:
            # At normal completion all submitted fits are already saved. On an
            # interruption unfinished fits remain pending. Either way, retire
            # all owned slots within one shared grace interval.
            stop_executor_workers(pools, futures, propagate_interruption=False)
        # Task writes precede delivery through the process pool. Recover records
        # even if a worker/parent was interrupted while returning its result.
        records = load_task_records(output_dir, manifest, expected)
        # Keep the submitted inventory separately: a crashed future has already
        # been popped when result() raises and still needs its marker removed.
        clear_running_markers(output_dir, submitted.values())
        counts = write_reports(output_dir, records, report_config, started, phase, args.workers)
    if interrupted:
        return PARTIAL_EXIT_CODE
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
    previous = {}
    try:
        previous = install_execution_signal_handlers()
        try:
            return _main_locked(args)
        except ExecutionInterrupted as exc:
            print(f"Execution stopped during preparation: {exc.phase}.", file=sys.stderr, flush=True)
            return PARTIAL_EXIT_CODE
    finally:
        restore_signal_handlers(previous)
        lock.release()


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
