"""Freeze and submit a resumable, synthetic-only MediEncoder experiment on BU SCC.

The submitted scripts use an immutable source snapshot and project-disk results.
No login-node training, secret export, email, or unrelated scheduler job changes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


SCIENCE_ARGS = [
    "--device", "cuda", "--n", "100,300,800,1200,2000,3000",
    "--p", "500", "--q", "500", "--reps", "200",
    "--loading-family", "polynomial", "--alpha-upper", "1",
    "--delta1-spread", "0.2", "--delta1-shift", "0", "--delta1-contrast", "0",
    "--lambda-mode", "fixed", "--lambda-fixed", "0.45,0.45,0.10",
    "--lambda-fixed-zero", "0.5,0.5,0", "--weight-decay", "0",
    "--seed-base", "20261019", "--mechanism-seed", "910000",
]
PHASES = ("prepare", "benchmark", "canary", "array", "merge")
GPU_POOLS = {"shared": (2, "l40s"), "public": (4, "ece-pub"),
             "combined": (6, "l40s,ece-pub")}


def science_args(device):
    if device not in ("cpu", "cuda"):
        raise ValueError("SCC device must be cpu or cuda")
    args = list(SCIENCE_ARGS)
    args[args.index("--device")+1] = device
    return args


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def safe_source(root, name):
    root = Path(root).resolve()
    text = str(name).replace("\\", "/")
    relative = PurePosixPath(text)
    if (relative.is_absolute() or ".." in relative.parts or ":" in text
            or not relative.parts or any(part in ("", ".") for part in relative.parts)):
        raise ValueError(f"Unsafe source path: {name}")
    path = root.joinpath(*relative.parts)
    if not path.is_file() or root not in path.resolve().parents:
        raise ValueError(f"Source file missing or escapes snapshot: {name}")
    current = path
    while current != root:
        if current.is_symlink():
            raise ValueError(f"Source symlinks are not accepted: {name}")
        current = current.parent
    return path


def source_inventory(source):
    source = Path(source).resolve()
    deployed = source / "SCC_DEPLOYMENT_MANIFEST.json"
    if deployed.is_file():
        manifest = read_json(deployed)
        commit, files = manifest["source_commit"], manifest["files"]
        inventory = {}
        for name, expected in files.items():
            path = safe_source(source, name)
            size = expected.get("size", expected.get("bytes"))
            if path.stat().st_size != size or sha256(path) != expected["sha256"]:
                raise ValueError(f"Deployment file differs from its manifest: {name}")
            inventory[name] = dict(size=size, sha256=expected["sha256"])
    else:
        def git(*args):
            return subprocess.check_output(["git", "-C", str(source), *args]).decode("utf-8").strip()
        if Path(git("rev-parse", "--show-toplevel")).resolve() != source:
            raise ValueError("Source directory must be the repository root")
        if git("status", "--porcelain", "--untracked-files=no"):
            raise ValueError("Commit tracked source changes before freezing an SCC run")
        commit = git("rev-parse", "HEAD")
        names = subprocess.check_output(["git", "-C", str(source), "ls-files", "-z"]).decode("utf-8").split("\0")
        inventory = {name: dict(size=safe_source(source, name).stat().st_size,
                                sha256=sha256(safe_source(source, name))) for name in names if name}
    if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise ValueError("Source commit must be a full hexadecimal Git object ID")
    if not inventory or "scripts/scc_simulation.py" not in inventory:
        raise ValueError("Deployment inventory must include the SCC execution tool")
    return commit, inventory


def validate_resources(resources):
    for field in ("workers", "concurrency", "cores", "mem_per_core_gb", "gpu_memory_gb", "wall_seconds", "cpu_slot_limit"):
        value = resources[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{field} must be a positive integer")
    if resources["workers"] > resources["cores"]:
        raise ValueError("Worker count exceeds reserved CPU cores")
    selection = resources.get("worker_selection", "benchmark")
    if selection not in ("benchmark", "fixed"):
        raise ValueError("worker_selection must be benchmark or fixed")
    if selection == "fixed" and resources["workers"] not in (1, 2, 4, 8, 16):
        raise ValueError("Fixed workers must be one of 1, 2, 4, 8, 16")
    if resources["device"] not in ("cpu", "cuda"):
        raise ValueError("SCC device must be cpu or cuda")
    if resources["gpu_pool"] not in GPU_POOLS:
        raise ValueError("Unknown SCC GPU queue pool")
    if resources["cores"] > resources["cpu_slot_limit"]:
        raise ValueError("CPU cores exceed the recorded execution slot budget")
    if resources["device"] == "cuda":
        if resources["workers"] != 1:
            raise ValueError("Safe SCC GPU execution requires one worker per GPU; GPU process sharing has not been validated")
        quota, _ = GPU_POOLS[resources["gpu_pool"]]
        if resources["concurrency"] > quota:
            raise ValueError(f"GPU concurrency exceeds the verified {resources['gpu_pool']} quota of {quota}")
        # The shared and public GPU pools have separate quota rules. A combined
        # array is also bounded by SGE's independent shared=2 and public=4 caps.
        pool_caps = ((2, 4) if resources["gpu_pool"] == "combined" else (quota,))
        if any(min(resources["concurrency"], cap)*resources["cores"] > resources["cpu_slot_limit"]
               for cap in pool_caps):
            raise ValueError("Concurrent GPU jobs exceed the recorded per-pool CPU slot budget")
        if resources["gpu_type"] != "L40S":
            raise ValueError("The verified GPU queue pools in this workflow are for L40S only")
    if not 900 <= resources["wall_seconds"] <= 43200:
        raise ValueError("Wall time must be between 15 minutes and 12 hours")
    for field in ("project", "gpu_type", "python_module"):
        if not re.fullmatch(r"[A-Za-z0-9_.+/-]+", resources[field]):
            raise ValueError(f"Invalid {field}")


def initialize(args):
    source, root = args.source_dir.resolve(), args.run_dir.resolve()
    if root == source or source in root.parents:
        raise ValueError("Run directory must be outside the mutable source checkout")
    if root.exists() and any(root.iterdir()):
        raise ValueError("Run directory must be empty; existing experiments are never overwritten")
    if not args.python.is_absolute():
        raise ValueError("Python executable must be an absolute path")
    slot_budget = args.cpu_slot_limit if args.cpu_slot_limit is not None else (16 if args.device == "cuda" else 256)
    resources = dict(project=args.project, gpu_type=args.gpu_type, device=args.device, gpu_pool=args.gpu_pool,
                     gpu_memory_gb=args.gpu_memory_gb, workers=args.workers,
                     worker_selection=getattr(args, "worker_selection", "benchmark"),
                     concurrency=(args.concurrency if args.concurrency is not None else
                                  GPU_POOLS[args.gpu_pool][0] if args.device == "cuda" else slot_budget),
                     cores=args.cores, cpu_slot_limit=slot_budget,
                     mem_per_core_gb=args.mem_per_core_gb,
                     wall_seconds=args.wall_seconds, python_module=args.python_module)
    validate_resources(resources)
    commit, inventory = source_inventory(source)
    root.mkdir(parents=True, exist_ok=True)
    for name in inventory:
        target = root / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(safe_source(source, name), target)
    plan = dict(schema_version=1, created_at=utc_now(), run_dir=str(root),
                source_dir=str(root / "source"), source_commit=commit,
                source_files=inventory, python=str(args.python), resources=resources,
                science_args=science_args(args.device), num_shards=200, target_reps=200)
    plan["plan_hash"] = object_hash(plan)
    atomic_json(root / "plan.json", plan)
    for phase in PHASES:
        path = root / "batch" / f"{phase}.sh"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(batch_script(plan, phase), encoding="utf-8", newline="\n")
    (root / "logs").mkdir()
    print(json.dumps(dict(run_dir=str(root), source_commit=commit,
                          plan_hash=plan["plan_hash"], resources=resources), indent=2))
    return 0


def load_plan(root):
    root = Path(root).resolve()
    plan = read_json(root / "plan.json")
    identity = {key: value for key, value in plan.items() if key != "plan_hash"}
    if object_hash(identity) != plan.get("plan_hash") or Path(plan["run_dir"]).resolve() != root:
        raise ValueError("Run plan was modified or moved after preparation")
    validate_resources(plan["resources"])
    if plan["science_args"] != science_args(plan["resources"]["device"]) or plan["num_shards"] != 200 or plan["target_reps"] != 200:
        raise ValueError("Unexpected scientific configuration in SCC plan")
    for name, expected in plan["source_files"].items():
        path = safe_source(plan["source_dir"], name)
        if path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Frozen source changed: {name}")
    for phase in PHASES:
        path = root / "batch" / f"{phase}.sh"
        if path.read_text(encoding="utf-8") != batch_script(plan, phase):
            raise ValueError(f"Frozen batch script changed: {phase}")
    return plan


def batch_script(plan, phase):
    q = shlex.quote
    resources = plan["resources"]
    executable = Path(plan["source_dir"]) / "scripts/scc_simulation.py"
    argv = [plan["python"], str(executable), "execute", "--run-dir", plan["run_dir"], "--phase", phase]
    seconds = resources["wall_seconds"]
    return ("#!/bin/bash -l\nset -euo pipefail\n"
            # GNU timeout must survive SGE warnings; the Python coordinators
            # install their own termination-warning handler while work runs.
            + "trap '' USR1 USR2\n"
            + f"module load {q(resources['python_module'])}\n"
            + f"export PYTHONPATH={q(str(Path(plan['source_dir']) / 'src'))}\n"
            + "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1\n"
            + "export PYTHONUNBUFFERED=1\n"
            + f"cd {q(plan['source_dir'])}\n"
            + f"exec timeout --signal=TERM --kill-after=45s {seconds-60}s "
            + " ".join(q(argument) for argument in argv) + "\n")


def sge_snapshot():
    completed = subprocess.run(["qstat", "-xml", "-u", os.environ.get("USER", "shibo")],
                               check=True, capture_output=True, text=True, timeout=30)
    return {job.findtext("JB_job_number"): dict(name=job.findtext("JB_name"),
              state=job.findtext("state", job.get("state")))
            for job in ET.fromstring(completed.stdout).iter("job_list")}


def marker(plan, name):
    path = Path(plan["run_dir"]) / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"{name} check has not completed successfully")
    value = read_json(path)
    if value.get("plan_hash") != plan["plan_hash"]:
        raise ValueError(f"{name} marker belongs to a different frozen run")
    manifest_path = Path(plan["run_dir"]) / "prepared/manifest.json"
    if value.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError(f"{name} marker does not match the prepared manifest")
    if name == "CANARY_PASSED":
        audit_path = Path(plan["run_dir"]) / "canary_audit.json"
        if value.get("audit_sha256") != sha256(audit_path):
            raise ValueError("Canary audit changed after its validation marker")
        audit = read_json(audit_path)
        if audit.get("plan_hash") != plan["plan_hash"] or not audit.get("all_records_validated"):
            raise ValueError("Canary audit is not a successful check of this plan")
        directory = Path(plan["run_dir"]) / "shards/shard-000"
        for filename, expected in audit["files"].items():
            if sha256(safe_source(directory, filename)) != expected:
                raise ValueError(f"Canary artifact changed after independent validation: {filename}")
    return value


def measured_workers(plan):
    verified = marker(plan, "BENCHMARK_PASSED")
    if verified.get("results_sha256") != sha256(Path(plan["run_dir"]) / "benchmark_results.json"):
        raise ValueError("Benchmark results changed after worker selection")
    selected = verified["selected_workers"]
    if verified.get("worker_selection", "benchmark") != plan["resources"].get("worker_selection", "benchmark"):
        raise ValueError("Benchmark worker-selection mode differs from the frozen plan")
    if selected not in benchmark_workers(plan):
        raise ValueError("Benchmark selected an unsupported worker count")
    return selected


def benchmark_workers(plan):
    # SCC may configure allocated GPUs in EXCLUSIVE_PROCESS mode. Until a
    # sharing mechanism is explicitly verified, one Python worker owns one GPU.
    resources = plan["resources"]
    candidates = ((resources["workers"],) if resources.get("worker_selection", "benchmark") == "fixed"
                  else (1,) if resources["device"] == "cuda" else (1, 2, 4, 8, 16))
    if resources["device"] == "cuda" and candidates != (1,):
        raise ValueError("Safe SCC GPU execution requires one worker per GPU")
    return [value for value in candidates if value <= plan["resources"]["cores"]]


def array_allocation(plan, workers):
    resources = plan["resources"]
    if workers not in benchmark_workers(plan):
        raise ValueError("Unsupported worker count for this SCC execution mode")
    gpu = resources["device"] == "cuda"
    cores = resources["cores"] if gpu else workers
    concurrency = (resources["concurrency"] if gpu else
                   min(resources["concurrency"], resources["cpu_slot_limit"] // cores))
    return dict(cores=cores, concurrency=min(concurrency, plan["num_shards"]))


def benchmark_metrics(plan, workers, completed, elapsed_seconds):
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ValueError("Benchmark duration must be finite and positive")
    if not isinstance(completed, int) or completed < 0:
        raise ValueError("Completed benchmark fits must be a nonnegative integer")
    allocation = array_allocation(plan, workers)
    rate = completed*3600/elapsed_seconds
    reserved_hours = plan["resources"]["cores"]*elapsed_seconds/3600
    return dict(fits_per_hour=rate, reserved_core_hours=reserved_hours,
                fits_per_core_hour=completed/reserved_hours,
                formal_cores_per_element=allocation["cores"],
                projected_formal_fits_per_core_hour=rate/allocation["cores"],
                projected_array_concurrency=allocation["concurrency"],
                projected_aggregate_fits_per_hour=rate*allocation["concurrency"],
                projection_note="Execution-budget projection; queue availability and full-setting throughput must be checked with the canary")


def process_tree_rss_mib(pid, proc_root=Path("/proc")):
    """Sample summed RSS, including worker descendants; shared pages can be counted twice."""
    seen, pending, kib = set(), [pid], 0
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            status = (proc_root / str(current) / "status").read_text()
            match = re.search(r"^VmRSS:\s+(\d+)\s+kB", status, re.MULTILINE)
            kib += int(match.group(1)) if match else 0
            children = (proc_root / str(current) / "task" / str(current) / "children").read_text()
            pending.extend(int(value) for value in children.split())
        except (OSError, ValueError):
            continue
    return kib / 1024 if seen and kib else None


def gpu_sample():
    allocated = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"(?:[0-9]+|GPU-[A-Za-z0-9-]+)", allocated):
        return None
    try:
        result = subprocess.run(["nvidia-smi", "--id", allocated,
                                 "--query-gpu=memory.used,utilization.gpu",
                                 "--format=csv,noheader,nounits"], check=True,
                                capture_output=True, text=True, timeout=5)
        memory, utilization = result.stdout.strip().split(",")
        return float(memory), float(utilization)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def profile_command(command, device):
    started = time.monotonic()
    process = None
    grouped = os.name == "posix"
    previous = {}
    interrupted = None
    stopping = False

    class ProfileInterrupted(BaseException):
        def __init__(self, signum):
            self.signum = signum

    def interrupt(signum, frame):
        if not stopping:
            raise ProfileInterrupted(signum)

    for name in ("SIGTERM", "SIGUSR2"):
        signum = getattr(signal, name, None)
        if signum is not None:
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, interrupt)
    peak_rss, peak_gpu, utilization = 0.0, 0.0, []
    try:
        # An owned session allows bounded cleanup of exactly this benchmark's
        # child tree, including grandchildren, without touching other jobs.
        process = subprocess.Popen(command, start_new_session=grouped)
        while process.poll() is None:
            rss = process_tree_rss_mib(process.pid)
            peak_rss = max(peak_rss, rss or 0)
            sample = gpu_sample() if device == "cuda" else None
            if sample:
                peak_gpu = max(peak_gpu, sample[0])
                utilization.append(sample[1])
            time.sleep(2)
    except ProfileInterrupted as exc:
        interrupted = signal.Signals(exc.signum).name
    finally:
        stopping = True
        try:
            if process is not None:
                stop_profile_process(process, grouped)
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
    return dict(exit_code=(4 if interrupted else process.returncode),
                interrupted_signal=interrupted, elapsed_seconds=time.monotonic()-started,
                peak_process_tree_rss_mib=peak_rss or None,
                peak_gpu_memory_mib=peak_gpu or None,
                mean_gpu_utilization_percent=(sum(utilization)/len(utilization) if utilization else None),
                profiling_note="RSS sums sampled process-tree RSS and can double-count shared pages; GPU metrics sample only the scheduler-assigned GPU")


def stop_profile_process(process, grouped, grace_seconds=15):
    """Terminate only the session spawned for this benchmark and bound cleanup."""
    def send(signum):
        try:
            if grouped:
                os.killpg(process.pid, signum)
            elif process.poll() is None:
                if signum == signal.SIGTERM:
                    process.terminate()
                else:
                    process.kill()
        except ProcessLookupError:
            pass
    if process.poll() is None:
        send(signal.SIGTERM)
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            send(getattr(signal, "SIGKILL", signal.SIGTERM))
            process.wait(timeout=5)
    if grouped:
        # The coordinator can exit while an interrupted worker is still alive.
        # These remaining processes belong to its private session only.
        send(getattr(signal, "SIGKILL", signal.SIGTERM))


def qsub_argv(plan, phase, job_name, dependency=None, retry_failed=False):
    resources = plan["resources"]
    hours, remainder = divmod(resources["wall_seconds"], 3600)
    minutes, seconds = divmod(remainder, 60)
    gpu = phase != "merge" and resources["device"] == "cuda"
    allocation = (array_allocation(plan, 1 if resources["device"] == "cuda" else measured_workers(plan))
                  if phase in ("canary", "array") else None)
    cores = (1 if phase == "merge" else allocation["cores"] if allocation else resources["cores"])
    argv = ["qsub", "-terse", "-sync", "n", "-r", "n", "-notify", "-P", resources["project"],
            "-N", job_name, "-j", "y", "-m", "n", "-o", str(Path(plan["run_dir"]) / "logs"),
            "-l", f"h_rt={hours:02d}:{minutes:02d}:{seconds:02d}",
            "-pe", "omp", str(cores),
            "-l", f"mem_per_core={resources['mem_per_core_gb']}G"]
    if gpu:
        argv += ["-q", GPU_POOLS[resources["gpu_pool"]][1]]
        argv += ["-l", f"gpus=1,gpu_type={resources['gpu_type']},gpu_memory={resources['gpu_memory_gb']}G"]
    if phase == "array":
        argv += ["-t", "1-200", "-tc", str(allocation["concurrency"])]
    if retry_failed:
        if phase not in ("canary", "array"):
            raise ValueError("Failed-fit retries apply only to canary or array execution")
        argv += ["-v", "MEDIENC_SCC_RETRY_FAILED=1"]
    if dependency:
        argv += ["-hold_jid", dependency]
    return argv + [str(Path(plan["run_dir"]) / "batch" / f"{phase}.sh")]


def submit(root, phase, dry_run=False, resume=False, retry_failed=False):
    from filelock import FileLock
    plan = load_plan(root)
    if phase in ("canary", "array", "merge"):
        marker(plan, "READY")
    if phase in ("canary", "array"):
        measured_workers(plan)
    if phase == "array":
        marker(plan, "CANARY_PASSED")
    path = Path(root) / "submissions.json"
    with FileLock(str(Path(root) / ".submit.lock"), timeout=0):
        state = read_json(path) if path.exists() else {"plan_hash": plan["plan_hash"], "jobs": []}
        if state.get("plan_hash") != plan["plan_hash"]:
            raise ValueError("Submission state belongs to another plan")
        live = sge_snapshot()
        # Unknown outcomes are not silently retried: the old job may already have run.
        for entry in state["jobs"]:
            if entry.get("status") in ("submitting", "ambiguous"):
                matches = [job for job in live.values() if job["name"] == entry["name"]]
                raise ValueError(f"Submission outcome needs reconciliation: {entry['name']}; live matches={len(matches)}")
            if entry["phase"] == phase and entry.get("job_id") in live:
                raise ValueError(f"This phase already has a live SCC job: {entry['job_id']}")
            if phase != "merge" and entry["phase"] != "merge" and entry.get("job_id") in live:
                raise ValueError(f"A compute phase already owns this frozen run: {entry['job_id']}")
        previous = [entry for entry in state["jobs"] if entry["phase"] == phase]
        if previous and not resume:
            raise ValueError("This phase was already submitted. Use --resume after checking its results")
        dependency = None
        if phase == "merge":
            arrays = [entry for entry in state["jobs"] if entry["phase"] == "array" and entry.get("job_id") in live]
            dependency = arrays[-1]["job_id"] if arrays else None
        name = f"me_{phase}_{plan['plan_hash'][:8]}_{len(state['jobs'])+1}"
        argv = qsub_argv(plan, phase, name, dependency, retry_failed=retry_failed)
        if dry_run:
            print(shlex.join(argv))
            return 0
        entry = dict(phase=phase, name=name, submitted_at=utc_now(), status="submitting", argv=argv,
                     retry_failed=retry_failed)
        state["jobs"].append(entry)
        atomic_json(path, state)
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=60)
            entry.update(stdout=result.stdout, stderr=result.stderr, qsub_exit_code=result.returncode)
            if result.returncode != 0:
                # An ordinary rejected submission creates no job. Do not suppress the rejection.
                entry["status"] = "rejected"
                atomic_json(path, state)
                raise RuntimeError(f"qsub rejected the submission: {result.stderr.strip()}")
            match = re.fullmatch(r"\s*(\d+)(?:\.[0-9:-]+)?\s*", result.stdout)
            if match is None:
                entry["status"] = "ambiguous"
                atomic_json(path, state)
                raise RuntimeError("Cannot parse qsub job ID; refusing automatic duplicate submission")
            entry.update(status="submitted", job_id=match.group(1))
            atomic_json(path, state)
            print(json.dumps(entry, indent=2))
            return 0
        except subprocess.TimeoutExpired:
            entry["status"] = "ambiguous"
            atomic_json(path, state)
            raise RuntimeError("qsub timed out; verify the recorded job name before any resubmission")


def runner_module(plan):
    sys.path.insert(0, str(Path(plan["source_dir"]) / "src"))
    from mediencoder.simulation import runner
    return runner


def prepared_copy(plan, destination):
    marker(plan, "READY")
    prepared = Path(plan["run_dir"]) / "prepared"
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "mechanism.json", "mechanism.npz"):
        target, source = destination / name, prepared / name
        if target.exists():
            if sha256(target) != sha256(source):
                raise ValueError(f"Shard prepared artifact changed: {name}")
        else:
            shutil.copy2(source, target)


def write_marker(plan, name, **extra):
    atomic_json(Path(plan["run_dir"]) / f"{name}.json", dict(
        plan_hash=plan["plan_hash"], manifest_sha256=sha256(Path(plan["run_dir"]) / "prepared/manifest.json"),
        verified_at=utc_now(), **extra))


def cloud_module(plan):
    path = Path(plan["source_dir"]) / "scripts/cloud_simulation.py"
    spec = importlib.util.spec_from_file_location("scc_cloud_merge", path)
    module = importlib.util.module_from_spec(spec)
    runner_module(plan)
    spec.loader.exec_module(module)
    return module


def audit_canary(plan, destination, runner):
    """Independently recompute every canary interval and check seeds/folds/pairing."""
    from mediencoder.simulation.dgp import load_mechanism
    root, destination = Path(plan["run_dir"]), Path(destination)
    cloud = cloud_module(plan)
    manifest = read_json(root / "prepared/manifest.json")
    runner.validate_manifest(manifest, read_json(destination / "manifest.json"))
    if runner.collect_code_hashes() != manifest["code_hashes"]:
        raise ValueError("Canary code fingerprint differs from the prepared source")
    mechanism = load_mechanism(root / "prepared/mechanism")
    if mechanism.mechanism_hash != manifest["mechanism_hash"]:
        raise ValueError("Canary fixed mechanism differs from the prepared manifest")
    expected = {task["task_id"]: task for task in
                runner.build_tasks(runner.execution_config(manifest["config"], 1))}
    paths = {path.stem: path for path in (destination / "tasks").glob("*.json")}
    if set(paths) != set(expected):
        raise ValueError("Canary task coverage differs from the complete paired replication")
    paired, files = {}, {}
    for key, task in expected.items():
        record = read_json(paths[key])
        if record.get("status") != "complete" or any(record.get(name) != value for name, value in task.items()):
            raise ValueError(f"Canary task status, original seeds, or metadata differ: {key}")
        if record.get("mechanism_hash") != mechanism.mechanism_hash:
            raise ValueError(f"Canary record uses a different fixed mechanism: {key}")
        if record.get("theta_population") != float(mechanism.truth.value):
            raise ValueError(f"Canary population truth differs from the fixed mechanism: {key}")
        data_hash = record.get("observed_data_sha256")
        pair = task["n"], task["rep"]
        if not data_hash or (pair in paired and paired[pair] != data_hash):
            raise ValueError(f"Canary methods do not use identical paired observed data: {key}")
        paired[pair] = data_hash
        relative = record["score_artifact"].replace("\\", "/")
        if relative != f"scores/{key}.npz":
            raise ValueError(f"Canary score artifact does not belong to its task: {key}")
        cloud.safe_file(destination, relative)
        runner.validate_checkpoint(record, manifest, destination)
        cloud.validate_scores(record, destination, mechanism.truth.value)
        files[f"tasks/{key}.json"] = sha256(paths[key])
        files[relative] = sha256(destination / relative)
    audit = dict(plan_hash=plan["plan_hash"], run_hash=manifest["run_hash"],
                 verified_at=utc_now(), verified_tasks=len(expected), paired_datasets=len(paired),
                 all_records_validated=True, files=files)
    atomic_json(root / "canary_audit.json", audit)
    return audit


def execute(root, phase):
    plan = load_plan(root)
    if not os.environ.get("JOB_ID"):
        raise ValueError("Training/preparation must run as an allocated SCC batch job")
    runner = runner_module(plan)
    root = Path(root).resolve()
    limit = plan["resources"]["wall_seconds"] - 600
    args = list(plan["science_args"]) + ["--workers", str(plan["resources"]["workers"]),
                                         "--tasks-per-worker", "1"]
    if phase == "prepare":
        code = runner.main(args + ["--output-dir", str(root / "prepared"), "--summarize-only"])
        if code != 0:
            return code
        manifest = read_json(root / "prepared/manifest.json")
        if (plan["resources"]["device"] == "cuda"
                and plan["resources"]["gpu_type"].lower() not in manifest["environment"]["gpu"].lower()):
            raise ValueError("Allocated GPU differs from the frozen SCC resource request")
        write_marker(plan, "READY", run_hash=manifest["run_hash"])
        return 0
    if phase == "benchmark":
        marker(plan, "READY")
        rows = []
        round_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        deadline = time.monotonic() + limit
        candidates = benchmark_workers(plan)
        if not candidates:
            raise ValueError("No supported benchmark worker count fits the CPU allocation")
        # The same balanced task mix is timed at every worker count, and the
        # largest pool gets at least two waves of work rather than idle slots.
        repetitions = (2*max(candidates)+4)//5
        expected_tasks = repetitions*5
        for workers in candidates:
            destination = root / "benchmarks" / round_name / f"workers-{workers}"
            benchmark_args = list(plan["science_args"])
            for option, value in (("--n", "3000"), ("--reps", str(repetitions))):
                benchmark_args[benchmark_args.index(option)+1] = value
            remaining = int(deadline-time.monotonic())
            if remaining < 60:
                return 124
            command = [plan["python"], str(Path(plan["source_dir"]) / "scripts/run_simulation.py"),
                       *benchmark_args, "--output-dir", str(destination), "--workers", str(workers),
                       "--tasks-per-worker", "1", "--max-tasks", str(expected_tasks),
                       "--pilot-epochs", "20", "--time-limit-seconds", str(remaining),
                       "--mechanism-from", str(root / "prepared/mechanism")]
            profile = profile_command(command, plan["resources"]["device"])
            status = read_json(destination / "status.json") if (destination / "status.json").exists() else {}
            completed = sum(read_json(path).get("status") == "complete"
                            for path in (destination / "tasks").glob("*.json"))
            rows.append(dict(workers=workers, **profile, epoch_cap=20, benchmark_only=True,
                             expected_tasks=expected_tasks, completed=completed,
                             **benchmark_metrics(plan, workers, completed, profile["elapsed_seconds"]),
                             status=status, command=command, output=str(destination)))
            atomic_json(root / "benchmark_results.json", dict(plan_hash=plan["plan_hash"], rows=rows))
            if profile["exit_code"] != 0 or completed != expected_tasks:
                return profile["exit_code"] or 2
        best = max(rows, key=lambda row: row["projected_aggregate_fits_per_hour"])
        write_marker(plan, "BENCHMARK_PASSED", selected_workers=best["workers"],
                     worker_selection=plan["resources"].get("worker_selection", "benchmark"),
                     results_sha256=sha256(root / "benchmark_results.json"),
                     fits_per_hour=best["fits_per_hour"],
                     projected_aggregate_fits_per_hour=best["projected_aggregate_fits_per_hour"],
                     projected_array_concurrency=best["projected_array_concurrency"],
                     selection_criterion=("prespecified_workers"
                         if plan["resources"].get("worker_selection", "benchmark") == "fixed"
                         else "projected_aggregate_fits_per_hour"))
        return 0
    if phase in ("canary", "array"):
        marker(plan, "READY")
        workers = measured_workers(plan)
        args[args.index("--workers")+1] = str(workers)
        if phase == "array":
            marker(plan, "CANARY_PASSED")
            index = int(os.environ.get("SGE_TASK_ID", "0"))-1
            if not 0 <= index < plan["num_shards"]:
                raise ValueError("Invalid scheduler array task index")
        else:
            index = 0
        destination = root / "shards" / f"shard-{index:03d}"
        prepared_copy(plan, destination)
        target = 1 if phase == "canary" else plan["target_reps"]
        retry_flag = os.environ.get("MEDIENC_SCC_RETRY_FAILED", "0")
        if retry_flag not in ("0", "1"):
            raise ValueError("Invalid failed-fit retry flag")
        retry_args = ["--retry-failed"] if retry_flag == "1" else []
        code = runner.main(args + ["--output-dir", str(destination), "--target-reps", str(target),
                                  "--num-shards", str(plan["num_shards"]), "--shard-index", str(index),
                                  "--time-limit-seconds", str(limit)] + retry_args)
        if phase == "canary":
            status = read_json(destination / "status.json")
            if code or status.get("shard_pending") or status.get("shard_failed"):
                return code or 2
            if status.get("shard_completed") != 30:
                raise ValueError("Canary must contain all 30 valid paired fits for replication zero")
            audit = audit_canary(plan, destination, runner)
            if audit["verified_tasks"] != 30:
                raise ValueError("Independent canary validation did not verify all 30 fits")
            write_marker(plan, "CANARY_PASSED", verified_tasks=30,
                         audit_sha256=sha256(root / "canary_audit.json"))
        return code
    return merge(root)


def merge(root):
    plan = load_plan(root)
    marker(plan, "READY")
    marker(plan, "CANARY_PASSED")
    module = cloud_module(plan)
    root = Path(root)
    # Preserve every preceding partial merge; never overwrite its failure evidence.
    destination = root / "merged" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    code = module.merge_artifacts(root / "prepared", root / "shards", destination,
                                 plan["target_reps"], plan["num_shards"], expected_device=plan["resources"]["device"],
                                 require_inventory=False, execution_platform="scc")
    atomic_json(root / "latest_merge.json", dict(plan_hash=plan["plan_hash"], output=str(destination), exit_code=code))
    if code == 0:
        write_marker(plan, "COMPLETED", output=str(destination))
    return code


def status(root):
    plan = load_plan(root)
    root = Path(root)
    state = read_json(root / "submissions.json") if (root / "submissions.json").exists() else {"jobs": []}
    live = sge_snapshot()
    own = {entry.get("job_id"): live[entry["job_id"]] for entry in state["jobs"] if entry.get("job_id") in live}
    complete = failed = 0
    for path in (root / "shards").glob("shard-*/tasks/*.json"):
        record = read_json(path)
        complete += record.get("status") == "complete"
        failed += record.get("status") == "failed"
    print(json.dumps(dict(plan_hash=plan["plan_hash"], source_commit=plan["source_commit"],
                          expected_tasks=6000, completed=complete, failed=failed,
                          pending=6000-complete-failed, own_live_jobs=own,
                          ready=(root / "READY.json").exists(),
                          benchmark_passed=(root / "BENCHMARK_PASSED.json").exists(),
                          canary_passed=(root / "CANARY_PASSED.json").exists()), indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--source-dir", type=Path, required=True)
    init.add_argument("--run-dir", type=Path, required=True)
    init.add_argument("--python", type=Path, required=True)
    init.add_argument("--python-module", default="python3/3.12.4")
    init.add_argument("--project", default="adsgrp")
    init.add_argument("--gpu-type", default="L40S")
    init.add_argument("--gpu-pool", choices=tuple(GPU_POOLS), default="shared")
    init.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    init.add_argument("--gpu-memory-gb", type=int, default=24)
    init.add_argument("--workers", type=int, default=1)
    init.add_argument("--worker-selection", choices=("benchmark", "fixed"), default="benchmark",
                      help="Select workers using the benchmark (default), or pin --workers while retaining real benchmark and full-setting canary gates")
    init.add_argument("--concurrency", type=int, default=None)
    init.add_argument("--cores", type=int, default=4)
    init.add_argument("--cpu-slot-budget", "--cpu-slot-limit", dest="cpu_slot_limit", type=int, default=None,
                      help="Operator execution budget, not a claim about a cluster-wide CPU quota; defaults CPU=256 or CUDA=16 per pool")
    init.add_argument("--mem-per-core-gb", type=int, default=4)
    init.add_argument("--wall-seconds", type=int, default=43200)
    for command in ("submit", "execute", "status", "merge"):
        sub = commands.add_parser(command)
        sub.add_argument("--run-dir", type=Path, required=True)
        if command in ("submit", "execute"):
            sub.add_argument("--phase", choices=PHASES, required=True)
        if command == "submit":
            sub.add_argument("--dry-run", action="store_true")
            sub.add_argument("--resume", action="store_true")
            sub.add_argument("--retry-failed", action="store_true",
                             help="Explicit same-seed retries; the runner preserves every prior failed attempt")
    args = parser.parse_args(argv)
    if args.command == "init":
        return initialize(args)
    if args.command == "submit":
        return submit(args.run_dir.resolve(), args.phase, args.dry_run, args.resume, args.retry_failed)
    if args.command == "execute":
        return execute(args.run_dir.resolve(), args.phase)
    return (status if args.command == "status" else merge)(args.run_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
