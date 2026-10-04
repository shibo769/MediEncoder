"""Package and validate synthetic-only simulation artifacts; never train a model."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path, PurePosixPath
import shutil

import numpy as np

from mediencoder.simulation import runner


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_file(root, relative):
    text = str(relative).replace("\\", "/")
    relative = PurePosixPath(text)
    if relative.is_absolute() or ".." in relative.parts or ":" in text or not relative.parts:
        raise ValueError(f"Unsafe artifact path: {text}")
    path = Path(root).joinpath(*relative.parts)
    if path.is_symlink() or not path.is_file() or Path(root).resolve() not in path.resolve().parents:
        raise ValueError(f"Missing or unsafe artifact: {text}")
    return path


def package_artifact(source, destination, kind, byte_limit):
    source, destination = Path(source), Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Artifact destination must be empty")
    files = ["manifest.json"]
    if kind in ("prepared", "merged"):
        files += ["mechanism.json", "mechanism.npz"]
    if kind in ("shard", "merged"):
        files += [str(p.relative_to(source)).replace("\\", "/")
                  for folder, pattern in (("tasks", "*.json"), ("scores", "*.npz"))
                  for p in sorted((source / folder).glob(pattern)) if p.is_file()]
        allowed = ("execution_plan.json", "status.json", "summary.csv", "main_table.tex",
                   "ablation_table.tex", "progress.html", "merge_audit.json", "merge_audit.md")
        files += [name for name in allowed if (source / name).is_file()]
    records = {}
    total = 0
    for name in files:
        path = safe_file(source, name)
        total += path.stat().st_size
        records[name] = {"bytes": path.stat().st_size, "sha256": runner.file_hash(path)}
    if total > byte_limit:
        raise ValueError(f"Artifact exceeds byte ceiling: {total} > {byte_limit}; refusing upload bundle")
    destination.mkdir(parents=True, exist_ok=True)
    for name in files:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(safe_file(source, name), target)
    runner.atomic_json(destination / "artifact_inventory.json", dict(kind=kind, bytes=total, files=records))
    print(f"Prepared {kind} artifact: {len(files)} files, {total:,} bytes")


def verify_inventory(directory, expected_kind):
    directory = Path(directory)
    inventory = _read(directory / "artifact_inventory.json")
    if inventory.get("kind") != expected_kind:
        raise ValueError("Artifact kind differs from its expected role")
    total = 0
    for name, expected in inventory["files"].items():
        path = safe_file(directory, name)
        if path.stat().st_size != expected["bytes"] or runner.file_hash(path) != expected["sha256"]:
            raise ValueError(f"Artifact inventory mismatch: {name}")
        total += path.stat().st_size
    if total != inventory["bytes"]:
        raise ValueError("Artifact byte count mismatch")
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != set(inventory["files"]) | {"artifact_inventory.json"}:
        raise ValueError("Artifact contains unlisted or missing files")
    return inventory


def validate_scores(record, directory, truth):
    """Verify numerical inference and original-index coverage, not just hashes."""
    n = record["n"]
    path = safe_file(directory, record["score_artifact"])
    with np.load(path, allow_pickle=False) as saved:
        scores = saved["crossfit_scores"]
        if scores.shape != (n,) or not np.isfinite(scores).all():
            raise ValueError("Scores must contain one finite value per original observation")
        if not np.array_equal(saved["subject_index"], np.arange(n)):
            raise ValueError("Saved subject indices do not cover the original observations")
        est = []
        roles = ("representation_train", "representation_validation", "nuisance", "estimation")
        for fold in range(4):
            indices = [saved[f"fold{fold}_{role}"] for role in roles]
            if any(a.ndim != 1 or not np.issubdtype(a.dtype, np.integer) for a in indices):
                raise ValueError("Fold indices must be integer vectors")
            if not np.array_equal(np.sort(np.concatenate(indices)), np.arange(n)):
                raise ValueError("Each fold's roles must partition all observations")
            for role in ("nuisance_train", "nuisance_validation"):
                if f"fold{fold}_{role}" not in saved:
                    raise ValueError("Missing nuisance fitting/validation indices")
            nuisance_parts = np.concatenate([saved[f"fold{fold}_nuisance_train"], saved[f"fold{fold}_nuisance_validation"]])
            if not np.array_equal(np.sort(nuisance_parts), np.sort(indices[2])):
                raise ValueError("Nuisance training/validation must partition the nuisance fold")
            est.append(indices[-1])
        if not np.array_equal(np.sort(np.concatenate(est)), np.arange(n)):
            raise ValueError("Estimation folds must cover every observation exactly once")
        mean, se = float(scores.mean()), float(scores.std(ddof=1)/np.sqrt(n))
    expected = dict(theta_hat=mean, theta_population=truth, error=mean-truth, se_IF=se,
                    ci_lower=mean-1.959963984540054*se, ci_upper=mean+1.959963984540054*se,
                    ci_length=2*1.959963984540054*se)
    for key, value in expected.items():
        actual = record.get(key)
        if not isinstance(actual, (int, float)) or not math.isfinite(actual) or not math.isclose(actual, value, rel_tol=1e-9, abs_tol=1e-10):
            raise ValueError(f"Record {key} differs from independently recomputed score inference")
    if record.get("covered") != (expected["ci_lower"] <= truth <= expected["ci_upper"]):
        raise ValueError("Record coverage differs from its own dataset-specific interval")


def merge_artifacts(prepared, shards, output, target_reps, num_shards):
    from mediencoder.simulation.dgp import load_mechanism
    from mediencoder.simulation.monitor import render_report
    prepared, shards, output = Path(prepared), Path(shards), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Merged output must be empty")
    verify_inventory(prepared, "prepared")
    manifest = _read(prepared / "manifest.json")
    identity = {key: manifest[key] for key in ("config", "code_hashes", "mechanism_hash", "environment")}
    if runner.digest(identity) != manifest["run_hash"] or runner.collect_code_hashes() != manifest["code_hashes"]:
        raise ValueError("Prepared scientific/source fingerprint is invalid or differs from this checkout")
    if manifest["config"]["device"] != "cpu":
        raise ValueError("Cloud aggregation only accepts its separate CPU experiment")
    mechanism = load_mechanism(prepared / "mechanism")
    if mechanism.mechanism_hash != manifest["mechanism_hash"]:
        raise ValueError("Prepared mechanism differs from the manifest")
    phase = runner.execution_config(manifest["config"], target_reps)
    expected = {task["task_id"]: task for task in runner.build_tasks(phase)}
    records, locations, seen_shards, paired = {}, {}, set(), {}
    shard_directories = sorted(p for p in shards.iterdir() if p.is_dir()) if shards.is_dir() else []
    for shard_dir in shard_directories:
        verify_inventory(shard_dir, "shard")
        runner.validate_manifest(manifest, _read(shard_dir / "manifest.json"))
        plan = _read(shard_dir / "execution_plan.json")
        index = plan["shard_index"]
        if (plan["num_shards"] != num_shards or plan["execution_target_reps"] != target_reps
                or plan["run_hash"] != manifest["run_hash"] or not 0 <= index < num_shards
                or index in seen_shards):
            raise ValueError("Inconsistent or duplicate shard execution plan")
        seen_shards.add(index)
        for path in sorted((shard_dir / "tasks").glob("*.json")):
            record = _read(path)
            key = record["task_id"]
            if key not in expected or key in records or record["rep"] % num_shards != index:
                raise ValueError("Unexpected, duplicate, or wrongly assigned task record")
            if any(record.get(k) != v for k, v in expected[key].items()):
                raise ValueError("Task seeds or metadata differ from the reserved experiment")
            if record.get("mechanism_hash") != mechanism.mechanism_hash:
                raise ValueError("Task mechanism hash differs from the shared mechanism")
            if record.get("theta_population") != float(mechanism.truth.value):
                raise ValueError("Task truth differs from the fixed population truth")
            data_hash = record.get("observed_data_sha256")
            if record["status"] == "complete" and not data_hash:
                raise ValueError("Completed fit is missing the observed-data pairing hash")
            if data_hash:
                pair = record["n"], record["rep"]
                if pair in paired and paired[pair] != data_hash:
                    raise ValueError("Methods were not fitted on identical paired observed data")
                paired[pair] = data_hash
            # Check the relative path before the generic checkpoint validator opens it.
            if record["status"] == "complete":
                if record["score_artifact"].replace("\\", "/") != f"scores/{key}.npz":
                    raise ValueError("Score artifact does not belong to its task")
                safe_file(shard_dir, record["score_artifact"])
            runner.validate_checkpoint(record, manifest, shard_dir)
            if record["status"] == "complete":
                validate_scores(record, shard_dir, mechanism.truth.value)
            records[key], locations[key] = record, path
    output.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "mechanism.json", "mechanism.npz"):
        shutil.copy2(prepared / name, output / name)
    for key, record in records.items():
        destination = output / "tasks" / (key + ".json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(locations[key], destination)
        if record["status"] == "complete":
            source_dir = locations[key].parent.parent
            relative = record["score_artifact"].replace("\\", "/")
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(safe_file(source_dir, relative), destination)
    missing = sorted(set(expected)-set(records))
    failed = sorted(key for key, record in records.items() if record["status"] == "failed")
    complete = not missing and len(seen_shards) == num_shards
    state = "finished" if complete and not failed else "complete_with_failures" if complete else "incomplete_cloud_run"
    runner.write_reports(output, records, phase, runner.utc_now(), state, 0)
    render_report(output)
    audit = dict(run_hash=manifest["run_hash"], reserved_reps=manifest["config"]["B_requested"],
                 target_reps=target_reps, expected_tasks=len(expected), retained_tasks=len(records),
                 completed=len(records)-len(failed), failed_tasks=failed, missing_tasks=missing,
                 shard_count=len(seen_shards), num_shards=num_shards, complete=complete,
                 all_records_validated=True, combined_with_local_gpu=False)
    runner.atomic_json(output / "merge_audit.json", audit)
    report = (f"# Synthetic CPU simulation\n\nScientific run: `{manifest['run_hash']}`\n\n"
              f"Target {target_reps}; reserved {manifest['config']['B_requested']} replications per size and arm.\n\n"
              f"Validated tasks: {len(records)}/{len(expected)}; failed fits: {len(failed)}; missing: {len(missing)}. "
              f"Shards: {len(seen_shards)}/{num_shards}.\n\n"
              "The fixed mechanism, source/run fingerprints, task seeds, score checksums, fold coverage, "
              "point estimates, standard errors and dataset-specific intervals were checked. "
              "Failed fits remain explicit. Partial results are not final paper tables. "
              "No local GPU results or real datasets were merged.\n")
    runner.atomic_text(output / "merge_audit.md", report)
    print(report)
    return 0 if complete and not failed else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bundle = commands.add_parser("package")
    bundle.add_argument("--source", type=Path, required=True)
    bundle.add_argument("--destination", type=Path, required=True)
    bundle.add_argument("--kind", choices=("prepared", "shard", "merged"), required=True)
    bundle.add_argument("--byte-limit", type=int, required=True)
    merge = commands.add_parser("merge")
    merge.add_argument("--prepared", type=Path, required=True)
    merge.add_argument("--shards", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--target-reps", type=int, required=True)
    merge.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "package":
        if args.byte_limit < 1:
            parser.error("byte-limit must be positive")
        package_artifact(args.source, args.destination, args.kind, args.byte_limit)
        return 0
    return merge_artifacts(args.prepared, args.shards, args.output, args.target_reps, args.num_shards)


if __name__ == "__main__":
    raise SystemExit(main())
