"""Require exact reproduction of every archived synthetic dataset before fitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from mediencoder.simulation import runner


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _sha256(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def validate_reference(reference):
    """Validate the entire declared n-by-replication inventory before sampling."""
    if not isinstance(reference, dict):
        raise ValueError("Reference must be a JSON object")
    _sha256(reference.get("mechanism_hash"), "mechanism_hash")
    seed_base = _integer(reference.get("seed_base"), "seed_base")
    replications = _integer(reference.get("replications"), "replications", 1)
    sizes = reference.get("n_values")
    if not isinstance(sizes, list) or not sizes:
        raise ValueError("n_values must be a nonempty list")
    for n in sizes:
        _integer(n, "sample size", 1)
    if len(set(sizes)) != len(sizes):
        raise ValueError("n_values must contain unique sample sizes")
    source_run = reference.get("source_run_id")
    if isinstance(source_run, bool) or not str(source_run).isdigit() or int(source_run) < 1:
        raise ValueError("source_run_id must identify a positive numeric archived run")
    records = reference.get("records")
    expected = {(n, rep) for n in sizes for rep in range(replications)}
    if not isinstance(records, list) or len(records) != len(expected):
        raise ValueError(f"Reference inventory must contain all {len(expected)} n/rep pairs exactly once")
    indexed = {}
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Every reference record must be a JSON object")
        n = _integer(row.get("n"), "record n", 1)
        rep = _integer(row.get("rep"), "record rep")
        pair = (n, rep)
        if pair not in expected:
            raise ValueError(f"Unexpected reference n/rep pair: {pair}")
        if pair in indexed:
            raise ValueError(f"Duplicate reference n/rep pair: {pair}")
        data_seed = _integer(row.get("data_seed"), "data_seed")
        training_seed = _integer(row.get("training_seed"), "training_seed")
        if (data_seed, training_seed) != runner.task_seeds(seed_base, n, rep):
            raise ValueError(f"Reference seeds do not match the declared seed base: n={n}, rep={rep}")
        _sha256(row.get("observed_data_sha256"), "observed_data_sha256")
        indexed[pair] = row
    if set(indexed) != expected:
        raise ValueError("Reference inventory omits declared n/rep pairs")
    return [indexed[(n, rep)] for rep in range(replications) for n in sizes]


def verify_fixture_data(mechanism_path, reference_path):
    # Set BLAS thread counts before importing the generator/NumPy in CLI use.
    # Importing runner itself is lightweight and does not import or fit Torch.
    runner.configure_environment("cpu")
    from mediencoder.simulation import dgp
    reference_path = Path(reference_path)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    rows = validate_reference(reference)
    prefix = runner.mechanism_prefix(mechanism_path)
    mechanism = dgp.load_mechanism(prefix)
    if mechanism.mechanism_hash != reference["mechanism_hash"]:
        raise ValueError("Saved mechanism hash differs from the archived dataset reference")
    failures = []
    for row in rows:
        data = dgp.sample_data(mechanism, row["n"], row["data_seed"])
        actual_hash = runner.observed_data_hash(data)
        if actual_hash != row["observed_data_sha256"]:
            failures.append(dict(n=row["n"], rep=row["rep"], reason="observed_data_hash_mismatch",
                                 expected_sha256=row["observed_data_sha256"], actual_sha256=actual_hash))
        # Keep memory bounded to one generated dataset rather than all 300.
        del data
    return dict(status="passed" if not failures else "failed", exact_match=not failures,
                source_run_id=reference["source_run_id"], mechanism_hash=mechanism.mechanism_hash,
                reference_sha256=runner.file_hash(reference_path),
                mechanism_artifact_sha256=runner.mechanism_artifact_hashes(prefix),
                seed_base=reference["seed_base"], n_values=reference["n_values"],
                replications=reference["replications"], expected_records=len(rows),
                checked_records=len(rows), matched_records=len(rows) - len(failures),
                failures=failures, numerical_tolerance=0, models_fitted=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("pairing-audit.json"),
                        help="Audit JSON, separate from the fresh prepared-run directory")
    args = parser.parse_args(argv)
    prefix = runner.mechanism_prefix(args.mechanism)
    protected = {args.reference.resolve(), Path(str(prefix) + ".json").resolve(),
                 Path(str(prefix) + ".npz").resolve()}
    if args.output.resolve() in protected:
        parser.error("--output must not overwrite a mechanism or reference artifact")
    try:
        audit = verify_fixture_data(args.mechanism, args.reference)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        audit = dict(status="failed", exact_match=False, error_type=type(exc).__name__,
                     error=str(exc), models_fitted=0)
    runner.atomic_json(args.output, audit)
    summary = dict(audit, audit_path=str(args.output.resolve()))
    if "failures" in summary:
        summary["failures"] = [{key: row[key] for key in ("n", "rep", "reason")}
                               for row in summary["failures"]]
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
    return 0 if audit["exact_match"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
