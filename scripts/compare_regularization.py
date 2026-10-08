"""Read-only audit of historical WD=0/B50 and WD=.01/B100 artifacts.

The nonzero value here validates archived scientific identities; it is not a
training default. Current code and workflow defaults use weight decay zero.

No model or scientific package is imported. Exit 0 means all planned records
and comparisons are complete; 2 means missing/failed fits; 1 means invalid input.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys

ARMS = ("mediencoder", "mediencoder_l3zero")
BASELINES = ("projection", "autoencoder", "vae")
SOURCES = {"simulation/runner.py", "simulation/dgp.py", "estimation.py",
           "training.py", "models.py", "nn_utils.py"}
PAIR_FIELDS = ("data_seed", "training_seed", "observed_data_sha256")
Z = 1.959963984540054


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_file(root, name):
    relative = PurePosixPath(str(name).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or ":" in str(name):
        raise ValueError(f"Unsafe artifact path: {name}")
    path = root.joinpath(*relative.parts)
    if not path.is_file() or path.is_symlink() or root.resolve() not in path.resolve().parents:
        raise ValueError(f"Missing or unsafe artifact: {name}")
    return path


def verify_inventory(root):
    inventory = read_json(root / "artifact_inventory.json")
    if inventory.get("kind") != "merged":
        raise ValueError("Inputs must be merged artifacts")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(inventory["files"]) | {"artifact_inventory.json"}:
        raise ValueError("Artifact has unlisted or missing files")
    total = 0
    for name, entry in inventory["files"].items():
        path = safe_file(root, name)
        total += path.stat().st_size
        if path.stat().st_size != entry["bytes"] or file_hash(path) != entry["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    if total != inventory["bytes"]:
        raise ValueError("Artifact byte count mismatch")


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Nonfinite or nonnumeric {label}")
    return float(value)


def close(actual, expected, label):
    if not math.isclose(finite(actual, label), expected, rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError(f"Inconsistent {label}")


def load_artifact(root, target):
    root = Path(root).resolve()
    verify_inventory(root)
    manifest = read_json(root / "manifest.json")
    identity = {k: manifest[k] for k in ("config", "code_hashes", "mechanism_hash", "environment")}
    if manifest["run_hash"] != digest(identity):
        raise ValueError("Manifest run fingerprint is inconsistent")
    config = manifest["config"]
    if config["run_kind"] != "FORMAL" or config["B_requested"] < target:
        raise ValueError("A formal run with sufficient reserved replications is required")
    ns, methods = config["n_values"], config["methods"]
    if (not ns or len(set(ns)) != len(ns) or any(type(n) is not int or n <= 0 for n in ns)
            or len(set(methods)) != len(methods)):
        raise ValueError("Invalid sample-size grid or duplicate methods")
    mechanism = read_json(root / "mechanism.json")
    if (mechanism["mechanism_hash"] != manifest["mechanism_hash"] or
            mechanism["config"] != config["dgp"] or
            mechanism["parameter_seed"] != config["mechanism_seed"] or
            mechanism["truth"] != manifest["truth"] or not mechanism["truth"]["converged"]):
        raise ValueError("Mechanism/DGP/truth differs from the manifest")
    truth = finite(mechanism["truth"]["value"], "population truth")
    expected = {(n, method, rep) for n in ns for method in methods for rep in range(target)}
    records = {}
    for path in sorted((root / "tasks").glob("*.json")):
        row = read_json(path)
        key = row["n"], row["method"], row["rep"]
        if key not in expected or key in records or type(row["rep"]) is not int:
            raise ValueError(f"Unexpected or duplicate task: {path.name}")
        task_id = f"n{row['n']:05d}_r{row['rep']:04d}_{row['method']}"
        if path.stem != task_id or row["task_id"] != task_id:
            raise ValueError(f"Task identity/file name mismatch: {path.name}")
        if row["run_hash"] != manifest["run_hash"] or row["mechanism_hash"] != manifest["mechanism_hash"]:
            raise ValueError(f"Mixed scientific run in {task_id}")
        close(row["theta_population"], truth, "population truth")
        for field in PAIR_FIELDS[:2]:
            if type(row[field]) is not int or row[field] < 0:
                raise ValueError(f"Invalid {field} in {task_id}")
        if row["status"] == "complete":
            resolved = row.get("estimator_metadata", {}).get("resolved_config", {})
            roles = (("nuisance", "nn_cfg"),)
            if row["method"] in ARMS:
                roles += (("representation_A", "me_cfg"), ("representation_B", "me_cfg"))
            for role, setting in roles:
                if resolved.get(role, {}).get("weight_decay") != config["training"][setting]["weight_decay"]:
                    raise ValueError(f"Resolved {role} weight decay differs from manifest in {task_id}")
            fingerprint = row.get("observed_data_sha256", "")
            if len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
                raise ValueError(f"Missing observed-data fingerprint in {task_id}")
            theta, se = finite(row["theta_hat"], "estimate"), finite(row["se_IF"], "SE")
            if se < 0:
                raise ValueError("Negative SE")
            for field, value in {"error": theta-truth, "ci_lower": theta-Z*se,
                                 "ci_upper": theta+Z*se, "ci_length": 2*Z*se}.items():
                close(row[field], value, field)
            if type(row["covered"]) is not bool or row["covered"] != (row["ci_lower"] <= truth <= row["ci_upper"]):
                raise ValueError(f"Inconsistent coverage in {task_id}")
            score = safe_file(root, row["score_artifact"])
            if file_hash(score) != row["score_artifact_sha256"]:
                raise ValueError(f"Score checksum mismatch in {task_id}")
        elif row["status"] != "failed":
            raise ValueError(f"Unknown task status in {task_id}")
        records[key] = row
    return dict(root=root, manifest=manifest, records=records, expected=expected, target=target)


def validate_comparability(old, new):
    a, b = old["manifest"], new["manifest"]
    ca, cb = copy.deepcopy(a["config"]), copy.deepcopy(b["config"])
    if set(ca["methods"]) != set(ARMS + BASELINES) or set(cb["methods"]) != set(ARMS):
        raise ValueError("Expected old five-arm and new two-arm MediEncoder experiments")
    for config, weight in ((ca, 0.0), (cb, 0.01)):
        config.pop("methods")
        for name in ("nn_cfg", "ae_cfg", "me_cfg"):
            if config["training"][name].pop("weight_decay") != weight:
                raise ValueError(f"Expected shared weight_decay={weight} in all neural learners")
    if ca != cb:
        raise ValueError("Scientific settings changed beyond shared weight decay and arms")
    if set(a["code_hashes"]) != SOURCES or set(b["code_hashes"]) != SOURCES:
        raise ValueError("Scientific source inventory differs")
    for name in SOURCES - {"simulation/runner.py"}:
        if a["code_hashes"][name] != b["code_hashes"][name]:
            raise ValueError(f"Scientific source changed: {name}")
    if a["mechanism_hash"] != b["mechanism_hash"] or a["truth"] != b["truth"]:
        raise ValueError("Mechanism or population truth changed")
    for name in ("mechanism.json", "mechanism.npz"):
        if file_hash(old["root"] / name) != file_hash(new["root"] / name):
            raise ValueError(f"Imported mechanism is not the exact saved artifact: {name}")
    # Platform/kernel descriptions may vary between otherwise identical cloud hosts.
    for name in ("numpy", "scipy", "scikit-learn", "torch"):
        if a["environment"]["packages"].get(name) != b["environment"]["packages"].get(name):
            raise ValueError(f"Numerical runtime changed: {name}")


def audit_pairs(left, right, pairs, label):
    complete, unavailable = 0, []
    for lk, rk in pairs:
        a, b = left.get(lk), right.get(rk)
        if a is None or b is None:
            unavailable.append(dict(left=list(lk), right=list(rk), reason="missing task"))
            continue
        for field in PAIR_FIELDS:
            if field not in a or field not in b:
                if a["status"] == b["status"] == "complete":
                    raise ValueError(f"Missing paired {field}: {label} {lk}")
                continue
            if a[field] != b[field]:
                raise ValueError(f"Mismatched paired {field}: {label} {lk}")
        if a["status"] != "complete" or b["status"] != "complete":
            unavailable.append(dict(left=list(lk), right=list(rk), reason="failed task"))
        else:
            complete += 1
    return dict(label=label, complete_pairs=complete, unavailable_pairs=unavailable)


def summary(artifact, n, method, start, stop, label):
    rows = [artifact["records"].get((n, method, rep)) for rep in range(start, stop)]
    done = sum(r is not None and r["status"] == "complete" for r in rows)
    failed = sum(r is not None and r["status"] == "failed" for r in rows)
    result = dict(cohort=label, n=n, method=method, rep_start=start, rep_stop_exclusive=stop,
                  B_requested=stop-start, B_completed=done, B_failed=failed,
                  B_missing=stop-start-done-failed, status="complete" if done == stop-start else "incomplete",
                  Bias=None, SD=None, RMSE=None, Mean_SE=None, RMS_SE=None,
                  CI_Length=None, Coverage=None, Coverage_MCSE=None)
    # Do not calculate success-conditional metrics for an incomplete cohort.
    if result["status"] == "complete":
        errors = [r["error"] for r in rows]
        coverage = statistics.mean(r["covered"] for r in rows)
        result.update(Bias=statistics.mean(errors), SD=statistics.stdev(errors),
                      RMSE=math.sqrt(statistics.mean(x*x for x in errors)),
                      Mean_SE=statistics.mean(r["se_IF"] for r in rows),
                      RMS_SE=math.sqrt(statistics.mean(r["se_IF"]**2 for r in rows)),
                      CI_Length=statistics.mean(r["ci_length"] for r in rows), Coverage=coverage,
                      Coverage_MCSE=math.sqrt(coverage*(1-coverage)/(stop-start)))
    return result


def paired_changes(left, right, keys, label, n, method):
    pairs = [(left.get(a), right.get(b)) for a, b in keys]
    done = sum(a is not None and b is not None and a["status"] == b["status"] == "complete" for a, b in pairs)
    row = dict(comparison=label, n=n, method=method, difference_direction="right_minus_left",
               pairs_requested=len(pairs), pairs_complete=done, pairs_unavailable=len(pairs)-done,
               status="complete" if done == len(pairs) else "incomplete")
    metrics = ("Bias", "MSE", "RMSE", "Coverage", "Mean_SE", "CI_Length", "SD")
    for metric in metrics:
        row[metric + "_change"] = None
        if metric != "SD":
            row[metric + "_change_MCSE"] = None
    if done != len(pairs):
        return row
    e0, e1 = [a["error"] for a, b in pairs], [b["error"] for a, b in pairs]
    rm0 = math.sqrt(statistics.mean(x*x for x in e0))
    rm1 = math.sqrt(statistics.mean(x*x for x in e1))
    differences = dict(Bias=[b-a for a,b in zip(e0,e1)], MSE=[b*b-a*a for a,b in zip(e0,e1)],
                       Coverage=[int(b["covered"])-int(a["covered"]) for a,b in pairs],
                       Mean_SE=[b["se_IF"]-a["se_IF"] for a,b in pairs],
                       CI_Length=[b["ci_length"]-a["ci_length"] for a,b in pairs])
    for name, values in differences.items():
        row[name+"_change"] = statistics.mean(values)
        row[name+"_change_MCSE"] = statistics.stdev(values)/math.sqrt(done)
    row["RMSE_change"] = rm1-rm0
    if rm0 > 0 and rm1 > 0:
        # Paired delta-method uncertainty of a difference of root mean squares.
        influence = [b*b/(2*rm1)-a*a/(2*rm0) for a,b in zip(e0,e1)]
        row["RMSE_change_MCSE"] = statistics.stdev(influence)/math.sqrt(done)
    row["SD_change"] = statistics.stdev(e1)-statistics.stdev(e0)
    return row


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare(old_dir, new_dir, output_dir):
    output = Path(output_dir).resolve()
    for source in (Path(old_dir).resolve(), Path(new_dir).resolve()):
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("Comparison output must be separate from both input artifacts")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Comparison output must be empty; previous reports are never overwritten")
    old, new = load_artifact(old_dir, 50), load_artifact(new_dir, 100)
    validate_comparability(old, new)
    ns = old["manifest"]["config"]["n_values"]
    overlap = [((n,m,r),(n,m,r)) for n in ns for m in ARMS for r in range(50)]
    alignment = [((n,ARMS[0],r),(n,ARMS[1],r)) for n in ns for r in range(100)]
    pair_audits = [audit_pairs(old["records"],new["records"],overlap,"old0_to_new.01_first50"),
                   audit_pairs(new["records"],new["records"],alignment,"new.01_zero_minus_positive_all100")]
    summaries, changes = [], []
    for n in ns:
        for method in ARMS:
            for artifact, start, stop, label in (
                (old,0,50,"old_wd0_first50"), (new,0,50,"new_wd.01_first50"),
                (new,0,100,"new_wd.01_all100"), (new,50,100,"new_wd.01_fresh50_99")):
                summaries.append(summary(artifact,n,method,start,stop,label))
            changes.append(paired_changes(old["records"],new["records"],
                           [((n,method,r),(n,method,r)) for r in range(50)],
                           "new_wd.01_minus_old_wd0_first50",n,method))
        changes.append(paired_changes(new["records"],new["records"],
                       [((n,ARMS[0],r),(n,ARMS[1],r)) for r in range(100)],
                       "new_wd.01_zero_minus_positive_all100",n,"alignment_ablation"))
        for method in BASELINES:
            summaries.append(summary(old,n,method,0,50,"historical_baseline_wd0_B50_reference_only"))
    audit = dict(complete=all(row["status"] == "complete" for row in summaries+changes),
                 old_run_hash=old["manifest"]["run_hash"], new_run_hash=new["manifest"]["run_hash"],
                 mechanism_hash=old["manifest"]["mechanism_hash"], population_truth=old["manifest"]["truth"]["value"],
                 old_target_reps=50, new_target_reps=100, paired_audits=pair_audits,
                 metrics_require_complete_cohort=True, scientific_sources_equal_except_runner=True,
                 exact_saved_mechanism_equal=True, cohorts_are_separate_not_pooled=True,
                 environment_difference=old["manifest"]["environment"] != new["manifest"]["environment"])
    for label, artifact in (("old",old),("new",new)):
        audit[label+"_missing_tasks"] = [list(k) for k in sorted(artifact["expected"]-set(artifact["records"]))]
        audit[label+"_failed_tasks"] = [dict(task_id=r["task_id"], error_type=r.get("error_type"),
                                          error_message=r.get("error_message"))
                                      for r in artifact["records"].values() if r["status"] == "failed"]
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/"summaries.csv",summaries)
    write_csv(output/"paired_changes.csv",changes)
    (output/"comparison_audit.json").write_text(json.dumps(audit,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    text = ("# Regularization comparison\n\n"
            + ("All planned comparisons are complete.\n\n" if audit["complete"] else
               "INCOMPLETE: failed/missing fits are listed in comparison_audit.json. Incomplete cohorts have counts only; no success-only estimates.\n\n")
            + "Old WD=0 and new shared WD=.01 are separate experiments with an identical saved DGP. "
            "Only the first 50 replicates support paired old/new comparisons. New positive/zero alignment use 100 paired replicates. "
            "Replicates 50-99 are separately reported as fresh validation after the original 50 informed this choice.\n\n"
            "Historical baseline rows contain B=50 only; they are not a new all-method B=100 head-to-head comparison. "
            "No winning method or 95% coverage is assumed. Failure/missing counts must accompany any report.\n\n"
            "Paired changes are right minus left, as named in each comparison. Negative RMSE change favors the right arm. "
            "MCSE uses paired replicate differences; RMSE MCSE uses the delta method and is undefined at zero RMSE. "
            "These are exploratory Monte Carlo uncertainties, sensitive to heavy tails, not multiplicity-adjusted claims. "
            "Mean_SE and RMS_SE are both reported to expose heterogeneity. "
            "Score bytes and original merged-artifact checksums are verified; this helper does not rerun model fits or the merge tool's numerical score checks.\n")
    (output/"comparison.md").write_text(text,encoding="utf-8")
    return 0 if audit["complete"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_artifact", type=Path)
    parser.add_argument("new_artifact", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return compare(args.old_artifact,args.new_artifact,args.output_dir)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"Comparison rejected: {exc}",file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
