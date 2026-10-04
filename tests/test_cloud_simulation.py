"""Synthetic artifact integrity and inference checks; no training or network."""
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from mediencoder.simulation import dgp, runner

SPEC = importlib.util.spec_from_file_location("cloud_simulation", Path(__file__).parents[1] / "scripts/cloud_simulation.py")
cloud = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cloud)


def make_artifacts(tmp_path, *, omit_task=False, changed_pairing=False, changed_interval=False):
    params = dgp.draw_parameters(dgp.DGPConfig(p=4, q=3, bar_p=2, bar_q=2, pilot_size=64), 17)
    config = dict(B_requested=4, n_values=[40], methods=["projection", "mediencoder"],
                  seed_base=99, device="cpu", dgp=asdict(params.config), tilde_p=2, tilde_q=2,
                  run_kind="TEST_FIXTURE_NOT_FOR_PAPER")
    identity = dict(config=config, code_hashes=runner.collect_code_hashes(),
                    mechanism_hash=params.mechanism_hash, environment={"fixture": "same"})
    manifest = dict(identity, run_hash=runner.digest(identity))
    prepared_source, prepared = tmp_path / "prepared-source", tmp_path / "prepared"
    prepared_source.mkdir()
    dgp.save_mechanism(params, prepared_source / "mechanism")
    runner.atomic_json(prepared_source / "manifest.json", manifest)
    cloud.package_artifact(prepared_source, prepared, "prepared", 2**20)
    shards = tmp_path / "shards"
    scores = params.truth.value + np.linspace(-0.2, 0.2, 40)
    mean, se = float(scores.mean()), float(scores.std(ddof=1)/np.sqrt(40))
    lower, upper = mean-1.959963984540054*se, mean+1.959963984540054*se
    split = np.array_split(np.arange(40), 4)
    arrays = dict(crossfit_scores=scores, subject_index=np.arange(40))
    for fold in range(4):
        for offset, role in enumerate(("representation_train", "representation_validation", "nuisance", "estimation")):
            arrays[f"fold{fold}_{role}"] = split[(fold+offset) % 4]
        nuisance = arrays[f"fold{fold}_nuisance"]
        arrays[f"fold{fold}_nuisance_train"] = nuisance[:5]
        arrays[f"fold{fold}_nuisance_validation"] = nuisance[5:]
    for index in range(2):
        source = tmp_path / f"source-{index}"
        source.mkdir()
        runner.atomic_json(source / "manifest.json", manifest)
        runner.atomic_json(source / "execution_plan.json", dict(shard_index=index, num_shards=2,
                           execution_target_reps=2, run_hash=manifest["run_hash"]))
        for task in runner.build_tasks(runner.execution_config(config, 2)):
            if task["rep"] != index or (omit_task and index == 1 and task["method"] == "mediencoder"):
                continue
            key = task["task_id"]
            artifact = source / "scores" / (key + ".npz")
            artifact.parent.mkdir(exist_ok=True)
            np.savez_compressed(artifact, **arrays)
            pair = f"paired-{index}"
            if changed_pairing and task["method"] == "mediencoder":
                pair = "different"
            record = dict(task, status="complete", run_hash=manifest["run_hash"], mechanism_hash=params.mechanism_hash,
                          theta_population=params.truth.value, theta_hat=mean, error=mean-params.truth.value,
                          se_IF=se, ci_lower=lower, ci_upper=upper, ci_length=upper-lower, covered=True,
                          score_artifact=f"scores/{key}.npz", score_artifact_sha256=runner.file_hash(artifact),
                          observed_data_sha256=hashlib.sha256(pair.encode()).hexdigest())
            if changed_interval:
                record["ci_upper"] += 0.1
            runner.atomic_json(source / "tasks" / (key + ".json"), record)
        cloud.package_artifact(source, shards / f"shard-{index}", "shard", 2**20)
    return prepared, shards


def test_complete_cloud_merge_checks_scores_and_preserves_reservation(tmp_path):
    prepared, shards = make_artifacts(tmp_path)
    output = tmp_path / "merged"
    assert cloud.merge_artifacts(prepared, shards, output, 2, 2) == 0
    audit = json.loads((output / "merge_audit.json").read_text())
    assert audit["complete"] and audit["completed"] == audit["expected_tasks"] == 4
    assert audit["reserved_reps"] == 4 and audit["target_reps"] == 2
    assert not audit["combined_with_local_gpu"]
    assert len(list((output / "scores").glob("*.npz"))) == 4
    assert (prepared / "manifest.json").read_bytes() == (output / "manifest.json").read_bytes()


def test_incomplete_cloud_merge_preserves_available_records_without_success_claim(tmp_path):
    prepared, shards = make_artifacts(tmp_path, omit_task=True)
    output = tmp_path / "merged"
    assert cloud.merge_artifacts(prepared, shards, output, 2, 2) == 2
    audit = json.loads((output / "merge_audit.json").read_text())
    assert not audit["complete"] and len(audit["missing_tasks"]) == 1
    assert audit["completed"] == 3
    assert (output / "main_table.tex").is_file()


@pytest.mark.parametrize("option, message", [("changed_pairing", "paired observed"), ("changed_interval", "ci_upper")])
def test_cloud_merge_rejects_pairing_or_interval_mismatch(tmp_path, option, message):
    prepared, shards = make_artifacts(tmp_path, **{option: True})
    with pytest.raises(ValueError, match=message):
        cloud.merge_artifacts(prepared, shards, tmp_path / "merged", 2, 2)


def test_cloud_package_rejects_storage_overage_and_artifact_tampering(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    runner.atomic_json(source / "manifest.json", {"example": 1})
    with pytest.raises(ValueError, match="byte ceiling"):
        cloud.package_artifact(source, tmp_path / "tiny", "shard", 1)
    bundle = tmp_path / "bundle"
    cloud.package_artifact(source, bundle, "shard", 1000)
    (bundle / "manifest.json").write_text("tampered")
    with pytest.raises(ValueError, match="inventory mismatch"):
        cloud.verify_inventory(bundle, "shard")
    with pytest.raises(ValueError, match="Unsafe artifact path"):
        cloud.safe_file(source, "../outside")


def test_cloud_score_validation_rejects_repeated_estimation_subjects(tmp_path):
    prepared, shards = make_artifacts(tmp_path)
    shard = shards / "shard-0"
    record = json.loads(next((shard / "tasks").glob("*.json")).read_text())
    artifact = shard / record["score_artifact"]
    with np.load(artifact) as saved:
        arrays = dict(saved)
    # Keep each fold locally partitioned but repeat another fold's evaluation set.
    for role in ("representation_train", "representation_validation", "nuisance", "estimation", "nuisance_train", "nuisance_validation"):
        arrays[f"fold1_{role}"] = arrays[f"fold0_{role}"]
    np.savez_compressed(artifact, **arrays)
    with pytest.raises(ValueError, match="every observation exactly once"):
        cloud.validate_scores(record, shard, record["theta_population"])
