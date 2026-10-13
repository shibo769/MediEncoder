"""SCC shard audits retain partial work and never combine different runtimes."""
import importlib.util
import json
from pathlib import Path

import pytest

from mediencoder.simulation import runner


SPEC = importlib.util.spec_from_file_location("cloud_merge_fixtures", Path(__file__).with_name("test_cloud_simulation.py"))
fixtures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixtures)
cloud = fixtures.cloud


def make_scc_artifacts(tmp_path, *, device="cuda", **kwargs):
    prepared, shards = fixtures.make_artifacts(tmp_path, **kwargs)
    manifest = json.loads((prepared / "manifest.json").read_text())
    manifest["config"]["device"] = device
    manifest["environment"] = dict(fixture="same", gpu="NVIDIA L40S", gpu_compute_capability=[8, 9])
    identity = {k: manifest[k] for k in ("config", "code_hashes", "mechanism_hash", "environment")}
    manifest["run_hash"] = runner.digest(identity)
    runner.atomic_json(prepared / "manifest.json", manifest)
    (prepared / "artifact_inventory.json").unlink()
    for directory in shards.iterdir():
        runner.atomic_json(directory / "manifest.json", manifest)
        plan = json.loads((directory / "execution_plan.json").read_text())
        plan["run_hash"] = manifest["run_hash"]
        runner.atomic_json(directory / "execution_plan.json", plan)
        (directory / "artifact_inventory.json").unlink()
        for path in (directory / "tasks").glob("*.json"):
            record = json.loads(path.read_text())
            record["run_hash"] = manifest["run_hash"]
            runner.atomic_json(path, record)
    return prepared, shards


def merge(prepared, shards, output, **kwargs):
    return cloud.merge_artifacts(prepared, shards, output, 2, 2,
                                 expected_device="cuda", require_inventory=False,
                                 execution_platform="scc", **kwargs)


def test_scc_cuda_merge_preserves_complete_inference_and_failed_attempt(tmp_path):
    prepared, shards = make_scc_artifacts(tmp_path)
    shard = shards / "shard-0"
    record = json.loads(next((shard / "tasks").glob("*.json")).read_text())
    previous = dict(record, status="failed", error_type="MemoryError", error_message="retained first attempt")
    previous.pop("score_artifact")
    previous.pop("score_artifact_sha256")
    relative = Path("attempts") / previous["task_id"] / (runner.digest(previous)[:16] + ".json")
    runner.atomic_json(shard / relative, previous)
    output = tmp_path / "merged"
    assert merge(prepared, shards, output) == 0
    audit = json.loads((output / "merge_audit.json").read_text())
    assert audit["complete"] and audit["completed"] == 4
    assert audit["device"] == "cuda" and audit["execution_platform"] == "scc"
    assert audit["environment"]["gpu"] == "NVIDIA L40S"
    assert audit["archived_failed_attempts"] == 1
    assert (output / relative).read_bytes() == (shard / relative).read_bytes()


def test_cloud_default_cannot_accept_gpu_artifacts(tmp_path):
    prepared, shards = make_scc_artifacts(tmp_path)
    with pytest.raises(ValueError, match="expected device cpu"):
        cloud.merge_artifacts(prepared, shards, tmp_path / "merged", 2, 2,
                              require_inventory=False)
    with pytest.raises(ValueError, match="separate CPU"):
        cloud.merge_artifacts(prepared, shards, tmp_path / "merged", 2, 2,
                              expected_device="cuda", require_inventory=False)


@pytest.mark.parametrize("change", ["device", "runtime", "score", "assignment"])
def test_scc_rejects_mixed_runtime_damaged_scores_and_wrong_task_assignment(tmp_path, change):
    prepared, shards = make_scc_artifacts(tmp_path)
    shard = shards / "shard-0"
    if change in ("device", "runtime"):
        path = shard / "manifest.json"
        manifest = json.loads(path.read_text())
        if change == "device":
            manifest["config"]["device"] = "cpu"
        else:
            manifest["environment"]["gpu"] = "NVIDIA A100"
        runner.atomic_json(path, manifest)
    elif change == "score":
        next((shard / "scores").glob("*.npz")).write_bytes(b"damaged")
    else:
        path = next((shard / "tasks").glob("*.json"))
        record = json.loads(path.read_text())
        record["data_seed"] = -1
        runner.atomic_json(path, record)
    with pytest.raises(ValueError):
        merge(prepared, shards, tmp_path / "merged")


def test_scc_incomplete_shards_and_empty_queued_directory_are_not_success(tmp_path):
    prepared, shards = make_scc_artifacts(tmp_path, omit_task=True)
    (shards / "queued-empty").mkdir()
    output = tmp_path / "merged"
    assert merge(prepared, shards, output) == 2
    audit = json.loads((output / "merge_audit.json").read_text())
    assert not audit["complete"] and audit["completed"] == 3
    assert len(audit["missing_tasks"]) == 1
    status = json.loads((output / "status.json").read_text())
    assert status["phase"] == "incomplete_scc_run"


def test_scc_unmanifested_nonempty_shard_fails_closed(tmp_path):
    prepared, shards = make_scc_artifacts(tmp_path)
    directory = shards / "unmanifested"
    directory.mkdir()
    (directory / "mystery.json").write_text("{}")
    with pytest.raises(ValueError, match="no scientific manifest"):
        merge(prepared, shards, tmp_path / "merged")


def test_scc_wrongly_named_or_corrupted_archived_attempt_rejected(tmp_path):
    prepared, shards = make_scc_artifacts(tmp_path)
    shard = shards / "shard-0"
    record = json.loads(next((shard / "tasks").glob("*.json")).read_text())
    previous = dict(record, status="failed")
    runner.atomic_json(shard / "attempts" / previous["task_id"] / "incorrect.json", previous)
    with pytest.raises(ValueError, match="Archived failed attempt"):
        merge(prepared, shards, tmp_path / "merged")


def test_package_keeps_failed_attempt_history(tmp_path):
    prepared, shards = make_scc_artifacts(tmp_path)
    shard = shards / "shard-0"
    record = json.loads(next((shard / "tasks").glob("*.json")).read_text())
    previous = dict(record, status="failed")
    relative = Path("attempts") / previous["task_id"] / (runner.digest(previous)[:16] + ".json")
    runner.atomic_json(shard / relative, previous)
    destination = tmp_path / "packaged"
    cloud.package_artifact(shard, destination, "shard", 2**20)
    cloud.verify_inventory(destination, "shard")
    assert (destination / relative).read_bytes() == (shard / relative).read_bytes()


def test_scc_archived_failed_attempt_must_use_original_paired_data(tmp_path):
    prepared, shards = make_scc_artifacts(tmp_path)
    shard = shards / "shard-0"
    record = json.loads(next((shard / "tasks").glob("*.json")).read_text())
    previous = dict(record, status="failed", observed_data_sha256="1" * 64)
    relative = Path("attempts") / previous["task_id"] / (runner.digest(previous)[:16] + ".json")
    runner.atomic_json(shard / relative, previous)
    with pytest.raises(ValueError, match="paired observed"):
        merge(prepared, shards, tmp_path / "merged")


def test_scc_shard_directory_link_cannot_import_artifacts_from_outside_run(tmp_path):
    prepared, shards = make_scc_artifacts(tmp_path)
    external = tmp_path / "external-shard"
    (shards / "shard-1").rename(external)
    try:
        (shards / "shard-1").symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("Creating directory links is not permitted on this host")
    with pytest.raises(ValueError, match="designated artifact root"):
        merge(prepared, shards, tmp_path / "merged")
