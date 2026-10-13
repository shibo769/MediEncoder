"""SCC scheduling, immutable-source, and launch-gate checks without SSH or training."""
from argparse import Namespace
import importlib.util
import json
from pathlib import Path
import subprocess
import signal
import sys

import pytest


SPEC = importlib.util.spec_from_file_location("scc_simulation", Path(__file__).parents[1] / "scripts/scc_simulation.py")
scc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scc)


def make_plan(tmp_path, **resource_overrides):
    source = tmp_path / "deployed"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts/scc_simulation.py").write_text("# frozen execution tool\n", encoding="utf-8")
    (source / "example.py").write_text("answer = 42\n", encoding="utf-8")
    inventory = {path.relative_to(source).as_posix(): dict(size=path.stat().st_size, sha256=scc.sha256(path))
                 for path in source.rglob("*") if path.is_file()}
    scc.atomic_json(source / "SCC_DEPLOYMENT_MANIFEST.json", dict(source_commit="a"*40, files=inventory))
    args = Namespace(source_dir=source, run_dir=tmp_path / "run", python=tmp_path / "venv/bin/python",
                     python_module="python3/3.12.4", project="adsgrp", gpu_type="L40S",
                     gpu_memory_gb=24, workers=1, concurrency=2, cores=4, mem_per_core_gb=4,
                     wall_seconds=43200, device="cuda", cpu_slot_limit=16, gpu_pool="shared")
    for name, value in resource_overrides.items():
        setattr(args, name, value)
    assert scc.initialize(args) == 0
    return args.run_dir, scc.load_plan(args.run_dir), source


def add_ready(plan):
    root = Path(plan["run_dir"])
    scc.atomic_json(root / "prepared/manifest.json", {"run_hash": "test-scientific-run"})
    scc.write_marker(plan, "READY", run_hash="test-scientific-run")


def add_checks(plan):
    add_ready(plan)
    root = Path(plan["run_dir"])
    scc.atomic_json(root / "benchmark_results.json", {"fixture": "checked"})
    scc.write_marker(plan, "BENCHMARK_PASSED", selected_workers=2,
                     results_sha256=scc.sha256(root / "benchmark_results.json"))
    scc.atomic_json(root / "canary_audit.json", dict(plan_hash=plan["plan_hash"],
                    all_records_validated=True, files={}))
    scc.write_marker(plan, "CANARY_PASSED", verified_tasks=30,
                     audit_sha256=scc.sha256(root / "canary_audit.json"))


def test_freeze_uses_deployment_manifest_and_survives_active_checkout_edits(tmp_path):
    root, plan, original = make_plan(tmp_path)
    (original / "example.py").write_text("answer = 0\n", encoding="utf-8")
    assert scc.load_plan(root)["source_commit"] == "a"*40
    frozen = Path(plan["source_dir"]) / "example.py"
    frozen.write_text("answer = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen source changed"):
        scc.load_plan(root)


def test_source_inventory_rejects_archive_traversal_and_tampering(tmp_path):
    root, plan, original = make_plan(tmp_path)
    manifest_path = original / "SCC_DEPLOYMENT_MANIFEST.json"
    manifest = scc.read_json(manifest_path)
    manifest["files"]["../outside.py"] = dict(size=1, sha256="bad")
    scc.atomic_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="Unsafe source path"):
        scc.source_inventory(original)
    manifest["files"].pop("../outside.py")
    manifest["files"]["example.py"]["sha256"] = "bad"
    scc.atomic_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="differs from its manifest"):
        scc.source_inventory(original)


def test_background_gpu_array_respects_verified_quota_and_logs_separately(tmp_path):
    root, plan, _ = make_plan(tmp_path)
    argv = scc.qsub_argv(plan, "array", "me_test")
    assert argv[argv.index("-sync")+1] == "n"
    assert argv[argv.index("-t")+1] == "1-200"
    assert argv[argv.index("-tc")+1] == "2"
    assert "gpus=1,gpu_type=L40S,gpu_memory=24G" in argv
    assert "mem_per_core=4G" in argv and "-V" not in argv
    assert "-notify" in argv
    assert argv[argv.index("-o")+1] == str(root / "logs")
    script = scc.batch_script(plan, "array")
    assert "CUDA_VISIBLE_DEVICES" not in script
    assert "timeout --signal=TERM --kill-after=45s" in script
    assert "trap '' USR1 USR2" in script
    assert str(Path(plan["source_dir"]) / "scripts/scc_simulation.py") in script


@pytest.mark.parametrize("field,value", [("workers", 5), ("concurrency", 3),
                                         ("wall_seconds", 43201), ("project", "adsgrp;bad")])
def test_invalid_resources_are_rejected_before_submission(tmp_path, field, value):
    _, plan, _ = make_plan(tmp_path)
    plan["resources"][field] = value
    with pytest.raises(ValueError):
        scc.validate_resources(plan["resources"])


def test_formal_array_cannot_skip_prepare_benchmark_or_canary(tmp_path, monkeypatch):
    root, plan, _ = make_plan(tmp_path)
    monkeypatch.setattr(scc, "sge_snapshot", lambda: {})
    with pytest.raises(ValueError, match="READY"):
        scc.submit(root, "array", dry_run=True)
    add_ready(plan)
    with pytest.raises(ValueError, match="BENCHMARK_PASSED"):
        scc.submit(root, "array", dry_run=True)
    add_checks(plan)
    (root / "CANARY_PASSED.json").unlink()
    with pytest.raises(ValueError, match="CANARY_PASSED"):
        scc.submit(root, "array", dry_run=True)


def test_worker_selection_rejects_tampered_benchmark(tmp_path):
    root, plan, _ = make_plan(tmp_path)
    add_checks(plan)
    assert scc.measured_workers(plan) == 2
    scc.atomic_json(root / "benchmark_results.json", {"fixture": "changed"})
    with pytest.raises(ValueError, match="Benchmark results changed"):
        scc.measured_workers(plan)


def test_eight_workers_per_gpu_fit_shared_gpu_cpu_slot_budget(tmp_path):
    root, plan, _ = make_plan(tmp_path, cores=8, concurrency=2, cpu_slot_limit=16)
    add_checks(plan)
    scc.write_marker(plan, "BENCHMARK_PASSED", selected_workers=8,
                     results_sha256=scc.sha256(root / "benchmark_results.json"))
    assert scc.measured_workers(plan) == 8
    argv = scc.qsub_argv(plan, "array", "me_eight_workers")
    assert argv[argv.index("-pe")+2] == "8"
    assert argv[argv.index("-tc")+1] == "2"
    assert 8 * int(argv[argv.index("-tc")+1]) == 16


def test_duplicate_live_job_refused_but_unrelated_jobs_are_not_touched(tmp_path, monkeypatch):
    root, plan, _ = make_plan(tmp_path)
    add_checks(plan)
    state = dict(plan_hash=plan["plan_hash"], jobs=[dict(phase="array", name="own", status="submitted", job_id="123")])
    scc.atomic_json(root / "submissions.json", state)
    monkeypatch.setattr(scc, "sge_snapshot", lambda: {"123": {"name": "own"}, "999": {"name": "unrelated"}})
    with pytest.raises(ValueError, match="already has a live"):
        scc.submit(root, "array", dry_run=True, resume=True)
    assert scc.read_json(root / "submissions.json") == state


def test_qsub_job_id_and_argv_are_saved_and_unknown_outcome_blocks_retries(tmp_path, monkeypatch):
    root, plan, _ = make_plan(tmp_path)
    monkeypatch.setattr(scc, "sge_snapshot", lambda: {})
    monkeypatch.setattr(scc.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 0, "8123456\n", ""))
    assert scc.submit(root, "prepare") == 0
    saved = scc.read_json(root / "submissions.json")
    assert saved["jobs"][0]["job_id"] == "8123456"
    assert saved["jobs"][0]["argv"][0] == "qsub"
    monkeypatch.setattr(scc.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 0, "unparseable\n", ""))
    with pytest.raises(RuntimeError, match="Cannot parse"):
        scc.submit(root, "prepare", resume=True)
    with pytest.raises(ValueError, match="needs reconciliation"):
        scc.submit(root, "prepare", resume=True)


def test_merge_dependency_is_not_a_success_gate_and_uses_no_gpu(tmp_path):
    _, plan, _ = make_plan(tmp_path)
    argv = scc.qsub_argv(plan, "merge", "me_merge", "8123456")
    assert argv[argv.index("-hold_jid")+1] == "8123456"
    assert argv[argv.index("-pe")+2] == "1"
    assert not any("gpus=" in value for value in argv)


def test_cpu_array_reserves_measured_cores_and_caps_parallelism_at_execution_budget(tmp_path):
    root, plan, _ = make_plan(tmp_path, device="cpu", cores=16, concurrency=20, cpu_slot_limit=32)
    add_checks(plan)
    argv = scc.qsub_argv(plan, "array", "me_cpu")
    assert not any("gpus=" in value for value in argv)
    assert argv[argv.index("-pe")+2] == "2"
    assert argv[argv.index("-tc")+1] == "16"
    assert plan["science_args"][plan["science_args"].index("--device")+1] == "cpu"
    benchmark = scc.qsub_argv(plan, "benchmark", "me_cpu_benchmark")
    assert benchmark[benchmark.index("-pe")+2] == "16"


def test_cpu_and_gpu_settings_cannot_be_combined_in_one_frozen_plan(tmp_path):
    root, plan, _ = make_plan(tmp_path, device="cpu")
    plan["science_args"] = scc.science_args("cuda")
    plan["plan_hash"] = scc.object_hash({key: value for key, value in plan.items() if key != "plan_hash"})
    scc.atomic_json(root / "plan.json", plan)
    with pytest.raises(ValueError, match="Unexpected scientific configuration"):
        scc.load_plan(root)


def test_combined_l40s_pool_is_one_array_with_separate_scheduler_quotas(tmp_path):
    _, plan, _ = make_plan(tmp_path, gpu_pool="combined", concurrency=6)
    argv = scc.qsub_argv(plan, "array", "me_combined")
    assert argv[argv.index("-q")+1] == "l40s,ece-pub"
    assert argv[argv.index("-tc")+1] == "6"
    assert "gpus=1,gpu_type=L40S,gpu_memory=24G" in argv
    plan["resources"]["concurrency"] = 7
    with pytest.raises(ValueError, match="quota of 6"):
        scc.validate_resources(plan["resources"])


def test_retry_is_explicit_and_only_exports_its_own_boolean(tmp_path):
    _, plan, _ = make_plan(tmp_path)
    normal = scc.qsub_argv(plan, "array", "me_normal")
    retry = scc.qsub_argv(plan, "array", "me_retry", retry_failed=True)
    assert "-v" not in normal
    assert retry[retry.index("-v")+1] == "MEDIENC_SCC_RETRY_FAILED=1"
    assert "-V" not in retry
    with pytest.raises(ValueError, match="only to canary or array"):
        scc.qsub_argv(plan, "prepare", "me_prepare", retry_failed=True)


def test_profile_samples_only_its_process_tree_and_handles_departed_workers(tmp_path):
    proc = tmp_path / "proc"
    for pid, rss, children in ((101, 2048, "102 103"), (102, 4096, "")):
        directory = proc / str(pid)
        (directory / "task" / str(pid)).mkdir(parents=True)
        (directory / "status").write_text(f"Name:\tworker\nVmRSS:\t{rss} kB\n")
        (directory / "task" / str(pid) / "children").write_text(children)
    assert scc.process_tree_rss_mib(101, proc_root=proc) == 6


def test_gpu_sampler_does_not_query_unassigned_or_multiple_devices(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(scc.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected GPU query"))
    assert scc.gpu_sample() is None
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert scc.gpu_sample() is None


def test_cpu_benchmark_gives_all_candidates_the_same_balanced_two_wave_task_mix(tmp_path, monkeypatch):
    root, plan, _ = make_plan(tmp_path, device="cpu", cores=16, concurrency=32, cpu_slot_limit=256)
    add_ready(plan)
    monkeypatch.setenv("JOB_ID", "fixture-job")
    monkeypatch.setattr(scc, "runner_module", lambda *args: Namespace())
    commands = []

    def profile(command, device):
        assert device == "cpu"
        commands.append(command)
        destination = Path(command[command.index("--output-dir")+1])
        completed = int(command[command.index("--max-tasks")+1])
        for index in range(completed):
            scc.atomic_json(destination / "tasks" / f"task-{index}.json", {"status": "complete"})
        scc.atomic_json(destination / "status.json", {"completed": completed})
        workers = int(command[command.index("--workers")+1])
        return dict(exit_code=0, elapsed_seconds=70/workers, peak_process_tree_rss_mib=10,
                    peak_gpu_memory_mib=None, mean_gpu_utilization_percent=None)

    monkeypatch.setattr(scc, "profile_command", profile)
    assert scc.execute(root, "benchmark") == 0
    assert len(commands) == 5
    for command in commands:
        assert command[command.index("--n")+1] == "3000"
        assert command[command.index("--reps")+1] == "7"
        assert command[command.index("--max-tasks")+1] == "35"
        assert command[command.index("--pilot-epochs")+1] == "20"
    assert scc.measured_workers(plan) == 16


@pytest.mark.parametrize("interruption", ["keyboard", "SIGTERM"])
def test_profile_interrupt_stops_its_child_and_restores_signal_handlers(monkeypatch, interruption):
    class Child:
        pid = 987654
        returncode = None
        def poll(self):
            return self.returncode
        def terminate(self):
            self.returncode = -15
        def kill(self):
            self.returncode = -9
        def wait(self, timeout):
            assert timeout in (15, 5)
            return self.returncode
    child = Child()
    previous = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(scc.subprocess, "Popen", lambda *a, **k: child)
    monkeypatch.setattr(scc, "process_tree_rss_mib", lambda *a, **k: 1)
    groups = []
    def kill_group(pid, signum):
        assert pid == child.pid
        groups.append((pid, signum))
        child.returncode = -int(signum)
    monkeypatch.setattr(scc.os, "killpg", kill_group, raising=False)
    def stop_sleep(seconds):
        if interruption == "keyboard":
            raise KeyboardInterrupt()
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    monkeypatch.setattr(scc.time, "sleep", stop_sleep)
    if interruption == "keyboard":
        with pytest.raises(KeyboardInterrupt):
            scc.profile_command(["owned-child"], "cpu")
    else:
        result = scc.profile_command(["owned-child"], "cpu")
        assert result["exit_code"] == 4 and result["interrupted_signal"] == "SIGTERM"
    assert child.returncode is not None
    assert signal.getsignal(signal.SIGTERM) == previous


def test_profile_cleanup_escalates_only_its_private_child_group(monkeypatch):
    class Child:
        pid = 987655
        returncode = None
        waits = []
        def poll(self):
            return self.returncode
        def wait(self, timeout):
            self.waits.append(timeout)
            if self.returncode is None:
                raise subprocess.TimeoutExpired("owned-child", timeout)
            return self.returncode
    child, groups = Child(), []
    kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
    def send(pid, signum):
        assert pid == child.pid
        groups.append((pid, signum))
        if len(groups) > 1:
            child.returncode = -9
    monkeypatch.setattr(scc.os, "killpg", send, raising=False)
    scc.stop_profile_process(child, grouped=True)
    assert child.waits == [15, 5]
    assert groups[0] == (child.pid, signal.SIGTERM)
    assert groups[1] == (child.pid, kill_signal)


def test_canary_gate_rejects_artifact_changes_after_independent_audit(tmp_path):
    root, plan, _ = make_plan(tmp_path)
    add_checks(plan)
    record = root / "shards/shard-000/tasks/checked.json"
    scc.atomic_json(record, {"theta_hat": 1, "se_IF": 0.2})
    audit = scc.read_json(root / "canary_audit.json")
    audit["files"] = {"tasks/checked.json": scc.sha256(record)}
    scc.atomic_json(root / "canary_audit.json", audit)
    scc.write_marker(plan, "CANARY_PASSED", verified_tasks=30,
                     audit_sha256=scc.sha256(root / "canary_audit.json"))
    assert scc.marker(plan, "CANARY_PASSED")["verified_tasks"] == 30
    scc.atomic_json(record, {"theta_hat": 1, "se_IF": 100})
    with pytest.raises(ValueError, match="changed after independent validation"):
        scc.marker(plan, "CANARY_PASSED")


def test_canary_audit_rejects_invalid_inference_even_when_completion_counts_pass(tmp_path, monkeypatch):
    root, plan, _ = make_plan(tmp_path)
    expected = [dict(task_id=f"task-{n}-{method}", n=n, rep=0, method=method,
                     data_seed=n, training_seed=9) for n in (100, 300, 800, 1200, 2000, 3000)
                for method in ("projection", "autoencoder", "vae", "mediencoder", "mediencoder_l3zero")]
    manifest = dict(config={"B_requested": 200}, code_hashes={"fixture": "same"},
                    mechanism_hash="mechanism", run_hash="scientific-run")
    destination = root / "shards/shard-000"
    scc.atomic_json(root / "prepared/manifest.json", manifest)
    scc.atomic_json(destination / "manifest.json", manifest)
    for task in expected:
        artifact = destination / f"scores/{task['task_id']}.npz"
        artifact.parent.mkdir(exist_ok=True)
        artifact.write_bytes(b"checked-score-fixture")
        scc.atomic_json(destination / f"tasks/{task['task_id']}.json", dict(task, status="complete",
                         mechanism_hash="mechanism", theta_population=0, se_IF=0.2,
                         observed_data_sha256=f"paired-data-{task['n']}",
                         score_artifact=f"scores/{task['task_id']}.npz"))
    fake_dgp = Namespace(load_mechanism=lambda *a: Namespace(mechanism_hash="mechanism", truth=Namespace(value=0)))
    monkeypatch.setitem(sys.modules, "mediencoder.simulation.dgp", fake_dgp)
    checks = []
    def validate_scores(record, directory, truth):
        checks.append(record["task_id"])
        if record["se_IF"] != 0.2:
            raise ValueError("Independent score inference rejects se_IF")
    fake_cloud = Namespace(safe_file=scc.safe_source, validate_scores=validate_scores)
    fake_runner = Namespace(validate_manifest=lambda *a: None, collect_code_hashes=lambda: {"fixture": "same"},
                            execution_config=lambda *a: {}, build_tasks=lambda *a: expected,
                            validate_checkpoint=lambda *a: None)
    monkeypatch.setattr(scc, "cloud_module", lambda *a: fake_cloud)
    audit = scc.audit_canary(plan, destination, fake_runner)
    assert len(checks) == audit["verified_tasks"] == 30
    assert audit["paired_datasets"] == 6 and len(audit["files"]) == 60
    record_path = destination / f"tasks/{expected[0]['task_id']}.json"
    record = scc.read_json(record_path)
    record["se_IF"] = 100
    scc.atomic_json(record_path, record)
    with pytest.raises(ValueError, match="Independent score inference"):
        scc.audit_canary(plan, destination, fake_runner)


def test_scheduler_xml_records_array_and_ordinary_job_ids(monkeypatch):
    xml = '<job_info><queue_info><job_list state="running"><JB_job_number>7</JB_job_number><JB_name>mine</JB_name><state>r</state><tasks>1</tasks></job_list></queue_info></job_info>'
    monkeypatch.setattr(scc.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 0, xml, ""))
    assert scc.sge_snapshot() == {"7": {"name": "mine", "state": "r"}}
