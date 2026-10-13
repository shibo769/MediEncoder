"""Exercise real spawned workers, durable resumption, and bounded interruption."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

from mediencoder.simulation import runner


def command(output, *extra):
    return [sys.executable, "-u", "-m", "mediencoder.simulation.runner",
            "--output-dir", str(output), "--device", "cpu", "--n", "100",
            "--p", "12", "--q", "12", "--reps", "1", "--pilot-epochs", "1",
            "--workers", "2", "--tasks-per-worker", "1", *map(str, extra)]


def run_command(cmd, timeout=120):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=timeout)
    assert result.returncode == 0, result.stdout
    assert "Traceback" not in result.stdout, result.stdout
    return result


@pytest.fixture(scope="module")
def prepared_run(tmp_path_factory):
    output = tmp_path_factory.mktemp("prepared_runner")
    run_command(command(output, "--summarize-only"))
    return output


def copy_prepared(prepared, destination):
    destination.mkdir()
    for name in ("manifest.json", "mechanism.json", "mechanism.npz"):
        shutil.copyfile(prepared / name, destination / name)
    return destination


def test_deadline_is_execution_only_and_validated():
    ordinary = runner.parse_args(["--output-dir", "unused"])
    timed = runner.parse_args(["--output-dir", "unused", "--time-limit-seconds", "120"])
    assert runner.make_config(ordinary) == runner.make_config(timed)
    for invalid in ("0", "-1", "nan", "inf"):
        with pytest.raises(SystemExit):
            runner.parse_args(["--output-dir", "unused", "--time-limit-seconds", invalid])


def test_fixed_weights_are_named_correctly(tmp_path):
    from mediencoder.simulation.monitor import render_report
    config = dict(n_values=[100], methods=list(runner.METHODS), B_requested=1,
                  run_kind="PILOT_NOT_FOR_PAPER", device="cpu", lambda_mode="fixed")
    rows = runner.summarize_records([], config)
    _, table = runner.table_texts(rows, config)
    assert "Fixed ME" in table and "Tuning" not in table
    runner.atomic_json(tmp_path / "manifest.json", {"config": config})
    runner.write_reports(tmp_path, {}, config, "now", "running", 1)
    render_report(tmp_path)
    report = (tmp_path / "progress.html").read_text(encoding="utf-8")
    assert "same fixed-weight MediEncoder fits" in report
    assert "Tuning" not in report and "tuned MediEncoder" not in report


def test_parent_retries_temporary_progress_file_sharing(tmp_path, monkeypatch):
    task = {"task_id": "shared"}
    marker = tmp_path / "running" / "shared.json"
    marker.parent.mkdir()
    marker.write_text("{}")
    original = Path.unlink
    attempts = 0
    def temporarily_shared(path, *args, **kwargs):
        nonlocal attempts
        if path == marker:
            attempts += 1
            if attempts <= 2:
                raise PermissionError("temporary Windows file sharing")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", temporarily_shared)
    runner.clear_running_markers(tmp_path, [task])
    assert attempts == 3 and not marker.exists()


def test_busy_progress_marker_never_discards_a_durable_outcome(tmp_path, monkeypatch, capsys):
    task = {"task_id": "busy"}
    marker = tmp_path / "running" / "busy.json"
    marker.parent.mkdir()
    marker.write_text("{}")
    outcome = tmp_path / "tasks" / "busy.json"
    outcome.parent.mkdir()
    outcome.write_text('{"status":"complete"}')
    def busy(path, *args, **kwargs):
        raise PermissionError("Windows file remains shared")
    monkeypatch.setattr(Path, "unlink", busy)
    runner.clear_running_markers(tmp_path, [task], grace_seconds=0)
    assert outcome.read_text() == '{"status":"complete"}'
    assert "Durable task outcome is retained" in capsys.readouterr().err


def test_two_real_workers_recycle_and_resume_without_refitting(prepared_run, tmp_path):
    import numpy as np
    output = copy_prepared(prepared_run, tmp_path / "complete")
    run_command(command(output))
    manifest = json.loads((output / "manifest.json").read_text())
    records = [json.loads(path.read_text()) for path in (output / "tasks").glob("*.json")]
    assert len(records) == 5
    assert all(row["status"] == "complete" for row in records)
    # More than two distinct workers proves that completed worker slots recycled.
    assert len({row["worker_pid"] for row in records}) > 2
    assert len({row["observed_data_sha256"] for row in records}) == 1
    for row in records:
        runner.validate_checkpoint(row, manifest, output)
    before = {path.name: path.read_bytes() for path in (output / "tasks").glob("*.json")}
    run_command(command(output, "--workers", "1", "--time-limit-seconds", "60"))
    after = {path.name: path.read_bytes() for path in (output / "tasks").glob("*.json")}
    assert before == after
    assert json.loads((output / "status.json").read_text())["pending"] == 0
    # Execution-only parallelism/recycling must leave scientific output intact.
    serial = copy_prepared(prepared_run, tmp_path / "serial")
    run_command(command(serial, "--workers", "1", "--tasks-per-worker", "5"))
    for row in records:
        with np.load(output / row["score_artifact"], allow_pickle=False) as parallel_scores, \
             np.load(serial / row["score_artifact"], allow_pickle=False) as serial_scores:
            assert set(parallel_scores.files) == set(serial_scores.files)
            assert all(np.array_equal(parallel_scores[name], serial_scores[name])
                       for name in parallel_scores.files)


def slow_worker_script(path):
    # A deliberate wait exercises actual process cleanup without depending on
    # machine speed. The resume test above runs the real estimator in every arm.
    path.write_text('''import os
import sys
import time
from mediencoder.simulation import runner

def slow_task(task):
    runner.atomic_json(runner._WORKER["output_dir"] / "running" / (task["task_id"] + ".json"),
                       dict(task, started_at=runner.utc_now(), worker_pid=os.getpid()))
    time.sleep(120)
    return runner._run_task(task)

if __name__ == "__main__":
    runner._run_task = slow_task
    raise SystemExit(runner.main(sys.argv[1:]))
''', encoding="utf-8")
    return path


def wait_for_workers(output, proc, expected=2):
    until = time.monotonic() + 60
    while time.monotonic() < until:
        paths = list((output / "running").glob("*.json"))
        if len(paths) >= expected:
            return [json.loads(path.read_text())["worker_pid"] for path in paths]
        if proc.poll() is not None:
            pytest.fail(f"Runner exited before spawning workers: {proc.returncode}")
        time.sleep(.05)
    pytest.fail("Spawned workers did not become ready")


def assert_process_stopped(pid):
    # psutil is optional; on Linux the /proc check covers the deployment runtime.
    if sys.platform == "linux":
        assert not Path(f"/proc/{pid}").exists()
    else:
        try:
            import psutil
        except ImportError:
            return
        assert not psutil.pid_exists(pid)


def test_deadline_stops_owned_workers_and_releases_lock(prepared_run, tmp_path):
    output = copy_prepared(prepared_run, tmp_path / "deadline")
    script = slow_worker_script(tmp_path / "slow_deadline.py")
    cmd = [sys.executable, "-u", str(script), *command(output)[4:],
           "--time-limit-seconds", "60"]
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    with (tmp_path / "deadline.log").open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            worker_pids = wait_for_workers(output, proc)
            # The real lock must reject another process while the workers run.
            duplicate = subprocess.run(command(output, "--summarize-only"),
                                       capture_output=True, text=True, timeout=20)
            assert duplicate.returncode == 3, duplicate.stdout + duplicate.stderr
            assert proc.wait(timeout=75) == runner.PARTIAL_EXIT_CODE
            assert unrelated.poll() is None
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
            unrelated.terminate()
            unrelated.wait(timeout=10)
    status = json.loads((output / "status.json").read_text())
    assert status["phase"] == "time_limit_reached"
    assert (status["completed"], status["failed"], status["pending"]) == (0, 0, 5)
    assert status["active_tasks"] == []
    assert not list((output / "running").glob("*.json"))
    for pid in worker_pids:
        assert_process_stopped(pid)
    original_manifest = (output / "manifest.json").read_bytes()
    run_command(command(output, "--summarize-only"))
    assert original_manifest == (output / "manifest.json").read_bytes()
    # Resumption of the interrupted tasks uses the real estimator and seeds.
    run_command(command(output))
    status = json.loads((output / "status.json").read_text())
    assert (status["completed"], status["failed"], status["pending"]) == (5, 0, 0)


def test_worker_death_is_partial_execution_not_a_failed_statistical_fit(prepared_run, tmp_path):
    output = copy_prepared(prepared_run, tmp_path / "worker_crash")
    script = tmp_path / "crash_worker.py"
    script.write_text('''import os
import sys
import time
from mediencoder.simulation import runner

def crash_task(task):
    output = runner._WORKER["output_dir"]
    print("TEST_WORKER", os.getpid(), flush=True)
    runner.atomic_json(output / "running" / (task["task_id"] + ".json"),
                       dict(task, started_at=runner.utc_now(), worker_pid=os.getpid()))
    if task["method"] == "projection":
        until = time.monotonic() + 60
        while len(list((output / "running").glob("*.json"))) < 2:
            if time.monotonic() >= until:
                raise RuntimeError("Second worker did not start")
            time.sleep(.05)
        os._exit(17)
    time.sleep(120)
    return runner._run_task(task)

if __name__ == "__main__":
    runner._run_task = crash_task
    raise SystemExit(runner.main(sys.argv[1:]))
''', encoding="utf-8")
    result = subprocess.run([sys.executable, "-u", str(script), *command(output)[4:]],
                            capture_output=True, text=True, timeout=100)
    assert result.returncode == runner.PARTIAL_EXIT_CODE, result.stdout + result.stderr
    worker_pids = [int(line.split()[1]) for line in result.stdout.splitlines()
                   if line.startswith("TEST_WORKER ")]
    assert len(worker_pids) == 2
    for pid in worker_pids:
        assert_process_stopped(pid)
    status = json.loads((output / "status.json").read_text())
    assert status["phase"] == "interrupted_or_worker_error"
    assert (status["completed"], status["failed"], status["pending"]) == (0, 0, 5)
    assert not list((output / "tasks").glob("*.json"))
    assert not list((output / "running").glob("*.json"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX scheduler signals need a POSIX runtime")
@pytest.mark.parametrize("signal_name", ["SIGTERM", "SIGUSR2"])
def test_scheduler_signals_stop_workers_and_restore_parent_handler(
        prepared_run, tmp_path, signal_name):
    output = copy_prepared(prepared_run, tmp_path / signal_name)
    script = slow_worker_script(tmp_path / "slow_signal.py")
    cmd = [sys.executable, "-u", str(script), *command(output)[4:]]
    with (tmp_path / "signal.log").open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            worker_pids = wait_for_workers(output, proc)
            proc.send_signal(getattr(signal, signal_name))
            assert proc.wait(timeout=20) == runner.PARTIAL_EXIT_CODE
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
    status = json.loads((output / "status.json").read_text())
    assert status["phase"] == "interrupted_" + signal_name
    assert status["failed"] == 0 and status["pending"] == 5
    for pid in worker_pids:
        assert_process_stopped(pid)
    before = {getattr(signal, name): signal.getsignal(getattr(signal, name))
              for name in ("SIGTERM", "SIGUSR2")}
    assert runner.main(command(output, "--summarize-only")[4:]) == 0
    assert all(signal.getsignal(signum) == handler for signum, handler in before.items())


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group cleanup needs a POSIX runtime")
def test_monitor_sigterm_stops_its_isolated_runner_and_workers(prepared_run, tmp_path):
    output = copy_prepared(prepared_run, tmp_path / "monitor_signal")
    slow_script = slow_worker_script(tmp_path / "slow_monitored.py")
    supervisor = tmp_path / "signal_monitor.py"
    supervisor.write_text('''import sys
from pathlib import Path
from mediencoder.simulation import monitor

if __name__ == "__main__":
    output = Path(sys.argv[1])
    child_command = sys.argv[2:]
    monitor.monitor_command = lambda argv: (output, child_command)
    raise SystemExit(monitor.main([]))
''', encoding="utf-8")
    cmd = [sys.executable, "-u", str(supervisor), str(output),
           sys.executable, "-u", str(slow_script), *command(output)[4:]]
    with (tmp_path / "monitor.log").open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            worker_pids = wait_for_workers(output, proc)
            runner_pid = json.loads((output / "supervisor.json").read_text())["runner_pid"]
            proc.send_signal(signal.SIGTERM)
            assert proc.wait(timeout=25) == runner.PARTIAL_EXIT_CODE
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
    status = json.loads((output / "status.json").read_text())
    assert status["phase"] == "interrupted_SIGTERM"
    assert status["failed"] == 0 and status["pending"] == 5
    for pid in [runner_pid, *worker_pids]:
        assert_process_stopped(pid)
