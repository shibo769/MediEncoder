"""Regression checks for interval aggregation, pairing, and restart provenance."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from mediencoder.simulation import runner


class CorrectedRunnerTests(unittest.TestCase):
    def config(self):
        return dict(n_values=[100], methods=["mediencoder", "mediencoder_l3zero"],
                    B_requested=4, seed_base=880000, run_kind="FORMAL")

    def record(self, rep, theta, lower, upper, method="mediencoder"):
        return dict(task_id=f"test{rep}", n=100, method=method, rep=rep,
                    status="complete", theta_hat=theta, theta_population=0.,
                    ci_lower=lower, ci_upper=upper, se_IF=(upper-lower)/3.919927969080108,
                    covered=lower <= 0 <= upper)

    def test_dataset_specific_intervals_not_monte_carlo_sd(self):
        # MC-SD intervals would cover both values; actual narrow intervals cover neither.
        records = [self.record(0, -1, -1.1, -.9), self.record(1, 1, .9, 1.1),
                   dict(n=100, method="mediencoder", status="failed")]
        row = runner.summarize_records(records, self.config())[0]
        self.assertEqual(row["Coverage"], 0.)
        self.assertAlmostEqual(row["CI_Length"], .2)
        self.assertAlmostEqual(row["RMSE"], 1.)
        self.assertAlmostEqual(row["SD"], 2**.5)
        self.assertEqual((row["B_requested"], row["B_completed"], row["B_failed"], row["B_pending"]), (4,2,1,1))

    def test_same_data_and_training_seeds_across_arms(self):
        config = self.config()
        tasks = list(runner.build_tasks(config))
        self.assertEqual(len(tasks), 8)
        self.assertEqual(tasks[0]["data_seed"], tasks[1]["data_seed"])
        self.assertEqual(tasks[0]["training_seed"], tasks[1]["training_seed"])
        self.assertNotEqual(tasks[0]["data_seed"], tasks[2]["data_seed"])
        self.assertEqual(tasks, list(runner.build_tasks(config)))

    def test_requested_observed_dimensions_are_recorded_without_changing_defaults(self):
        runner.configure_environment("cpu")
        from mediencoder.estimation import SHARED_TRAIN_CFG
        previous = dict(SHARED_TRAIN_CFG)
        try:
            args = runner.parse_args(["--output-dir", "unused", "--p", "1000", "--q", "1000",
                                      "--n", "800,1500", "--reps", "50",
                                      "--arms", "mediencoder,mediencoder_l3zero"])
            config = runner.make_config(args)
            self.assertEqual((config["dgp"]["p"], config["dgp"]["q"]), (1000, 1000))
            self.assertEqual(len(list(runner.build_tasks(config))), 200)
            default = runner.make_config(runner.parse_args(["--output-dir", "unused"]))
            self.assertEqual((default["dgp"]["p"], default["dgp"]["q"]), (2000, 1000))
            for option in ("--p", "--q"):
                with self.assertRaises(SystemExit):
                    runner.parse_args(["--output-dir", "unused", option, "0"])
        finally:
            SHARED_TRAIN_CFG.clear()
            SHARED_TRAIN_CFG.update(previous)

    def test_execution_target_preserves_reservation_and_existing_task_seeds(self):
        config = self.config()
        before = json.dumps(config, sort_keys=True)
        first = runner.execution_config(config, 2)
        later = runner.execution_config(config, 4)
        self.assertEqual(first["B_requested"], 2)
        self.assertEqual(first["B_reserved"], 4)
        self.assertEqual(json.dumps(config, sort_keys=True), before)
        self.assertEqual(list(runner.build_tasks(first)), list(runner.build_tasks(later))[:4])
        records = {str(rep): self.record(rep, 0, -1, 1) | {"rep": rep} for rep in range(4)}
        selected = runner.phase_records(records, first)
        self.assertEqual(len(selected), 2)
        self.assertEqual(len(records), 4)
        self.assertEqual(runner.summarize_records(list(selected.values()), first)[0]["B_pending"], 0)
        with self.assertRaises(ValueError):
            runner.execution_config(config, 5)

    def test_phase_cli_and_monitor_forwarding(self):
        from mediencoder.simulation.monitor import monitor_command
        args = runner.parse_args(["--output-dir", "unused", "--reps", "200", "--target-reps", "50"])
        self.assertEqual((args.reps, args.target_reps), (200, 50))
        _, command = monitor_command(["--output-dir", "unused", "--reps", "200", "--target-reps", "50", "--n", "100"])
        self.assertEqual(command[command.index("--target-reps") + 1], "50")
        self.assertEqual(command[command.index("--n") + 1], "100")
        with self.assertRaises(SystemExit):
            runner.parse_args(["--output-dir", "unused", "--reps", "50", "--target-reps", "100"])

    def test_phase_reports_show_target_and_reservation(self):
        from mediencoder.simulation.monitor import render_report
        config = self.config() | dict(dgp={"p": 20, "q": 10, "bar_p": 2, "bar_q": 2}, tilde_p=3, tilde_q=3, device="cpu")
        target = runner.execution_config(config, 2)
        records = {str(rep): self.record(rep, 0, -1, 1) | {"rep": rep} for rep in range(4)}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            runner.atomic_json(output / "manifest.json", {"config": config})
            runner.write_reports(output, records, target, "now", "finished", 1)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual((status["execution_target_reps"], status["reserved_reps"]), (2, 4))
            self.assertEqual((status["completed"], status["requested"], status["reserved_fits"]), (2, 4, 8))
            render_report(output)
            html = (output / "progress.html").read_text(encoding="utf-8")
            self.assertIn("2 replications per size and arm", html)
            self.assertIn("with 4 reserved", html)
            self.assertNotIn("200 replications", html)

    def test_monitor_stops_only_its_live_owned_windows_child_tree(self):
        from unittest.mock import Mock, patch
        from mediencoder.simulation import monitor
        child = Mock(pid=12345)
        child.poll.return_value = None
        with patch.object(monitor.os, "name", "nt"), patch.object(monitor.subprocess, "run") as stop:
            monitor.stop_owned_process_tree(child)
            self.assertEqual(stop.call_args.args[0], ['taskkill', '/PID', '12345', '/T', '/F'])
            child.wait.assert_called_once_with(timeout=10)
        child.poll.return_value = 0
        with patch.object(monitor.subprocess, "run") as stop:
            monitor.stop_owned_process_tree(child)
            stop.assert_not_called()

    def test_shared_tuned_results_in_both_table_outputs(self):
        records = [self.record(0, -1, -2, .5), self.record(1, 1, -.5, 2),
                   self.record(0, -2, -3, 1, "mediencoder_l3zero"),
                   self.record(1, 2, -1, 3, "mediencoder_l3zero")]
        rows = runner.summarize_records(records, self.config())
        main, ablation = runner.table_texts(rows, self.config())
        self.assertIn("MediEncoder & 1.414 & 1.000 & 2.500 & 1.000", main)
        self.assertIn("100 & 2.828 & 1.414 & 2.000 & 1.000 & 4.000 & 2.500", ablation)

    def test_resume_rejects_source_or_config_drift(self):
        manifest = dict(config={"B":200}, code_hashes={"source.py":"abc"},
                        mechanism_hash="fixed", run_hash="same")
        runner.validate_manifest(manifest, manifest)
        modified = dict(manifest, code_hashes={"source.py":"different"})
        with self.assertRaisesRegex(ValueError, "code_hashes changed"):
            runner.validate_manifest(manifest, modified)
        modified = dict(manifest, config={"B":201})
        with self.assertRaisesRegex(ValueError, "config changed"):
            runner.validate_manifest(manifest, modified)

    def test_completed_checkpoint_requires_intact_score_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "score.npz"
            path.write_bytes(b"example saved scores")
            record = dict(status="complete", task_id="sample", run_hash="run",
                          score_artifact="score.npz", score_artifact_sha256=runner.file_hash(path))
            runner.validate_checkpoint(record, {"run_hash":"run"}, directory)
            path.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "damaged score artifact"):
                runner.validate_checkpoint(record, {"run_hash":"run"}, directory)

    def test_atomic_json_has_no_nonstandard_nan_and_replaces(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"task.json"
            runner.atomic_json(path, {"x":float("nan"), "complete":False})
            self.assertEqual(json.loads(path.read_text()), {"x":None, "complete":False})
            runner.atomic_json(path, {"complete":True})
            self.assertEqual(json.loads(path.read_text()), {"complete":True})
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_no_success_is_explicit_not_dropped(self):
        rows = runner.summarize_records([], self.config())
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["B_pending"], 4)
        self.assertIsNone(rows[0]["Coverage"])

    def test_output_directory_lock_excludes_duplicate_run(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = runner.acquire_run_lock(directory)
            try:
                with self.assertRaisesRegex(RuntimeError, "refusing duplicate jobs"):
                    runner.acquire_run_lock(directory)
            finally:
                lock.release()
            resumed_lock = runner.acquire_run_lock(directory)
            resumed_lock.release()

    def test_actual_config_construction_and_shared_pilot_epochs(self):
        runner.configure_environment("cpu")
        from mediencoder.estimation import SHARED_TRAIN_CFG, _shared_cfg, _merge_method_cfg
        previous_epochs = SHARED_TRAIN_CFG["epochs"]
        try:
            args = runner.parse_args(["--output-dir", "unused", "--pilot-epochs", "2"])
            pilot = runner.make_config(args)
            self.assertEqual(pilot["run_kind"], "PILOT_NOT_FOR_PAPER")
            self.assertEqual(len(pilot["lambda_grid"]), 36)
            self.assertEqual(len(pilot["lambda_grid_zero"]), 9)
            runtime = runner.runtime_training_configuration(pilot["training"])
            for name in ("nn_cfg", "ae_cfg", "me_cfg"):
                self.assertEqual(pilot["training"][name]["epochs"], 2)
                effective = _merge_method_cfg(runtime[name], _shared_cfg(), method=name)
                self.assertEqual(effective["epochs"], 2)
            formal = runner.make_config(runner.parse_args(["--output-dir", "unused"]))
            self.assertEqual(formal["run_kind"], "FORMAL")
            for name in ("nn_cfg", "ae_cfg", "me_cfg"):
                self.assertEqual(formal["training"][name]["epochs"], 300)
            self.assertEqual(SHARED_TRAIN_CFG["epochs"], 300)
        finally:
            SHARED_TRAIN_CFG["epochs"] = previous_epochs


if __name__ == "__main__":
    unittest.main()
