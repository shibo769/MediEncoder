"""Distributed execution partitions fixed scientific tasks without changing them."""
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from mediencoder.simulation import runner


class SimulationShardingTests(unittest.TestCase):
    def config(self):
        return dict(n_values=[100, 300, 800, 1200, 2000, 3000],
                    methods=list(runner.METHODS), B_requested=200,
                    seed_base=880000, run_kind="FORMAL", device="cpu")

    def test_twenty_shards_exactly_cover_fifty_paired_replications(self):
        config = self.config()
        before = runner.canonical_json(config)
        phase = runner.execution_config(config, 50)
        tasks = list(runner.build_tasks(phase))
        groups = [runner.select_shard_tasks(tasks, index, 20) for index in range(20)]
        self.assertEqual(len(tasks), 50 * 6 * 5)
        observed = [task["task_id"] for group in groups for task in group]
        self.assertEqual(len(observed), len(set(observed)))
        self.assertEqual(set(observed), {task["task_id"] for task in tasks})
        for index, group in enumerate(groups):
            reps = set(range(index, 50, 20))
            self.assertEqual({task["rep"] for task in group}, reps)
            for rep in reps:
                paired = [task for task in group if task["rep"] == rep]
                self.assertEqual(len(paired), 6 * 5)
                for n in phase["n_values"]:
                    arms = [task for task in paired if task["n"] == n]
                    self.assertEqual({task["method"] for task in arms}, set(runner.METHODS))
                    self.assertEqual(len({task["data_seed"] for task in arms}), 1)
                    self.assertEqual(len({task["training_seed"] for task in arms}), 1)
            expected = [task for task in tasks if task["rep"] in reps]
            self.assertEqual(group, expected)
        self.assertEqual({len(group) for group in groups}, {60, 90})
        self.assertEqual(runner.canonical_json(config), before)

    def test_shard_cli_validation_and_identity_neutrality(self):
        common = ["--output-dir", "unused", "--reps", "200", "--target-reps", "50"]
        base = runner.parse_args(common)
        shard = runner.parse_args(common + ["--shard-index", "7", "--num-shards", "20"])
        self.assertEqual((base.shard_index, base.num_shards), (0, 1))
        self.assertEqual((shard.shard_index, shard.num_shards), (7, 20))
        with patch.object(runner, "lambda_grids", return_value=([], [])), \
             patch.object(runner, "training_configuration", return_value={}):
            self.assertEqual(runner.make_config(base), runner.make_config(shard))
        for options in (["--num-shards", "0"], ["--num-shards", "3", "--shard-index", "3"],
                        ["--shard-index", "-1"]):
            with self.subTest(options=options), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.parse_args(common + options)
        for index, count in ((-1, 2), (2, 2), (0, 0), (False, 2), (0, True)):
            with self.assertRaises(ValueError):
                runner.select_shard_tasks([], index, count)

    def test_shard_reports_keep_full_phase_target_and_local_progress(self):
        phase = runner.execution_config(self.config(), 50)
        phase.update(shard_index=0, num_shards=20)
        tasks = runner.select_shard_tasks(runner.build_tasks(phase), 0, 20)
        records = {task["task_id"]: dict(task, status="complete", theta_hat=.1,
                    theta_population=0., ci_lower=-.1, ci_upper=.3, se_IF=.2/1.959963984540054,
                    covered=True) for task in tasks}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            counts = runner.write_reports(output, records, phase, "now", "shard_finished", 1)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(counts, (90, 0, 1500))
            self.assertEqual(status["report_scope"], "shard_partial")
            self.assertEqual((status["execution_target_reps"], status["reserved_reps"]), (50, 200))
            self.assertEqual((status["shard_requested"], status["shard_completed"], status["shard_pending"]), (90, 90, 0))
            self.assertEqual((status["pending"], status["reserved_fits"]), (1410, 6000))
            with (output / "summary.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 30)
            self.assertTrue(all(row["B_requested"] == "50" and row["B_completed"] == "3" for row in rows))

    def test_distributed_run_requires_prepared_shared_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            args = runner.parse_args(["--output-dir", directory, "--num-shards", "20"])
            with patch.object(runner, "configure_environment"), \
                 patch.object(runner, "make_config", return_value=self.config()), \
                 patch.object(runner, "environment_identity") as environment:
                with self.assertRaisesRegex(ValueError, "shared prepared manifest"):
                    runner._main_locked(args)
                environment.assert_not_called()


if __name__ == "__main__":
    unittest.main()
