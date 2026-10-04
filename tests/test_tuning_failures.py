"""Tuning failure handling with stub fits; no neural training or GPU allocation."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from mediencoder import estimation


class TuningFailureTests(unittest.TestCase):
    def setUp(self):
        self.X = np.ones((8, 2))
        self.M = np.ones((8, 2))
        self.A = np.arange(8) % 2
        self.Y = np.arange(8, dtype=float)
        self.roles = dict(tilde_p=1, tilde_q=1, subtrain_idx=np.arange(4), val_idx=np.arange(4, 8))

    def vae(self, grid):
        return estimation._learn_representations_fixed_split(
            self.X, self.M, self.A, Y=self.Y, factor_method="vae",
            ae_cfg={"beta_kl_grid": grid}, **self.roles)

    def test_invalid_beta_grids_fail_before_training(self):
        with patch.object(estimation, "train_autoencoder") as fit:
            for grid in ([], [-1], [np.nan], [np.inf]):
                with self.subTest(grid=grid), self.assertRaises(ValueError):
                    self.vae(grid)
            fit.assert_not_called()

    def test_beta_candidate_failures_are_retained_and_all_failed_raises(self):
        with patch.object(estimation, "train_autoencoder", return_value=object()), \
             patch.object(estimation, "encode_with_autoencoder", return_value=np.ones((8, 1))), \
             patch.object(estimation, "_fit_outcome_predictor_and_eval",
                          side_effect=[ValueError("synthetic failure"), {"prediction_mse": .3}]):
            result = self.vae([.1, .2])
        info = result["rep_fit_info"]
        self.assertEqual(info["selected_beta_kl"], .2)
        self.assertEqual(info["beta_tuning_rows"][0]["failure"]["type"], "ValueError")
        self.assertIsNone(info["beta_tuning_rows"][1]["failure"])
        with patch.object(estimation, "train_autoencoder", side_effect=ValueError("all fail")):
            with self.assertRaisesRegex(RuntimeError, "All 2 beta candidates failed") as caught:
                self.vae([.1, .2])
        self.assertEqual(len(caught.exception.tuning_rows), 2)

    def test_nonfinite_beta_score_does_not_silently_choose_first(self):
        with patch.object(estimation, "train_autoencoder", return_value=object()), \
             patch.object(estimation, "encode_with_autoencoder", return_value=np.ones((8, 1))), \
             patch.object(estimation, "_fit_outcome_predictor_and_eval", return_value={"prediction_mse": np.nan}):
            with self.assertRaises(RuntimeError) as caught:
                self.vae([.1])
        self.assertEqual(caught.exception.tuning_rows[0]["failure"]["type"], "NonfinitePredictionError")

    def test_vae_resource_failure_aborts_instead_of_trying_next_candidate(self):
        for error in (MemoryError("CPU allocation"), torch.cuda.OutOfMemoryError("CUDA out of memory")):
            with self.subTest(error=type(error).__name__), \
                 patch.object(estimation, "train_autoencoder", side_effect=error) as fit:
                with self.assertRaises(type(error)) as caught:
                    self.vae([.1, .2])
            self.assertEqual(fit.call_count, 1)
            self.assertEqual(len(caught.exception.tuning_rows), 1)

    def test_lambda_oom_aborts_even_after_a_valid_candidate(self):
        first = dict(f_X_all=self.X, f_M_all=self.M)
        error = torch.cuda.OutOfMemoryError("CUDA out of memory")
        with patch.object(estimation, "_learn_representations_fixed_split", side_effect=[first, error]) as fit, \
             patch.object(estimation, "_fit_outcome_predictor_and_eval", return_value={"prediction_mse": .1}):
            with self.assertRaises(torch.cuda.OutOfMemoryError) as caught:
                estimation._select_lambda_for_fold(
                    self.X, self.M, self.A, self.Y, factor_method="mediencoder",
                    lambda_grid=[(.2, .5, .3), (.3, .4, .3), (.1, .5, .4)], **self.roles)
        self.assertEqual(fit.call_count, 2)
        self.assertEqual(len(caught.exception.tuning_rows), 2)
        self.assertIsNone(caught.exception.tuning_rows[0]["failure"])


if __name__ == "__main__":
    unittest.main()
