"""Inference and observed-data contracts; no simulation-scale training runs."""
import inspect
import os
import unittest
from unittest.mock import patch

import numpy as np
import torch

from mediencoder import training as trainer
from mediencoder import estimation as estimator


class EstimationContractTests(unittest.TestCase):
    def test_no_oracle_arguments_on_fit_or_tuning_functions(self):
        for function in (
            estimator.estimate_triply_IF,
            estimator._select_lambda_for_fold,
            estimator._learn_representations_fixed_split,
            trainer.compute_loss_scales,
            trainer.train_mediencoder,
            trainer.train_mediencoder_vae,
        ):
            names = inspect.signature(function).parameters
            self.assertFalse(set(names) & {
                "f_M_true_all", "f_M_train", "f_M_val", "mu10_true_vals",
            }, function.__name__)
            self.assertFalse(any(p.kind == p.VAR_KEYWORD for p in names.values()))

    def test_observable_scales_are_training_mediator_variance(self):
        X = np.arange(20, dtype=float).reshape(5, 4)
        M = np.arange(15, dtype=float).reshape(5, 3) / 2
        scales = trainer.compute_loss_scales(X, M)
        self.assertAlmostEqual(scales["var_X"], np.var(X))
        self.assertAlmostEqual(scales["var_M"], np.var(M))
        self.assertEqual(scales["var_align"], scales["var_M"])
        with self.assertRaises(TypeError):
            trainer.compute_loss_scales(X, M, f_M_train=M * 1e9)
        with self.assertRaises(ValueError):
            trainer.compute_loss_scales(X, np.full_like(M, np.nan))

    @staticmethod
    def _score_fixture():
        scores = np.array([-4., 1., 7., 2., 9., -2., 4., 8., 3.])
        indices = [np.array([8, 1, 4]), np.array([0, 6]),
                   np.array([2, 7]), np.array([3, 5])]
        outputs = [dict(phi=scores[i].copy(), theta_hat_IF=float(scores[i].mean()))
                   for i in indices]
        return scores, indices, outputs

    def test_population_score_se_and_unequal_fold_weighting(self):
        scores, indices, outputs = self._score_fixture()
        result = estimator._aggregate_crossfit_scores(len(scores), indices, outputs)
        expected_se = np.std(scores, ddof=1) / np.sqrt(len(scores))
        np.testing.assert_array_equal(result["crossfit_scores"], scores)
        self.assertAlmostEqual(result["theta_hat_IF"], scores.mean())
        self.assertAlmostEqual(result["se_IF"], expected_se)
        self.assertAlmostEqual(result["ci_upper"] - result["ci_lower"],
                               2 * 1.959963984540054 * expected_se)
        self.assertEqual(result["n_scores"], len(scores))
        self.assertNotAlmostEqual(scores.mean(), np.mean([o["theta_hat_IF"] for o in outputs]))

    def test_bad_scores_and_bad_coverage_fail_instead_of_dropping_subjects(self):
        for corruption in ("nan", "inf", "duplicate", "missing", "shape", "fold_mean"):
            with self.subTest(corruption=corruption):
                scores, indices, outputs = self._score_fixture()
                if corruption in {"nan", "inf"}:
                    outputs[0]["phi"][0] = float(corruption)
                elif corruption == "duplicate":
                    indices[0][0] = indices[1][0]
                elif corruption == "missing":
                    indices, outputs = indices[:-1], outputs[:-1]
                elif corruption == "shape":
                    outputs[0]["phi"] = outputs[0]["phi"][:-1]
                else:
                    outputs[0]["theta_hat_IF"] += 1
                with self.assertRaises(ValueError):
                    estimator._aggregate_crossfit_scores(len(scores), indices, outputs)

    def test_estimator_contract_with_stub_fits_and_original_subject_order(self):
        n = 11
        X = np.arange(n * 3, dtype=float).reshape(n, 3)
        M = np.arange(n * 2, dtype=float).reshape(n, 2)
        A = np.arange(n) % 2
        Y = np.arange(n, dtype=float) ** 2

        def representations(X, M, A, **kwargs):
            return dict(f_X_all=X[:, :1], f_M_all=M[:, :1],
                        rep_fit_info={"method": "projection"}, resolved_config={})

        def nuisances(f_X, f_M, A, Y, *, target_idx, **kwargs):
            phi = Y[target_idx]
            return dict(phi=phi, theta_hat_IF=float(phi.mean()))

        def nuisance_split(indices, A, **kwargs):
            return indices[:1], indices[1:]

        with patch.object(estimator, "_learn_representations_fixed_split", side_effect=representations), \
             patch.object(estimator, "_fit_nuisances_and_eval_theta", side_effect=nuisances), \
             patch.object(estimator, "_split_nuisance_fold", side_effect=nuisance_split), \
             patch.object(estimator, "_fit_outcome_predictor_and_eval", return_value={"prediction_mse": .1}):
            result = estimator.estimate_triply_IF(X, M, A, Y, tilde_p=1, tilde_q=1)
        np.testing.assert_array_equal(result["crossfit_scores"], Y)
        self.assertNotIn("theta_true", result)
        self.assertAlmostEqual(result["theta_hat_IF"], Y.mean())
        used = np.concatenate([f["estimation"] for f in result["fold_indices"]])
        np.testing.assert_array_equal(np.sort(used), np.arange(n))
        for fold in result["fold_indices"]:
            self.assertFalse(set(fold["estimation"]) & set(fold["nuisance"]))
        with self.assertRaises(TypeError):
            estimator.estimate_triply_IF(X, M, A, Y, tilde_p=1, tilde_q=1,
                                        mu10_true_vals=Y)

    def test_actual_alignment_gradient_contract(self):
        torch.manual_seed(123)
        model = trainer.MediEncoder(4, 3, 2, 2, (4,), (4,), (4,), activation="tanh")
        output = model(torch.randn(6, 4), torch.randn(6, 3), torch.arange(6) % 2)
        trainer._alignment_loss(output["z_M"], output["z_M_pred"]).backward()
        self.assertTrue(all(p.grad is None for p in model.encoder_M.parameters()))
        self.assertTrue(all(p.grad is None for p in model.decoder_M.parameters()))
        for network in (model.encoder_X, model.g_XM):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in network.parameters()
                                   if p.grad is not None), 0.)

    def test_candidate_failures_remain_auditable(self):
        X, M = np.ones((8, 2)), np.ones((8, 2))
        A, Y = np.arange(8) % 2, np.arange(8, dtype=float)
        kwargs = dict(tilde_p=1, tilde_q=1, factor_method="mediencoder",
                      lambda_grid=[(.2, .5, .3), (.3, .4, .3)],
                      subtrain_idx=np.arange(4), val_idx=np.arange(4, 8))
        representation = dict(f_X_all=X, f_M_all=M)
        with patch.object(estimator, "_learn_representations_fixed_split",
                          side_effect=[ValueError("synthetic candidate failure"), representation]), \
             patch.object(estimator, "_fit_outcome_predictor_and_eval", return_value={"prediction_mse": .1}):
            selected, rows = estimator._select_lambda_for_fold(X, M, A, Y, **kwargs)
        self.assertEqual(selected, (.3, .4, .3))
        self.assertEqual(rows[0]["failure"]["type"], "ValueError")
        self.assertIsNone(rows[1]["failure"])
        with patch.object(estimator, "_learn_representations_fixed_split", side_effect=ValueError("all fail")):
            with self.assertRaises(RuntimeError) as caught:
                estimator._select_lambda_for_fold(X, M, A, Y, **kwargs)
        self.assertEqual(len(caught.exception.tuning_rows), 2)
        self.assertIn("all fail", str(caught.exception))

    def test_one_epoch_uses_observed_scales_and_fixed_reconstruction_monitor(self):
        rng = np.random.default_rng(18)
        X, M = rng.normal(size=(8, 4)), rng.normal(size=(8, 3))
        Xv, Mv = 5 * rng.normal(size=(6, 4)), 9 * rng.normal(size=(6, 3))
        Av = np.arange(6) % 2
        with patch.dict(os.environ, {"MEDIENC_MONITOR_ALIGN": "1"}), \
             patch.object(trainer, "_alignment_loss", wraps=trainer._alignment_loss) as alignment:
            model, history, info = trainer.train_mediencoder(
                X, M, np.arange(8) % 2, latent_p=2, latent_q=2,
                X_val=Xv, M_val=Mv, A_val=Av, epochs=1, batch_size=4,
                hidden_dims_X=(4,), hidden_dims_M=(4,), hidden_dims_XM=(4,),
                activation="tanh", scheduler_type="none", verbose=False)
        self.assertGreater(alignment.call_count, 0)
        self.assertAlmostEqual(info["var_align"], np.var(M))
        self.assertEqual(info["checkpoint_criterion"], "reconstruction")
        device = next(model.parameters()).device
        with torch.no_grad():
            out = model(torch.tensor(Xv, dtype=torch.float32, device=device),
                        torch.tensor(Mv, dtype=torch.float32, device=device),
                        torch.tensor(Av, dtype=torch.float32, device=device))
            expected = .2 * torch.mean((out["X_recon"] - torch.tensor(Xv, dtype=torch.float32, device=device)) ** 2).item() / np.var(X)
            expected += .5 * torch.mean((out["M_recon"] - torch.tensor(Mv, dtype=torch.float32, device=device)) ** 2).item() / np.var(M)
        self.assertAlmostEqual(info["best_weighted_loss"], expected, places=5)

    def test_unsafe_legacy_evaluation_is_disabled(self):
        with self.assertRaises(RuntimeError):
            estimator.simulate_one_run(())
        with self.assertRaises(RuntimeError):
            estimator.evaluate_estimator_performance([1., 2.], [0., 0.])


if __name__ == "__main__":
    unittest.main()
