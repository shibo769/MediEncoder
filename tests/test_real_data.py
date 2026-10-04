import copy
import json
import os
from pathlib import Path
import random
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from mediencoder import estimation
from mediencoder.real_data import analyze, load_npz, load_rds, validate_observed_data
from mediencoder.real_data.cli import main as real_data_main, _resolved_device_metadata


class RealDataTests(unittest.TestCase):
    @staticmethod
    def observed(n=40):
        rng = np.random.default_rng(8)
        return dict(X=rng.normal(size=(n, 3)), M=rng.normal(size=(n, 2)),
                    A=np.arange(n) % 2, Y=rng.normal(size=n))

    def test_observed_validation_and_no_global_scaling(self):
        data = self.observed()
        checked = validate_observed_data(**data)
        for key in data:
            np.testing.assert_array_equal(checked[key], data[key])
        for corruption in ("missing", "one_arm", "length", "empty_features"):
            bad = copy.deepcopy(data)
            if corruption == "missing":
                bad["Y"][0] = np.nan
            elif corruption == "one_arm":
                bad["A"][:] = 1
            elif corruption == "length":
                bad["M"] = bad["M"][:-1]
            else:
                bad["X"] = bad["X"][:, :0]
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                validate_observed_data(**bad)

    def test_preprocessing_uses_only_representation_training_rows(self):
        data = self.observed(12)
        train = np.arange(4)
        X1, M1, info1 = estimation._preprocess_representation_inputs(data["X"], data["M"], train, mode="standardize")
        changed = copy.deepcopy(data)
        changed["X"][4:] += 1e5
        changed["M"][4:] -= 1e5
        X2, M2, info2 = estimation._preprocess_representation_inputs(changed["X"], changed["M"], train, mode="standardize")
        np.testing.assert_array_equal(X1[train], X2[train])
        np.testing.assert_array_equal(M1[train], M2[train])
        for name in ("X", "M"):
            np.testing.assert_array_equal(info1[name]["mean"], info2[name]["mean"])
            np.testing.assert_array_equal(info1[name]["scale"], info2[name]["scale"])
        np.testing.assert_allclose(X1[train].mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(M1[train].std(axis=0), 1)

    def test_covariance_uses_matched_subject_score_contrasts(self):
        s11 = np.array([0., 1., 4., 8., 7.])
        s10 = np.array([1., 2., 5., 2., 6.])
        s00 = np.array([2., 0., 2., 1., 3.])
        result = estimation.summarize_effect_scores(s11, s10, s00)
        contrasts = np.column_stack((s11-s10, s10-s00, s11-s00))
        np.testing.assert_allclose(result["effect_covariance"], np.cov(contrasts, rowvar=False) / len(s11))
        np.testing.assert_allclose(result["effect_scores"]["NIE"] + result["effect_scores"]["NDE"], result["effect_scores"]["TE"])
        self.assertAlmostEqual(result["effects"]["TE"], result["effects"]["NIE"] + result["effects"]["NDE"])
        self.assertAlmostEqual(result["effect_se"]["NIE"] ** 2, np.var(s11-s10, ddof=1) / len(s11))
        self.assertNotAlmostEqual(result["effect_se"]["NIE"] ** 2, (np.var(s11, ddof=1)+np.var(s10, ddof=1))/len(s11))
        with self.assertRaises(ValueError):
            estimation.summarize_effect_scores(s11, np.array([1., 2., np.nan, 2., 6.]), s00)

    def test_marginal_outcome_models_use_only_pretreatment_features(self):
        fX = np.arange(24, dtype=float).reshape(12, 2)
        A = np.arange(12) % 2
        Y = np.arange(12, dtype=float)
        tr, va, te = np.arange(4), np.arange(4, 8), np.arange(8, 12)
        fitted = []

        def fit(Xtr, Ytr, Xval, Yval, **kwargs):
            fitted.append((Xtr.copy(), Xval.copy()))
            return (float(Ytr.mean()),)

        with patch.object(estimation, "train_nuisance_nn", side_effect=fit), \
             patch.object(estimation, "predict_nn", side_effect=lambda model, X, **kwargs: np.full(len(X), model)):
            result = estimation._fit_marginal_outcome_scores(fX, A, Y, tr, va, te, np.full(len(te), .5))
        self.assertEqual(len(fitted), 2)
        for Xtr, Xval in fitted:
            self.assertEqual(Xtr.shape[1], 2)
            self.assertTrue(set(Xtr[:, 0]) <= set(fX[tr, 0]))
            self.assertTrue(set(Xval[:, 0]) <= set(fX[va, 0]))
        np.testing.assert_allclose(result["theta11"], 2 + 2 * (A[te] == 1) * (Y[te]-2))
        np.testing.assert_allclose(result["theta00"], 1 + 2 * (A[te] == 0) * (Y[te]-1))

    def test_effects_preserve_theta10_and_original_fold_order_without_training(self):
        data = self.observed(41)

        def representation(X, M, A, **kwargs):
            return dict(f_X_all=X, f_M_all=M, rep_fit_info={}, resolved_config={})

        def nuisance(fX, fM, A, Y, *, target_idx, **kwargs):
            scores = Y[target_idx] + .25
            return dict(phi=scores, theta_hat_IF=float(scores.mean()), propensity=np.full(len(target_idx), .5))

        def marginal(fX, A, Y, tr, va, te, propensity, **kwargs):
            self.assertFalse(set(te) & (set(tr) | set(va)))
            return dict(theta11=Y[te]+1, theta00=Y[te]-2)

        with patch.object(estimation, "_learn_representations_fixed_split", side_effect=representation), \
             patch.object(estimation, "_fit_nuisances_and_eval_theta", side_effect=nuisance), \
             patch.object(estimation, "_fit_outcome_predictor_and_eval", return_value={"prediction_mse": .1}), \
             patch.object(estimation, "_fit_marginal_outcome_scores", side_effect=marginal):
            baseline = estimation.estimate_triply_IF(**data, tilde_p=2, tilde_q=2, seed=21)
            extended = estimation.estimate_triply_IF(**data, tilde_p=2, tilde_q=2, seed=21, return_effects=True)
        for key in ("theta_hat_IF", "se_IF", "ci_lower", "ci_upper"):
            self.assertEqual(baseline[key], extended[key])
        np.testing.assert_array_equal(extended["component_scores"]["theta10"], data["Y"]+.25)
        np.testing.assert_array_equal(extended["component_scores"]["theta11"], data["Y"]+1)
        np.testing.assert_array_equal(extended["component_scores"]["theta00"], data["Y"]-2)

    def test_additional_fits_restore_random_state(self):
        random.seed(91); np.random.seed(91); torch.manual_seed(91)
        expected = (random.random(), np.random.rand(), torch.rand(1))
        random.seed(91); np.random.seed(91); torch.manual_seed(91)
        with estimation._isolated_random_state(10):
            random.random(); np.random.rand(); torch.rand(3)
        actual = (random.random(), np.random.rand(), torch.rand(1))
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2])

    def test_npz_rejects_oracle_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.npz"
            data = self.observed()
            np.savez(path, **data)
            for key, value in load_npz(path).items():
                np.testing.assert_array_equal(value, data[key])
            np.savez(path, **data, true_factors=np.ones((40, 1)))
            with self.assertRaises(ValueError):
                load_npz(path)

    def test_explicit_safeguards_ignore_ambient_environment(self):
        with patch.dict(os.environ, {"MEDIENC_CLIP_EPS": ".2", "MEDIENC_PI2_SOFT": "4", "MEDIENC_PI2_CAP": "0"}):
            self.assertEqual(estimation._resolve_numerical_safeguards({}),
                             {"clip_eps": .01, "pi2_soft": 0., "pi2_cap": 0.})
            with patch.object(estimation, "estimate_triply_IF", return_value={}) as fitted:
                analyze(**self.observed(), tilde_p=2, tilde_q=2, seed=1,
                        factor_method="projection")
            self.assertEqual(fitted.call_args.kwargs["numerical_safeguards"], {})
        for bad in ({"clip_eps": .5}, {"pi2_soft": -1},
                    {"pi2_cap": 2, "pi2_soft": 3}, {"clip_eps": np.nan}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                estimation._resolve_numerical_safeguards(bad)

    def test_rds_requires_row_alignment_and_rejects_treatment_feature(self):
        import pandas as pd
        data = self.observed(8)
        tables = {"X": pd.DataFrame(data["X"]), "M": pd.DataFrame(data["M"]),
                  "Y": pd.DataFrame(data["Y"]),
                  "A": pd.DataFrame({"GDTOTAL": 5 + data["A"]})}
        reader = types.SimpleNamespace(read_r=lambda path: {None: tables[path]})
        arguments = dict(covariates="X", mediators="M", outcome="Y", treatment_table="A")
        with patch.dict("sys.modules", {"pyreadr": reader}):
            with self.assertRaisesRegex(ValueError, "row identifiers"):
                load_rds(**arguments)
            loaded = load_rds(**arguments, assume_row_aligned=True)
            np.testing.assert_array_equal(loaded["A"], data["A"])
            tables["X"].index = [f"subject{k}" for k in range(8)]
            tables["M"].index = list(reversed(tables["X"].index))
            with self.assertRaisesRegex(ValueError, "do not agree"):
                load_rds(**arguments, assume_row_aligned=True)
            tables["X"]["GDTOTAL"] = 5 + data["A"]
            with self.assertRaisesRegex(ValueError, "treatment-defining"):
                load_rds(**arguments, assume_row_aligned=True)

    def test_realdata_cli_outputs_no_observed_records_and_hashes_input(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path, output = directory / "input.npz", directory / "results"
            data = self.observed()
            np.savez(path, **data)
            result = estimation.summarize_effect_scores(data["Y"]+1, data["Y"], data["Y"]-1)
            result.update(fold_indices=[], crossfit_scores=data["Y"], theta_hat_IF=float(data["Y"].mean()))
            with patch("mediencoder.real_data.analyze", return_value=result) as mocked:
                status = real_data_main(["--input", str(path), "--output", str(output),
                                         "--tilde-p", "2", "--tilde-q", "2", "--seed", "1", "--device", "cpu"])
            self.assertEqual(status, 0)
            np.testing.assert_array_equal(mocked.call_args.kwargs["Y"], data["Y"])
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(len(manifest["inputs"][0]["sha256"]), 64)
            self.assertIn("estimation.py", manifest["code_sha256"])
            self.assertEqual(manifest["numerical_safeguards"], {"clip_eps": .01, "pi2_soft": 0., "pi2_cap": 0.})
            self.assertEqual(manifest["treatment_definition"]["source"], "supplied_binary_A")
            self.assertIn("numpy", manifest["package_versions"])
            self.assertEqual(manifest["resolved_device"], "cpu")
            with np.load(output / "scores.npz") as saved:
                self.assertFalse({"X", "M", "A", "Y"} & set(saved.files))

    def test_resolved_device_is_recorded_and_explicit_mismatch_fails(self):
        with patch("mediencoder.nn_utils.device", torch.device("cpu")):
            self.assertEqual(_resolved_device_metadata("auto"), {"resolved_device": "cpu"})
            with self.assertRaisesRegex(RuntimeError, "restart the process"):
                _resolved_device_metadata("cuda")
        with patch("mediencoder.nn_utils.device", torch.device("cuda:0")), \
             patch.object(torch.cuda, "get_device_name", return_value="synthetic GPU"), \
             patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)):
            info = _resolved_device_metadata("auto")
        self.assertEqual(info["resolved_device"], "cuda:0")
        self.assertEqual(info["gpu_name"], "synthetic GPU")
        self.assertEqual(info["gpu_compute_capability"], [8, 9])


if __name__ == "__main__":
    unittest.main()
