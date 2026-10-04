# Validation of the corrected release

Local validation on October 3, 2026 used Python 3.12, PyTorch 2.8.0, and one CPU
thread for tests, with the environment recorded in `requirements/`.

- 63 tests passed, including 20 parameterized unittest subchecks. Checks cover
  population truth, fixed mechanisms, oracle exclusion, sample splitting,
  score-based uncertainty, correlated effect contrasts, preprocessing boundaries,
  missing/invalid rows, RDS row alignment, failed tuning, memory failures, paired
  comparisons, provenance, run locking, and replication-phase extension.
- An actual four-method comparison smoke run completed MediEncoder, Nath-adapted,
  DP2LM-adapted, and IMAVAE-adapted on one shared synthetic dataset.
- Actual tiny neural fits verified that adding marginal effect components leaves
  theta10 scores unchanged. The observed-data CLI loaded synthetic NPZ input,
  fitted its regressions, and saved NIE/NDE/TE scores whose means and standard
  errors were independently recomputed in the integration test.
- The main runner completed Projection and the full MediEncoder tuning grid in
  a one-epoch execution check using spawned, recycled worker processes.
- A separate actual run extended a reserved two-replication pilot from target one
  to target two. The first task JSON and the original manifest retained their
  exact SHA-256 hashes, and only the new replication ran.
- Python syntax validation, package wheel construction, and Git whitespace checks
  passed. A CPU correctness workflow is included for GitHub Actions.

All small-epoch runs are execution checks, not estimates of inferential coverage
or manuscript results. The full main/ablation computation runs separately with a
frozen source snapshot; repackaging does not overwrite its provenance. The actual
private real-data analysis and full external-baseline experiment have not been
recomputed in this validation. The manuscript and proofs were not changed.
