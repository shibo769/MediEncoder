# Validation of the corrected release

## October 14, 2026 joint alignment and exploratory measurement maps

Alignment now propagates through both encoders and the cross-factor predictor.
The row-L2 loss, epsilon, coordinate averaging, reconstruction losses, and fixed
lambda weights are unchanged. Opt-in injective mixed/deep tanh maps change only
the measurement maps, with fixed independent pilot calibration to the original
polynomial signal means and variances. Legacy polynomial/Haar configuration
payloads retain their hashes. Validation for this update runs on allocated SCC
compute nodes; the historical checks below do not validate these new changes.

Subsequent exploratory options expose the tanh input frequency and the existing
shared representation-input standardization in the simulation CLI. A nonzero
mediator contrast now retains the specified common treatment shift; shift-zero
mechanisms retain their previous numerical values. These options require their
own SCC validation and must not be represented as tests of the earlier snapshot.

## October 13, 2026 synchronized experiment configuration

The main defaults now use p=q=500, alpha Uniform(0,1), independent delta_0/delta_1
entries Uniform(0.9,1.1), and fixed ME weights (.45,.45,.10) / ablation (.5,.5,0).
Nuisance networks have two hidden layers (300,300). The alignment loss compares
per-observation row-L2-normalized mediator and predicted codes with epsilon 1e-8
and a detached target. Reconstruction stays in raw MSE units.

All 222 tests passed in 79.55 seconds on Python 3.11.11, PyTorch 2.7.0 CPU,
NumPy 2.2.5, and pytest 7.4.0. Checks include independent positive rescaling,
zero/near-zero rows, gradient routing, actual tiny MediEncoder/MediVAE and
fixed-weight estimator fits, skipping candidate selection, the smaller nuisance
architecture, DGP ranges, and legacy mechanism hash compatibility. These fits
are software checks; no formal simulation was launched. GPU runtime validation
and inferential performance of the directional alignment remain separate.

## October 6, 2026 code audit

The previously agreed within-fold variance estimator is now implemented in
simulation, observed-data effect contrasts, comparison outputs, and cloud score
verification. For evaluation-fold size `n_k`, the estimated covariance of the
overall score mean is `sum_k(n_k * sample_covariance(scores_k)) / n**2`.
Each sample covariance uses denominator `n_k - 1`. The point estimator, training
procedure, population truth, loss normalization, and stop-gradient are unchanged
by this variance update. The lambda simplex validation added earlier is retained.

Validation used the existing Python 3.11.11 environment with PyTorch 2.7.0 CPU,
NumPy 2.2.5, and the existing pure-Python pytest 7.4.0 installation. No packages
were installed or upgraded. The full suite passed 172 tests in 87.80 seconds;
after adding four more checks and retaining comparison-fold metadata, all 31
tests in the affected fold-variance, cloud, comparison, and integration modules
passed in 16.81 seconds. These overlapping counts must not be added together.

Checks include hand-calculated equal/unequal-fold variances, rejection of invalid
partitions and the former pooled SE, covariance-preserving effect contrasts,
saved-score/fold reconstruction, and the observed-data CLI using tiny synthetic
test inputs. No formal simulation or private real-data run was started. Full SCC
runtime validation and the next run's scientific configuration remain separate.
At that audit, observed `Var(X)`/`Var(M)` loss scales were retained. A subsequent
user instruction on the same day removed all variance divisors from the coupled
training and validation losses. Raw MSE retains its existing averaging over
subjects and coordinates; lambda weights must still sum to one. Input
preprocessing remains a separate, unchanged option.

After removing variance normalization, all 187 tests passed in 77.47 seconds.
Actual tiny MediEncoder and MediVAE fits verified raw training and validation
losses with ordinary and constant inputs, including the zero-alignment ablation;
invalid lambda sums remained rejected. No formal experiments were launched.

## October 3, 2026 release validation

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
