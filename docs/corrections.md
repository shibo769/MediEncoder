# Corrections and interpretation

## Dataset-specific inference

Every observation must contribute exactly one finite uncentered cross-fitted
score in its original row position. With scores `s_i`, the estimate is their mean
and its estimated standard error is
`sqrt(sum((s_i - mean(s))**2) / (n * (n - 1)))`.
The 95% interval uses the standard-normal multiplier 1.959963984540054.
The standard deviation across simulation replications is an evaluation metric;
it is never substituted for a dataset's standard error.

Coverage uses each valid replication's own interval and the fixed population
target. Requested, valid, failed, and pending counts are saved separately.
Coverage conditional on successful fits does not eliminate the need to report
failures. Monte Carlo precision is separate from estimator uncertainty.

## Simulated truth and training data

The main experiment draws a mechanism once using parameter seed 910000, saves it,
and samples independent datasets conditional on that mechanism. Structural and
measurement coefficients and Haar atoms do not change with sample size or repeat.
An independent 4,096-row latent pilot fixes atom support, removing the old
measurement-map dependence on the estimation sample.

The population target `E[Y(1,M(0))]` uses analytic polynomial moments and converged
low-dimensional Gauss-Legendre quadrature. The estimation sample's mean true
conditional outcome is not substituted for the population target. Oracle values
remain in simulation metadata and are excluded from fitting and tuning.

Coupled-loss scales use observed X and M from the representation-training fold.
Validation reuses those scales. Real-data preprocessing also fits within that
training fold; outcome units are preserved.

## Natural effect contrasts

Marginal potential-outcome means require outcome regressions on pretreatment
covariates. Conditioning on the actual posttreatment mediator is not a substitute
for that baseline in an AIPW mean using propensity `P(A=1|X)`.
The effect path constructs theta11 and theta00 scores on the same cross-fitting
assignments as theta10. NIE, NDE, and TE use per-person score differences, followed
by variance estimation that retains covariance. Variation over random splits is
not an inferential standard error.

## Training and remaining scientific limitations

Retained stop-gradient alignment updates the predictor of the mediator
representation without propagating that term through the mediator target.
Checkpoint selection uses the weighted reconstruction criterion. The code does
not identify this procedure with solving an unconstrained joint scalar objective,
nor establish a representation recovery rate.

The main rerun preserves the historical training settings: 300 maximum epochs,
encoder widths 300/200, cross-factor widths 50/50, nuisance widths 300/300/300,
zero weight decay, and VAE KL weight 0.5. Tuning evaluates 36 strictly positive
alignment-weight candidates; the ablation evaluates nine zero-alignment candidates.
These choices are recorded rather than silently reconciled with manuscript text.

The fixed seed's Haar atoms have limited support: only one of five overlaps the
covariate latent support. The prespecified draw is retained; it is not replaced
based on estimator performance. A finite coarse Haar map does not itself establish
latent identifiability. Simulation results do not verify representation assumptions.

Corrections change the population target and normalization, so old estimates are
not pooled into new results. Unfavorable rankings or coverage are not grounds for
changing seeds or omitting tasks. The manuscript and proofs need separate review.
