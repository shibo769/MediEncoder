# Corrections and interpretation

## Dataset-specific inference

Every observation must contribute exactly one finite uncentered cross-fitted
score in its original row position. With scores `s_i`, the estimate is their mean
and its estimated standard error is
`sqrt(sum_k(n_k * var(scores_in_fold_k, ddof=1)) / n**2)`.
Scores are centered separately within each evaluation fold. With four equally
sized folds, this is the square root of the average fold score variance divided
by `n`. Unequal folds use their sizes as weights; every fold needs at least two
subjects. This asymptotic estimator does not presume finite-sample independence
between estimates fitted on overlapping training data.
The 95% interval uses the standard-normal multiplier 1.959963984540054.
The standard deviation across simulation replications is an evaluation metric;
it is never substituted for a dataset's standard error.

Coverage uses each valid replication's own interval and the fixed population
target. Requested, valid, failed, and pending counts are saved separately.
Coverage conditional on successful fits does not eliminate the need to report
failures. Monte Carlo precision is separate from estimator uncertainty.

## Simulated truth and training data

The current main defaults use p=q=500, alpha entries Uniform(0,1), and independent
delta_0/delta_1 entries Uniform(0.9,1.1), with no contrast perturbation.
The main experiment draws a mechanism once using parameter seed 910000, saves it,
and samples independent datasets conditional on that mechanism. Structural and
measurement coefficients do not change with sample size or repeat. New runs use
additive polynomial loadings of degree 3, with independent Gaussian coefficients
of variance `coef_scale^2 / 3` for each block. The current default also uses the narrow independent delta and alpha ranges
specified above. Existing numerical mechanisms remain loadable with their
original arrays; the current defaults do not reconstruct the old frozen run.
Explicit legacy Haar runs use an independent 4,096-row latent pilot to fix atom
support, removing the old dependence on the estimation sample. Saved mechanisms
retain their original family and content hash; new polynomial runs reject an
incompatible Haar import rather than silently reuse it.

The population target `E[Y(1,M(0))]` uses analytic polynomial moments and converged
low-dimensional Gauss-Legendre quadrature. The estimation sample's mean true
conditional outcome is not substituted for the population target. Oracle values
remain in simulation metadata and are excluded from fitting and tuning.

The coupled loss is `lambda1 * MSE_X + lambda2 * MSE_M + lambda3 * MSE_align`.
Reconstruction uses raw MSE. The alignment term compares the mediator
code and its prediction after each row is divided by its own L2 norm, bounded
below by 1e-8. Positive rescaling of either nonzero row leaves this directional
loss unchanged when its norm is above the epsilon threshold. It no longer
penalizes discrepancies in code magnitude. Only the alignment loss uses these
normalized codes; reconstruction and nuisance estimation retain the original
encoder outputs. MSE retains its mean over subjects and coordinates; the nonnegative
lambda weights sum to one, with positive reconstruction weights. Optional input
standardization is a separate preprocessing setting fitted on representation-
training rows; outcome units are preserved.

## Natural effect contrasts

Marginal potential-outcome means require outcome regressions on pretreatment
covariates. Conditioning on the actual posttreatment mediator is not a substitute
for that baseline in an AIPW mean using propensity `P(A=1|X)`.
The effect path constructs theta11 and theta00 scores on the same cross-fitting
assignments as theta10. NIE, NDE, and TE use per-person score differences, followed
by the same size-weighted within-fold covariance calculation. Variation over random splits is
not an inferential standard error.

## Training and remaining scientific limitations

Alignment gradients update both encoders and the cross-factor predictor.
The mediator target is no longer detached. Checkpoint selection still uses the
weighted reconstruction criterion. Finite gradient training does not establish
a global optimum or a representation recovery rate.

The main defaults use 300 maximum epochs,
encoder widths 300/200, cross-factor widths 50/50, nuisance widths 300/300,
zero weight decay, and VAE KL weight 0.5. Main defaults fix lambda to
(.45,.45,.10), with ablation weights (.5,.5,0), and skip candidate selection.
The explicit grid option evaluates 36 positive-alignment candidates and nine
zero-alignment candidates.
These choices are recorded rather than silently reconciled with manuscript text.

In the retained historical Haar design, the fixed seed's atoms have limited support: only one of five overlaps the
covariate latent support. The prespecified draw is retained; it is not replaced
based on estimator performance. A finite coarse Haar map does not itself establish
latent identifiability. Simulation results do not verify representation assumptions.
The new cubic maps also do not by themselves prove that learned representations
recover the latent factors or that the auxiliary propensity has uniform overlap.

Scientific configuration and normalization changes define a new experiment, so old estimates are
not pooled into new results. Unfavorable rankings or coverage are not grounds for
changing seeds or omitting tasks. The manuscript and proofs need separate review.
