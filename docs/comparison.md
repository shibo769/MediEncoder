# Natural-effect comparisons

Run the comparison through the installed package:

```sh
python scripts/run_comparison.py --smoke --output outputs/comparison-smoke
python scripts/run_comparison.py --replications 200 --n-grid 500,800,1500,2000 --output outputs/comparison-pilot
```

The default execution device is CPU and execution is serial with one Torch
thread. Use `--device cuda` only on an available device. The smoke mode is a
tiny integration check: one replication, n=160, reduced dimensions, one epoch,
and one MediEncoder tuning candidate. It is not an experiment for the paper.
All new runs are labeled **pilot**, `publication_ready=false`, until the
scientific protocol and resulting diagnostics have been reviewed.

## Fixed polynomial mechanism

This comparison preserves the paper's additive, no-interaction comparison
model, separate from the main wavelet simulation:

```
f_X ~ Uniform([-1,1]^3)
A | f_X ~ Bernoulli(expit(alpha' f_X))
f_M = delta (f_X ** 2) + A alpha_M + Normal(0, Sigma_U)
Y = sin(5 f_X)' beta + softplus(kappa' f_X) + f_M' gamma + A alpha_Y + epsilon
```

The observed X and M are degree-three additive polynomial maps plus independent
Gaussian measurement errors. Defaults are p=q=500, latent dimensions 3, learned
dimensions 7, all three observation/outcome noise SDs 1, and effect scale 0.5.
The mediator-error covariance eigenvalues are Uniform(1,2); treatment-effect
coefficients are 0.5 times Uniform(0.5,1.5). Structural and measurement
coefficients are drawn once with parameter seed 910000 and saved in
`mechanism.json` and `mechanism.npz`. A loaded mechanism can be reused with
`--mechanism PATH_PREFIX`.

The exact population truths are
`NIE = alpha_M @ gamma`, `NDE = alpha_Y`, `TE = NIE + NDE`.
No estimation sample or finite integration sample defines these targets.
The fixed coefficients are an intentional correction to the old workflow that
redrew coefficients for every replication. These runs do not reproduce the
old tables numerically.

## Methods and implementation provenance

| CLI name | Implementation | Inference supplied here |
| --- | --- | --- |
| `mediencoder` | Canonical corrected representation learning and four-fold EIF estimation | Score-based dataset-specific intervals |
| `projection`, `autoencoder`, `vae` | Alternative representations in the same canonical EIF pipeline | Score-based dataset-specific intervals |
| `nath-adapted` | Project-local alternating neural mediation implementation, with a learned covariate summary added to both regressions | Point estimates only |
| `dp2lm-adapted` | Project-local partially linear neural/SCAD implementation with HBIC penalty selection | Point estimates only |
| `imavae-adapted` | Project-local conditional-prior/posterior VAE and Monte Carlo counterfactual plug-in | Point estimates only |
| `lsem-ridge` | Project-local PCA/ridge linear-mediation reference | Point estimates only |

The three named adaptations come from the author's local experimental
`Baselines_and_Train.py` and `IMAVAE_and_Train.py`. They are **not claimed to be
verified reproductions of the original authors' software**. In particular,
Nath's covariate summary is an extension, the DP2LM adapter does not implement
the original publication's inferential procedure, and the IMAVAE adaptation
conditions its prior and outcome network on the observed X. Interpret the
comparison as one between these documented implementations.

Reference implementations consulted by the earlier project are
[deep-mediation](https://github.com/meet10may/deep-mediation),
[DP2LM](https://github.com/ShuoyangWang/dp2lm), and
[crumble](https://github.com/nt-williams/crumble).
None of those repositories, their private data, notebooks, results, or model
artifacts are vendored. The local deep-mediation checkout carries the MIT
license. No license file was present in the inspected DP2LM checkout, so this
package does not redistribute its source. The unverified old `crumble_py`
approximation is not presented as the official crumble estimator.

Defaults use 150 epochs per neural fit, width (300,300), and patience 25.
Nath uses 20 alternating iterations, DP2LM fits multiple stages and penalty
candidates, and IMAVAE uses its own alpha=beta=1 objective weights and 200
counterfactual latent draws. These budgets are not equal total compute, and
identical optimizer settings are not proof of fair method-specific tuning.
Use `--options method-options.json` to prespecify method-specific choices;
the complete supplied options are stored in the manifest. No tuning rule reads
the simulation truth. Final Nath coefficients are refitted on the final learned
summaries; collapsed summaries fail rather than receiving artificial noise.

## Paired data and inference accounting

Canonical adapters explicitly pin propensity clipping to 0.01 and both density
ratio caps to zero, independently of ambient environment variables. Any supplied
`numerical_safeguards` override is recorded in the method options. The serial
runner temporarily applies its declared epoch/width/patience budget to the
canonical shared configuration and restores the defaults after each fit.

Every method receives the same observed dataset and its own private copies.
Stable method-specific training seeds do not depend on method execution order.
No latent factor or population truth enters an estimator's input API. Dataset
hashes, mechanism hash, parameter/data/training seeds, package versions and source
hashes accompany the outputs. Default MediEncoder tuning includes lambda3=0
alongside positive values; the smoke-only singleton grid is explicitly recorded.

The canonical estimator constructs all three counterfactual scores using the
same folds. It subtracts scores **per observation** for NIE/NDE/TE before
estimating their variance, preserving covariance between component estimates.
All n observations must have finite scores; failed folds are not silently
discarded. Full effect scores are saved under `scores/` for successful EIF fits.

`CI_Length` averages each reported dataset-specific interval's length.
`Coverage` counts whether that interval contains that replication's exact
population effect. Monte Carlo error SD is a separate descriptive quantity
(`SD_error`), and is never substituted into an interval.

For point-only methods, interval columns are **Unavailable**. No Monte Carlo
interval is fabricated to fill the table. A method must supply a defensible
native interval or a separately designed, recorded bootstrap before inferential
claims can be compared.

Failures produce explicit rows with error messages, and the command exits with
status 1 if any fit failed. A bad method fit neither causes resampling nor
changes the other methods' dataset. Summaries separate planned, attempted,
pending, failed, successful, and interval-producing counts. Bias/SD/RMSE are
conditional on successful point fits. Coverage is conditional on valid
intervals, with `Coverage_all_attempts` also provided for methods advertising
inference. All failed interval-producing attempts count as noncoverage in that
second diagnostic. Do not hide failure rates when interpreting either summary.

At 200 successful replications, coverage near 0.95 has Monte Carlo SE about
0.0154. Small coverage differences are therefore not precise evidence of a
method ranking. `Coverage_MCSE` reports this binomial simulation uncertainty;
it is not an estimator standard error.

## External implementations and outputs

An optional official implementation can be wrapped without vendoring it:

```sh
python scripts/run_comparison.py --methods official-method --external-adapter official-method=my_adapter:fit --output outputs/external-pilot
```

The independently installed `my_adapter.fit(X, M, A, Y, *, seed, options)`
receives observed arrays only. It must return `effects` keyed by NIE/NDE/TE,
`effect_ci` and optionally `effect_se` with those keys (use `None` if unavailable),
plus `inference_available` and an `implementation` provenance string. Provide
the official repository/package version and all implementation settings there.
No external code is downloaded or installed automatically. R-based crumble,
for example, requires its own R dependencies and an audited array-to-R wrapper;
the Python package does not silently substitute the old approximation.

Each output directory contains the mechanism artifact, `manifest.json`, an
append-only `replications.jsonl`, `summary.json`, `table.md`, and available score
arrays. Existing nonempty output directories are rejected to preserve results.
`mediencoder.comparison.runner.merge_records(paths)` validates explicit JSONL
shards, rejects duplicate method/replication keys and mismatched paired dataset
hashes, and retains failures. Each CLI record includes a study fingerprint and a
method fingerprint; merging rejects changed source/runtime or method settings
and rows without this provenance. Shard bounds are excluded from these
fingerprints, while requested replication counts must still agree. It does not guess which historical output files
belong together. The old comparison CSV/LaTeX tables are not copied or relabeled
as corrected results.
