# Real-data analysis

`mediencoder-real-data` is the supported observed-data workflow. It calls the same
`estimate_triply_IF` implementation used by the corrected simulations. It has no
access to latent factors or a simulation truth. Private data files are not included
in this repository, downloaded by the program, or uploaded anywhere. The ADNI
name below follows the manuscript and original analysis scripts; the loader does
not authenticate a supplied file's study provenance.

## Input and execution

Install the package with `pip install -e .`. Supply a local NPZ containing exactly:

| Array | Shape | Meaning |
|---|---|---|
| `X` | `(n, p)` | Numeric pretreatment covariates/proxies |
| `M` | `(n, q)` | Numeric mediator measurements |
| `A` | `(n,)` | Binary treatment, encoded 0 and 1 |
| `Y` | `(n,)` | Numeric outcome, retained on its supplied scale |

All four arrays must use the same subject order. Inputs with nonfinite values,
inconsistent row counts, missing treatment arms, or empty feature sets fail
explicitly. There is no automatic row deletion, imputation, feature screening,
outcome rescaling, or sample selection. Define the cohort and scientifically
appropriate pretreatment covariates before creating the input. Do not include
treatment-defining variables, their transformations, or post-treatment variables
in `X`. Deterministic feature construction specified in advance can be external;
sample-fitted transformations or feature selection must not use evaluation rows.

```bash
mediencoder-real-data --input /local/private/analysis.npz \
  --tilde-p 18 --tilde-q 2 --seed 1998 --device cpu \
  --output /local/private/results/mediencoder_seed1998
```

The dimensions above illustrate an explicit configuration, not an automatically
validated choice. Select dimensions and the partition seed before inspecting the
effect estimates. Both dimensions and the seed are required arguments. The default
method is MediEncoder and the default tuning grid has positive alignment weights;
`--grid-step` changes the grid. The `projection`, `autoencoder`, and `vae` methods
are also available. A JSON `--config` may supply `nn_cfg`, `ae_cfg`, `me_cfg`, and
`encode_cfg`; the core records the resolved representation configuration.

Alternatively, `python scripts/run_real_data.py` accepts the same arguments.

### Explicit RDS inputs

Install the optional `pyreadr` dependency to read local RDS files. For the original
ADNI file convention, use:

```bash
mediencoder-real-data \
  --covariates-rds /local/private/X_sixthorder.rds \
  --mediators-rds /local/private/M_AD_EWAS_01_lunnon_crosscortex.rds \
  --outcome-rds /local/private/Y.rds \
  --treatment-rds /local/private/X.rds \
  --treatment-column GDTOTAL --threshold 6 \
  --tilde-p 18 --tilde-q 2 --seed 1998 --device cpu \
  --output /local/private/results/mediencoder_seed1998
```

The treatment is defined as `GDTOTAL >= 6` in this example. The loader checks
matching row identifiers when they are available. If RDS objects have only default
row indexes, it requires `--assume-row-aligned`, which declares that their common
subject order was verified during data preparation. Conflicting identifiers are
always rejected. A literal treatment-defining column in the covariate table is
rejected; the user must also exclude derived treatment features when constructing
the design matrix. No raw data records are printed by the loader.

## Cross-fitting, preprocessing, and targets

Each of the four rotations has disjoint representation-training,
representation-validation, nuisance, and estimation folds. Tuning is performed
separately within each representation half; it does not select a single lambda
using the full analysis sample. `--preprocessing standardize` is the default:
the mean and standard deviation of each `X` and `M` feature are fitted on the
representation-training fold and then reused for its validation, nuisance, and
estimation rows. Constant training features use scale one. No full-sample
centering or standardization is performed. `--preprocessing none` retains the
supplied measurements. Outcomes are never standardized by this workflow.

The real-data API and CLI explicitly set probability clipping to 0.01 and disable
density-ratio soft/hard caps. These values do not inherit ambient `MEDIENC_*`
environment settings. Prespecified changes use `--clip-eps`, `--pi2-soft`, or
`--pi2-cap` and are saved in the manifest. A positive soft cap and hard cap cannot
be selected together. Such truncation choices can affect bias and must be
reported with the analysis; finite weights do not establish population overlap.

The coupled loss scales also use only observed representation-training data.
Stop-gradient alignment and reconstruction-based checkpointing are retained;
this release does not claim that those updates solve the ordinary joint scalar
minimization written in older manuscript versions.

The core first estimates `theta10 = E[Y(1, M(0))]`. With `return_effects=True`, it
also fits, within the same nuisance folds and learned pretreatment representation,
the marginal outcome regressions

`m_a(f_X) = E[Y | A=a, f_X]`, for `a=0,1`.

Their AIPW scores are

`S11 = m_1(f_X) + A/pi(f_X) * (Y - m_1(f_X))`

and

`S00 = m_0(f_X) + (1-A)/(1-pi(f_X)) * (Y - m_0(f_X))`.

The mediator is deliberately absent from these marginal outcome regressions:
using `mu_a(f_X, M_observed)` as their augmentation baseline does not integrate
the post-treatment mediator under intervention `a`. The additional fits occur
after the original `theta10` fitting path and use isolated random state.

Natural effects follow the manuscript's treatment-1 convention:

| Effect | Subject-level score | Estimand |
|---|---|---|
| NIE | `S11 - S10` | `E[Y(1,M(1)) - Y(1,M(0))]` |
| NDE | `S10 - S00` | `E[Y(1,M(0)) - Y(0,M(0))]` |
| TE | `S11 - S00` | `E[Y(1) - Y(0)]` |

Every subject must contribute exactly one finite score to each component.
Nonfinite scores fail the analysis; they are never silently removed. For each
effect, the estimate is its score mean and the SE is the sample standard deviation
of those scores divided by `sqrt(n)`. The program saves the full effect covariance
matrix `sample_covariance(effect_scores) / n`. This includes covariance between
components; it does not add component variances as if they were independent.
Intervals use the standard normal 0.975 quantile. Identification and asymptotic
validity still require the causal, overlap, representation, and nuisance-rate
conditions; a finite numeric interval alone does not verify those conditions.

## Outputs and provenance

- `manifest.json`: input-file SHA-256 hashes, package-source hashes, library
  versions, binary treatment definition/threshold, numerical safeguards,
  dimensions, seed, configuration, environment, and complete/failed status.
- `summary.json`: component means, NIE/NDE/TE, SEs, intervals, joint covariance,
  per-half tuning and fit information, and preprocessing statistics.
- `scores.npz`: component/effect scores in original subject order, numerical
  subject positions, and the actual fold indices. It does not contain X/M/A/Y.

Scores and fitted preprocessing statistics are still sensitive analysis output.
Keep them in an approved local location; do not commit them to a public repository.
The CLI refuses to overwrite a nonempty output directory. There is no partial-fit
resume facility: use a new directory to retry a failed fit with documented inputs,
code, and configuration. Hashes make that retry auditable.

## Earlier real-data entry points

The local prototypes are not alternate supported implementations:

| Earlier entry points | Replacement/status |
|---|---|
| `RealData/run_realdata_algorithm*.py`, `run_realdata_crossfit.py`, `test_crossfit_1seed.py` | Use this CLI, with within-half tuning, train-fold preprocessing, and matched score inference. |
| `RealData/realdataADNI.py`, `readData_train.py`, duplicated `run_and_eval.py` / `NNModel_and_Train.py` | Use explicit input loading and the shared package estimator/trainers. No private paths are embedded. |
| `RealData/run_realdata_multiseed.py`, `test_lambda*.py` | Different seeds/configurations may be assessed for stability, but across-seed SD is not a sampling SE; do not select a result by effect size or significance. |
| `RealData/run_realdata_bootstrap*.py` (B200/B300, fixed-lambda, linear, final variants) | Bootstrap, fixed-representation bootstrap, m-out-of-n, and averaged-partition bootstrap are not implemented by this canonical release. They are not relabeled as IF intervals. |

Historical numeric results cannot be certified or reproduced merely by retaining
their filenames. This workflow produces a newly traceable analysis and does not
claim that the old real-data tables have already been rerun.
