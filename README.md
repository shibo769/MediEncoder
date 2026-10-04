# MediEncoder

Representation learning and cross-fitted estimation for causal mediation analysis.
This research implementation separates the estimator from simulated truth and uses
observation-level cross-fitted scores for dataset-specific uncertainty estimates.

## Repository layout

```text
src/mediencoder/
  estimation.py       # Shared cross-fitting and influence-score estimation
  training.py         # Coupled encoders, training losses, and tuning
  models.py           # Autoencoders, VAEs, and nuisance regressions
  nn_utils.py         # Neural-network construction and device selection
  simulation/         # Fixed wavelet mechanism; main and ablation experiments
  comparison/         # Separate polynomial experiment and comparison methods
  real_data/          # Observed-data loading, validation, and effect estimation
scripts/              # Command-line entry points
tests/                # Statistical, data-boundary, and execution regression tests
docs/                 # Methods, corrections, migration, and workflow notes
requirements/         # Validated environment snapshot
```

Generated results and real datasets stay outside version control. The earlier
`mediencoder_exmaple_code/` layout is retired; see [migration](docs/migration.md).

## Install

Use Python 3.11 or later in a virtual environment. Install the PyTorch build
appropriate for your hardware, then install this package:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

The development environment used Python 3.12 and PyTorch 2.8.0 with CUDA 12.6.
Its exact snapshot is in
[`requirements/validated-windows-cu126.txt`](requirements/validated-windows-cu126.txt).
That file describes the validated Windows environment; its CUDA-specific torch
wheel requires the corresponding PyTorch wheel index. CPU installations can use
the ordinary package dependencies. Other version combinations are not certified.

## Main simulation and alignment ablation

Start with a small execution check:

```bash
python -m mediencoder.simulation.runner --output-dir results/pilot --n 100 --reps 1 --pilot-epochs 2 --workers 1 --device cpu
```

For the full experiment:

```bash
python -m mediencoder.simulation.runner --output-dir results/formal --workers 1 --device cuda
```

To compute the first 50 replications now and extend the same run later, reserve
200 from the outset and change only the execution target:

```bash
python -m mediencoder.simulation.runner --output-dir results/formal --reps 200 --target-reps 50 --workers 1 --device cuda
python -m mediencoder.simulation.runner --output-dir results/formal --reps 200 --target-reps 100 --workers 1 --device cuda
python -m mediencoder.simulation.runner --output-dir results/formal --reps 200 --target-reps 200 --workers 1 --device cuda
```

Each phase includes every selected sample size and arm. Completed fits retain
their original seeds, scores, and scientific fingerprint. The manifest records
the reserved 200; summaries and status show the current target (50, 100, or 200).
Changing `--reps` itself changes the reserved scientific configuration and is
rejected when resuming. Omit `--target-reps` to compute the full reservation.

For parallel execution on standard GitHub-hosted CPU runners, see the manual
[cloud simulation workflow](docs/cloud-simulation.md). It prepares one shared
mechanism, computes the first 50 replications across 50 shards, and validates
the saved scores before assembling either table.

Defaults are six sample sizes (100, 300, 800, 1200, 2000, 3000), 200 replications,
and five arms: Projection, Autoencoder, VAE, tuned MediEncoder, and MediEncoder
with zero alignment weight. This is 6,000 fits, each potentially containing many
neural-network training runs. Both tables share the same tuned MediEncoder fits.
One worker is the conservative default. Workers are recycled between fits to
release allocations; increase concurrency only after checking resources.

The same command resumes completed checkpoints only when scientific settings,
source hashes, the saved mechanism, and runtime identity match. Small-epoch pilot
results are marked and cannot be merged with the formal experiment.
Failures retain their seeds and traces; `--retry-failed` archives the previous
attempt before rerunning the same task. It does not replace difficult datasets.

Outputs include a configuration manifest, task JSON records, per-person score NPZ
files, `summary.csv`, `main_table.tex`, `ablation_table.tex`, and `status.json`.
Requested, valid, failed, and pending counts are explicit. Partial summaries are
not final manuscript results. For durable logs and a refreshing local HTML report:

```bash
python -m mediencoder.simulation.monitor --output-dir results/formal --reps 200 --target-reps 50 --workers 1 --device cuda
```

The monitor launches the runner; use it instead of launching a second runner
against the same output directory. It forwards other runner options and displays
the saved configuration and current phase. Raise its target to 100 or 200 to
extend the same run after the previous phase has stopped.
Interrupting the monitor stops its owned runner and worker processes; completed
checkpoints are retained, and an interrupted fit is rerun with the same seed.

## Real data and comparison experiments

- [Real-data workflow and input contract](docs/real_data.md)
- [Polynomial comparison experiment and external baselines](docs/comparison.md)
- [Scientific corrections and remaining limitations](docs/corrections.md)
- [Validation performed for this release](docs/validation.md)

All supported MediEncoder workflows call the shared estimator. External baseline
adaptations are identified explicitly; unavailable standard errors are not
replaced by Monte Carlo error SDs. Real-data inputs, identifiers, private paths,
and third-party source repositories are not distributed here.

## Interpretation

Confidence intervals use each dataset's own cross-fitted score variability.
They remain asymptotic intervals and depend on identification, overlap,
representation, and nuisance-estimation assumptions. Passing software tests does
not establish these assumptions or guarantee nominal coverage.

The mediator alignment target intentionally uses stop-gradient; checkpoint
selection uses reconstruction loss. The implementation is not asserted to solve
an unconstrained joint scalar-objective argmin. These corrections do not revise
or certify any manuscript proof. Historical tables must be recomputed; this code
release does not validate their reported numbers.
