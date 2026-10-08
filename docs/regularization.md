# Paired MediEncoder with a retained mechanism

The manual **Paired MediEncoder on retained mechanism (weight decay 0)**
workflow (`.github/workflows/regularization.yml`) uses weight decay `0.0` for
every neural learner, matching the current training defaults. It fits
MediEncoder and its zero-alignment ablation on the same datasets and partitions.
It does not search over regularization strengths or rerun Projection,
Autoencoder, or VAE. This optional workflow explicitly retains the historical
Haar mechanism. The main simulation runner defaults to cubic polynomial loadings.

## Fixed experimental design

The scientific mechanism is loaded from
`experiments/mechanisms/wavelet_seed910000/`, which contains the exact synthetic
mechanism used in cloud run `37174159429`. It is never regenerated if the fixture
is missing or inconsistent. Keeping the fixture in the repository avoids a
dependency on the previous run's expiring GitHub artifact.

| Setting | Value |
|---|---|
| Mechanism hash | `20e82d3e80e64bbcec053828a145ece1f93932ecd0b81376c937cbb37ebb2e69` |
| Mechanism seed / data-seed base | `910000` / `880000` |
| Sample sizes | 100, 300, 800, 1200, 2000, 3000 |
| Observed dimensions | p=2000, q=1000 |
| Latent / learned dimensions | 5/5 and 10/10 |
| Replications in this execution phase | 100 per sample size and arm |
| Reserved replications | 200, allowing a later extension using unchanged configuration |
| Fitted arms | `mediencoder`, `mediencoder_l3zero` |
| Weight decay | 0 for all shared neural learners |
| Maximum epochs | 300, with the existing early stopping and scheduler |
| Tuning grids | 36 positive-alignment candidates; 9 zero-alignment candidates |
| Population target | `E[Y(1,M(0))] = 4.216720624294777` for this fixed mechanism |

Zero weight decay applies to the representation networks, auxiliary outcome
regressions used for tuning, and final nuisance regressions. Current coupled
losses use raw MSE with simplex-constrained lambda weights and stop-gradient
alignment. Each dataset's confidence interval uses its own cross-fitted scores
and within-fold score variance. The manifest records effective training settings
and scientific source hashes. A run using the current source requires a new
run identity and output directory; it cannot resume historical checkpoints
created from different source or settings.

The retained mechanism has a known limitation: only one of its five shared Haar
atoms has nonzero support on the covariate factors' interval `[-1,1]`. Its
covariate measurement mean depends on binary indicators of those factors, rather
than their within-interval values. This experiment deliberately retains that
mechanism for paired comparisons on the same saved data. Changing weight decay
does not repair the information lost by the measurement map, and results should
not be described as establishing factor recovery or general performance across
different mechanisms.

## Pairing and comparison

The data and training seeds are determined by the unchanged seed base, sample
size, and replication index. They do not depend on which methods are requested,
the weight decay, worker assignment, or completion order. Within this run, the
two arms share observed data and cross-fitting partitions for every `(n, rep)`.

Replications 0–49 also retain the prior run's data and training seeds. Before preparing the new
manifest, the cloud workflow regenerates all 300 previous datasets (50
replications at six sample sizes) from the retained mechanism and checks their
seeds and bitwise observed-data hashes against `reference_data_hashes.json`.
This check performs no neural-network fitting. A mismatch stops the workflow
before the canary or formal fits; matching seeds alone do not establish bitwise
pairing across platforms. Replications 50–99 have no
matching fit in the prior 50-replication run. Comparing an old 50-replication
summary directly against a new 100-replication summary does not isolate code or
setting changes from Monte Carlo variation. Both current arms use weight decay
0, so the within-run alignment comparison uses the same regularization setting.

## Historical regularization comparison

Earlier experiments compared shared weight decay 0 against 0.01. Their saved
manifests, results, and frozen source snapshots retain those original values.
`scripts/compare_regularization.py` audits exactly that historical comparison;
its check for 0.01 is not a current training setting. The historical runs used
the loss scaling and inference code in their respective source snapshots.
Current code has changed beyond weight decay and cannot be presented as a
weight-decay-only reproduction of those runs.

## Execution and artifacts

Dispatch `regularization.yml` manually with `target_reps=100` and
`weight_decay=0.0`. The workflow first verifies the previous datasets on the
current cloud runtime, then builds a fresh scientific manifest from the retained
mechanism. A full-grid, full-epoch n=100 MediEncoder canary uses the
same mechanism and regularization; it must succeed before the matrix starts.
This canary checks execution and resources, not statistical performance or a
guaranteed completion time. Its summary includes the observed-data hash for
direct inspection.

There are 50 shards, with at most 20 running concurrently. Shard `s` receives
replications satisfying `rep % 50 == s`: two replications, each containing all
six sample sizes and both arms. Each shard executes 24 fits using two CPU workers
that are recycled after every fit. The complete phase has 1,200 fits. A shard's
reports remain explicitly partial and retain the full 100-replication target.

Preparation, shards, and aggregation use the pinned CPU runtime. Runtime or
source mismatches fail rather than merging incompatible checkpoints. Completed
synthetic checkpoints are retained after a failed or timed-out shard, and the
aggregation checks run identity, mechanism, seeds, paired-data hashes, score
checksums, fold coverage, and score-based estimates and intervals. Missing or
failed fits remain explicit; an incomplete run cannot report success.

Artifact names are scoped to the new GitHub run:

- `synthetic-prepared`: fresh manifest and the retained mechanism.
- `synthetic-shard-<index>`: per-shard records and score files.
- `synthetic-main-ablation-validated`: validated merged records, summary, tables,
  and merge audit.

The generic main-table export leaves the unfitted baseline rows blank. Those
rows do not represent new baseline results; use the MediEncoder row, the paired
ablation table, and `summary.csv` for this experiment. Download artifacts within
their one-day retention period. No real data or private study files are used.
The workflow shares the existing simulation concurrency group, so a new manual
dispatch does not cancel an active simulation.
