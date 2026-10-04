# Manual synthetic simulation on GitHub Actions

The `Synthetic main and ablation simulation` workflow computes the first 50
replications of the main wavelet experiment and alignment ablation. It is a
separate Linux CPU experiment. It never loads real data or combines the local
CUDA experiment with cloud results.

The workflow is manual (`workflow_dispatch`). It uses standard `ubuntu-24.04`
runners only, with a maximum of 20 concurrent jobs; it does not request a GPU,
larger runner, or self-hosted machine. GitHub documents standard hosted runners
as free for public repositories. Artifact storage is accounted for separately;
check repository visibility and account limits before dispatching. See
[hosted runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
and [Actions limits](https://docs.github.com/en/actions/reference/limits).

## Scientific configuration and pairing

The preparation job creates the mechanism **once**, from parameter seed 910000,
and saves a common manifest, `mechanism.json`, and `mechanism.npz`. Every shard
downloads these exact files. The scientific manifest reserves 200 replications,
but the execution target is 50. Both lambda grids (36 positive-alignment and 9
zero-alignment candidates), the 300-epoch cap, dimensions, noise parameters,
data seeds, and training seeds are unchanged.

There are 50 shards. Shard `i` computes replication indices congruent to `i`
modulo 50, covering all six sample sizes and all five arms. For the initial
target of 50, each shard has 30 fits. Together they contain exactly 1,500 fits.
Both tables reuse the same tuned MediEncoder fit. Each runner uses two workers,
one CPU thread per worker, and a fresh worker process after every fit.

Python 3.12.13, the CPU-only PyTorch 2.8.0 wheel, and the dependency versions in
`requirements/cloud-cpu.txt` are pinned. Before fitting, each shard must match
the prepared source, mechanism, scientific configuration, and runtime
fingerprints. A hosted-runner image/kernel mismatch is an explicit error, not
permission to mix environments. No persistent dependency cache is created by
this workflow.

## Canary, limits, and incomplete jobs

Before launching the matrix, one separate canary runs a full-grid, full-epoch
MediEncoder fit at n=100. Its manifest is marked `BENCHMARK_NOT_FOR_PAPER`; its
result is not included in the formal 50 replications. Failure or a 25-minute
canary timeout prevents the matrix from starting. A single timing measurement
does not establish the time required for larger sample sizes.

Each shard has a 300-minute computation deadline and a 330-minute job limit,
leaving time to upload completed checkpoints after interruption. A failed fit
keeps its original seed and failure record. Timeouts preserve completed task
records and scores; they do not invent missing values, replace seeds, reduce
training epochs, or skip lambda candidates. Missing tasks make aggregation fail
its completion check while preserving validated partial outputs. A new manual
dispatch starts a separate run; automatic cross-run cloud resumption is not
implemented in this first workflow.

## Results and validation

The aggregate job verifies matching manifests and source fingerprints, task
seeds, shard assignment, no duplicate tasks, and paired observed-data hashes.
For successful fits it checks score-file hashes, all n finite subject scores,
original subject indices, disjoint fold roles, once-only evaluation coverage,
and nuisance-training/validation splits. It independently recomputes the point
estimate, standard error, confidence interval, and coverage from saved scores.
The expected task inventory must be complete before completion is reported.

The final artifact is named **`synthetic-main-ablation-validated`**. It includes
the shared manifest and mechanism, every retained task JSON and score NPZ,
`summary.csv`, `main_table.tex`, `ablation_table.tex`, `progress.html`, and
`merge_audit.json`/`.md`. The audit distinguishes failed fits and missing tasks.
Coverage remains conditional on valid completed fits; incomplete tables are
provisional. Source data matrices, local files, model checkpoints, and private
datasets are not uploaded.

Every artifact expires after **one day**. Download the final artifact promptly
and retain the full directory, including task records and scores, to preserve
the option to extend the reserved experiment later under matching source/runtime
conditions. If aggregation failed before producing a final artifact, preserve
`synthetic-prepared` and all `synthetic-shard-*` artifacts instead.

Uploads have explicit uncompressed byte ceilings: 2 MiB for preparation, 3 MiB
per shard, and 100 MiB for the merged artifact. At most 252 MiB plus small file
inventories is uploaded by one complete run before compression. Exceeding a
ceiling fails packaging rather than uploading an unexpectedly large bundle.
One-day retention and these bounds limit storage exposure; they do not replace
checking existing account usage or guarantee that all account storage is free.
The workflow exposes only the initial target of 50; larger cloud targets need
an explicit resume and storage plan.

To validate downloaded artifacts locally without training, use the helper from
the same source revision. Its `merge` command requires the prepared artifact,
the directory containing all separate shard artifact directories, target 50,
and shard count 50. Keep these Linux CPU outputs separate from CUDA outputs.
