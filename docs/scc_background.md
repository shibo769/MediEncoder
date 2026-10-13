# MediEncoder background experiments on BU SCC

This workflow uses SCC batch jobs. After `qsub` accepts a job, a laptop can sleep,
disconnect SSH, or close its terminal. Training runs on allocated compute nodes.
The project, environment, frozen source, logs, scores, and task records remain on
project storage; an interactive login session is not the supervisor.

## Resource findings on 2026-10-13

Live scheduler configuration was checked for `shibo` and project `adsgrp`.

- Shared GPU queues permit two GPUs per user; the relevant shared GPU queues
  permit 16 CPU slots per user. Request one GPU per array element. More than one
  allocated GPU would not accelerate the current single-device runner.
- Public GPU queues have a separate four-GPU allowance, but the inspected L40S
  public queue instances were disabled. The default is therefore shared L40S.
  `--gpu-pool public` and `combined` are expansion options for when those queues
  become available; their quota allowances do not guarantee allocation.
- No matching global CPU-slot quota for `shibo` or `adsgrp` was found. The 896/
  1024-slot rules inspected apply to MPI queues and are not an OMP allowance.
- OMP allocates every core of a job on one node. Use many independent array jobs
  to distribute replications across nodes. Prefer requests of 4, 8, or 16 cores.
  The scheduler's limits on queued jobs are not a promise of available cores.
- `--cpu-slot-budget` is an execution budget chosen for this run, not an asserted
  SCC quota. A budget of 128 caps the CPU array at 128 cores across running
  elements; a budget of 256 permits up to 256. Other scheduler limits still apply.

Queue occupancy and limits can change. Refresh `qgpus -s`, `qstat -g c`, and the
relevant resource-quota rules before increasing concurrency. Never sum idle
slots from overlapping queues to estimate distinct available cores.

## Frozen experiment and validation gates

`scripts/scc_simulation.py init` verifies the source checkout or its SCC deployment
manifest, then copies all tracked files into an immutable run directory. Every
later phase verifies the source and generated batch scripts against this plan.
Local, GitHub, and SCC source must be synchronized before initialization.

The scientific configuration remains fixed: six sample sizes, five methods,
200 replications, p=q=500, cubic measurement loadings, independently drawn
delta matrices in [0.9,1.1], nuisance layers (300,300), row-L2 alignment,
fixed lambda weights, weight decay zero, and a maximum of 300 training epochs
with the existing early-stopping rule. Each CPU/GPU plan has its own runtime
fingerprint. Results from different devices or environments are not merged.

The phases are:

1. `prepare`: save one fixed mechanism and verified runtime manifest.
2. `benchmark`: measure balanced five-method workloads with a 20-epoch cap,
   enough tasks to exercise each candidate worker count, and recorded memory
   usage. These outputs are performance tests and cannot enter formal results.
3. `canary`: run all 30 fits for replication zero under the full training settings.
   This must pass before the formal array can be submitted.
4. `array`: 200 elements, one replication per element, 30 paired fits per element.
   Each has its own directory and lock. The completed canary fits are reused.
5. `merge`: independently check source/runtime fingerprints, the fixed mechanism,
   seeds, pairing, score checksums, fold coverage, estimates, within-fold standard
   errors, and confidence intervals. Missing fits and failures remain explicit.

## Example CPU plan

Run these commands on SCC after loading the Python module. Run paths must be
fresh and outside the mutable source checkout. The commands below are examples;
use the existing run paths reported for the current experiment instead of creating
duplicates.

```bash
module load python3/3.12.4
PY=/restricted/project/adsgrp/Allcodes/.mediencoder-scc-py312-cu126/bin/python
CODE=/restricted/project/adsgrp/Allcodes/MediEncoder
RUN=/restricted/project/adsgrp/Allcodes/mediencoder_runs/cpu_example

"$PY" "$CODE/scripts/scc_simulation.py" init \
  --source-dir "$CODE" --run-dir "$RUN" --python "$PY" \
  --device cpu --cores 16 --concurrency 16 --cpu-slot-budget 128 \
  --mem-per-core-gb 4 --wall-seconds 43200

"$PY" "$RUN/source/scripts/scc_simulation.py" submit --run-dir "$RUN" --phase prepare
```

The CPU benchmark compares 1/2/4/8/16 workers. Every worker uses one Torch/BLAS
thread. Formal elements request the selected number of cores, and the array
concurrency is capped by both `--concurrency` and the total CPU-slot budget.
For example, 8 selected workers and 16 simultaneous elements use up to 128 cores.
The requested budget may spend time queued rather than being allocated at once.

## Example GPU plan

```bash
RUN=/restricted/project/adsgrp/Allcodes/mediencoder_runs/gpu_example
"$PY" "$CODE/scripts/scc_simulation.py" init \
  --source-dir "$CODE" --run-dir "$RUN" --python "$PY" \
  --device cuda --gpu-type L40S --gpu-pool shared \
  --cores 8 --concurrency 2 --cpu-slot-budget 16 \
  --gpu-memory-gb 24 --mem-per-core-gb 4 --wall-seconds 43200

"$PY" "$RUN/source/scripts/scc_simulation.py" submit --run-dir "$RUN" --phase prepare
```

This compares 1/2/4/8 workers sharing one allocated L40S. The scheduler's
`CUDA_VISIBLE_DEVICES` is preserved. Formal concurrency is two GPUs, with the
number of workers per GPU selected from the benchmark. Compare aggregate fits
per hour and queue delay with the CPU plan before selecting a formal backend.
A short-epoch benchmark is provisional; the full-setting canary provides the
actual training and inference check.

## Submit later phases, inspect, resume, and stop

After each required gate passes, submit its successor. These submissions are
separate from the current read-only status commands; each returns a job ID.

```bash
TOOL="$RUN/source/scripts/scc_simulation.py"
"$PY" "$TOOL" status --run-dir "$RUN"
"$PY" "$TOOL" submit --run-dir "$RUN" --phase benchmark
"$PY" "$TOOL" submit --run-dir "$RUN" --phase canary
"$PY" "$TOOL" submit --run-dir "$RUN" --phase array
"$PY" "$TOOL" submit --run-dir "$RUN" --phase merge
```

Success markers, rather than an earlier job merely disappearing from the queue,
authorize later scientific phases. Submission records retain job IDs. Ambiguous
submission outcomes block automatic duplicates and require manual reconciliation
of the submission record against scheduler/accounting evidence; merely checking
`qstat` does not clear that block.
Use `--dry-run` with `submit` to inspect resource requests without submitting.

```bash
qstat -u shibo
"$PY" "$TOOL" status --run-dir "$RUN"
"$PY" "$TOOL" submit --run-dir "$RUN" --phase array --resume
```

Resume only after the previous array has ended. Completed task/score files are
retained and skipped. Interrupted fits remain pending and use their original
seeds. Failed fits are retained without silently replacing them; adding
`--retry-failed` explicitly retries those same seeds and archives failed attempts.
The runner stops its owned workers before its deadline and rescans durable
records. Timeout does not create a fake failed statistical replication.

To stop a run, identify its job IDs in that run's `submissions.json`, then use
`qdel JOB_ID` for those exact jobs. Do not cancel all of a user's jobs by default.
Check the queue afterward. There is no automatic resubmission service; uncompleted
work after a wall-time limit requires the explicit resume command.

Keep all versioned partial merges. Only a successful independent merge of all
6,000 valid fits marks the B200 experiment complete. Network checkpoints are not
stored; scores, inference, configuration, provenance, attempts, and logs are.

Official references: [BU batch submission](https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/),
[parallel jobs](https://www.bu.edu/tech/support/research/system-usage/running-jobs/parallel-batch/),
and [resources/runtime](https://www.bu.edu/tech/support/research/system-usage/running-jobs/resources-jobs/).
