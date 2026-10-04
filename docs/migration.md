# Migration from exploratory scripts

The public `mediencoder_exmaple_code/` directory is replaced by an installable
package. Its history remains in Git. Use a fresh results directory: source hashes
and scientific definitions changed, so the reorganized package refuses to resume
historical checkpoints silently.

| Earlier component | Supported replacement |
| --- | --- |
| `NNModel_and_Train.py` | `mediencoder.models` |
| `MediEncoder_and_Train.py` | `mediencoder.training` |
| `nn_utils.py` | `mediencoder.nn_utils` |
| `run_and_eval.py` | `mediencoder.estimation` |
| main wavelet generator in `DGP_and_estimate.py` | `mediencoder.simulation.dgp` |
| `simulation/run_tables_b400.py` | `python -m mediencoder.simulation.runner` (200 default) |
| local real-data crossfit/multiseed/bootstrap prototypes | documented `real_data` workflow |
| local comparison driver and MC-default table writer | documented `comparison` workflow |

Old diagnostic launchers, unpublished tables, private datasets, duplicate modules,
and vendored external repositories are not supported release entry points. They
are not relabeled as corrected code. The old bootstrap and multiseed prototypes
do not establish that the new IF workflow reproduces historical numbers.

The separately launched main/ablation run keeps its frozen source snapshot and
provenance. Packaging does not mutate that run. The default theta10 algorithm is
retained; this package adds strict input checks, effect components, and resource
handling. Tests check that optional effect computation preserves theta10 scores.
