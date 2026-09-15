# MediEncoder

**MediEncoder** is a representation-learning framework for nonlinear, high-dimensional causal mediation analysis. It jointly learns covariate and mediator representations with a coupled encoder–decoder architecture, then uses a cross-fitted efficient influence-function estimator for natural direct, natural indirect, and total effects.

This repository contains the core Python implementation and simulation scripts.

## Code structure

The implementation is in [`mediencoder_exmaple_code/`](mediencoder_exmaple_code/) (the directory name is spelled this way in the repository).

| Module | Purpose |
| --- | --- |
| `nn_utils.py` | MLP construction and device selection |
| `NNModel_and_Train.py` | Autoencoder/VAE models and nuisance-network trainers |
| `MediEncoder_and_Train.py` | Coupled model, reconstruction and alignment objectives, and tuning grid |
| `DGP_and_estimate.py` | Data-generating processes, factor projections, and target effects |
| `run_and_eval.py` | Cross-fitted estimation, tuning, and single-replication evaluation |
| `simulation/run_tables_b400.py` | Method comparison and alignment-loss ablation |

## Setup

The source imports **NumPy, pandas, SciPy, scikit-learn, and PyTorch**. Use a Python environment compatible with these packages; dependency versions are not currently pinned in the repository.

```bash
git clone https://github.com/shibo769/MediEncoder.git
cd MediEncoder/mediencoder_exmaple_code
python -m pip install numpy pandas scipy scikit-learn torch
```

## Run the simulation

The simulation compares projection, autoencoder, VAE, and MediEncoder representations, and includes an ablation with the alignment weight fixed to zero.

From `mediencoder_exmaple_code/`, run the following example configuration in **Bash**:

```bash
PYTHONPATH=. TAB_P=2000 TAB_Q=1000 TAB_SIGMA_X=2.0 TAB_TILDE=10 TAB_B=200 \
  python simulation/run_tables_b400.py
```

`PYTHONPATH=.` makes the core modules in the current directory available to the simulation script and its worker processes. The example uses wavelet loadings, 2,000 covariates, 1,000 mediators, and 200 replications per method and sample-size setting.

### Configuration

| Environment variable | Meaning | Default |
| --- | --- | --- |
| `TAB_P`, `TAB_Q` | Observed covariate and mediator dimensions | `800`, `200` |
| `TAB_TILDE` | Working representation dimension for each variable group | `10` |
| `TAB_N` | Comma-separated sample sizes | `100,300,800,1200,2000,3000` |
| `TAB_B` | Monte Carlo replications per method and sample size | `400` |
| `TAB_SIGMA_X`, `TAB_SIGMA_M`, `TAB_SIGMA_Y` | Noise standard deviations | `2.0`, `1.0`, `1.0` |
| `TAB_SEED` | Base random seed | `880000` |
| `TAB_TAG` | Output filename prefix | `TABB400` |

Full comparison runs involve repeated neural-network training. The runner can start up to 56 worker processes. Use `TAB_N` and `TAB_B` to restrict the experiment size when checking an environment or exploring a configuration.

### Outputs

Results are written to `mediencoder_exmaple_code/simulation/rebuttal_results/`:

- CSV files with per-replication results and summary statistics;
- a LaTeX table body for the method comparison;
- a LaTeX table body for the alignment-loss ablation.

The summaries report standard deviation, RMSE, confidence-interval length, and coverage. Performance comparisons depend on the simulation configuration and sample size.
