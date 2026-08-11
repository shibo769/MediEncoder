# MediEncoder — core code

Nonlinear representation learning for high-dimensional causal mediation:
estimation and inference for natural direct/indirect effects (NDE/NIE/TE)
via a cross-fitted efficient influence function.

## Core modules
- `nn_utils.py`             — MLP builder, device
- `NNModel_and_Train.py`    — autoencoder / VAE + nuisance-network trainers
- `MediEncoder_and_Train.py`— the MediEncoder model, its 3-term objective
                              (recon_X + recon_M + A-alignment), lambda grid,
                              and encode helpers
- `DGP_and_estimate.py`     — data-generating process (wavelet/poly/spline
                              factor loadings) + estimand truths
- `run_and_eval.py`         — Algorithm 1 (4-fold cross-fit EIF) + Algorithm 2
                              (per-fold lambda selection by held-out prediction);
                              simulate_one_run worker

## Simulation
`simulation/run_tables_b400.py` reproduces the main comparison table
(Projection / Autoencoder / VAE / MediEncoder) plus the lambda3 ablation.

The setting where MediEncoder wins cleanly at every large n (RMSE lowest,
coverage ~0.95):  p=2000, q=1000, bar_p=bar_q=5, tilde_p=tilde_q=10,
wavelet loadings, sigma_X=2, sigma_M=sigma_Y=1. Run:

    TAB_P=2000 TAB_Q=1000 TAB_SIGMA_X=2.0 TAB_TILDE=10 TAB_B=200 \
      python simulation/run_tables_b400.py

Env knobs: TAB_P, TAB_Q, TAB_TILDE, TAB_SIGMA_X/M/Y, TAB_B (replications),
TAB_N (sample sizes), TAB_SEED. Outputs a wave table + ablation LaTeX body.
