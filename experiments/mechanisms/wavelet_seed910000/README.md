# Fixed synthetic wavelet mechanism

These two files preserve the exact synthetic mechanism from the completed
50-replication CPU experiment in GitHub Actions run
[37174159429](https://github.com/shibo769/MediEncoder/actions/runs/37174159429).
They contain generated structural coefficients and measurement loadings, not
subject-level observations or real data.

- Parameter seed: `910000`.
- Mechanism content hash:
  `20e82d3e80e64bbcec053828a145ece1f93932ecd0b81376c937cbb37ebb2e69`.
- Population target: `4.216720624294777`.
- Observed dimensions: `p=2000`, `q=1000`; latent dimensions: `5`, `5`.
- The files are byte-identical to that run's `synthetic-prepared` artifact.

The subsequent regularization experiment imports this saved mechanism to change
only shared neural-network weight decay from `0` to `0.01`, using the same data
and training seed rules. It has its own manifest and outputs. The first 50
replications overlap the original experiment; another 50 extend the data seeds.

This mechanism is retained for a controlled regularization sensitivity analysis.
It has a known information limitation: only one of its five Haar atoms overlaps
the covariate factors' support `[-1,1]`. The conditional mean of X depends only
on the five indicators `1{f_X,j < -0.5}`. Weight decay cannot restore the missing
within-bin continuous factor information. This fixture does not establish
latent recovery or causal identification assumptions, and it is not redrawn to
obtain favorable rankings or coverage.
