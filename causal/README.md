# Causal Discovery & Inference (causal)

Scope: structure discovery and effect estimation with simple, dependency-light methods.

## Recommended examples to implement
- PC algorithm for DAG discovery with partial correlation CI tests
- Linear SEM backdoor adjustment; total/average treatment effect (ATE)
- Propensity score estimation + IPW and matching
- Doubly robust (AIPW) estimator on synthetic data
- NOTEARS (linear) scoring objective (optional, small-scale)

## Suggested layout
- `include/`: CI tests, graph structures, estimators
- `src/`: implementations and synthetic data generators
- Tests under top-level `tests/` prefixed with `test_causal_*.cpp`

Notes: Use Eigen for covariance/precision estimation and regression. Keep datasets small and synthetic for clarity and speed.
