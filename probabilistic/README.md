# Probabilistic Modeling & Inference (probabilistic)

Scope: graphical models and approximate inference algorithms.

## Recommended examples to implement
- Hidden Markov Model: forward–backward, Viterbi decoding
- Kalman Filter (linear-Gaussian), Extended KF (optional)
- Particle Filter (sequential Monte Carlo) for nonlinear state-space
- Bayesian Networks / Factor Graphs: variable elimination belief updates
- MCMC: Metropolis–Hastings and Gibbs sampling on simple models
- Variational Inference: mean-field Gaussian for a conjugate model

## Suggested layout
- `include/`: distributions, models (HMM, KF), inference interfaces
- `src/`: implementations and numerical utilities (log-sum-exp, stabilization)
- Tests under top-level `tests/` prefixed with `test_prob_*.cpp`

Notes: Use Eigen for linear-Gaussian algebra and log-domain computations to maintain numerical stability.
