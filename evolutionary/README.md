# Evolutionary Methods (evolutionary)

Scope: black-box optimization and search using population-based methods.

## Recommended examples to implement
- Genetic Algorithm (binary and real-coded) on Rastrigin/Sphere
- (μ, λ)-Evolution Strategies on Sphere/Ackley
- CMA-ES for ill-conditioned quadratic minimization
- Differential Evolution (DE/rand/1/bin)
- Simple neuroevolution: evolve linear policy for a toy control task

## Suggested layout
- `include/`: optimizers, selection/mutation/crossover operators
- `src/`: algorithm implementations and benchmark functions
- Tests under top-level `tests/` prefixed with `test_evo_*.cpp`

Notes: Use Eigen for vectorized populations and covariance handling (CMA-ES). Keep benchmarks deterministic for reproducible tests.
