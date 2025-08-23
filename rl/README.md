# Reinforcement Learning (rl)

Scope: tabular and linear-function-approximation RL baselines implemented in C++ with Eigen.

## Recommended examples to implement
- Tabular Q-learning (ε-greedy) on gridworld
- SARSA (on-policy TD control)
- Monte Carlo control (first-visit)
- REINFORCE (policy gradient) for bandits
- Semi-gradient actor–critic with linear value function
- Dyna-Q (model-based planning with a learned model)

## Suggested layout
- `include/`: public headers for agents, policies, value functions
- `src/`: implementations and utilities (e.g., experience buffers)
- Tests under top-level `tests/` prefixed with `test_rl_*.cpp`

Notes: Keep environments minimal (e.g., tiny gridworld/bandit) to avoid extra deps. Use Eigen for vectorized value updates and linear approximators.
