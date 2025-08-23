# Graph Learning & Signal Processing (graph)

Scope: classical graph algorithms and lightweight graph learning methods.

## Recommended examples to implement
- PageRank with power iteration
- Spectral clustering (Laplacian eigenmaps + k-means)
- Label propagation for semi-supervised classification
- Graph Laplacian regularization for smoothing/denoising
- Simple message passing: one GCN-style layer without backprop stack

## Suggested layout
- `include/`: graph data structures, operators (Laplacian), algorithms
- `src/`: implementations and spectral utilities (eigendecompositions)
- Tests under top-level `tests/` prefixed with `test_graph_*.cpp`

Notes: Represent sparse graphs with Eigen sparse matrices where helpful; prefer numerically stable eigensolvers.
