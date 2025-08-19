### Contributing to cpp-ai

Thanks for your interest in contributing! This repository is a C++23 project implementing classic ML and DL algorithms from scratch with `Eigen3`, built with CMake and using `vcpkg` for dependencies. Contributions should keep the code clear, well‑documented, and easy to build on all platforms.

## Quick start

Development uses CMake ≥ 3.25, a C++23 compiler, and vcpkg (manifest mode via `vcpkg.json`).

```bash
# 1) Install toolchain (macOS example)
brew install cmake ninja llvm || true

# 2) Install vcpkg (if not already installed)
git clone https://github.com/microsoft/vcpkg.git "$HOME/vcpkg"
"$HOME/vcpkg"/bootstrap-vcpkg.sh
export VCPKG_ROOT="$HOME/vcpkg"

# Build and run
./build.sh            # Release, ML-only (default)
./build.sh Debug      # Debug, ML-only
./build.sh --with-dl  # Release, include WIP DL sources (may not compile yet)
```

Notes:
- If the hardcoded `CMAKE_TOOLCHAIN_FILE` path in `CMakeLists.txt` does not match your system, pass your toolchain via `-DCMAKE_TOOLCHAIN_FILE=...` as shown above.
- On Windows, use a recent MSVC (Visual Studio 2022) generator instead of Ninja, e.g. `-G "Visual Studio 17 2022"`.
- On Linux, install a recent GCC (≥ 13) or Clang (≥ 16).

## Project layout (high-level)

- `ml/` – classic machine‑learning algorithms
  - `include/` – public headers for ML algorithms (e.g., `LinearRegression.h`)
  - `src/` – implementations (e.g., `LinearRegression.cpp`)
- `dl/` – deep‑learning algorithm stubs/implementations
  - `include/` – public headers for DL modules
  - `src/` – implementations
- `main.cpp` – entrypoint with small demo code
- `CMakeLists.txt` – top‑level CMake build configuration
- `vcpkg.json` – manifest dependencies (currently `eigen3`)

## Coding guidelines

- **Language**: C++23. Prefer `const` correctness, RAII, and clear ownership.
- **Math**: Use `Eigen` types for linear algebra (`Eigen::MatrixXd`, `Eigen::VectorXd`).
- **API shape**: Algorithms generally provide `train(...)` and `predict(...)`. Keep interfaces minimal and well‑documented in headers.
- **Style/formatting**: Use `clang-format` before committing. If no repo config exists, follow LLVM or Google style consistently.
- **Headers vs. sources**: Public types in `include/`, implementations in `src/`. Keep headers lean; avoid unnecessary includes.
- **Performance**: Avoid needless copies; pass large objects by `const&`. Prefer fixed‑size matrices where appropriate.
- **Examples**: Keep `main.cpp` runnable. Avoid adding heavy datasets or large binaries to the repo.

Common commands:

```bash
# Configure with vcpkg toolchain (Debug)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
  -DVCPKG_FEATURE_FLAGS=manifests

# Build
cmake --build build -j

# Format (example)
clang-format -i $(git ls-files "*.h" "*.hpp" "*.cpp" "*.cc")
```

## Conventional Commits

This project uses Conventional Commits. Use the form:

```
<type>(<scope>): <subject>

[optional body]

[optional footer(s)]
```

### Commit message character set

- Encoding: UTF‑8 is allowed and preferred across subjects and bodies.
- Subjects may include UTF‑8 symbols when they add clarity; keep the subject ≤ 72 chars and avoid emoji.
- If maximum legacy compatibility is needed, prefer ASCII in the subject and use UTF‑8 in the body.

Example (UTF‑8 subject):

```
feat(ml): add “ridge” regularization option to LinearRegression::train
```

Accepted types (stick to the standard):

- `build` – build system or external dependencies (e.g., CMake, vcpkg)
- `chore` – maintenance (no library behavior change)
- `ci` – continuous integration configuration
- `docs` – documentation only
- `feat` – user‑facing feature or capability
- `fix` – bug fix
- `perf` – performance improvements
- `refactor` – code change that neither fixes a bug nor adds a feature
- `revert` – revert of a previous commit
- `style` – formatting/whitespace (no code behavior)
- `test` – add/adjust tests only

Recommended scopes (choose the smallest, most accurate unit; prefer module or directory names):

- Module scopes:
  - `ml` – classic ML algorithms under `ml/`
  - `dl` – deep‑learning modules under `dl/`
  - `main` – `main.cpp` entrypoint and demos
  - `cmake` – build configuration (`CMakeLists.txt`)
  - `vcpkg` – dependency manifest (`vcpkg.json`) or toolchain integration

- Other scopes:
  - `deps` – dependency updates and version pins
  - `repo` – top‑level repository metadata (`README.md`, `CONTRIBUTING.md`, license)
  - `docs` – broader documentation updates
  - `tests` – unit/integration tests under `tests/`
  - `workflows` – CI pipelines under `.github/workflows/` (if/when added)

Note: Avoid redundant type==scope pairs (e.g., `docs(docs)`). Prefer a module scope (e.g., `docs(ml)`) or `docs(repo)` for top‑level updates.

Examples:

```text
build(cmake): enable -Wall -Wextra -Wpedantic for all targets
chore(repo): add CODE_OF_CONDUCT and security policy
docs(ml): document LinearRegression API and usage in main.cpp
feat(ml): implement train/predict for KNearestNeighbors
feat(dl): scaffold basic NeuralNetwork interface
fix(ml): correct gradient step size in LinearRegression::train
perf(ml): use noalias() in Eigen operations to reduce temporaries
refactor(cmake): split target options into function
revert(ml): revert KMeans initialization change
```

Examples (no scope):

```text
build: switch default build type to Release
chore: refresh .gitignore patterns
docs: add build and vcpkg setup notes
style: format codebase with clang-format
```

Breaking changes:

- Use `!` after the type/scope or a `BREAKING CHANGE:` footer.

```text
feat(ml)!: change LinearRegression::train signature to include regularization

BREAKING CHANGE: train(X, y, learningRate, iterations, lambda) adds a new parameter.
```

### Multiple scopes (optional)

- Comma‑separate scopes without spaces: `type(scope1,scope2): ...`
- Prefer a single scope when possible; use multiple only when the change genuinely spans tightly related areas.

Example:

```text
feat(ml,main): add demo for new KMeansClustering implementation
```

## Pull request checklist

- Builds locally with CMake on your platform; add platform notes if platform‑specific.
- Lint/format: run `clang-format` on changed files; keep includes ordered.
- Docs: update headers and `README.md` when behavior or setup changes.
- Build config: update `CMakeLists.txt` when adding new sources/headers.
- Dependencies: update `vcpkg.json` only when necessary; avoid committing large artifacts.
- Secrets: no secrets or private keys committed.

## Adding a new algorithm (quick recipe)

- Create header in `ml/include/` or `dl/include/` with a minimal, documented API.
- Implement in `ml/src/` or `dl/src/`.
- Wire files into the target in `CMakeLists.txt` (add to `add_executable(cpp_ai ...)`).
- Add a small usage demo in `main.cpp` that can run quickly without external datasets.
- Include basic comments on assumptions, numeric stability, and expected input shapes.

## Versioning and releases

- Maintainers manage releases. Use SemVer semantics when tagging versions.
- Include brief release notes in PRs that introduce notable changes.

## Security and data

- Do not commit private keys, secrets, or sensitive personal data.
- Keep generated or large binary artifacts out of the repo; prefer small, synthetic examples.

## License

By contributing, you agree that your contributions are licensed under the repository’s MIT License.
