# Repository Guidelines

## Project goal

enczoo is a stateless **library** with a single, simple goal: to compute intermediate representations from common, off-the-shelf vision models as np.ndarrays, given PIL.Images. 

### What enczoo should do
* `enczoo` should just work. 
* `enczoo` should handle annoying gotchas, such as model-specific image normalization, dtypes, batching, randomness etc.
* `enczoo` should and have good (but overridable) defaults for many things, like image cropping, and dimensionality reduction.
* `enczoo` should be _deterministic_ (ish), to some epsilon, for any given version. 

### What enczoo does not do
* `enczoo` is not meant to be used as components of a larger machine learning graph; it really is for off-the-shelf image representations only. 
* `enczoo` isn't focused on performance, though it does its best. It's meant to be used occasionally, and have its results cached by the caller.


## Project Structure & Module Organization
- `src/enczoo/`: library source code and subpackages (e.g., `neural_networks/`, `transforms/`, `mref/`).
- `tests/`: unit tests for core modules.
- `regression_tests/`: regression suite with reference images and target arrays under `regression_tests/test_images/` and `regression_tests/test_targets/`.
- `examples/`: runnable scripts (e.g., `examples/get_resnet50_features.py`).
- `dist/`: build artifacts created by `make build`.

## Build, Test, and Development Commands
- `make lint`: auto-fix style issues with `ruff check --fix` and format with `ruff format`.
- `make check`: run type checks (`ty`) and verify formatting/linting without fixes.
- `make test`: run the unit test suite via `pytest tests`.
- `make build`: clean `dist/` and build the package with `uv build`.

## Coding Style & Naming Conventions
- Python 3.12+, 4-space indentation, standard PEP 8 naming.
- Modules/functions/variables use `snake_case`; classes use `CamelCase`.
- Formatting and linting are enforced by `ruff format` and `ruff check`; type checks use `ty`.

## Testing Guidelines
- Test framework: `pytest`.
- Unit tests live in `tests/` and follow `test_*.py` naming.
- Regression tests live in `regression_tests/` and use on-disk fixtures; keep new golden data under `regression_tests/test_targets/`.

## Commit & Pull Request Guidelines
- Commit messages in history are short, imperative, and lowercase (e.g., "lint", "optimize imports").
- PRs should include a brief summary, testing performed (commands and results), and links to relevant issues or datasets when applicable.

## Configuration Tips
- Use `uv run ...` for tool invocations to ensure the project environment is consistent (e.g., `uv run pytest tests`).
