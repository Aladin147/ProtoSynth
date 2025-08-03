# Contributing to ProtoSynth

Thank you for your interest in contributing to ProtoSynth! This guide will help you get started.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Aladin147/ProtoSynth.git
   cd ProtoSynth
   ```

2. **Install in development mode:**
   ```bash
   pip install -e .
   pip install pytest black isort pre-commit
   ```

3. **Set up pre-commit hooks:**
   ```bash
   pre-commit install
   ```

## Running Tests

### Full Test Suite
```bash
pytest tests/
```

### Specific Test Categories
```bash
# Core functionality
pytest tests/test_core.py tests/test_eval.py

# Evolution system
pytest tests/test_evolve.py tests/test_mutation.py

# Integration tests
pytest tests/test_integration.py
```

### Parity Harness (Critical)
Before submitting any changes that affect evaluation, run the parity harness:

```bash
python scripts/staged_parity_harness.py
```

This verifies that evaluation results match theoretical expectations (F_eval ≈ F* within ±0.05).

## Code Style

We use Black for formatting and isort for import sorting:

```bash
# Format code
black protosynth/ tests/ scripts/

# Sort imports
isort protosynth/ tests/ scripts/

# Check formatting (CI will fail if not formatted)
black --check protosynth/ tests/ scripts/
```

Pre-commit hooks will automatically format your code, but you can run these manually.

## Submitting Changes

### Bug Reports
When reporting bugs, please include:
- **Reproduction command** with exact parameters
- **Random seed** used (if applicable)
- **Save bundle** from `--save-bundle` flag (attach .pkl file)
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, ProtoSynth version)

### Pull Requests
1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines

3. **Add tests** for new functionality

4. **Run the test suite:**
   ```bash
   pytest tests/
   python scripts/staged_parity_harness.py
   ```

5. **Commit with descriptive messages:**
   ```bash
   git commit -m "Add feature: brief description
   
   - Detailed explanation of changes
   - Any breaking changes or migration notes
   - Fixes #issue-number (if applicable)"
   ```

6. **Push and create a pull request**

### Save Bundles for Debugging

When working with evolution experiments, always use `--save-bundle` to create reproducible test cases:

```bash
# Save experiment state
protosynth-evolve --env periodic_k4 --gens 50 --seed 42 --save-bundle experiment.pkl

# Replay for debugging
protosynth-replay experiment.pkl
```

Include these bundles when reporting issues or requesting features.

## Architecture Overview

- **`protosynth/core.py`** - AST representation and Lisp interpreter
- **`protosynth/eval.py`** - Fitness evaluation and entropy calculations
- **`protosynth/evolve.py`** - Evolution engine and population management
- **`protosynth/mutation.py`** - AST mutation operators
- **`protosynth/predictor.py`** - Program-to-probability adapter
- **`protosynth/envs.py`** - Sequence generation environments
- **`scripts/`** - CLI tools and harnesses

## Key Principles

1. **Mathematical Correctness**: All evaluation must be mathematically sound
2. **Reproducibility**: Use seeds and save bundles for all experiments
3. **Testing**: Maintain 100% test pass rate
4. **Documentation**: Update docstrings and README for user-facing changes

## Questions?

- Check existing [issues](https://github.com/Aladin147/ProtoSynth/issues)
- Create a new issue with the appropriate template
- Review the [README](README.md) for usage examples

Thank you for contributing to ProtoSynth! 🚀
