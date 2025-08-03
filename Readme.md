# ProtoSynth

[![CI](https://img.shields.io/github/actions/workflow/status/Aladin147/ProtoSynth/ci.yml?branch=master)](https://github.com/Aladin147/ProtoSynth/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/Aladin147/ProtoSynth)](https://github.com/Aladin147/ProtoSynth/releases)

ProtoSynth is a research prototype for evolving sequence prediction programs through mutation and selection. The system uses a Lisp-like interpreter that can modify its own abstract syntax trees to discover predictive patterns in binary sequences.

## Overview

This implementation explores program synthesis for sequence prediction tasks using:

- **Evolutionary programming**: Mutation and selection of Lisp-like ASTs
- **Self-modification**: Programs can inspect and modify their own structure  
- **Verified evaluation**: Mathematical validation of fitness measurements
- **Reproducible experiments**: Complete save/replay system for scientific reproducibility

The system has been tested on periodic sequences and first-order Markov chains, achieving performance near theoretical limits on both tasks.

## Installation

```bash
pip install protosynth
```

Or from source:
```bash
git clone https://github.com/Aladin147/ProtoSynth.git
cd ProtoSynth
pip install -e .
```

## Quick Start

### Basic Usage

```bash
# Evolve programs for periodic pattern prediction
protosynth-evolve --env periodic_k4 --gens 100 --k 4

# Evolve programs for Markov chain prediction  
protosynth-evolve --env markov_k2 --gens 200 --k 2

# Save results for later analysis
protosynth-evolve --env markov_k2 --gens 100 --k 2 --save-bundle results.json

# Replay saved results
protosynth-replay --bundle results.json
```

### Programmatic Usage

```python
from protosynth import LispInterpreter, const, var, op
from protosynth.envs import periodic

# Create a simple program
program = op('xor', var('prev'), var('prev2'))

# Evaluate on a sequence  
interpreter = LispInterpreter()
stream = periodic([1, 0, 1, 0])
# ... evaluation code
```

## Architecture

### Core Components

- **`LispNode`**: AST representation for programs
- **`LispInterpreter`**: Execution engine with resource constraints
- **`EvolutionEngine`**: Mutation and selection logic
- **`PredictorAdapter`**: Interface between programs and evaluation

### Environments

- **`periodic`**: Repeating bit patterns
- **`markov_k1`**: First-order Markov chains  
- **`random_bits`**: Independent random bits

### Evaluation

Programs are scored using compression-based fitness:
```
F = H₀ - H_model
```
where H₀ is baseline entropy and H_model is model entropy.

## Benchmarks

The system has been validated on two standard tasks:

| Environment | Target F | Achieved F | F* (theory) | Notes |
|-------------|----------|------------|-------------|-------|
| periodic_k4 | ≥ 0.25   | 0.811      | —           | Pattern: [1,0,1,0] |
| markov_k2   | ≥ 0.10   | 0.288      | 0.289       | Near theoretical limit |

## Implementation Notes

### Key Design Decisions

1. **Interpreter state management**: Each prediction resets execution state to prevent accumulation artifacts
2. **Probabilistic program detection**: Programs outputting probabilities skip binary calibration  
3. **Penalty handling**: Exceptions replace cross-entropy loss rather than adding to it
4. **Context preservation**: Evolution maintains programs that use historical context

### Known Limitations

- Currently limited to binary sequences
- Evaluation is single-threaded
- Memory usage scales with population size
- No persistent program libraries between runs

## Development

### Running Tests

```bash
pytest tests/
python scripts/staged_parity_harness.py  # Validates evaluation correctness
```

### Project Structure

```
protosynth/
├── core.py          # AST and interpreter
├── evolve.py        # Evolution engine  
├── eval.py          # Fitness evaluation
├── envs.py          # Test environments
├── mutation.py      # AST mutations
├── predictor.py     # Prediction interface
└── cli.py           # Command-line tools
```

## Contributing

This is a research prototype. Contributions are welcome, particularly:

- Additional sequence environments
- Performance optimizations  
- Multi-alphabet support
- Visualization tools

## License

MIT License. See LICENSE file for details.

## Citation

If you use ProtoSynth in research, please cite:

```bibtex
@software{protosynth2024,
  title={ProtoSynth: Evolutionary Program Synthesis for Sequence Prediction},
  author={ProtoSynth Team},
  year={2024},
  url={https://github.com/Aladin147/ProtoSynth}
}
```
