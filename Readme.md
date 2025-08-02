# ProtoSynth: A Self-Modifying AI Prototype

**ProtoSynth** is an experimental prototype exploring a radically different approach to machine intelligence—one that learns through self-modification, environmental interaction, and compression-based utility, rather than gradient descent or human-generated datasets.

This project investigates whether an autonomous, self-rewriting system can evolve intelligent behavior from first principles by manipulating its own source code within safe, verifiable boundaries.

## 🧠 Core Concepts

- **Self-Modifying Interpreter**: Programs can inspect and modify their own abstract syntax tree (AST).
- **Mutation Engine**: Generates new program variants by altering ASTs with safe, minimal transformations.
- **Verification Layer**: Ensures mutated programs remain syntactically valid and resource-safe.
- **Evaluation Loop**: Scores programs based on their ability to compress or model synthetic environments.
- **Emergent Modularity**: High-utility substructures may be reused across generations, leading to structured intelligence.

## ⚙️ Architecture

The initial prototype is implemented in Python using:

- A Lisp-like AST structure (`LispNode`)
- A resource-constrained interpreter (`LispInterpreter`)
- Lightweight symbolic environments for evaluation (e.g. pattern prediction)

Future extensions may include:
- Modular memory and code libraries
- Curriculum generation
- Verification via dependent types or symbolic execution
- Migration to more performant runtimes (e.g. WebAssembly)

## 🚧 Status

This project is in **early experimental stages**. Core interpreter and self-inspection are implemented. Mutation, verification, and evaluation pipelines are in progress.

## 📜 License

MIT or Unlicense (to be decided)

---
