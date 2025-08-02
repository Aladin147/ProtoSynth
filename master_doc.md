# 🧠 ProtoSynth Master Document

## Overview

**ProtoSynth** is a self-modifying, non-gradient-based AI architecture. It builds internal knowledge and utility through interaction, self-inspection, mutation, and compression-driven evaluation—without reliance on human-generated data or traditional machine learning methods.

---

## Core Architecture

### 1. Interpreter Core

- **Language Primitives**:
  - `const`: constant value
  - `var`: variable reference
  - `let name val body`: local binding
  - `if cond then else`: conditional logic
  - `op name args`: operations (+, *, ==, etc.)
  - `fn params body` *(TBD)*
  - `call fn args` *(TBD)*

- **Evaluation Mechanics**:
  - Depth-limited recursive interpreter
  - Operation count tracking
  - Timeout enforcement (in seconds)

- **Environment Scope**:
  - Flat lexical bindings
  - Read-only `get_self_ast()` capability

---

### 2. Mutation Engine

- **Mutation Types**:
  - Replace operator (`+` → `*`)
  - Constant tweaking (e.g., ±1)
  - AST subtree swapping
  - Variable renaming
  - Subtree deletion/insertion

- **Mutation Parameters**:
  - `mutation_rate`: probability of mutation per node (e.g., 0.15)
  - `max_mutation_depth`: how deep mutations can reach (e.g., 3)
  - `max_mutation_attempts`: fallback retry limit (e.g., 10)

- **Mutation Constraints**:
  - Only mutate verified ASTs
  - Preserve syntactic validity
  - Avoid name collisions for `let`
  - Arity and type compatibility for `op`

---

### 3. Verification Layer

- **Syntactic Verification**:
  - Valid `op` names and arity
  - No free variables
  - Balanced tree structure

- **Resource Verification**:
  - Max recursion depth: `D_max` (default: 10)
  - Max evaluation steps: `S_max` (default: 100)
  - Timeout: `T_max` seconds (default: 1.0s)

- *(Future)*: Symbolic or type-based verification

---

### 4. Evaluation System

- **Fitness Objective**:
  - Primary: **compression utility** on structured streams
  - Alternative: prediction accuracy, error minimization

- **Evaluation Function**:
  ```python
  fitness = baseline_entropy - program_compressed_entropy


Where entropy is measured using n-gram frequency models or sliding-window compressibility.

    Environment Examples:

        Binary sequences with repeating or recursive patterns

        Generated math series: [1, 2, 4, 8, 16...]

        Logic puzzles (AND/OR/NAND truth tables)

        Symbolic event traces

Formal Notation (WIP)

Let:

    P = program AST

    μ(P) = mutated form of P

    V(P) = 1 if P passes verification, 0 otherwise

    E(P, x) = output of program P on input x

    F(P) = fitness score of program P

Then:

    Mutation-Selection Cycle:

    P₀ → μ₁ → V(μ₁) = 1 → F(μ₁) > F(P₀) → accept
                             ↓
                        F(μ₁) ≤ F(P₀) → reject

Naming Conventions
Term	Definition
Agent	An instance of a self-evaluating, self-modifying AST
Genome	The entire AST representing a program
Mutator	The logic responsible for modifying ASTs
Fitness	Scalar score measuring utility (e.g., compression gain)
Verifier	Ensures structural and safety constraints
Submodule	A verified AST subtree reused in other programs
Open Research Questions

    How to encourage emergence of modular code structures?

    Can curriculum evolve autonomously?

    How to verify semantic safety of logic (e.g., no infinite loops)?

    What environments best support early-stage emergence?

Credits

Prototype design by @Aladin147