# ProtoSynth v0.1.0 — Self-Modifying AI with Verified Evaluation

ProtoSynth is an experimental, **non-gradient** learning system. It improves by **rewriting its own code** under verification and is scored by **compression / prediction utility** on synthetic environments. No human text datasets. No backprop.

This README gives you a clean, copy-pasteable quick start, results, and the few “gotchas” that matter.

---

## Table of Contents
- [Features](#features)
- [Quick Start](#quick-start)
- [Benchmarks](#benchmarks)
- [Core Concepts](#core-concepts)
- [Architecture](#architecture)
- [Reproducibility](#reproducibility)
- [Configuration](#configuration)
- [Tests & CI](#tests--ci)
- [Gotchas (Read This)](#gotchas-read-this)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [License & Citation](#license--citation)

---

## Features
- **Self-modifying Lisp-like interpreter** with resource limits (steps, depth, timeout).
- **Mutation + verification** pipeline (arity, scope, reserved names, resource hints).
- **Compression/prediction fitness** (cross-entropy, bits saved per symbol).
- **Curriculum & novelty** (optional): bandits, novelty archive, module mining via MDL.
- **Parity-checked evaluation**: measured fitness matches analytic/theoretical \(F^\*\) within ±0.01.
- **Deterministic bundles** for replay, audit, and publication.

---

## Quick Start

> Requires Python 3.10–3.12.

```bash
# Install
pip install -e .

# Run quick benchmark demos (finish fast)
protosynth-evolve --env periodic_k4 --gens 50 --k 4
protosynth-evolve --env markov_k2   --gens 50 --k 2

Reproduce v0.1.0 results:

# Full runs used for the release
protosynth-evolve --env periodic_k4 --gens 200 --k 4 --mu 16 --lambda 48 --seed 42 --save-bundle bundles/v0.1.0/periodic.json
protosynth-evolve --env markov_k2   --gens 300 --k 2 --mu 16 --lambda 48 --seed 43 --save-bundle bundles/v0.1.0/markov.json

# Replay any bundle (deterministic)
protosynth-replay --bundle bundles/v0.1.0/markov.json

Benchmarks
Benchmark	Target (bits/sym)	Achieved	F* (theory)	Status
periodic_k4	≥ 0.25	0.811312	—	✅ Pass
markov_k2	≥ 0.10	0.287868	0.289726	✅ Pass
Teacher parity (Markov)	—	0.286797	0.289726	✅

Takeaway: evaluation parity is locked: F_eval ≈ F* ± 0.01.
Core Concepts

    Bits-saved fitness:
    F=H2(q)−1N∑tCE(yt,pt)F=H2​(q)−N1​∑t​CE(yt​,pt​).
    Higher is better. Positive FF means you compress/predict better than the baseline entropy.

    Probabilistic vs binary programs:
    Programs that already output probabilities (e.g., markov_table) skip calibration. Binary programs may be calibrated on the train slice only.

    Safety rails:
    Per-prediction reset_tracker(); strict penalties replace CE on exception/timeout.

Architecture

Interpreter & AST

    Minimal Lisp-like nodes: const, var, let, if, op, call.

    Resource tracker: max_steps, max_depth, timeout (reset before every prediction).

Mutation Engine

    Operator swap, constant perturb, variable rename, subtree insert/delete.

    Context-preserving mutations (lower rate inside ctx-dependent subtrees).

Verification

    Arity checks, scope & reserved names (ctx), resource hints, module contracts.

Evaluation

    Single pipeline for all cases (population + teacher).

    Ensemble buffers (optional) with union-train calibration for binary programs.

    Parity harness: analytic F\*F\* from counts vs measured F_eval.

Selection

    ES (μ+λ) with optional context niche (keeps ctx users alive early).

    Optional bandits & novelty archive.

Reproducibility

    Replay bundles include: seeds, config, best ASTs, module library, metrics, and verify reports.

    Parity tests guarantee F_eval ≈ F* on the same validation buffers.

    Determinism: seeded RNG; no global state.

Configuration

Typical knobs:

env: markov_k2
k: 2
gens: 300
mu: 16
lambda: 48
mutation_rate: 0.10
time_limits:
  max_steps: 1000
  timeout_sec: 10.0
evaluation:
  ensemble_buffers: 3
  N_train: 4096
  N_val: 4096
  penalty_bits: 1.5  # replaces CE on exception only
selection:
  context_quota: 8   # adaptive decay gated by readiness
  force_ctx_parent_frac: 0.5

Tests & CI

Run everything:

pytest -q
python scripts/staged_parity_harness.py   # asserts F_eval ≈ F* ± 0.01

Core tests include:

    Teacher parity on Markov (analytic vs measured).

    Alternating stream: prev2 ≥ 0.95 bits saved.

    Markov(0.8): prev > 0 bits saved.

    Reserved var & out-of-range probability errors.

Gotchas (Read This)

    Reset tracker before every prediction
    Call interpreter.reset_tracker() for each time step. This prevents hidden state accumulation and timeouts.

    Single evaluation path
    Population and teacher must go through the exact same adapter/evaluator.

    Probabilistic programs skip calibration
    Modules like markov_table, soft_prev, soft_flip output probabilities already—do not remap them via binary calibration.

    Penalties replace CE
    On exception/timeout, use the penalty instead of cross-entropy, never added on top.

    Context handling
    Do not shadow ctx. Use negative indices for past bits: index(ctx, -1) is last bit.

