# ProtoSynth v0.1.0 — Benchmarks Passed, Parity Locked

## Highlights
- ✅ **periodic_k4**: F = 0.811 (≥0.25 target)
- ✅ **markov_k2**: F = 0.288 ≈ F* = 0.289 (theoretical limit)
- ✅ **Teacher parity**: F_eval ≈ F* ± 0.01 (CI)

## What's locked in
- Interpreter tracker reset (no state bleed)
- Prob-vs-binary detection (no miscalibration on probabilistic outputs)
- Penalty replaces CE (never added)
- Single evaluation path + staged parity harness in CI
- Robust context niche/selection (no masking of eval errors)

## Known gotchas
- Always `reset_tracker()` *before each prediction*
- Do not calibrate probabilistic programs (`markov_table`, soft-prev/flip)
- Use the same buffers for analytic vs measured evals

## Thanks
First public milestone of a self-modifying, verifiably-evaluated system.
