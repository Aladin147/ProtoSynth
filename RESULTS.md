# RESULTS

## Benchmarks
| Benchmark    | Target  | Achieved | F* (theory) | Notes |
|--------------|---------|----------|-------------|------|
| periodic_k4  | ≥0.25   | 0.811    | —           | prev4 |
| markov_k2    | ≥0.10   | 0.288    | 0.289       | soft-prev / MLE table |

## Curves
- `results/v0.1.0/periodic_k4_curve.png`
- `results/v0.1.0/markov_k2_curve.png`

## Parity
- Teacher F_eval: 0.2868
- Theoretical F*: 0.2897
- Δ = 0.0029 (within tolerance)

## Ablations (short)
- Without tracker reset → markov_k2 collapses to ~−1.5 F
- Calibrating prob outputs → F → ~0.0
- Penalty added (not replaced) → F → ~−0.5

## Repro
```bash
protosynth-evolve --env periodic_k4 --gens 200 --mu 16 --lambda 48 --k 4 --seed 42
protosynth-evolve --env markov_k2   --gens 300 --mu 16 --lambda 48 --k 2 --seed 43
protosynth-replay --bundle bundles/v0.1.0/markov_k2_bundle.json
```
