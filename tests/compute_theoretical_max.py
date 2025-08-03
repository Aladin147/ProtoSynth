#!/usr/bin/env python3
"""Compute the theoretical maximum bits-saved on the actual Markov_k2 stream."""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.evolve import eval_candidate
from protosynth.envs import markov_k1
import itertools
import math

def H2(p):
    """Binary entropy function."""
    p = max(1e-9, min(1-1e-9, p))
    return -(p * math.log2(p) + (1-p) * math.log2(1-p))

def bits_saved_upper_bound_from_buffer(buf, k=2):
    """Compute theoretical upper bound F* for any predictor on this buffer."""
    N = len(buf) - k
    counts = {(0,0): [0,0], (0,1): [0,0], (1,0): [0,0], (1,1): [0,0]}  # [n_s, c1_s]
    ones = 0
    
    for i in range(k, len(buf)):
        s = (buf[i-2], buf[i-1])
        y = buf[i]
        counts[s][0] += 1
        counts[s][1] += y
        ones += y
    
    q = ones / max(1, N)  # Global 1-rate
    H0 = H2(q)  # Baseline entropy
    
    Hprog = 0.0  # Optimal predictor entropy
    for s, (n, c1) in counts.items():
        if n == 0:
            continue
        qs = c1 / n  # Per-state next-1 rate
        pi_s = n / N  # Stationary weight
        Hprog += pi_s * H2(qs)
    
    return H0 - Hprog, counts, q

def compare_analytic_vs_eval():
    """Compare F_analytic vs F_eval on the teacher."""
    print("THEORETICAL MAXIMUM ANALYSIS")
    print("=" * 35)
    
    # Generate validation buffer (same as benchmark uses)
    print("Generating Markov validation buffer...")
    m1_stream = markov_k1(p_stay=0.8, seed=42)
    val_buf = list(itertools.islice(m1_stream, 8000))
    
    # Compute theoretical upper bound
    F_star, counts, q_global = bits_saved_upper_bound_from_buffer(val_buf, k=2)
    
    print(f"\n1) THEORETICAL UPPER BOUND")
    print(f"   Buffer size: {len(val_buf)} bits")
    print(f"   Global 1-rate: q = {q_global:.4f}")
    print(f"   Baseline entropy: H0 = {H2(q_global):.6f}")
    
    print(f"\n   Per-state analysis:")
    N = len(val_buf) - 2
    Hprog_components = 0.0
    
    for s, (n, c1) in counts.items():
        if n == 0:
            continue
        qs = c1 / n
        pi_s = n / N
        H_s = H2(qs)
        contribution = pi_s * H_s
        Hprog_components += contribution
        
        print(f"     State {s}: n={n:4d}, c1={c1:4d}, q_s={qs:.4f}, π_s={pi_s:.4f}, H2(q_s)={H_s:.6f}, contrib={contribution:.6f}")
    
    print(f"\n   Optimal predictor entropy: H_prog = {Hprog_components:.6f}")
    print(f"   THEORETICAL MAXIMUM: F* = {F_star:.6f}")
    
    # Interpretation
    if F_star >= 0.10:
        print(f"   → F* ≥ 0.10: Target is achievable! Pipeline dilution suspected.")
    elif F_star >= 0.05:
        print(f"   → F* ≥ 0.05: Target challenging but possible with perfect predictor.")
    elif F_star >= 0.02:
        print(f"   → F* ≥ 0.02: Target very difficult, consider lowering threshold.")
    else:
        print(f"   → F* < 0.02: Target impossible, stream has insufficient signal.")
    
    # Compare with MLE teacher
    print(f"\n2) MLE TEACHER ANALYSIS")
    
    # Build MLE parameters from counts
    mle_params = {}
    for s, (n, c1) in counts.items():
        if n == 0:
            continue
        # MLE with Laplace smoothing: P(next=1|s) = (c1 + 1) / (n + 2)
        p1_mle = (c1 + 1) / (n + 2)
        param_key = f'p{s[0]}{s[1]}'
        mle_params[param_key] = 1.0 - p1_mle  # Store P(next=0|s) for compatibility
    
    print(f"   MLE parameters: {mle_params}")
    
    # Compute analytic fitness with MLE parameters
    H_mle = 0.0
    for s, (n, c1) in counts.items():
        if n == 0:
            continue
        qs = c1 / n  # Empirical P(next=1|s)
        param_key = f'p{s[0]}{s[1]}'
        p0_mle = mle_params[param_key]
        p1_mle = 1.0 - p0_mle
        
        pi_s = n / N
        if p1_mle > 0 and p1_mle < 1:
            H_s_mle = -(qs * math.log2(p1_mle) + (1-qs) * math.log2(p0_mle))
        else:
            H_s_mle = 10.0
        
        H_mle += pi_s * H_s_mle
    
    F_analytic = H2(q_global) - H_mle
    print(f"   F_analytic (MLE): {F_analytic:.6f}")
    
    # Compare with evaluator
    markov_prog = op('markov_table', op('+', op('*', const(2), var('prev2')), var('prev')))
    F_eval, metrics = eval_candidate(markov_prog, "markov_k2", val_buf, 2)
    
    print(f"   F_eval (pipeline): {F_eval:.6f}")
    print(f"   Difference: {F_eval - F_analytic:.6f}")
    
    if abs(F_eval - F_analytic) > 0.01:
        print(f"   → PIPELINE DILUTION: Evaluator differs from analytic by {abs(F_eval - F_analytic):.6f}")
    else:
        print(f"   → Pipeline matches analytic within tolerance")
    
    # Test with multiple buffers (ensemble)
    print(f"\n3) ENSEMBLE ANALYSIS")
    ensemble_F_stars = []
    ensemble_buffers = []
    
    for i in range(5):
        stream = markov_k1(p_stay=0.8, seed=42 + i)
        buf = list(itertools.islice(stream, 8000))
        F_star_i, _, _ = bits_saved_upper_bound_from_buffer(buf, k=2)
        ensemble_F_stars.append(F_star_i)
        ensemble_buffers.append(buf)
    
    avg_F_star = sum(ensemble_F_stars) / len(ensemble_F_stars)
    std_F_star = (sum((f - avg_F_star)**2 for f in ensemble_F_stars) / len(ensemble_F_stars))**0.5
    
    print(f"   Individual F* values: {[f'{f:.6f}' for f in ensemble_F_stars]}")
    print(f"   Average F*: {avg_F_star:.6f}")
    print(f"   Std F*: {std_F_star:.6f}")
    
    # Test ensemble evaluation
    F_eval_ensemble, _ = eval_candidate(markov_prog, "markov_k2", val_buf, 2, ensemble_buffers[1:])
    print(f"   F_eval (ensemble): {F_eval_ensemble:.6f}")
    
    print(f"\n4) SUMMARY")
    print(f"   Theoretical maximum: F* = {F_star:.6f}")
    print(f"   MLE analytic: F_analytic = {F_analytic:.6f}")
    print(f"   Pipeline single: F_eval = {F_eval:.6f}")
    print(f"   Pipeline ensemble: F_eval_ensemble = {F_eval_ensemble:.6f}")
    print(f"   Current benchmark result: F ≈ -0.002")
    
    # Recommendations
    print(f"\n5) RECOMMENDATIONS")
    if F_star >= 0.10:
        print(f"   ✓ Target F ≥ 0.10 is achievable (F* = {F_star:.6f})")
        if F_analytic < F_star * 0.8:
            print(f"   → Fix MLE fitting (F_analytic too low)")
        elif abs(F_eval - F_analytic) > 0.01:
            print(f"   → Fix pipeline dilution (evaluator vs analytic)")
        else:
            print(f"   → Add logit bias or improve selection dynamics")
    elif F_star >= 0.05:
        print(f"   ~ Target challenging (F* = {F_star:.6f}), consider F ≥ {F_star * 0.8:.3f}")
    else:
        print(f"   ✗ Target impossible (F* = {F_star:.6f}), increase env signal or lower target")
    
    return F_star, F_analytic, F_eval

def main():
    F_star, F_analytic, F_eval = compare_analytic_vs_eval()
    
    # Return success if we have a clear path forward
    success = F_star >= 0.05 or abs(F_eval - F_analytic) > 0.01
    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
