#!/usr/bin/env python3
"""5-minute diagnostic to pinpoint the Markov evaluation fault."""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.evolve import EvolutionEngine, eval_candidate
from protosynth.envs import markov_k1
import itertools
import math

def analytic_teacher_vs_evaluator():
    """Check 1: Analytic teacher vs evaluator (no interpreter involved)."""
    print("CHECK 1: Analytic Teacher vs Evaluator")
    print("-" * 45)
    
    # Generate validation buffer
    m1_stream = markov_k1(p_stay=0.8, seed=42)
    val_buf = list(itertools.islice(m1_stream, 8000))
    
    # Collect per-state counts on validation buffer
    k = 2
    state_counts = {(a, b): {'n': 0, 'c1': 0} for a in (0, 1) for b in (0, 1)}
    
    for i in range(k, len(val_buf)):
        ctx = val_buf[i-k:i]
        y = val_buf[i]
        
        if len(ctx) >= 2:
            s = (ctx[-2], ctx[-1])
            state_counts[s]['n'] += 1
            if y == 1:
                state_counts[s]['c1'] += 1
    
    print("  Per-state counts on validation:")
    total_n = sum(state_counts[s]['n'] for s in state_counts)
    total_ones = sum(state_counts[s]['c1'] for s in state_counts)
    q_overall = total_ones / total_n if total_n > 0 else 0.5
    
    for s in state_counts:
        n_s = state_counts[s]['n']
        c1_s = state_counts[s]['c1']
        q_s = c1_s / n_s if n_s > 0 else 0.5
        print(f"    State {s}: n={n_s}, ones={c1_s}, q={q_s:.4f}")
    
    print(f"  Overall: n={total_n}, ones={total_ones}, q={q_overall:.4f}")
    
    # Teacher parameters (stay-biased for p_stay=0.8)
    teacher_params = {
        (0, 0): 0.8,  # p00: P(next=0 | prev2=0, prev=0) = 0.8
        (0, 1): 0.2,  # p01: P(next=0 | prev2=0, prev=1) = 0.2  
        (1, 0): 0.2,  # p10: P(next=0 | prev2=1, prev=0) = 0.2
        (1, 1): 0.8,  # p11: P(next=0 | prev2=1, prev=1) = 0.8
    }
    
    # Compute analytic fitness from counts
    H_prog = 0.0
    for s in state_counts:
        n_s = state_counts[s]['n']
        if n_s == 0:
            continue
            
        q_s = state_counts[s]['c1'] / n_s  # Empirical P(next=1 | state=s)
        p_s = 1 - teacher_params[s]  # Teacher's P(next=1 | state=s)
        
        # Cross-entropy: H_s = -[q_s * log2(p_s) + (1-q_s) * log2(1-p_s)]
        if p_s > 0 and p_s < 1:
            H_s = -(q_s * math.log2(p_s) + (1 - q_s) * math.log2(1 - p_s))
        else:
            H_s = 10.0  # Large penalty for invalid probabilities
        
        H_prog += (n_s / total_n) * H_s
    
    # Baseline entropy (uniform predictor)
    if q_overall > 0 and q_overall < 1:
        H0 = -(q_overall * math.log2(q_overall) + (1 - q_overall) * math.log2(1 - q_overall))
    else:
        H0 = 1.0
    
    F_analytic = H0 - H_prog
    
    print(f"  Analytic calculation:")
    print(f"    H0 (baseline): {H0:.6f}")
    print(f"    H_prog (teacher): {H_prog:.6f}")
    print(f"    F_analytic: {F_analytic:.6f}")
    
    # Compare with evaluator
    markov_prog = op('markov_table', op('+', op('*', const(2), var('prev2')), var('prev')))
    F_evaluator, metrics = eval_candidate(markov_prog, "markov_k2", val_buf, k)
    
    print(f"    F_evaluator: {F_evaluator:.6f}")
    print(f"    Difference: {F_evaluator - F_analytic:.6f}")
    
    # Success if analytic is positive but evaluator is near zero
    analytic_positive = F_analytic > 0.05
    evaluator_near_zero = abs(F_evaluator) < 0.01
    
    print(f"  Result: Analytic={'POSITIVE' if analytic_positive else 'NEGATIVE'}, "
          f"Evaluator={'NEAR_ZERO' if evaluator_near_zero else 'SIGNIFICANT'}")
    
    return F_analytic, F_evaluator

def binary_calibration_guard():
    """Check 2: Binary-only calibration guard."""
    print("\nCHECK 2: Binary-Only Calibration Guard")
    print("-" * 40)
    
    # Generate buffer
    m1_stream = markov_k1(p_stay=0.8, seed=42)
    buf = list(itertools.islice(m1_stream, 8000))
    
    # Test prev with calibration (should be > 0)
    prev_prog = var('prev')
    F_prev, _ = eval_candidate(prev_prog, "markov_k2", buf, 2)
    print(f"  prev with calibration: F={F_prev:.6f} (should be > 0)")
    
    # Test markov_table with calibration (current behavior)
    markov_prog = op('markov_table', op('+', op('*', const(2), var('prev2')), var('prev')))
    F_markov_cal, _ = eval_candidate(markov_prog, "markov_k2", buf, 2)
    print(f"  markov_table with calibration: F={F_markov_cal:.6f}")
    
    # Test markov_table without calibration (would need eval path modification)
    # For now, just report the issue
    print(f"  Issue: If markov_table outputs probabilities but gets calibrated,")
    print(f"         it gets remapped toward 0.5, destroying the advantage.")
    
    return F_prev, F_markov_cal

def state_index_correctness():
    """Check 3: State index correctness."""
    print("\nCHECK 3: State Index Correctness")
    print("-" * 35)
    
    # Generate buffer and check first 20 (state, y_next) pairs
    m1_stream = markov_k1(p_stay=0.8, seed=42)
    buf = list(itertools.islice(m1_stream, 100))
    
    print("  First 20 (state, y_next) pairs:")
    k = 2
    for i in range(k, min(k + 20, len(buf))):
        ctx = buf[i-k:i]
        y_next = buf[i]
        
        if len(ctx) >= 2:
            state = (ctx[-2], ctx[-1])  # (prev2, prev)
            state_idx = 2 * state[0] + state[1]  # 2*prev2 + prev
            print(f"    i={i}: ctx={ctx}, state={state}, idx={state_idx}, y_next={y_next}")
    
    # Check for common errors
    print("  Checking for common errors:")
    print("    - State should be (ctx[-2], ctx[-1]), not (ctx[-1], y)")
    print("    - State index should be 2*prev2 + prev")
    print("    - No current-bit leakage into state")
    
    return True

def main():
    print("MARKOV EVALUATION DIAGNOSTIC")
    print("=" * 30)
    
    # Run all three checks
    F_analytic, F_evaluator = analytic_teacher_vs_evaluator()
    F_prev, F_markov = binary_calibration_guard()
    state_ok = state_index_correctness()
    
    print(f"\nDIAGNOSTIC SUMMARY")
    print(f"=" * 20)
    print(f"F_analytic: {F_analytic:.6f}")
    print(f"F_evaluator: {F_evaluator:.6f}")
    print(f"F_prev: {F_prev:.6f}")
    print(f"F_markov: {F_markov:.6f}")
    
    # Diagnose the issue
    if F_analytic > 0.05 and abs(F_evaluator) < 0.01:
        print("\nDIAGNOSIS: Evaluator is diluting the advantage!")
        print("- Analytic calculation shows positive fitness")
        print("- Evaluator reports near-zero fitness")
        print("- Likely cause: Calibration applied to probabilistic outputs")
    elif F_prev <= 0:
        print("\nDIAGNOSIS: Calibration system broken!")
        print("- Even 'prev' doesn't achieve positive fitness")
        print("- Calibration logic needs fixing")
    else:
        print("\nDIAGNOSIS: Unclear - need deeper investigation")
    
    return F_analytic > 0.05

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
