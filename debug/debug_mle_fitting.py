#!/usr/bin/env python3
"""Debug MLE fitting to see if parameters are correct."""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.evolve import EvolutionEngine
from protosynth.envs import markov_k1
import itertools
import math

def debug_mle_fitting():
    """Debug the MLE fitting process."""
    print("DEBUG: MLE Fitting Process")
    print("-" * 30)
    
    # Generate Markov buffer with known parameters
    print("Generating Markov chain with p_stay=0.8...")
    m1_stream = markov_k1(p_stay=0.8, seed=42)
    buf = list(itertools.islice(m1_stream, 8000))
    
    # Collect per-state counts
    k = 2
    state_counts = {(a, b): {'n': 0, 'c1': 0} for a in (0, 1) for b in (0, 1)}
    
    for i in range(k, len(buf)):
        ctx = buf[i-k:i]
        y = buf[i]
        
        if len(ctx) >= 2:
            s = (ctx[-2], ctx[-1])
            state_counts[s]['n'] += 1
            if y == 1:
                state_counts[s]['c1'] += 1
    
    print("\nEmpirical state statistics:")
    for s in state_counts:
        n_s = state_counts[s]['n']
        c1_s = state_counts[s]['c1']
        c0_s = n_s - c1_s
        p1_empirical = c1_s / n_s if n_s > 0 else 0.5
        p0_empirical = c0_s / n_s if n_s > 0 else 0.5
        
        print(f"  State {s}: n={n_s}, c0={c0_s}, c1={c1_s}")
        print(f"    P(next=0|{s}) = {p0_empirical:.4f}")
        print(f"    P(next=1|{s}) = {p1_empirical:.4f}")
    
    # Expected theoretical values for p_stay=0.8
    print("\nTheoretical values for p_stay=0.8:")
    theoretical = {
        (0, 0): {'p0': 0.8, 'p1': 0.2},  # Stay at 0
        (0, 1): {'p0': 0.2, 'p1': 0.8},  # Flip to 1
        (1, 0): {'p0': 0.8, 'p1': 0.2},  # Flip to 0  
        (1, 1): {'p0': 0.2, 'p1': 0.8},  # Stay at 1
    }
    
    for s in theoretical:
        print(f"  State {s}: P(next=0|{s}) = {theoretical[s]['p0']:.4f}, P(next=1|{s}) = {theoretical[s]['p1']:.4f}")
    
    # MLE fitting with Laplace smoothing
    print("\nMLE fitting with Laplace smoothing:")
    mle_params = {}
    for s in state_counts:
        n_s = state_counts[s]['n']
        c1_s = state_counts[s]['c1']
        
        # MLE with Laplace: p_s = (c1_s + 1) / (n_s + 2)
        p1_mle = (c1_s + 1) / (n_s + 2)
        p0_mle = 1 - p1_mle
        
        param_key = f'p{s[0]}{s[1]}'
        mle_params[param_key] = p0_mle  # Store P(next=0|state)
        
        print(f"  State {s}: MLE P(next=0|{s}) = {p0_mle:.4f}, P(next=1|{s}) = {p1_mle:.4f}")
    
    print(f"\nMLE parameters: {mle_params}")
    
    # Test MLE parameters vs theoretical
    print("\nComparison with theoretical:")
    for s in theoretical:
        param_key = f'p{s[0]}{s[1]}'
        mle_val = mle_params[param_key]
        theo_val = theoretical[s]['p0']
        diff = abs(mle_val - theo_val)
        
        print(f"  {param_key}: MLE={mle_val:.4f}, Theoretical={theo_val:.4f}, Diff={diff:.4f}")
    
    # Compute fitness with MLE parameters
    print("\nComputing fitness with MLE parameters:")
    total_n = sum(state_counts[s]['n'] for s in state_counts)
    total_ones = sum(state_counts[s]['c1'] for s in state_counts)
    q_overall = total_ones / total_n
    
    # Baseline entropy
    H0 = -(q_overall * math.log2(q_overall) + (1 - q_overall) * math.log2(1 - q_overall))
    
    # MLE entropy
    H_mle = 0.0
    for s in state_counts:
        n_s = state_counts[s]['n']
        if n_s == 0:
            continue
            
        q_s = state_counts[s]['c1'] / n_s  # Empirical P(next=1|state=s)
        param_key = f'p{s[0]}{s[1]}'
        p0_mle = mle_params[param_key]
        p1_mle = 1 - p0_mle
        
        # Cross-entropy
        if p1_mle > 0 and p1_mle < 1:
            H_s = -(q_s * math.log2(p1_mle) + (1 - q_s) * math.log2(p0_mle))
        else:
            H_s = 10.0
        
        H_mle += (n_s / total_n) * H_s
        print(f"  State {s}: q_s={q_s:.4f}, p1_mle={p1_mle:.4f}, H_s={H_s:.4f}")
    
    F_mle = H0 - H_mle
    print(f"\nFitness calculation:")
    print(f"  H0 (baseline): {H0:.6f}")
    print(f"  H_mle: {H_mle:.6f}")
    print(f"  F_mle: {F_mle:.6f}")
    
    return F_mle > 0

def main():
    success = debug_mle_fitting()
    print(f"\nResult: {'PASS' if success else 'FAIL'}")
    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
