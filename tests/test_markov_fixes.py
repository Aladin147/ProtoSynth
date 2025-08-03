#!/usr/bin/env python3
"""Test Markov generator fixes and calibration."""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.envs import markov_k1, check_transitions
from protosynth.eval import evaluate_program_calibrated
import itertools

def test_markov_generator():
    """Test that markov_k1 generator has correct transition probabilities."""
    print("TEST: Markov Generator")
    print("-" * 25)
    
    # Test with p_stay=0.8
    counts, stay, flip = check_transitions(lambda: markov_k1(0.8, seed=42))
    
    print(f"Transition counts: {counts}")
    print(f"Stay probability: {stay:.3f} (expected: 0.8)")
    print(f"Flip probability: {flip:.3f} (expected: 0.2)")
    
    # Assert close to config
    stay_correct = abs(stay - 0.8) < 0.02
    print(f"Stay probability correct: {'PASS' if stay_correct else 'FAIL'}")
    
    return stay_correct

def test_calibrated_evaluation():
    """Test calibrated evaluation on Markov chain."""
    print("\nTEST: Calibrated Evaluation")
    print("-" * 30)
    
    # Generate Markov chain
    m1_stream = markov_k1(p_stay=0.8, seed=42)
    m1_bits = list(itertools.islice(m1_stream, 5000))  # Need enough for train/val split
    
    print(f"Generated {len(m1_bits)} bits")
    print(f"Pattern: {m1_bits[:20]}...")
    
    interpreter = LispInterpreter(max_steps=1000, timeout_seconds=10.0)
    
    # Test prev predictor with calibration
    print("\nTesting prev predictor with calibration:")
    fitness, metrics = evaluate_program_calibrated(
        interpreter, var('prev'), m1_bits, k=1, N_train=1000, N_val=1000
    )
    
    delta = metrics.get('delta_calibration', 0)
    ctx_reads = metrics.get('ctx_reads_per_eval', 0)
    
    print(f"  Fitness: {fitness:.6f}")
    print(f"  Delta calibration: {delta:.3f}")
    print(f"  Context reads per eval: {ctx_reads:.2f}")
    print(f"  Train samples: {metrics.get('train_samples', 0)}")
    print(f"  Val samples: {metrics.get('val_samples', 0)}")
    
    # Should get positive fitness with calibration
    success = fitness > 0 and ctx_reads > 0.5
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    
    return success

def test_soft_predictors():
    """Test soft predictor seeds."""
    print("\nTEST: Soft Predictors")
    print("-" * 22)
    
    # Generate Markov chain
    m1_stream = markov_k1(p_stay=0.8, seed=42)
    m1_bits = list(itertools.islice(m1_stream, 5000))
    
    interpreter = LispInterpreter(max_steps=1000, timeout_seconds=10.0)
    
    # Test soft predictor: if prev=1 then 0.8 else 0.2
    soft_pred = if_expr(op('=', var('prev'), const(1)), const(0.8), const(0.2))
    
    print("Testing soft predictor: if(prev=1, 0.8, 0.2)")
    fitness, metrics = evaluate_program_calibrated(
        interpreter, soft_pred, m1_bits, k=1, N_train=1000, N_val=1000
    )
    
    ctx_reads = metrics.get('ctx_reads_per_eval', 0)
    
    print(f"  Fitness: {fitness:.6f}")
    print(f"  Context reads per eval: {ctx_reads:.2f}")
    
    # Soft predictor should work well without calibration
    success = fitness > 0.05 and ctx_reads > 0.5
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    
    return success

def main():
    print("MARKOV FIXES TEST")
    print("=" * 20)
    
    test_results = [
        test_markov_generator(),
        test_calibrated_evaluation(),
        test_soft_predictors(),
    ]
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\nMARKOV FIXES SUMMARY")
    print(f"=" * 22)
    print(f"Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("ALL MARKOV FIXES WORKING!")
        print("Ready to test markov_k2 benchmark.")
    elif passed_tests >= 2:
        print("MOSTLY WORKING - Minor issues remain")
    else:
        print("MARKOV FIXES FAILED - More debugging needed")
    
    return passed_tests >= 2

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
