#!/usr/bin/env python3
"""Sanity tests for context binding fixes."""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.eval import evaluate_program_on_window
from protosynth.predictor import PredictorAdapter
import itertools

def alternating_stream():
    """Generate alternating 0,1,0,1,... stream."""
    b = 0
    while True:
        yield b
        b ^= 1

def markov_k1_stream(p_stay=0.8, seed=42):
    """Generate Markov chain with k=1 memory."""
    import random
    random.seed(seed)
    
    state = 0
    while True:
        yield state
        # Stay in same state with probability p_stay
        if random.random() < p_stay:
            # Stay
            pass
        else:
            # Flip
            state = 1 - state

def test_alternating_prev2():
    """Test prev2 on alternating pattern - should get F ~ +1.0."""
    print("TEST 1: prev2 on alternating pattern")
    print("-" * 40)
    
    # Generate alternating sequence
    alternating = list(itertools.islice(alternating_stream(), 200))
    print(f"  Pattern: {alternating[:10]}...")
    
    interpreter = LispInterpreter()
    
    # Test prev2 with sufficient context (k=3 to ensure prev2 is available)
    fitness, metrics = evaluate_program_on_window(interpreter, var('prev2'), alternating, k=3)
    
    ctx_reads = metrics.get('ctx_reads', 0)
    ctx_reads_per_eval = metrics.get('ctx_reads_per_eval', 0)
    penalty = metrics.get('penalty_bits', 0)
    num_predictions = metrics.get('num_predictions', 0)
    
    print(f"  Results:")
    print(f"    Fitness: {fitness:.6f}")
    print(f"    Context reads: {ctx_reads} total, {ctx_reads_per_eval:.2f} per eval")
    print(f"    Penalty: {penalty:.1f} bits")
    print(f"    Predictions: {num_predictions}")
    
    # Expected: F ~ +1.0 (nearly perfect), ctx_reads_per_eval ~ 1.0, penalty = 0
    success = fitness > 0.8 and ctx_reads_per_eval > 0.9 and penalty == 0
    
    print(f"  Expected: F > 0.8, ctx_reads_per_eval > 0.9, penalty = 0")
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    
    return success

def test_markov_prev():
    """Test prev on Markov chain - should beat baseline."""
    print("\nTEST 2: prev on Markov chain")
    print("-" * 35)
    
    # Generate Markov sequence with high stay probability
    markov = list(itertools.islice(markov_k1_stream(p_stay=0.8, seed=42), 200))
    print(f"  Pattern: {markov[:20]}...")
    
    interpreter = LispInterpreter()
    
    # Test prev with k=2 (sufficient for prev)
    fitness, metrics = evaluate_program_on_window(interpreter, var('prev'), markov, k=2)
    
    ctx_reads = metrics.get('ctx_reads', 0)
    ctx_reads_per_eval = metrics.get('ctx_reads_per_eval', 0)
    penalty = metrics.get('penalty_bits', 0)
    num_predictions = metrics.get('num_predictions', 0)
    
    print(f"  Results:")
    print(f"    Fitness: {fitness:.6f}")
    print(f"    Context reads: {ctx_reads} total, {ctx_reads_per_eval:.2f} per eval")
    print(f"    Penalty: {penalty:.1f} bits")
    print(f"    Predictions: {num_predictions}")
    
    # Expected: F > 0 (beats baseline), ctx_reads_per_eval ~ 1.0, penalty = 0
    success = fitness > 0.0 and ctx_reads_per_eval > 0.9 and penalty == 0
    
    print(f"  Expected: F > 0.0, ctx_reads_per_eval > 0.9, penalty = 0")
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    
    return success

def test_context_access_with_index():
    """Test context access using index operation."""
    print("\nTEST 3: Context access with index")
    print("-" * 40)
    
    # Test sequence
    test_bits = [1, 0, 1, 0, 1, 0, 1, 0] * 10
    
    interpreter = LispInterpreter()
    
    # Create program: ctx[-2] (equivalent to prev2)
    ctx_index_prog = op('index', var('ctx'), const(-2))
    
    fitness, metrics = evaluate_program_on_window(interpreter, ctx_index_prog, test_bits, k=3)
    
    ctx_reads = metrics.get('ctx_reads', 0)
    ctx_reads_per_eval = metrics.get('ctx_reads_per_eval', 0)
    penalty = metrics.get('penalty_bits', 0)
    
    print(f"  Results:")
    print(f"    Program: {pretty_print_ast(ctx_index_prog)}")
    print(f"    Fitness: {fitness:.6f}")
    print(f"    Context reads: {ctx_reads} total, {ctx_reads_per_eval:.2f} per eval")
    print(f"    Penalty: {penalty:.1f} bits")
    
    # Should work similarly to prev2
    success = fitness > 0.5 and penalty == 0
    
    print(f"  Expected: F > 0.5, penalty = 0")
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    
    return success

def test_baseline_comparison():
    """Test that context programs beat baseline."""
    print("\nTEST 4: Baseline comparison")
    print("-" * 35)
    
    # Alternating pattern
    alternating = [1, 0] * 50
    
    interpreter = LispInterpreter()
    
    # Test baseline
    baseline_fitness, _ = evaluate_program_on_window(interpreter, const(0.5), alternating, k=3)
    
    # Test prev2
    prev2_fitness, _ = evaluate_program_on_window(interpreter, var('prev2'), alternating, k=3)
    
    improvement = prev2_fitness - baseline_fitness
    
    print(f"  Results:")
    print(f"    Baseline (const 0.5): {baseline_fitness:.6f}")
    print(f"    Context (prev2): {prev2_fitness:.6f}")
    print(f"    Improvement: {improvement:.6f}")
    
    # prev2 should dramatically beat baseline on alternating
    success = improvement > 0.8
    
    print(f"  Expected: improvement > 0.8")
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    
    return success

def test_direct_prediction():
    """Test direct prediction calls."""
    print("\nTEST 5: Direct prediction calls")
    print("-" * 40)
    
    interpreter = LispInterpreter()
    adapter = PredictorAdapter(interpreter)
    
    # Test prev2 prediction directly
    context = [1, 0, 1]  # Last 3 bits
    
    try:
        prediction = adapter.predict(var('prev2'), context)
        ctx_reads = getattr(interpreter, 'ctx_reads', 0)
        
        print(f"  Direct prediction:")
        print(f"    Context: {context}")
        print(f"    Program: prev2")
        print(f"    Prediction: {prediction:.6f}")
        print(f"    Context reads: {ctx_reads}")
        
        # prev2 should return context[-2] = 0
        expected = 0.0
        success = abs(prediction - expected) < 0.01 and ctx_reads == 1
        
        print(f"  Expected: prediction ≈ {expected}, ctx_reads = 1")
        print(f"  Result: {'PASS' if success else 'FAIL'}")
        
    except Exception as e:
        print(f"  Direct prediction failed: {e}")
        success = False
        print(f"  Result: FAIL")
    
    return success

def main():
    print("CONTEXT BINDING FIXES - SANITY TESTS")
    print("=" * 45)
    
    test_results = [
        test_alternating_prev2(),
        test_markov_prev(),
        test_context_access_with_index(),
        test_baseline_comparison(),
        test_direct_prediction()
    ]
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\nSUMMARY: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests >= 4:
        print("CONTEXT FIXES WORKING - Ready for evolution!")
    elif passed_tests >= 2:
        print("PARTIAL SUCCESS - Some fixes working")
    else:
        print("CONTEXT FIXES FAILED - Need debugging")
    
    return passed_tests >= 4

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
