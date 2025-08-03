#!/usr/bin/env python3
"""
Litmus Test: Compare manual vs evolution evaluation paths
"""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.eval import evaluate_program_on_window, evaluate_program
from protosynth.evolve import EvolutionEngine
from protosynth.envs import periodic
import itertools

def test_evaluation_paths():
    """Test that manual and evolution evaluation paths give same results."""
    print("LITMUS TEST: Manual vs Evolution Evaluation")
    print("=" * 45)
    
    # Test program: prev2 on alternating
    prev2_ast = var('prev2')
    alternating = [1, 0] * 100  # 200 bits
    
    print(f"Test program: {pretty_print_ast(prev2_ast)}")
    print(f"Test data: alternating pattern, {len(alternating)} bits")
    
    # Manual evaluation path
    print("\n1. Manual evaluation path:")
    interpreter_manual = LispInterpreter()
    F_manual, metrics_manual = evaluate_program_on_window(
        interpreter_manual, prev2_ast, alternating, k=3
    )
    
    ctx_reads_manual = metrics_manual.get('ctx_reads_per_eval', 0)
    penalty_manual = metrics_manual.get('penalty_bits', 0)
    
    print(f"   F_manual = {F_manual:.6f}")
    print(f"   ctx_reads = {ctx_reads_manual:.2f}")
    print(f"   penalty = {penalty_manual:.1f}")
    
    # Evolution evaluation path (simulate what evolution does)
    print("\n2. Evolution evaluation path:")
    interpreter_evolve = LispInterpreter()
    
    # Use the same evaluation function that evolution uses
    stream_iter = iter(alternating)
    F_evolve, metrics_evolve = evaluate_program(
        interpreter_evolve, prev2_ast, stream_iter, k=3, N=len(alternating)-3
    )
    
    ctx_reads_evolve = metrics_evolve.get('ctx_reads_per_eval', 0)
    penalty_evolve = metrics_evolve.get('penalty_bits', 0)
    
    print(f"   F_evolve = {F_evolve:.6f}")
    print(f"   ctx_reads = {ctx_reads_evolve:.2f}")
    print(f"   penalty = {penalty_evolve:.1f}")
    
    # Compare results
    print("\n3. Comparison:")
    fitness_diff = abs(F_manual - F_evolve)
    ctx_diff = abs(ctx_reads_manual - ctx_reads_evolve)
    
    print(f"   Fitness difference: {fitness_diff:.6f}")
    print(f"   Context reads difference: {ctx_diff:.2f}")
    print(f"   Manual penalty: {penalty_manual:.1f}")
    print(f"   Evolution penalty: {penalty_evolve:.1f}")
    
    # Success criteria
    fitness_match = F_evolve > 0.95 and fitness_diff < 1e-3
    ctx_match = ctx_diff < 0.1
    no_penalties = penalty_manual == 0 and penalty_evolve == 0
    
    print(f"\n4. Results:")
    print(f"   Fitness match (F > 0.95, diff < 1e-3): {'PASS' if fitness_match else 'FAIL'}")
    print(f"   Context reads match (diff < 0.1): {'PASS' if ctx_match else 'FAIL'}")
    print(f"   No penalties: {'PASS' if no_penalties else 'FAIL'}")
    
    overall_success = fitness_match and ctx_match and no_penalties
    print(f"   Overall: {'PASS' if overall_success else 'FAIL'}")
    
    if not overall_success:
        print("\nDIAGNOSIS: Evaluation paths differ - this explains evolution failure!")
        
        if not fitness_match:
            print("  - Fitness mismatch indicates different evaluation logic")
        if not ctx_match:
            print("  - Context reads mismatch indicates different adapter usage")
        if not no_penalties:
            print("  - Penalties indicate evaluation failures")
    
    return overall_success

def test_stream_consumption():
    """Test if stream consumption is the issue."""
    print("\nSTREAM CONSUMPTION TEST")
    print("=" * 25)
    
    # Create a generator
    def alternating_gen():
        while True:
            yield 1
            yield 0
    
    stream = alternating_gen()
    
    # Simulate multiple evaluations on same stream (what evolution might do)
    print("Simulating multiple evaluations on same generator:")
    
    for i in range(3):
        # Take some bits from the stream
        bits = list(itertools.islice(stream, 10))
        print(f"  Evaluation {i+1}: got {len(bits)} bits: {bits[:5]}...")
        
        if len(bits) < 10:
            print(f"    WARNING: Stream exhausted after {len(bits)} bits!")
            break
    
    print("\nThis demonstrates stream consumption issue if evolution shares generators.")
    
    return True

def main():
    print("EVOLUTION DEBUGGING - LITMUS TESTS")
    print("=" * 40)
    
    test_results = [
        test_evaluation_paths(),
        test_stream_consumption(),
    ]
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\nLITMUS TEST SUMMARY")
    print(f"=" * 20)
    print(f"Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("EVALUATION PATHS MATCH - Issue is elsewhere")
    else:
        print("EVALUATION PATH MISMATCH FOUND - This is the root cause!")
    
    return passed_tests == total_tests

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
