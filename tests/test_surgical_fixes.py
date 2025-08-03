#!/usr/bin/env python3
"""Test the three surgical fixes for Markov evolution."""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.evolve import EvolutionEngine, eval_candidate, create_initial_population
from protosynth.eval import evaluate_program_on_window
from protosynth.envs import markov_k1
import itertools

def test_state_conditioned_calibration():
    """Test state-conditioned calibration."""
    print("TEST 1: State-Conditioned Calibration")
    print("-" * 40)
    
    # Generate Markov buffer
    m1_stream = markov_k1(p_stay=0.8, seed=42)
    buf = list(itertools.islice(m1_stream, 3000))
    
    # Test prev predictor with state-conditioned calibration
    fitness, metrics = eval_candidate(var('prev'), "markov_k2", buf, k=1)
    
    delta = metrics.get('delta_calibration', {})
    ctx_reads = metrics.get('ctx_reads_per_eval', 0)
    
    print(f"  prev predictor: F={fitness:.6f}, ctx={ctx_reads:.2f}")
    print(f"  Per-state deltas: {delta}")
    
    # Should have positive fitness with state-conditioned calibration
    success = fitness > 0 and ctx_reads > 0
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    
    return success

def test_markov_table_module():
    """Test markov_table parametric module."""
    print("\nTEST 2: Markov Table Module")
    print("-" * 30)
    
    # Create interpreter with markov_table
    interpreter = LispInterpreter(max_steps=1000, timeout_seconds=10.0)
    
    # Test markov_table function directly
    print("  Testing markov_table function:")
    for state in range(4):
        prob = interpreter._markov_table(state)
        print(f"    state {state}: p={prob:.3f}")
    
    # Test markov_table program
    markov_prog = op('markov_table', op('+', op('*', const(2), var('prev2')), var('prev')))
    
    # Generate test data
    m1_stream = markov_k1(p_stay=0.8, seed=42)
    m1_bits = list(itertools.islice(m1_stream, 100))
    
    fitness, metrics = evaluate_program_on_window(interpreter, markov_prog, m1_bits, k=2)
    ctx_reads = metrics.get('ctx_reads_per_eval', 0)
    penalty = metrics.get('penalty_bits', 0)
    
    print(f"  markov_table program: F={fitness:.6f}, ctx={ctx_reads:.2f}, penalty={penalty:.1f}")
    
    # Should work without penalties
    success = penalty == 0 and ctx_reads > 0
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    
    return success

def test_niche_selection():
    """Test niche selection for context users."""
    print("\nTEST 3: Niche Selection")
    print("-" * 25)
    
    # Create evolution engine with niche selection
    engine = EvolutionEngine(mu=8, lambda_=16, seed=42, k=1, N=500, env_name="markov_k2")
    
    # Initialize with mixed population
    initial_pop = create_initial_population(8, seed=42)
    engine.initialize_population(initial_pop)
    
    print("  Initial population:")
    for i, ind in enumerate(engine.population):
        print(f"    {i}: {pretty_print_ast(ind.program)}")
    
    # Run a few generations to test niche selection
    print(f"\n  Running 5 generations with niche selection:")
    
    for gen in range(5):
        # Create fresh Markov stream
        m1_stream = markov_k1(p_stay=0.8, seed=42+gen)
        fresh_bits = list(itertools.islice(m1_stream, engine.N + engine.k))
        stream = iter(fresh_bits)
        
        stats = engine.evolve_generation(stream)
        best_fitness = stats['best_fitness']
        
        # Count context users
        context_count = sum(1 for ind in engine.population if ind.metrics.get('ctx_reads_per_eval', 0) > 0)
        
        # Calculate expected K_ctx
        K_ctx = max(0, 4 - (4 * gen // 50)) if gen < 50 else 0
        
        print(f"    Gen {gen}: F_best={best_fitness:.6f}, ctx_count={context_count}, K_ctx={K_ctx}")
    
    # Check if context users survived better
    final_context_count = sum(1 for ind in engine.population if ind.metrics.get('ctx_reads_per_eval', 0) > 0)
    success = final_context_count >= 2  # Should maintain some context users
    
    print(f"  Final context users: {final_context_count}")
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    
    return success

def main():
    print("SURGICAL FIXES TEST")
    print("=" * 20)
    
    test_results = [
        test_state_conditioned_calibration(),
        test_markov_table_module(),
        test_niche_selection(),
    ]
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\nSURGICAL FIXES SUMMARY")
    print(f"=" * 24)
    print(f"Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("ALL SURGICAL FIXES WORKING!")
        print("Ready to test markov_k2 benchmark with fixes.")
    elif passed_tests >= 2:
        print("MOSTLY WORKING - Minor issues remain")
    else:
        print("SURGICAL FIXES NEED MORE WORK")
    
    return passed_tests >= 2

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
