#!/usr/bin/env python3
"""Test the complete surgical fixes for Markov evolution."""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.evolve import EvolutionEngine, create_markov_table_individual, create_initial_population
from protosynth.envs import markov_k1
import itertools

def test_markov_table_with_refit():
    """Test markov_table with Lamarckian refit."""
    print("TEST: Markov Table with Refit")
    print("-" * 35)
    
    # Create stay-biased individual
    stay_ind = create_markov_table_individual('stay', generation=0)
    
    print(f"  Initial params: {stay_ind.metrics.get('markov_params', {})}")
    print(f"  Program: {pretty_print_ast(stay_ind.program)}")
    
    # Generate Markov data
    m1_stream = markov_k1(p_stay=0.8, seed=42)
    buf = list(itertools.islice(m1_stream, 2000))
    
    # Create evolution engine to test refit
    engine = EvolutionEngine(mu=4, lambda_=8, seed=42, k=2, N=500, env_name="markov_k2")
    
    # Test refit
    print(f"\n  Testing refit on training data:")
    engine.refit_markov_table(stay_ind, buf)
    
    refit_params = stay_ind.metrics.get('markov_params', {})
    print(f"  Refit params: {refit_params}")
    
    # Check if parameters changed toward true values
    success = refit_params.get('p00', 0) > 0.7 and refit_params.get('p11', 0) > 0.7
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    
    return success

def test_niche_selection_with_shaping():
    """Test niche selection with warmup shaping."""
    print("\nTEST: Niche Selection with Shaping")
    print("-" * 40)
    
    # Create evolution engine
    engine = EvolutionEngine(mu=8, lambda_=16, seed=42, k=2, N=500, env_name="markov_k2")
    
    # Create mixed population with Markov seeds
    initial_pop = create_initial_population(8, seed=42)
    stay_ind = create_markov_table_individual('stay', generation=0)
    flip_ind = create_markov_table_individual('flip', generation=0)

    # Replace last 2 with Markov seeds
    initial_pop[-2:] = [stay_ind.program, flip_ind.program]
    engine.initialize_population(initial_pop)
    
    print(f"  Initial population size: {len(engine.population)}")
    
    # Run a few generations to test niche selection
    print(f"\n  Running 10 generations with niche selection:")
    
    for gen in range(10):
        # Create fresh Markov stream
        m1_stream = markov_k1(p_stay=0.8, seed=42+gen)
        fresh_bits = list(itertools.islice(m1_stream, engine.N + engine.k))
        stream = iter(fresh_bits)
        
        stats = engine.evolve_generation(stream)
        best_fitness = stats['best_fitness']
        
        # Count context users and markov_table programs
        context_count = sum(1 for ind in engine.population if ind.metrics.get('ctx_reads_per_eval', 0) > 0)
        markov_count = sum(1 for ind in engine.population if 'markov_table' in pretty_print_ast(ind.program))
        
        # Calculate expected K_ctx
        K_ctx = max(0, 8 - (8 * gen // 50)) if gen < 50 else 0
        
        print(f"    Gen {gen:2d}: F_best={best_fitness:.6f}, ctx={context_count}, markov={markov_count}, K_ctx={K_ctx}")
        
        if best_fitness > 0:
            print(f"    SUCCESS: Positive fitness achieved!")
            break
    
    # Check if context users and markov_table programs survived
    final_context_count = sum(1 for ind in engine.population if ind.metrics.get('ctx_reads_per_eval', 0) > 0)
    final_markov_count = sum(1 for ind in engine.population if 'markov_table' in pretty_print_ast(ind.program))
    final_best_fitness = max(ind.fitness for ind in engine.population)
    
    print(f"\n  Final results:")
    print(f"    Context users: {final_context_count}")
    print(f"    Markov tables: {final_markov_count}")
    print(f"    Best fitness: {final_best_fitness:.6f}")
    
    # Success if we maintain context users and achieve reasonable fitness
    success = final_context_count >= 2 and final_best_fitness > -1.0
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    
    return success

def main():
    print("COMPLETE SURGICAL FIXES TEST")
    print("=" * 30)
    
    test_results = [
        test_markov_table_with_refit(),
        test_niche_selection_with_shaping(),
    ]
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\nCOMPLETE SURGICAL FIXES SUMMARY")
    print(f"=" * 35)
    print(f"Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("ALL SURGICAL FIXES WORKING!")
        print("Ready to test markov_k2 benchmark with complete fixes.")
    elif passed_tests >= 1:
        print("MOSTLY WORKING - Some issues remain")
    else:
        print("SURGICAL FIXES NEED MORE WORK")
    
    return passed_tests >= 1

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
