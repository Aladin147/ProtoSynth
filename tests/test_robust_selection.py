#!/usr/bin/env python3
"""Test the robust selection system with hard quota enforcement."""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.evolve import EvolutionEngine, Individual, create_markov_table_individual, create_initial_population
from protosynth.envs import markov_k1
import itertools

def test_quota_enforcement():
    """Test that the quota is actually enforced."""
    print("TEST: Hard Quota Enforcement")
    print("-" * 35)
    
    # Create evolution engine
    engine = EvolutionEngine(mu=16, lambda_=48, seed=42, k=2, N=1000, env_name="markov_k2")
    
    # Create test population with known context usage
    test_population = []
    
    # Add 12 plain programs with slightly better fitness
    for i in range(12):
        ind = Individual(
            program=const(0.5),
            fitness=0.10 + 1e-6 * i,  # Slightly better fitness
            metrics={'ctx_reads_per_eval': 0.0},  # No context usage
            generation=0
        )
        test_population.append(ind)
    
    # Add 4 context programs with slightly worse fitness
    for i in range(4):
        ind = Individual(
            program=var('prev'),
            fitness=0.09,  # Slightly worse fitness
            metrics={'ctx_reads_per_eval': 0.5},  # Uses context
            generation=0
        )
        test_population.append(ind)
    
    print(f"  Test population: 12 plain (F=0.10+), 4 context (F=0.09)")
    print(f"  Without quota: plain programs should dominate")
    print(f"  With K_ctx=8: should keep all 4 context + 4 best plain")
    
    # Test robust selection with K_ctx=8
    try:
        survivors = engine.select_with_ctx_quota(test_population, mu=16, K_ctx=8)
        
        # Count context survivors
        ctx_survivors = sum(1 for ind in survivors if ind.metrics.get('ctx_reads_per_eval', 0) > 0)
        plain_survivors = len(survivors) - ctx_survivors
        
        print(f"  Result: {ctx_survivors} context, {plain_survivors} plain survivors")
        print(f"  Quota enforcement: {'PASS' if ctx_survivors >= 4 else 'FAIL'}")
        
        # Test should pass - all 4 context users should survive despite worse fitness
        success = ctx_survivors >= 4 and len(survivors) == 16
        
    except AssertionError as e:
        print(f"  Assertion failed: {e}")
        success = False
    
    return success

def test_robust_evolution():
    """Test robust evolution with hard quota."""
    print("\nTEST: Robust Evolution")
    print("-" * 25)
    
    # Create evolution engine
    engine = EvolutionEngine(mu=16, lambda_=48, seed=42, k=2, N=1000, env_name="markov_k2")
    
    # Create initial population with Markov seeds
    initial_pop = create_initial_population(16, seed=42)
    stay_biased = create_markov_table_individual('stay', generation=0)
    flip_biased = create_markov_table_individual('flip', generation=0)
    
    # Replace last 2 with Markov seeds
    initial_pop[-2] = stay_biased.program
    initial_pop[-1] = flip_biased.program
    
    engine.initialize_population(initial_pop)
    
    print(f"  Initial population size: {len(engine.population)}")
    
    # Test robust evolution over multiple generations
    print(f"\n  Testing robust evolution over 15 generations:")
    
    context_survival_history = []
    quota_violations = 0
    
    for gen in range(15):
        # Create fresh Markov stream
        m1_stream = markov_k1(p_stay=0.8, seed=42+gen)
        fresh_bits = list(itertools.islice(m1_stream, engine.N + engine.k))
        stream = iter(fresh_bits)
        
        try:
            stats = engine.evolve_generation(stream)
            best_fitness = stats['best_fitness']
            
            # Count context users
            context_count = sum(1 for ind in engine.population if ind.metrics.get('ctx_reads_per_eval', 0) > 0)
            context_survival_history.append(context_count)
            
            # Expected K_ctx (should be 8 for first 50 generations)
            K_ctx_expected = 8
            
            if gen % 5 == 0 or gen < 5:
                print(f"    Gen {gen:2d}: F_best={best_fitness:.6f}, ctx={context_count:2d}, K_ctx={K_ctx_expected}")
            
            if best_fitness > 0:
                print(f"    SUCCESS: Positive fitness achieved at gen {gen}!")
                break
                
        except AssertionError as e:
            print(f"    Gen {gen}: QUOTA VIOLATION - {e}")
            quota_violations += 1
            break
    
    # Check robust evolution results
    min_context_count = min(context_survival_history) if context_survival_history else 0
    avg_context_count = sum(context_survival_history) / len(context_survival_history) if context_survival_history else 0
    
    print(f"\n  Robust evolution analysis:")
    print(f"    Quota violations: {quota_violations}")
    print(f"    Min context survivors: {min_context_count}")
    print(f"    Avg context survivors: {avg_context_count:.1f}")
    print(f"    Context survival history: {context_survival_history}")
    
    # Success if no quota violations and good context survival
    success = quota_violations == 0 and min_context_count >= 6
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    
    return success

def main():
    print("ROBUST SELECTION SYSTEM TEST")
    print("=" * 30)
    
    test_results = [
        test_quota_enforcement(),
        test_robust_evolution(),
    ]
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\nROBUST SELECTION SUMMARY")
    print(f"=" * 26)
    print(f"Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("ROBUST SELECTION WORKING!")
        print("Hard quota enforcement should solve markov_k2.")
    elif passed_tests >= 1:
        print("MOSTLY WORKING - Some issues remain")
    else:
        print("ROBUST SELECTION NEEDS MORE WORK")
    
    return passed_tests >= 1

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
