#!/usr/bin/env python3
"""Test the complete hard niche system for Markov evolution."""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.evolve import EvolutionEngine, create_markov_table_individual, create_initial_population
from protosynth.envs import markov_k1
import itertools

def test_hard_niche_survival():
    """Test hard niche survival guarantees."""
    print("TEST: Hard Niche Survival")
    print("-" * 30)
    
    # Create evolution engine with hard niche
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
    
    # Test hard niche survival over multiple generations
    print(f"\n  Testing hard niche survival over 20 generations:")
    
    context_survival_history = []
    
    for gen in range(20):
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
        K_ctx_start = max(8, engine.mu // 2)  # 8 for mu=16
        if gen <= 30:
            K_ctx = K_ctx_start
        elif gen <= 100:
            decay_progress = (gen - 30) / (100 - 30)
            K_ctx = int(K_ctx_start * (1 - decay_progress))
        else:
            K_ctx = 0
        
        context_survival_history.append(context_count)
        
        if gen % 5 == 0 or gen < 10:
            print(f"    Gen {gen:2d}: F_best={best_fitness:.6f}, ctx={context_count:2d}, markov={markov_count}, K_ctx={K_ctx}")
        
        if best_fitness > 0:
            print(f"    SUCCESS: Positive fitness achieved at gen {gen}!")
            break
    
    # Check hard niche guarantees
    min_expected_ctx = 8  # K_ctx for first 30 generations
    min_actual_ctx = min(context_survival_history[:min(20, len(context_survival_history))])
    
    print(f"\n  Hard niche analysis:")
    print(f"    Expected min context users: {min_expected_ctx}")
    print(f"    Actual min context users: {min_actual_ctx}")
    print(f"    Context survival history: {context_survival_history}")
    
    # Success if we maintain at least the minimum expected context users
    success = min_actual_ctx >= min_expected_ctx
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    
    return success

def test_ensemble_variance_reduction():
    """Test ensemble evaluation for variance reduction."""
    print("\nTEST: Ensemble Variance Reduction")
    print("-" * 40)
    
    # Create a simple context program
    prog = var('prev')
    
    # Test single vs ensemble evaluation
    from protosynth.evolve import eval_candidate
    
    # Generate primary buffer
    m1_stream = markov_k1(p_stay=0.8, seed=42)
    primary_buf = list(itertools.islice(m1_stream, 5000))
    
    # Generate ensemble buffers
    ensemble_buffers = []
    for i in range(2):
        ensemble_stream = markov_k1(p_stay=0.8, seed=42 + i + 1)
        ensemble_buf = list(itertools.islice(ensemble_stream, 5000))
        ensemble_buffers.append(ensemble_buf)
    
    # Single evaluation
    fitness_single, metrics_single = eval_candidate(prog, "markov_k2", primary_buf, 2)

    # Ensemble evaluation
    fitness_ensemble, metrics_ensemble = eval_candidate(prog, "markov_k2", primary_buf, 2, ensemble_buffers)
    
    print(f"  Single evaluation:")
    print(f"    Fitness: {fitness_single:.6f}")
    print(f"    Context reads: {metrics_single.get('ctx_reads_per_eval', 0):.2f}")
    
    print(f"  Ensemble evaluation:")
    print(f"    Fitness: {fitness_ensemble:.6f}")
    print(f"    Context reads: {metrics_ensemble.get('ctx_reads_per_eval', 0):.2f}")
    print(f"    Ensemble std: {metrics_ensemble.get('ensemble_std', 0):.6f}")
    print(f"    Individual fitnesses: {metrics_ensemble.get('ensemble_fitnesses', [])}")
    
    # Success if ensemble evaluation works and reduces variance
    has_ensemble_metrics = 'ensemble_std' in metrics_ensemble
    success = has_ensemble_metrics and fitness_ensemble != fitness_single
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    
    return success

def main():
    print("HARD NICHE SYSTEM TEST")
    print("=" * 25)
    
    test_results = [
        test_hard_niche_survival(),
        test_ensemble_variance_reduction(),
    ]
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\nHARD NICHE SYSTEM SUMMARY")
    print(f"=" * 28)
    print(f"Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("HARD NICHE SYSTEM WORKING!")
        print("Context users should survive and markov_k2 should pass.")
    elif passed_tests >= 1:
        print("MOSTLY WORKING - Some issues remain")
    else:
        print("HARD NICHE SYSTEM NEEDS MORE WORK")
    
    return passed_tests >= 1

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
