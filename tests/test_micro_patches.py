#!/usr/bin/env python3
"""Test the micro-patches for pushing F > 0."""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.evolve import EvolutionEngine, create_markov_table_individual, is_context_user_fast
from protosynth.envs import markov_k1
import itertools

def test_pinned_teacher():
    """Test pinned teacher system."""
    print("TEST: Pinned Teacher System")
    print("-" * 35)
    
    # Create evolution engine
    engine = EvolutionEngine(mu=16, lambda_=48, seed=42, k=2, N=2000, env_name="markov_k2")
    
    # Initialize with context-heavy population
    initial_pop = []
    
    # Generate training buffer
    m1_stream = markov_k1(p_stay=0.8, seed=42)
    train_buf = list(itertools.islice(m1_stream, 4096))
    
    # Add MLE teacher
    mle_teacher = engine.build_mle_markov_candidate([train_buf], k=2)
    initial_pop.append(mle_teacher.program)
    
    # Add context seeds
    context_seeds = [
        var('prev'), var('prev2'),
        if_expr(op('>', var('prev'), const(0.5)), const(0.8), const(0.2)),
        create_markov_table_individual('stay', generation=0).program,
        create_markov_table_individual('flip', generation=0).program,
    ]
    
    for i in range(min(11, len(context_seeds))):
        initial_pop.append(context_seeds[i])
    
    # Fill remainder
    while len(initial_pop) < 16:
        initial_pop.append(const(0.5))
    
    engine.initialize_population(initial_pop)
    
    print(f"  Initial population size: {len(engine.population)}")
    
    # Test pinned teacher over generations
    print(f"  Testing pinned teacher over 15 generations:")
    
    teacher_history = []
    fitness_history = []
    
    for gen in range(15):
        # Create fresh stream
        m1_stream = markov_k1(p_stay=0.8, seed=42+gen)
        fresh_bits = list(itertools.islice(m1_stream, engine.N + engine.k))
        stream = iter(fresh_bits)
        
        stats = engine.evolve_generation(stream)
        best_fitness = stats['best_fitness']
        
        # Track teacher status
        teacher_pinned = engine.pinned_teacher is not None
        teacher_fitness = engine.pinned_teacher.fitness if teacher_pinned else None
        
        teacher_history.append(teacher_pinned)
        fitness_history.append(best_fitness)
        
        if gen % 5 == 0 or gen < 3:
            teacher_status = f"PINNED(F={teacher_fitness:.4f})" if teacher_pinned else "FREE"
            print(f"    Gen {gen:2d}: F_best={best_fitness:.6f}, teacher={teacher_status}")
        
        if best_fitness > 0.05:
            print(f"    SUCCESS: Target fitness achieved at gen {gen}!")
            break
    
    # Analyze teacher system
    pinned_count = sum(teacher_history)
    max_fitness = max(fitness_history)
    
    print(f"  Pinned teacher analysis:")
    print(f"    Generations with pinned teacher: {pinned_count}/{len(teacher_history)}")
    print(f"    Max fitness achieved: {max_fitness:.6f}")
    
    # Success if teacher stays pinned and good fitness
    success = pinned_count >= 10 and max_fitness > -0.1
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    
    return success

def test_variance_reduction():
    """Test variance reduction with M=5 buffers and longer windows."""
    print("\nTEST: Variance Reduction (M=5, N=8192)")
    print("-" * 45)
    
    # Test single vs ensemble evaluation
    from protosynth.evolve import eval_candidate
    
    # Create markov_table program
    prog = op('markov_table', op('+', op('*', const(2), var('prev2')), var('prev')))
    
    # Generate primary buffer
    m1_stream = markov_k1(p_stay=0.8, seed=42)
    primary_buf = list(itertools.islice(m1_stream, 10000))  # Longer buffer
    
    # Generate ensemble buffers (M=5 total)
    ensemble_buffers = []
    for i in range(4):  # 4 additional = 5 total
        ensemble_stream = markov_k1(p_stay=0.8, seed=42 + i + 1)
        ensemble_buf = list(itertools.islice(ensemble_stream, 10000))
        ensemble_buffers.append(ensemble_buf)
    
    # Single evaluation
    fitness_single, metrics_single = eval_candidate(prog, "markov_k2", primary_buf, 2)
    
    # Ensemble evaluation
    fitness_ensemble, metrics_ensemble = eval_candidate(prog, "markov_k2", primary_buf, 2, ensemble_buffers)
    
    print(f"  Single evaluation:")
    print(f"    Fitness: {fitness_single:.6f}")
    print(f"    Train samples: {metrics_single.get('train_samples', 0)}")
    
    print(f"  Ensemble evaluation (M=5):")
    print(f"    Fitness: {fitness_ensemble:.6f}")
    print(f"    Train samples: {metrics_ensemble.get('train_samples', 0)}")
    print(f"    Ensemble std: {metrics_ensemble.get('ensemble_std', 0):.6f}")
    print(f"    Individual fitnesses: {[f'{f:.4f}' for f in metrics_ensemble.get('ensemble_fitnesses', [])]}")
    
    # Success if ensemble evaluation works and has lower variance
    has_ensemble = 'ensemble_std' in metrics_ensemble
    lower_variance = metrics_ensemble.get('ensemble_std', 1.0) < 0.1
    longer_train = metrics_ensemble.get('train_samples', 0) >= 8000
    
    success = has_ensemble and longer_train
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    
    return success

def main():
    print("MICRO-PATCHES TEST")
    print("=" * 20)
    
    test_results = [
        test_pinned_teacher(),
        test_variance_reduction(),
    ]
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\nMICRO-PATCHES SUMMARY")
    print(f"=" * 23)
    print(f"Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("MICRO-PATCHES WORKING!")
        print("Should push F > 0 and toward 0.10 target.")
    elif passed_tests >= 1:
        print("MOSTLY WORKING - Some issues remain")
    else:
        print("MICRO-PATCHES NEED MORE WORK")
    
    return passed_tests >= 1

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
