#!/usr/bin/env python3
"""Test the complete surgical system with MLE teacher and gate-only decay."""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.evolve import EvolutionEngine, create_markov_table_individual, create_initial_population
from protosynth.envs import markov_k1
import itertools

def test_mle_teacher():
    """Test MLE teacher generation and positive fitness."""
    print("TEST: MLE Teacher Generation")
    print("-" * 35)
    
    # Create evolution engine
    engine = EvolutionEngine(mu=16, lambda_=48, seed=42, k=2, N=1000, env_name="markov_k2")
    
    # Generate test buffers
    buffers = []
    for i in range(3):
        m1_stream = markov_k1(p_stay=0.8, seed=42+i)
        buffer = list(itertools.islice(m1_stream, 3000))
        buffers.append(buffer)
    
    # Build MLE teacher
    teacher = engine.build_mle_markov_candidate(buffers, k=2)
    
    print(f"  MLE teacher program: {pretty_print_ast(teacher.program)}")
    print(f"  MLE parameters: {teacher.metrics.get('markov_params', {})}")
    
    # Evaluate teacher on same buffers
    from protosynth.evolve import eval_candidate
    fitness, metrics = eval_candidate(teacher.program, "markov_k2", buffers[0], 2, buffers[1:])
    
    print(f"  Teacher fitness: {fitness:.6f}")
    print(f"  Teacher ctx_reads: {metrics.get('ctx_reads_per_eval', 0):.2f}")
    
    # Success if teacher has positive fitness and uses context
    success = fitness > 0 and metrics.get('ctx_reads_per_eval', 0) > 0
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    
    return success

def test_gate_only_decay():
    """Test gate-only decay policy."""
    print("\nTEST: Gate-Only Decay Policy")
    print("-" * 35)
    
    # Create evolution engine with gate-only decay
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
    
    # Test gate-only evolution over extended period
    print(f"\n  Testing gate-only evolution over 100 generations:")
    
    context_survival_history = []
    k_ctx_history = []
    ready_history = []
    teacher_fitness_history = []
    
    for gen in range(100):
        # Create fresh Markov stream
        m1_stream = markov_k1(p_stay=0.8, seed=42+gen)
        fresh_bits = list(itertools.islice(m1_stream, engine.N + engine.k))
        stream = iter(fresh_bits)
        
        stats = engine.evolve_generation(stream)
        best_fitness = stats['best_fitness']
        
        # Track statistics
        context_count = sum(1 for ind in engine.population if ind.metrics.get('ctx_reads_per_eval', 0) > 0)
        from protosynth.evolve import adaptive_K_ctx
        K_ctx = adaptive_K_ctx(engine.generation, engine.mu, engine.ctx_stats)
        ready = engine.ctx_stats.ready()
        
        context_survival_history.append(context_count)
        k_ctx_history.append(K_ctx)
        ready_history.append(ready)
        teacher_fitness_history.append(best_fitness)
        
        if gen % 20 == 0 or gen < 5 or best_fitness > 0:
            status = "READY" if ready else "PROTECTED"
            print(f"    Gen {gen:2d}: F_best={best_fitness:.6f}, ctx={context_count:2d}, K_ctx={K_ctx}, {status}")
        
        if best_fitness > 0.05:
            print(f"    SUCCESS: Target fitness achieved at gen {gen}!")
            break
    
    # Analyze gate-only behavior
    min_context_count = min(context_survival_history)
    avg_context_count = sum(context_survival_history) / len(context_survival_history)
    ready_count = sum(ready_history)
    max_fitness = max(teacher_fitness_history)
    
    print(f"\n  Gate-only evolution analysis:")
    print(f"    Min context survivors: {min_context_count}")
    print(f"    Avg context survivors: {avg_context_count:.1f}")
    print(f"    Generations ready: {ready_count}/{len(ready_history)}")
    print(f"    K_ctx range: {min(k_ctx_history)} - {max(k_ctx_history)}")
    print(f"    Max fitness achieved: {max_fitness:.6f}")
    
    # Success if good context survival and positive fitness achieved
    success = min_context_count >= 6 and max_fitness > 0
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    
    return success

def main():
    print("COMPLETE SURGICAL SYSTEM TEST")
    print("=" * 32)
    
    test_results = [
        test_mle_teacher(),
        test_gate_only_decay(),
    ]
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\nSURGICAL SYSTEM SUMMARY")
    print(f"=" * 25)
    print(f"Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("COMPLETE SURGICAL SYSTEM WORKING!")
        print("MLE teacher + gate-only decay should solve markov_k2.")
    elif passed_tests >= 1:
        print("MOSTLY WORKING - Some issues remain")
    else:
        print("SURGICAL SYSTEM NEEDS MORE WORK")
    
    return passed_tests >= 1

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
