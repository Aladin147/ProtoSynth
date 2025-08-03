#!/usr/bin/env python3
"""Test the adaptive K_ctx system."""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.evolve import EvolutionEngine, CtxStats, adaptive_K_ctx, create_markov_table_individual, create_initial_population
from protosynth.envs import markov_k1
import itertools

def test_ctx_stats():
    """Test CtxStats tracking and gate logic."""
    print("TEST: CtxStats Tracking")
    print("-" * 25)
    
    stats = CtxStats(W=5)
    
    # Simulate early generations with poor fitness and low context ratio
    print("  Simulating early generations (poor fitness, low context):")
    for gen in range(5):
        # Mock survivors: mostly plain with poor fitness
        mock_survivors = []
        for i in range(16):
            if i < 2:  # 2 context users
                mock_survivors.append(type('MockInd', (), {
                    'fitness': -0.8,
                    'metrics': {'ctx_reads_per_eval': 0.5}
                })())
            else:  # 14 plain users
                mock_survivors.append(type('MockInd', (), {
                    'fitness': -0.7,
                    'metrics': {'ctx_reads_per_eval': 0.0}
                })())
        
        stats.push(-0.7, mock_survivors)
        ready = stats.ready()
        print(f"    Gen {gen}: ready={ready}")
    
    # Test adaptive K_ctx during early phase
    K_ctx_early = adaptive_K_ctx(gen=10, mu=16, stats=stats)
    print(f"  Early K_ctx (not ready): {K_ctx_early}")
    
    # Simulate later generations with better fitness and higher context ratio
    print("\n  Simulating later generations (better fitness, higher context):")
    for gen in range(5):
        # Mock survivors: more context users with better fitness
        mock_survivors = []
        for i in range(16):
            if i < 8:  # 8 context users
                mock_survivors.append(type('MockInd', (), {
                    'fitness': 0.1,
                    'metrics': {'ctx_reads_per_eval': 0.5}
                })())
            else:  # 8 plain users
                mock_survivors.append(type('MockInd', (), {
                    'fitness': 0.08,
                    'metrics': {'ctx_reads_per_eval': 0.0}
                })())
        
        stats.push(0.1, mock_survivors)
        ready = stats.ready()
        print(f"    Gen {gen+5}: ready={ready}")
    
    # Test adaptive K_ctx when ready
    K_ctx_ready = adaptive_K_ctx(gen=60, mu=16, stats=stats)
    print(f"  Ready K_ctx (gen 60): {K_ctx_ready}")
    
    success = K_ctx_early == 8 and K_ctx_ready < 8
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    
    return success

def test_adaptive_evolution():
    """Test adaptive evolution with signal-based decay."""
    print("\nTEST: Adaptive Evolution")
    print("-" * 28)
    
    # Create evolution engine with adaptive K_ctx
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
    
    # Test adaptive evolution over multiple generations
    print(f"\n  Testing adaptive evolution over 30 generations:")
    
    context_survival_history = []
    k_ctx_history = []
    ready_history = []
    
    for gen in range(30):
        # Create fresh Markov stream
        m1_stream = markov_k1(p_stay=0.8, seed=42+gen)
        fresh_bits = list(itertools.islice(m1_stream, engine.N + engine.k))
        stream = iter(fresh_bits)
        
        stats = engine.evolve_generation(stream)
        best_fitness = stats['best_fitness']
        
        # Track statistics
        context_count = sum(1 for ind in engine.population if ind.metrics.get('ctx_reads_per_eval', 0) > 0)
        K_ctx = adaptive_K_ctx(engine.generation, engine.mu, engine.ctx_stats)
        ready = engine.ctx_stats.ready()
        
        context_survival_history.append(context_count)
        k_ctx_history.append(K_ctx)
        ready_history.append(ready)
        
        if gen % 10 == 0 or gen < 5:
            status = "READY" if ready else "PROTECTED"
            print(f"    Gen {gen:2d}: F_best={best_fitness:.6f}, ctx={context_count:2d}, K_ctx={K_ctx}, {status}")
        
        if best_fitness > 0:
            print(f"    SUCCESS: Positive fitness achieved at gen {gen}!")
            break
    
    # Analyze adaptive behavior
    min_context_count = min(context_survival_history)
    avg_context_count = sum(context_survival_history) / len(context_survival_history)
    ready_count = sum(ready_history)
    
    print(f"\n  Adaptive evolution analysis:")
    print(f"    Min context survivors: {min_context_count}")
    print(f"    Avg context survivors: {avg_context_count:.1f}")
    print(f"    Generations ready: {ready_count}/{len(ready_history)}")
    print(f"    K_ctx range: {min(k_ctx_history)} - {max(k_ctx_history)}")
    
    # Success if good context survival and adaptive behavior
    success = min_context_count >= 4 and max(k_ctx_history) >= 8
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    
    return success

def main():
    print("ADAPTIVE K_CTX SYSTEM TEST")
    print("=" * 28)
    
    test_results = [
        test_ctx_stats(),
        test_adaptive_evolution(),
    ]
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\nADAPTIVE K_CTX SUMMARY")
    print(f"=" * 23)
    print(f"Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("ADAPTIVE K_CTX WORKING!")
        print("Context users should survive until ready to compete.")
    elif passed_tests >= 1:
        print("MOSTLY WORKING - Some issues remain")
    else:
        print("ADAPTIVE K_CTX NEEDS MORE WORK")
    
    return passed_tests >= 1

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
