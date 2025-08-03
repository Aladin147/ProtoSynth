#!/usr/bin/env python3
"""Quick benchmark test for periodic_k4."""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.evolve import EvolutionEngine, create_initial_population
from protosynth.envs import periodic_k4
import itertools

def main():
    print('QUICK BENCHMARK TEST: periodic_k4')
    print('=' * 35)

    # Create evolution engine with prev4 in seeds
    engine = EvolutionEngine(mu=8, lambda_=16, seed=42, k=4, N=500)
    initial_pop = create_initial_population(8, seed=42)
    engine.initialize_population(initial_pop)

    print('Initial population (first 4):')
    for i, ind in enumerate(engine.population[:4]):
        print(f'  {i}: {pretty_print_ast(ind.program)}')

    # Run evolution on periodic_k4
    print(f'\nRunning evolution on periodic_k4 (target: F >= 0.25):')

    for gen in range(20):
        # Create fresh stream for each generation
        p4_stream = periodic_k4(seed=42)
        fresh_bits = list(itertools.islice(p4_stream, engine.N + engine.k))
        stream = iter(fresh_bits)
        
        stats = engine.evolve_generation(stream)
        best_fitness = stats['best_fitness']
        
        if gen % 5 == 0 or best_fitness >= 0.25:
            best_ind = max(engine.population, key=lambda x: x.fitness)
            ctx_reads = best_ind.metrics.get('ctx_reads_per_eval', 0)
            penalty = best_ind.metrics.get('penalty_bits', 0)
            
            print(f'  Gen {gen:2d}: F={best_fitness:.6f}, ctx={ctx_reads:.2f}, penalty={penalty:.1f}')
            print(f'    Best: {pretty_print_ast(best_ind.program)}')
            
            if best_fitness >= 0.25:
                print(f'\nTARGET REACHED! F={best_fitness:.6f} >= 0.25 in {gen+1} generations')
                break

    final_fitness = max(ind.fitness for ind in engine.population)
    print(f'\nFinal result: F={final_fitness:.6f}')

    if final_fitness >= 0.25:
        print('BENCHMARK PASSED: periodic_k4 target achieved!')
        return True
    else:
        print('BENCHMARK FAILED: Need more generations or better seeds')
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
