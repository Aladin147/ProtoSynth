#!/usr/bin/env python3
"""Quick benchmark test for markov_k2."""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.evolve import EvolutionEngine, create_initial_population
from protosynth.envs import markov_k2
import itertools

def main():
    print('QUICK BENCHMARK TEST: markov_k2')
    print('=' * 32)

    # First, let's analyze what the markov_k2 environment generates
    print('1. Analyzing markov_k2 environment:')
    m2_stream = markov_k2(p_stay=0.8, seed=42)
    m2_bits = list(itertools.islice(m2_stream, 100))
    print(f'   Pattern: {m2_bits[:20]}...')
    
    # Count transitions
    transitions = {}
    for i in range(1, len(m2_bits)):
        prev_bit = m2_bits[i-1]
        curr_bit = m2_bits[i]
        key = f'{prev_bit}->{curr_bit}'
        transitions[key] = transitions.get(key, 0) + 1
    
    print(f'   Transitions: {transitions}')
    
    # Test what programs work on this pattern
    interpreter = LispInterpreter(max_steps=1000, timeout_seconds=10.0)
    
    test_programs = [
        ('const(0.5)', const(0.5)),
        ('prev', var('prev')),
        ('prev2', var('prev2')),
    ]
    
    print('\n   Testing programs on markov_k2:')
    for name, prog in test_programs:
        try:
            fitness, metrics = evaluate_program_on_window(interpreter, prog, m2_bits, k=2)
            ctx_reads = metrics.get('ctx_reads_per_eval', 0)
            penalty = metrics.get('penalty_bits', 0)
            print(f'     {name:12s}: F={fitness:8.6f}, ctx={ctx_reads:.2f}, penalty={penalty:.1f}')
        except Exception as e:
            print(f'     {name:12s}: ERROR - {e}')

    # Create evolution engine
    print('\n2. Running evolution on markov_k2:')
    engine = EvolutionEngine(mu=8, lambda_=16, seed=42, k=2, N=500)
    initial_pop = create_initial_population(8, seed=42)
    engine.initialize_population(initial_pop)

    print('   Initial population (first 4):')
    for i, ind in enumerate(engine.population[:4]):
        print(f'     {i}: {pretty_print_ast(ind.program)}')

    # Run evolution
    print(f'\n   Running evolution (target: F >= 0.10):')

    for gen in range(30):
        # Create fresh stream for each generation
        m2_stream = markov_k2(p_stay=0.8, seed=42+gen)  # Different seed each gen
        fresh_bits = list(itertools.islice(m2_stream, engine.N + engine.k))
        stream = iter(fresh_bits)
        
        stats = engine.evolve_generation(stream)
        best_fitness = stats['best_fitness']
        
        if gen % 10 == 0 or best_fitness >= 0.10:
            best_ind = max(engine.population, key=lambda x: x.fitness)
            ctx_reads = best_ind.metrics.get('ctx_reads_per_eval', 0)
            penalty = best_ind.metrics.get('penalty_bits', 0)
            
            print(f'     Gen {gen:2d}: F={best_fitness:.6f}, ctx={ctx_reads:.2f}, penalty={penalty:.1f}')
            print(f'       Best: {pretty_print_ast(best_ind.program)}')
            
            if best_fitness >= 0.10:
                print(f'\n   TARGET REACHED! F={best_fitness:.6f} >= 0.10 in {gen+1} generations')
                break

    final_fitness = max(ind.fitness for ind in engine.population)
    print(f'\nFinal result: F={final_fitness:.6f}')

    if final_fitness >= 0.10:
        print('BENCHMARK PASSED: markov_k2 target achieved!')
        return True
    else:
        print('BENCHMARK FAILED: Need more generations or better approach')
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
