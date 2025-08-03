#!/usr/bin/env python3
"""Test the system improvements: context bonus, lexicase selection."""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.eval import evaluate_program_on_window
from protosynth.evolve import EvolutionEngine, create_initial_population
import itertools

def main():
    print('TESTING IMPROVED SYSTEM')
    print('=' * 25)

    # Test 1: Context bonus
    print('Test 1: Context Bonus')
    test_bits = [1, 0] * 50
    interpreter = LispInterpreter()

    # Test constant vs context-using program
    const_fitness, const_metrics = evaluate_program_on_window(interpreter, const(0.5), test_bits, k=3)
    prev2_fitness, prev2_metrics = evaluate_program_on_window(interpreter, var('prev2'), test_bits, k=3)

    print(f'  const(0.5): F={const_fitness:.6f}, bonus={const_metrics.get("context_bonus", 0):.6f}')
    print(f'  prev2: F={prev2_fitness:.6f}, bonus={prev2_metrics.get("context_bonus", 0):.6f}')

    # Test 2: Evolution with improvements
    print('\nTest 2: Evolution with Improvements')
    engine = EvolutionEngine(mu=6, lambda_=12, seed=42, k=3, N=80)
    initial_pop = create_initial_population(6, seed=42)
    engine.initialize_population(initial_pop)

    print('  Initial population:')
    for i, ind in enumerate(engine.population):
        print(f'    {i}: {pretty_print_ast(ind.program)}')

    # Run a few generations
    best_fitness_history = []
    for gen in range(10):
        stats = engine.evolve_generation(iter(test_bits))
        best_fitness = stats['best_fitness']
        best_fitness_history.append(best_fitness)
        
        if gen % 3 == 0:
            best_ind = max(engine.population, key=lambda x: x.fitness)
            ctx_reads = best_ind.metrics.get('ctx_reads_per_eval', 0)
            context_bonus = best_ind.metrics.get('context_bonus', 0)
            print(f'  Gen {gen:2d}: F={best_fitness:.6f}, ctx={ctx_reads:.2f}, bonus={context_bonus:.6f}')

    final_fitness = max(best_fitness_history)
    improvement = final_fitness - best_fitness_history[0] if best_fitness_history else 0

    print(f'\nResults:')
    print(f'  Final fitness: {final_fitness:.6f}')
    print(f'  Improvement: {improvement:.6f}')

    if final_fitness > 0.1:
        print('  Status: EXCELLENT - Found positive fitness!')
    elif improvement > 0.01:
        print('  Status: GOOD - Showing improvement')
    else:
        print('  Status: NEEDS WORK - Limited progress')

    print('\nImproved system test complete.')

if __name__ == '__main__':
    main()
