#!/usr/bin/env python3
"""Test the core fixes for ProtoSynth context access and fitness evaluation."""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.eval import evaluate_program_on_window
from protosynth.envs import periodic
import itertools

def main():
    print('ROBUST VALIDATION: Core Fixes')
    print('=' * 35)

    # Test data
    test_bits = list(itertools.islice(periodic([1, 0], seed=42), 50))
    interpreter = LispInterpreter()

    test_cases = [
        ('const(0.5)', const(0.5), 'baseline constant'),
        ('const(0)', const(0), 'always wrong'),
        ('prev', var('prev'), 'previous bit'),
        ('prev2', var('prev2'), 'bit from 2 steps ago'),
    ]

    print(f'Test sequence length: {len(test_bits)}')
    print(f'Pattern: {test_bits[:10]}...')
    print()

    results = []
    for name, program, description in test_cases:
        try:
            fitness, metrics = evaluate_program_on_window(interpreter, program, test_bits, k=2)
            
            ctx_reads = metrics.get('ctx_reads', 0)
            penalty = metrics.get('penalty_bits', 0)
            num_predictions = metrics.get('num_predictions', 0)
            
            results.append({
                'name': name,
                'fitness': fitness,
                'ctx_reads': ctx_reads,
                'penalty': penalty,
                'num_predictions': num_predictions,
                'description': description
            })
            
            print(f'{name:12s}: F={fitness:8.6f}, ctx_reads={ctx_reads}, penalty={penalty:.1f}, preds={num_predictions}')
            
        except Exception as e:
            print(f'{name:12s}: ERROR - {e}')

    print()
    print('Analysis:')

    # Check if context tracking works
    ctx_programs = [r for r in results if 'prev' in r['name']]
    if all(r['ctx_reads'] > 0 for r in ctx_programs):
        print('  Context tracking: WORKING')
    else:
        print('  Context tracking: FAILED')

    # Check if penalties are applied correctly
    working_programs = [r for r in results if r['name'] in ['const(0.5)']]
    if all(r['penalty'] == 0 for r in working_programs):
        print('  Penalty system: WORKING')
    else:
        print('  Penalty system: FAILED')

    # Check if fitness differences exist
    fitnesses = [r['fitness'] for r in results]
    if len(set(f'{f:.3f}' for f in fitnesses)) > 1:
        print('  Fitness differences: WORKING')
    else:
        print('  Fitness differences: FAILED')

    # Check if prev2 beats baseline (should be very good on alternating sequence)
    baseline_fitness = next((r['fitness'] for r in results if r['name'] == 'const(0.5)'), None)
    prev2_fitness = next((r['fitness'] for r in results if r['name'] == 'prev2'), None)

    if baseline_fitness is not None and prev2_fitness is not None:
        improvement = prev2_fitness - baseline_fitness
        print(f'  prev2 vs baseline: {improvement:.6f} bits improvement')
        if improvement > 0.5:
            print('  Pattern recognition: WORKING')
        else:
            print('  Pattern recognition: NEEDS ATTENTION')

    print('\nCore fixes validation complete.')

if __name__ == '__main__':
    main()
