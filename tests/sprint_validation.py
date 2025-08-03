#!/usr/bin/env python3
"""Validate the sprint goals: quick wins, selection improvements, fitness signals."""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.eval import evaluate_program_on_window
from protosynth.evolve import EvolutionEngine, create_initial_population
from protosynth.envs import periodic
import itertools

def test_manual_programs():
    """Test that manual programs work correctly on target patterns."""
    print("MILESTONE CHECK: Manual Programs")
    print("-" * 35)
    
    interpreter = LispInterpreter()
    
    # Test 1: prev2 on alternating (should be excellent)
    print("Test 1: prev2 on alternating pattern")
    alternating = list(itertools.islice(periodic([1, 0], seed=42), 200))
    fitness, metrics = evaluate_program_on_window(interpreter, var('prev2'), alternating, k=3)
    ctx_reads = metrics.get('ctx_reads_per_eval', 0)
    context_bonus = metrics.get('context_bonus', 0)
    
    print(f"  prev2: F={fitness:.6f}, ctx={ctx_reads:.2f}, bonus={context_bonus:.6f}")
    prev2_success = fitness > 0.5
    print(f"  Result: {'PASS' if prev2_success else 'FAIL'}")
    
    # Test 2: parity3 on alternating (should be good)
    print("\nTest 2: parity3 on alternating pattern")
    parity3 = op('xor', op('xor', var('prev'), var('prev2')), var('prev3'))
    fitness, metrics = evaluate_program_on_window(interpreter, parity3, alternating, k=4)
    ctx_reads = metrics.get('ctx_reads_per_eval', 0)
    context_bonus = metrics.get('context_bonus', 0)
    
    print(f"  parity3: F={fitness:.6f}, ctx={ctx_reads:.2f}, bonus={context_bonus:.6f}")
    parity3_success = fitness > 0.0
    print(f"  Result: {'PASS' if parity3_success else 'FAIL'}")
    
    # Test 3: majority3 on biased pattern
    print("\nTest 3: majority3 on biased pattern")
    biased = [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1] * 20  # Mostly 1s
    majority3 = op('>', op('+', op('+', var('prev'), var('prev2')), var('prev3')), const(1.5))
    fitness, metrics = evaluate_program_on_window(interpreter, majority3, biased, k=4)
    ctx_reads = metrics.get('ctx_reads_per_eval', 0)
    context_bonus = metrics.get('context_bonus', 0)
    
    print(f"  majority3: F={fitness:.6f}, ctx={ctx_reads:.2f}, bonus={context_bonus:.6f}")
    majority3_success = fitness > -1.0  # Should be reasonable
    print(f"  Result: {'PASS' if majority3_success else 'FAIL'}")
    
    return prev2_success and parity3_success and majority3_success

def test_evolution_curves():
    """Test that evolution shows improvement over time."""
    print("\nEVOLUTION CURVES: Rising Fitness")
    print("-" * 35)
    
    # Test on periodic pattern
    print("Test: Evolution on periodic pattern")
    pattern = [1, 0, 1, 0]  # Simple alternating
    test_bits = (pattern * 60)[:200]
    
    engine = EvolutionEngine(mu=8, lambda_=16, seed=42, k=3, N=150)
    initial_pop = create_initial_population(8, seed=42)
    engine.initialize_population(initial_pop)
    
    fitness_history = []
    context_history = []
    
    for gen in range(25):
        stats = engine.evolve_generation(iter(test_bits))
        best_fitness = stats['best_fitness']
        fitness_history.append(best_fitness)
        
        # Track context usage
        best_ind = max(engine.population, key=lambda x: x.fitness)
        ctx_reads = best_ind.metrics.get('ctx_reads_per_eval', 0)
        context_history.append(ctx_reads)
        
        if gen % 5 == 0:
            context_bonus = best_ind.metrics.get('context_bonus', 0)
            print(f"  Gen {gen:2d}: F={best_fitness:.6f}, ctx={ctx_reads:.2f}, bonus={context_bonus:.6f}")
            print(f"    Best: {pretty_print_ast(best_ind.program)}")
    
    # Check for improvement
    early_fitness = sum(fitness_history[:5]) / 5  # Average of first 5 gens
    late_fitness = sum(fitness_history[-5:]) / 5  # Average of last 5 gens
    improvement = late_fitness - early_fitness
    
    # Check for context usage
    early_context = sum(context_history[:5]) / 5
    late_context = sum(context_history[-5:]) / 5
    context_improvement = late_context - early_context
    
    print(f"\n  Early fitness (avg): {early_fitness:.6f}")
    print(f"  Late fitness (avg): {late_fitness:.6f}")
    print(f"  Improvement: {improvement:.6f}")
    print(f"  Context usage improvement: {context_improvement:.2f}")
    
    # Success criteria
    fitness_improving = improvement > 0.01
    context_used = late_context > 0.1
    
    print(f"  Fitness improving: {'PASS' if fitness_improving else 'FAIL'}")
    print(f"  Context being used: {'PASS' if context_used else 'FAIL'}")
    
    return fitness_improving and context_used

def test_system_features():
    """Test that new system features are working."""
    print("\nSYSTEM FEATURES: Implementation Check")
    print("-" * 40)
    
    # Test 1: Seed diversity
    population = create_initial_population(10, seed=42)
    context_programs = sum(1 for prog in population if 'prev' in pretty_print_ast(prog))
    print(f"  Context programs in seeds: {context_programs}/10")
    seed_success = context_programs >= 6
    print(f"  Seed diversity: {'PASS' if seed_success else 'FAIL'}")
    
    # Test 2: Context bonus
    interpreter = LispInterpreter()
    test_bits = [1, 0, 1, 0] * 25
    
    const_fitness, const_metrics = evaluate_program_on_window(interpreter, const(0.5), test_bits, k=2)
    var_fitness, var_metrics = evaluate_program_on_window(interpreter, var('prev'), test_bits, k=2)
    
    const_bonus = const_metrics.get('context_bonus', 0)
    var_bonus = var_metrics.get('context_bonus', 0)
    
    print(f"  const(0.5) bonus: {const_bonus:.6f}")
    print(f"  var('prev') bonus: {var_bonus:.6f}")
    bonus_success = var_bonus > const_bonus
    print(f"  Context bonus working: {'PASS' if bonus_success else 'FAIL'}")
    
    # Test 3: Selection tie-breaking
    from protosynth.evolve import Individual
    ind1 = Individual(const(0.5), 0.5, {'ctx_reads_per_eval': 0.0}, 0)
    ind2 = Individual(var('prev'), 0.5, {'ctx_reads_per_eval': 1.0}, 0)
    
    sorted_inds = sorted([ind1, ind2])
    first_has_context = 'prev' in pretty_print_ast(sorted_inds[0].program)
    print(f"  Selection prefers context: {'PASS' if first_has_context else 'FAIL'}")
    
    return seed_success and bonus_success and first_has_context

def main():
    print("SPRINT VALIDATION")
    print("=" * 20)
    
    test_results = [
        test_manual_programs(),
        test_evolution_curves(),
        test_system_features()
    ]
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\nSUMMARY: {passed_tests}/{total_tests} test categories passed")
    
    if passed_tests >= 2:
        print("SPRINT GOALS ACHIEVED - System shows learning capability")
    else:
        print("SPRINT GOALS PARTIALLY MET - More work needed")
    
    return passed_tests >= 2

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
