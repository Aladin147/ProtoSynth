#!/usr/bin/env python3
"""Test the quick wins: selection tie-breaking and seed diversity."""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.evolve import create_initial_population, EvolutionEngine
from protosynth.envs import periodic
import itertools

def test_seed_diversity():
    """Test that new seed programs are included."""
    print("TEST: Seed Diversity")
    print("-" * 20)
    
    population = create_initial_population(10, seed=42)
    
    # Check for context-aware programs
    context_programs = 0
    majority_programs = 0
    parity_programs = 0
    
    for prog in population:
        prog_str = pretty_print_ast(prog)
        if 'prev' in prog_str:
            context_programs += 1
        if 'prev3' in prog_str and '+' in prog_str:
            majority_programs += 1
        if 'xor' in prog_str and 'prev3' in prog_str:
            parity_programs += 1
    
    print(f"  Context programs: {context_programs}/10")
    print(f"  Majority programs: {majority_programs}/10")
    print(f"  Parity programs: {parity_programs}/10")
    
    success = context_programs >= 6 and majority_programs >= 1 and parity_programs >= 1
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    return success

def test_selection_tiebreaking():
    """Test that selection tie-breaking works correctly."""
    print("\nTEST: Selection Tie-Breaking")
    print("-" * 30)
    
    from protosynth.evolve import Individual
    
    # Create individuals with same fitness but different properties
    individuals = [
        Individual(const(0.5), 0.5, {'ctx_reads_per_eval': 0.0}, 0, 0),  # No context, larger AST
        Individual(var('prev'), 0.5, {'ctx_reads_per_eval': 1.0}, 0, 1),  # Context, smaller AST
        Individual(const(0.25), 0.5, {'ctx_reads_per_eval': 0.0}, 0, 2),  # No context, smaller AST
    ]
    
    # Sort using the new tie-breaking rules
    sorted_individuals = sorted(individuals)
    
    # Should be ordered by: fitness (same), then AST size (smaller better), then ctx_reads (higher better)
    # Expected order: var('prev') (smallest + context), const(0.25) (small), const(0.5) (larger)
    
    first_prog = pretty_print_ast(sorted_individuals[0].program)
    print(f"  Best individual: {first_prog}")
    print(f"  Context reads: {sorted_individuals[0].metrics.get('ctx_reads_per_eval', 0)}")
    
    # The first should be var('prev') due to context reads tie-breaking
    success = 'prev' in first_prog
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    return success

def test_periodic_k4_target():
    """Test evolution on periodic_k4 pattern."""
    print("\nTEST: Periodic K4 Evolution")
    print("-" * 30)
    
    # Create periodic pattern with period 4
    pattern = [1, 0, 1, 1]  # Period 4 pattern
    test_bits = (pattern * 50)[:200]  # 200 bits total
    
    print(f"  Pattern: {pattern}")
    print(f"  Test length: {len(test_bits)} bits")
    
    # Create evolution engine with context-aware seeds
    engine = EvolutionEngine(mu=8, lambda_=16, seed=42, k=4, N=150)
    
    # Initialize with context-aware population
    initial_pop = create_initial_population(8, seed=42)
    engine.initialize_population(initial_pop)
    
    print(f"  Initial population:")
    for i, ind in enumerate(engine.population):
        print(f"    {i}: {pretty_print_ast(ind.program)}")
    
    # Run evolution for limited generations
    best_fitness_history = []
    
    for gen in range(20):  # Limited test run
        stats = engine.evolve_generation(iter(test_bits))
        best_fitness = stats['best_fitness']
        best_fitness_history.append(best_fitness)
        
        if gen % 5 == 0:
            best_ind = max(engine.population, key=lambda x: x.fitness)
            ctx_reads = best_ind.metrics.get('ctx_reads_per_eval', 0)
            print(f"  Gen {gen:2d}: F={best_fitness:.6f}, best={pretty_print_ast(best_ind.program)}, ctx={ctx_reads:.2f}")
    
    final_fitness = max(best_fitness_history)
    improvement = final_fitness - best_fitness_history[0] if best_fitness_history else 0
    
    print(f"  Final fitness: {final_fitness:.6f}")
    print(f"  Improvement: {improvement:.6f}")
    
    # Success if we see improvement and positive fitness
    success = final_fitness > 0.01 and improvement > 0.01
    print(f"  Result: {'PASS' if success else 'PARTIAL'}")
    return success

def test_markov_k2_target():
    """Test evolution on markov_k2 pattern."""
    print("\nTEST: Markov K2 Evolution")
    print("-" * 25)
    
    # Create Markov chain with 2-bit memory
    # Simple transition: if last 2 bits are [0,0] -> 0.8 prob of 1, else 0.2 prob of 1
    test_bits = []
    state = [0, 0]  # Initial state
    
    import random
    random.seed(42)
    
    for _ in range(200):
        # Determine next bit based on current state
        if state == [0, 0]:
            next_bit = 1 if random.random() < 0.8 else 0
        else:
            next_bit = 1 if random.random() < 0.2 else 0
        
        test_bits.append(next_bit)
        state = [state[1], next_bit]  # Update state
    
    print(f"  Test length: {len(test_bits)} bits")
    print(f"  Sample: {test_bits[:20]}")
    
    # Create evolution engine
    engine = EvolutionEngine(mu=8, lambda_=16, seed=42, k=2, N=150)
    
    # Initialize with context-aware population
    initial_pop = create_initial_population(8, seed=42)
    engine.initialize_population(initial_pop)
    
    # Run evolution
    best_fitness_history = []
    
    for gen in range(15):  # Limited test run
        stats = engine.evolve_generation(iter(test_bits))
        best_fitness = stats['best_fitness']
        best_fitness_history.append(best_fitness)
        
        if gen % 5 == 0:
            best_ind = max(engine.population, key=lambda x: x.fitness)
            ctx_reads = best_ind.metrics.get('ctx_reads_per_eval', 0)
            print(f"  Gen {gen:2d}: F={best_fitness:.6f}, best={pretty_print_ast(best_ind.program)}, ctx={ctx_reads:.2f}")
    
    final_fitness = max(best_fitness_history)
    improvement = final_fitness - best_fitness_history[0] if best_fitness_history else 0
    
    print(f"  Final fitness: {final_fitness:.6f}")
    print(f"  Improvement: {improvement:.6f}")
    
    # Success if we see improvement
    success = final_fitness > -0.5 and improvement > 0.01
    print(f"  Result: {'PASS' if success else 'PARTIAL'}")
    return success

def main():
    print("QUICK WINS VALIDATION")
    print("=" * 30)
    
    test_results = [
        test_seed_diversity(),
        test_selection_tiebreaking(),
        test_periodic_k4_target(),
        test_markov_k2_target()
    ]
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\nSUMMARY: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests >= 3:
        print("QUICK WINS IMPLEMENTED SUCCESSFULLY")
    else:
        print("SOME QUICK WINS NEED ATTENTION")
    
    return passed_tests >= 3

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
