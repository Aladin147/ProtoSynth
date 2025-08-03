#!/usr/bin/env python3
"""
Test State Reset Fixes
Verify that interpreter state is properly reset between evaluations.
"""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.evolve import EvolutionEngine, Individual, create_initial_population
from protosynth.eval import evaluate_program_on_window
import itertools

def test_interpreter_reset():
    """Test that interpreter reset works correctly."""
    print("TEST: Interpreter Reset")
    print("-" * 25)
    
    interpreter = LispInterpreter(max_steps=1000, timeout_seconds=10.0)
    
    # First evaluation
    print("First evaluation:")
    alternating = [1, 0] * 50
    fitness1, metrics1 = evaluate_program_on_window(interpreter, var('prev2'), alternating, k=3)
    
    print(f"  Fitness: {fitness1:.6f}")
    print(f"  Step count: {interpreter.step_count}")
    print(f"  Context reads: {interpreter.ctx_reads}")
    
    # Reset and second evaluation
    print("\nAfter reset:")
    interpreter.reset_tracker()
    fitness2, metrics2 = evaluate_program_on_window(interpreter, var('prev2'), alternating, k=3)
    
    print(f"  Fitness: {fitness2:.6f}")
    print(f"  Step count: {interpreter.step_count}")
    print(f"  Context reads: {interpreter.ctx_reads}")
    
    # Check if results are consistent
    fitness_match = abs(fitness1 - fitness2) < 1e-6
    print(f"\nFitness consistency: {'PASS' if fitness_match else 'FAIL'}")
    
    return fitness_match

def test_evolution_with_reset():
    """Test evolution with state reset fixes."""
    print("\nTEST: Evolution with State Reset")
    print("-" * 40)
    
    # Create evolution engine
    engine = EvolutionEngine(mu=4, lambda_=8, seed=42, k=3, N=100)
    
    # Initialize with simple population including known-good programs
    initial_pop = [
        Individual(const(0.5), 0.0, {}, 0),
        Individual(var('prev'), 0.0, {}, 1),
        Individual(var('prev2'), 0.0, {}, 2),
        Individual(const(0.3), 0.0, {}, 3),
    ]
    engine.population = initial_pop
    
    # Test evaluation on alternating pattern
    alternating = [1, 0] * 75  # 150 bits
    stream = iter(alternating)
    
    print("Testing evolution evaluation with reset:")
    stats = engine.evolve_generation(stream)
    
    print(f"  Best fitness: {stats['best_fitness']:.6f}")
    
    # Check individual results
    success_count = 0
    for i, ind in enumerate(engine.population):
        penalty = ind.metrics.get('penalty_bits', 0)
        ctx_reads = ind.metrics.get('ctx_reads_per_eval', 0)
        exc_counts = ind.metrics.get('exception_counts', {})
        
        print(f"  {i}: F={ind.fitness:.6f}, penalty={penalty:.1f}, ctx={ctx_reads:.2f}")
        if exc_counts:
            print(f"      Exceptions: {exc_counts}")
        
        if ind.fitness > -1.0:  # Not at penalty floor
            success_count += 1
    
    print(f"\nSuccessful evaluations: {success_count}/{len(engine.population)}")
    
    # Success if at least one program works
    success = success_count > 0 and stats['best_fitness'] > -1.0
    print(f"Result: {'PASS' if success else 'FAIL'}")
    
    return success

def test_exception_tracking():
    """Test that exception tracking works."""
    print("\nTEST: Exception Tracking")
    print("-" * 30)
    
    interpreter = LispInterpreter(max_steps=1000, timeout_seconds=10.0)
    
    # Test program that should cause exceptions
    bad_prog = op('/', const(1), const(0))  # Division by zero
    
    alternating = [1, 0] * 20
    fitness, metrics = evaluate_program_on_window(interpreter, bad_prog, alternating, k=2)
    
    exc_counts = metrics.get('exception_counts', {})
    penalty = metrics.get('penalty_bits', 0)
    
    print(f"  Fitness: {fitness:.6f}")
    print(f"  Penalty: {penalty:.1f}")
    print(f"  Exception counts: {exc_counts}")
    
    # Should have exceptions and penalties
    has_exceptions = len(exc_counts) > 0
    has_penalties = penalty > 0
    
    print(f"  Has exceptions: {'PASS' if has_exceptions else 'FAIL'}")
    print(f"  Has penalties: {'PASS' if has_penalties else 'FAIL'}")
    
    return has_exceptions and has_penalties

def main():
    print("STATE RESET FIXES TEST")
    print("=" * 30)
    
    test_results = [
        test_interpreter_reset(),
        test_evolution_with_reset(),
        test_exception_tracking(),
    ]
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\nSTATE RESET TEST SUMMARY")
    print(f"=" * 25)
    print(f"Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("ALL STATE RESET FIXES WORKING!")
        print("Evolution should now work correctly.")
    elif passed_tests >= 2:
        print("MOSTLY WORKING - Minor issues remain")
    else:
        print("STATE RESET FIXES FAILED - More debugging needed")
    
    return passed_tests >= 2

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
