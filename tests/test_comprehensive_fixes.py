#!/usr/bin/env python3
"""Comprehensive test of all ProtoSynth fixes."""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.eval import evaluate_program_on_window
from protosynth.envs import periodic
import itertools

def test_context_access():
    """Test that context access tracking works correctly."""
    print("TEST 1: Context Access Tracking")
    print("-" * 35)
    
    test_bits = list(itertools.islice(periodic([1, 0], seed=42), 50))
    interpreter = LispInterpreter()
    
    # Test programs with expected context reads
    test_cases = [
        ('const(0.5)', const(0.5), 0),  # No context access
        ('prev', var('prev'), 1),       # 1 context read per prediction
        ('prev2', var('prev2'), 1),     # 1 context read per prediction
    ]
    
    all_passed = True
    for name, program, expected_reads_per_pred in test_cases:
        fitness, metrics = evaluate_program_on_window(interpreter, program, test_bits, k=2)
        
        ctx_reads = metrics.get('ctx_reads', 0)
        num_predictions = metrics.get('num_predictions', 0)
        reads_per_pred = ctx_reads / num_predictions if num_predictions > 0 else 0
        
        expected_total = expected_reads_per_pred * num_predictions
        tolerance = 0.1  # Allow some tolerance for edge cases
        
        passed = abs(reads_per_pred - expected_reads_per_pred) <= tolerance
        
        print(f"  {name:12s}: {ctx_reads:3d} reads / {num_predictions:2d} preds = {reads_per_pred:.2f} per pred (expected {expected_reads_per_pred}) - {'PASS' if passed else 'FAIL'}")
        
        if not passed:
            all_passed = False
    
    print(f"  Overall: {'PASS' if all_passed else 'FAIL'}")
    return all_passed

def test_penalty_system():
    """Test that the penalty system works correctly."""
    print("\nTEST 2: Penalty System")
    print("-" * 25)
    
    test_bits = list(itertools.islice(periodic([1, 0], seed=42), 50))
    interpreter = LispInterpreter()
    
    # Test programs with expected penalty behavior
    test_cases = [
        ('const(0.5)', const(0.5), 0.0),    # Should work without penalty
        ('const(0)', const(0), 0.0),        # Should work without penalty
        ('prev', var('prev'), 0.0),         # Should work with k=2
        ('prev2', var('prev2'), 0.0),       # Should work with k=2
    ]
    
    all_passed = True
    for name, program, expected_penalty in test_cases:
        fitness, metrics = evaluate_program_on_window(interpreter, program, test_bits, k=2)
        
        penalty = metrics.get('penalty_bits', 0)
        passed = penalty == expected_penalty
        
        print(f"  {name:12s}: penalty={penalty:.1f} (expected {expected_penalty}) - {'PASS' if passed else 'FAIL'}")
        
        if not passed:
            all_passed = False
    
    print(f"  Overall: {'PASS' if all_passed else 'FAIL'}")
    return all_passed

def test_fitness_differences():
    """Test that real fitness differences exist."""
    print("\nTEST 3: Fitness Differences")
    print("-" * 30)
    
    test_bits = list(itertools.islice(periodic([1, 0], seed=42), 50))
    interpreter = LispInterpreter()
    
    # Test programs with expected fitness relationships
    programs = [
        ('const(0.5)', const(0.5)),     # Baseline
        ('const(0)', const(0)),         # Very bad
        ('prev2', var('prev2')),        # Should be excellent on alternating
    ]
    
    results = {}
    for name, program in programs:
        fitness, metrics = evaluate_program_on_window(interpreter, program, test_bits, k=2)
        results[name] = fitness
        print(f"  {name:12s}: F={fitness:.6f}")
    
    # Check expected relationships
    checks = [
        ('prev2 > const(0.5)', results['prev2'] > results['const(0.5)'] + 0.5),
        ('const(0.5) > const(0)', results['const(0.5)'] > results['const(0)'] + 5.0),
        ('prev2 > const(0)', results['prev2'] > results['const(0)'] + 5.0),
    ]
    
    all_passed = True
    for check_name, passed in checks:
        print(f"  {check_name:20s}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_passed = False
    
    print(f"  Overall: {'PASS' if all_passed else 'FAIL'}")
    return all_passed

def test_evolution_readiness():
    """Test that the system is ready for evolution."""
    print("\nTEST 4: Evolution Readiness")
    print("-" * 30)
    
    from protosynth.evolve import create_initial_population, EvolutionEngine
    
    # Test that context-aware programs are in initial population
    population = create_initial_population(8, seed=42)
    
    context_programs = 0
    for prog in population:
        prog_str = pretty_print_ast(prog)
        if 'prev' in prog_str:
            context_programs += 1
    
    print(f"  Context programs in initial pop: {context_programs}/8")
    
    # Test that evolution engine can evaluate programs
    engine = EvolutionEngine(mu=4, lambda_=8, seed=42, k=2, N=40)
    engine.initialize_population(population[:4])
    
    test_bits = list(itertools.islice(periodic([1, 0], seed=42), 50))
    
    try:
        stats = engine.evolve_generation(iter(test_bits))
        best_fitness = stats['best_fitness']
        print(f"  Evolution test: F={best_fitness:.6f} - {'PASS' if best_fitness > -10 else 'FAIL'}")
        evolution_works = True
    except Exception as e:
        print(f"  Evolution test: ERROR - {e}")
        evolution_works = False
    
    readiness_passed = context_programs >= 3 and evolution_works
    print(f"  Overall: {'PASS' if readiness_passed else 'FAIL'}")
    return readiness_passed

def main():
    print("COMPREHENSIVE FIXES VALIDATION")
    print("=" * 40)
    
    test_results = [
        test_context_access(),
        test_penalty_system(), 
        test_fitness_differences(),
        test_evolution_readiness()
    ]
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\nSUMMARY: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("ALL FIXES WORKING - READY FOR EVOLUTION TESTING")
    else:
        print("SOME FIXES NEED ATTENTION")
    
    return passed_tests == total_tests

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
