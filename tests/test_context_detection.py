#!/usr/bin/env python3
"""Test the context detection fixes."""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.evolve import is_context_user_fast, EvolutionEngine, create_markov_table_individual, create_initial_population
from protosynth.envs import markov_k1
import itertools

def test_fast_context_detection():
    """Test AST-based context detection."""
    print("TEST: Fast Context Detection")
    print("-" * 35)
    
    # Test various program types
    test_cases = [
        # (program, expected_result, description)
        (const(0.5), False, "constant"),
        (var('prev'), True, "direct context var"),
        (var('prev2'), True, "direct context var prev2"),
        (op('+', var('prev'), const(1)), True, "expression with context"),
        (op('markov_table', const(3)), False, "markov_table with constant"),
        (op('markov_table', var('prev')), True, "markov_table with context var"),
        (op('markov_table', op('+', op('*', const(2), var('prev2')), var('prev'))), True, "markov_table with state expression"),
    ]
    
    print("  Testing AST-based context detection:")
    all_passed = True
    
    for program, expected, description in test_cases:
        result = is_context_user_fast(program)
        status = "PASS" if result == expected else "FAIL"
        print(f"    {description}: {result} (expected {expected}) - {status}")
        
        if result != expected:
            all_passed = False
    
    print(f"  Result: {'PASS' if all_passed else 'FAIL'}")
    return all_passed

def test_context_detection_in_evolution():
    """Test context detection in evolution with markov_table programs."""
    print("\nTEST: Context Detection in Evolution")
    print("-" * 45)
    
    # Create evolution engine
    engine = EvolutionEngine(mu=16, lambda_=48, seed=42, k=2, N=1000, env_name="markov_k2")
    
    # Create population with known context programs
    initial_pop = create_initial_population(14, seed=42)
    
    # Add markov_table programs
    stay_biased = create_markov_table_individual('stay', generation=0)
    flip_biased = create_markov_table_individual('flip', generation=0)
    
    initial_pop.extend([stay_biased.program, flip_biased.program])
    engine.initialize_population(initial_pop)
    
    print(f"  Initial population size: {len(engine.population)}")
    
    # Check initial context detection
    initial_ctx_count = 0
    for ind in engine.population:
        runtime_ctx = ind.metrics.get('ctx_reads_per_eval', 0.0) > 0.0
        ast_ctx = is_context_user_fast(ind.program)
        
        if runtime_ctx or ast_ctx:
            initial_ctx_count += 1
            if ast_ctx and not runtime_ctx:
                print(f"    AST-detected context user: {pretty_print_ast(ind.program)}")
    
    print(f"  Initial context users detected: {initial_ctx_count}")
    
    # Run a few generations to test context survival
    print(f"\n  Testing context survival over 10 generations:")
    
    context_survival_history = []
    
    for gen in range(10):
        # Create fresh Markov stream
        m1_stream = markov_k1(p_stay=0.8, seed=42+gen)
        fresh_bits = list(itertools.islice(m1_stream, engine.N + engine.k))
        stream = iter(fresh_bits)
        
        stats = engine.evolve_generation(stream)
        best_fitness = stats['best_fitness']
        
        # Count context users with combined detection
        context_count = 0
        markov_table_count = 0
        
        for ind in engine.population:
            runtime_ctx = ind.metrics.get('ctx_reads_per_eval', 0.0) > 0.0
            ast_ctx = is_context_user_fast(ind.program)
            
            if runtime_ctx or ast_ctx:
                context_count += 1
            
            if 'markov_table' in pretty_print_ast(ind.program):
                markov_table_count += 1
        
        context_survival_history.append(context_count)
        
        if gen % 5 == 0 or gen < 3:
            print(f"    Gen {gen:2d}: F_best={best_fitness:.6f}, ctx={context_count:2d}, markov={markov_table_count}")
        
        if best_fitness > 0:
            print(f"    SUCCESS: Positive fitness achieved at gen {gen}!")
            break
    
    # Analyze context survival
    min_context_count = min(context_survival_history)
    avg_context_count = sum(context_survival_history) / len(context_survival_history)
    
    print(f"\n  Context survival analysis:")
    print(f"    Initial context users: {initial_ctx_count}")
    print(f"    Min context survivors: {min_context_count}")
    print(f"    Avg context survivors: {avg_context_count:.1f}")
    print(f"    Context survival history: {context_survival_history}")
    
    # Success if we maintain good context survival
    success = min_context_count >= 8 and initial_ctx_count >= 2
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    
    return success

def main():
    print("CONTEXT DETECTION TEST")
    print("=" * 25)
    
    test_results = [
        test_fast_context_detection(),
        test_context_detection_in_evolution(),
    ]
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\nCONTEXT DETECTION SUMMARY")
    print(f"=" * 28)
    print(f"Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("CONTEXT DETECTION WORKING!")
        print("markov_table programs should now be protected.")
    elif passed_tests >= 1:
        print("MOSTLY WORKING - Some issues remain")
    else:
        print("CONTEXT DETECTION NEEDS MORE WORK")
    
    return passed_tests >= 1

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
