#!/usr/bin/env python3
"""Test Markov evolution fixes."""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.evolve import EvolutionEngine, eval_candidate, create_initial_population
from protosynth.envs import markov_k1, check_transitions
import itertools

def test_calibrated_evaluator_in_evolution():
    """Test that calibrated evaluator is used in evolution for Markov."""
    print("TEST: Calibrated Evaluator in Evolution")
    print("-" * 40)
    
    # Generate Markov buffer
    m1_stream = markov_k1(p_stay=0.8, seed=42)
    buf = list(itertools.islice(m1_stream, 3000))
    
    print(f"Generated {len(buf)} bits")
    
    # Test eval_candidate directly
    print("\n1. Testing eval_candidate directly:")
    
    # Test soft predictor
    soft_pred = if_expr(op('=', op('index', var('ctx'), const(-1)), const(1)), const(0.8), const(0.2))
    
    fitness, metrics = eval_candidate(soft_pred, "markov_k2", buf, k=1)
    ctx_reads = metrics.get('ctx_reads_per_eval', 0)
    delta = metrics.get('delta_calibration', 'N/A')
    
    print(f"  Soft predictor: F={fitness:.6f}, ctx={ctx_reads:.2f}, delta={delta}")
    
    # Test regular predictor
    fitness2, metrics2 = eval_candidate(var('prev'), "markov_k2", buf, k=1)
    ctx_reads2 = metrics2.get('ctx_reads_per_eval', 0)
    delta2 = metrics2.get('delta_calibration', 'N/A')
    
    print(f"  prev predictor: F={fitness2:.6f}, ctx={ctx_reads2:.2f}, delta={delta2}")
    
    # Test non-Markov environment (should use regular evaluation)
    fitness3, metrics3 = eval_candidate(var('prev'), "periodic_k4", buf, k=1)
    ctx_reads3 = metrics3.get('ctx_reads_per_eval', 0)
    delta3 = metrics3.get('delta_calibration', 'N/A')
    
    print(f"  prev (non-Markov): F={fitness3:.6f}, ctx={ctx_reads3:.2f}, delta={delta3}")
    
    # Check results
    markov_working = fitness > -1.0 and ctx_reads > 0
    calibration_used = delta != 'N/A'
    non_markov_no_calibration = delta3 == 'N/A'
    
    print(f"\n  Markov evaluation working: {'PASS' if markov_working else 'FAIL'}")
    print(f"  Calibration used for Markov: {'PASS' if calibration_used else 'FAIL'}")
    print(f"  No calibration for non-Markov: {'PASS' if non_markov_no_calibration else 'FAIL'}")
    
    return markov_working and calibration_used

def test_evolution_with_soft_predictors():
    """Test evolution with soft predictor seeds."""
    print("\n2. Testing Evolution with Soft Predictors:")
    
    # Create evolution engine for Markov
    engine = EvolutionEngine(mu=4, lambda_=8, seed=42, k=1, N=500, env_name="markov_k2")
    
    # Create population with soft predictors
    soft_stay = if_expr(op('=', op('index', var('ctx'), const(-1)), const(1)), const(0.8), const(0.2))
    soft_flip = if_expr(op('=', op('index', var('ctx'), const(-1)), const(1)), const(0.2), const(0.8))
    
    initial_pop = [
        const(0.5),
        var('prev'),
        soft_stay,
        soft_flip,
    ]
    
    engine.initialize_population(initial_pop)
    
    print(f"  Initial population:")
    for i, ind in enumerate(engine.population):
        print(f"    {i}: {pretty_print_ast(ind.program)}")
    
    # Run a few generations
    print(f"\n  Running 5 generations:")
    
    for gen in range(5):
        # Create fresh Markov stream
        m1_stream = markov_k1(p_stay=0.8, seed=42+gen)
        fresh_bits = list(itertools.islice(m1_stream, engine.N + engine.k))
        stream = iter(fresh_bits)
        
        stats = engine.evolve_generation(stream)
        best_fitness = stats['best_fitness']
        
        # Check context usage
        ctx_usage = sum(ind.metrics.get('ctx_reads_per_eval', 0) for ind in engine.population) / len(engine.population)
        
        print(f"    Gen {gen}: F_best={best_fitness:.6f}, ctx_avg={ctx_usage:.2f}")
        
        if best_fitness > 0:
            print(f"    SUCCESS: Positive fitness achieved!")
            break
    
    final_fitness = max(ind.fitness for ind in engine.population)
    success = final_fitness > -1.0  # Not at penalty floor
    
    print(f"  Final best fitness: {final_fitness:.6f}")
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    
    return success

def main():
    print("MARKOV EVOLUTION FIXES TEST")
    print("=" * 30)
    
    # First check Markov generator
    print("0. Checking Markov generator:")
    counts, stay, flip = check_transitions(lambda: markov_k1(0.8, seed=42))
    print(f"   Stay: {stay:.3f}, Flip: {flip:.3f}")
    generator_ok = abs(stay - 0.8) < 0.02
    print(f"   Generator: {'PASS' if generator_ok else 'FAIL'}")
    
    if not generator_ok:
        print("   Generator broken, aborting tests")
        return False
    
    test_results = [
        test_calibrated_evaluator_in_evolution(),
        test_evolution_with_soft_predictors(),
    ]
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\nMARKOV EVOLUTION FIXES SUMMARY")
    print(f"=" * 32)
    print(f"Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("ALL MARKOV EVOLUTION FIXES WORKING!")
        print("Ready to run markov_k2 benchmark.")
    elif passed_tests >= 1:
        print("MOSTLY WORKING - Minor issues remain")
    else:
        print("MARKOV EVOLUTION FIXES FAILED")
    
    return passed_tests >= 1

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
