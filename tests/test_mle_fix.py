#!/usr/bin/env python3
"""Test the MLE parameter fitting fix."""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.eval import evaluate_program_calibrated, _fit_mle_parameters
from protosynth.envs import markov_k1
import itertools

def test_mle_fix():
    """Test that MLE parameter fitting works in evaluate_program_calibrated."""
    print("TEST: MLE Parameter Fitting Fix")
    print("-" * 35)
    
    # Generate buffer
    m1_stream = markov_k1(p_stay=0.8, seed=42)
    buf = list(itertools.islice(m1_stream, 2000))
    
    # Create markov_table program
    markov_prog = op('markov_table', op('+', op('*', const(2), var('prev2')), var('prev')))
    print(f"Program: {pretty_print_ast(markov_prog)}")
    
    # Test 1: Direct MLE fitting
    print(f"\n1) Direct MLE fitting test:")
    interpreter = LispInterpreter(max_steps=1000, timeout_seconds=10.0)
    
    # Fit parameters manually
    _fit_mle_parameters(interpreter, buf, k=2, N_train=1000)
    print(f"   Fitted parameters: {interpreter.markov_params}")
    
    # Test a prediction
    interpreter.reset_tracker()
    env = {'prev2': 0, 'prev': 0}
    result = interpreter.evaluate(markov_prog, env)
    print(f"   Test prediction for state (0,0): {result:.4f}")
    
    # Test 2: evaluate_program_calibrated
    print(f"\n2) evaluate_program_calibrated test:")
    interpreter2 = LispInterpreter(max_steps=1000, timeout_seconds=10.0)
    
    fitness, metrics = evaluate_program_calibrated(
        interpreter2, markov_prog, buffer=buf, k=2, N_train=1000, N_val=1000
    )
    
    print(f"   Fitness: {fitness:.6f}")
    print(f"   Model entropy: {metrics.get('model_entropy', 'N/A'):.6f}")
    print(f"   Baseline entropy: {metrics.get('baseline_entropy', 'N/A'):.6f}")
    print(f"   Empirical 1-rate: {metrics.get('empirical_1_rate', 'N/A'):.4f}")
    
    # Check if parameters were set
    if hasattr(interpreter2, 'markov_params') and interpreter2.markov_params:
        print(f"   Parameters set: {interpreter2.markov_params}")
    else:
        print(f"   Parameters NOT set!")
    
    # Success if fitness is positive
    success = fitness > 0.0
    print(f"   Result: {'PASS' if success else 'FAIL'}")
    
    return success

def main():
    success = test_mle_fix()
    print(f"\nOverall result: {'PASS' if success else 'FAIL'}")
    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
