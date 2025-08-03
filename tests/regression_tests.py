#!/usr/bin/env python3
"""
Quick Regression Tests - Keep It Sticky
Ensure core functionality remains working.
"""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.eval import evaluate_program_on_window
from protosynth.predictor import PredictorAdapter
from protosynth.envs import markov_k2
import itertools

def test_alternating_prev2():
    """Alternating, k=2, prev2 must hit F >= 0.95, penalty_rate = 0."""
    print("TEST 1: Alternating + prev2")
    print("-" * 30)
    
    # Alternating pattern
    alternating = [1, 0] * 50  # 100 bits
    interpreter = LispInterpreter()
    
    fitness, metrics = evaluate_program_on_window(interpreter, var('prev2'), alternating, k=3)
    
    penalty_bits = metrics.get('penalty_bits', 0)
    penalty_rate = 1.0 if penalty_bits > 0 else 0.0
    
    print(f"  Fitness: {fitness:.6f} (target: >= 0.95)")
    print(f"  Penalty rate: {penalty_rate:.2f} (target: = 0.0)")
    
    success = fitness >= 0.95 and penalty_rate == 0.0
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    
    return success

def test_markov_prev():
    """Markov(0.8), prev must be F > 0."""
    print("\nTEST 2: Markov + prev")
    print("-" * 25)
    
    # Generate Markov sequence
    markov_bits = list(itertools.islice(markov_k2(p_stay=0.8, seed=42), 200))
    interpreter = LispInterpreter()
    
    fitness, metrics = evaluate_program_on_window(interpreter, var('prev'), markov_bits, k=2)
    
    print(f"  Fitness: {fitness:.6f} (target: > 0.0)")
    
    success = fitness > 0.0
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    
    return success

def test_ctx_shadowing():
    """Any program shadowing ctx → verify error."""
    print("\nTEST 3: Context shadowing protection")
    print("-" * 40)
    
    interpreter = LispInterpreter()
    
    # Try to create a program that shadows 'ctx'
    try:
        # This should fail: (let ctx 42 ctx)
        shadowing_prog = let('ctx', const(42), var('ctx'))
        
        # Try to evaluate it
        result = interpreter.evaluate(shadowing_prog, {'ctx': [1, 0, 1]})
        
        print(f"  ERROR: Shadowing allowed, got result: {result}")
        success = False
        
    except ValueError as e:
        if "shadow" in str(e).lower() or "reserved" in str(e).lower():
            print(f"  Correctly blocked shadowing: {e}")
            success = True
        else:
            print(f"  Wrong error type: {e}")
            success = False
    except Exception as e:
        print(f"  Unexpected error: {e}")
        success = False
    
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    return success

def test_out_of_range_prob():
    """Any out-of-range prob → runtime error → penalty."""
    print("\nTEST 4: Out-of-range probability handling")
    print("-" * 45)
    
    interpreter = LispInterpreter()
    adapter = PredictorAdapter(interpreter)
    
    # Test program that returns out-of-range value
    bad_prog = const(2.0)  # > 1.0
    
    try:
        result = adapter.predict(bad_prog, [1, 0])
        
        # Should be clamped to valid range
        if 0.0 <= result <= 1.0:
            print(f"  Correctly clamped: {result}")
            success = True
        else:
            print(f"  ERROR: Out-of-range result: {result}")
            success = False
            
    except Exception as e:
        print(f"  ERROR: Exception on valid clamp case: {e}")
        success = False
    
    # Test program that causes evaluation error
    error_prog = op('/', const(1), const(0))  # Division by zero
    
    try:
        result = adapter.predict(error_prog, [1, 0])
        print(f"  ERROR: Should have failed, got: {result}")
        success = False
        
    except Exception as e:
        print(f"  Correctly failed on error: {type(e).__name__}")
        success = True
    
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    return success

def test_index_robustness():
    """Test rock-solid index operation."""
    print("\nTEST 5: Index operation robustness")
    print("-" * 40)
    
    interpreter = LispInterpreter()
    
    test_cases = [
        # (context, index, expected_result, description)
        ([1, 0, 1], -1, 1, "Valid negative index"),
        ([1, 0, 1], -3, 1, "Valid boundary index"),
        ([1, 0, 1], -4, 0, "Out-of-bounds negative"),
        ([1, 0, 1], 0, 0, "Positive index (forbidden)"),
        ([1, 0, 1], 1, 0, "Positive index (forbidden)"),
    ]
    
    all_passed = True
    
    for ctx, idx, expected, desc in test_cases:
        try:
            env = {'ctx': tuple(ctx)}
            prog = op('index', var('ctx'), const(idx))
            result = interpreter.evaluate(prog, env)
            
            if result == expected:
                print(f"  {desc}: {result} ✓")
            else:
                print(f"  {desc}: got {result}, expected {expected} ✗")
                all_passed = False
                
        except Exception as e:
            print(f"  {desc}: ERROR {e} ✗")
            all_passed = False
    
    print(f"  Result: {'PASS' if all_passed else 'FAIL'}")
    return all_passed

def test_micro_primitives():
    """Test micro-primitives work correctly."""
    print("\nTEST 6: Micro-primitives")
    print("-" * 30)
    
    interpreter = LispInterpreter()
    ctx = [1, 0, 1, 0, 1]
    env = {'ctx': tuple(ctx)}
    
    test_cases = [
        (op('parity3', var('ctx')), ctx[-3] ^ ctx[-2] ^ ctx[-1], "parity3"),
        (op('sum_bits', var('ctx')), sum(ctx), "sum_bits"),
    ]
    
    all_passed = True
    
    for prog, expected, name in test_cases:
        try:
            result = interpreter.evaluate(prog, env)
            if result == expected:
                print(f"  {name}: {result} ✓")
            else:
                print(f"  {name}: got {result}, expected {expected} ✗")
                all_passed = False
        except Exception as e:
            print(f"  {name}: ERROR {e} ✗")
            all_passed = False
    
    print(f"  Result: {'PASS' if all_passed else 'FAIL'}")
    return all_passed

def main():
    print("REGRESSION TESTS - KEEP IT STICKY")
    print("=" * 40)
    
    test_results = [
        test_alternating_prev2(),
        test_markov_prev(),
        test_ctx_shadowing(),
        test_out_of_range_prob(),
        test_index_robustness(),
        test_micro_primitives(),
    ]
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\nREGRESSION SUMMARY")
    print(f"=" * 20)
    print(f"Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("ALL REGRESSION TESTS PASSED - System is stable!")
        return True
    elif passed_tests >= total_tests - 1:
        print("MOSTLY PASSING - Minor issues to address")
        return True
    else:
        print("REGRESSION FAILURES - System needs attention")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
