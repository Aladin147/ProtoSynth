#!/usr/bin/env python3
"""Debug the infinite loop in the interpreter."""

import sys
sys.path.append('.')

from protosynth import *

def debug_interpreter_loop():
    """Debug what's causing the infinite loop."""
    print("DEBUG: Interpreter Infinite Loop")
    print("-" * 35)
    
    # Create interpreter with MLE parameters
    interpreter = LispInterpreter(max_steps=100, timeout_seconds=5.0)  # Low limit for debugging
    
    # Set MLE parameters
    mle_params = {
        'p00': 0.7983784802966502,
        'p01': 0.20770186335403729, 
        'p10': 0.7967196819085487,
        'p11': 0.1989394884591391
    }
    interpreter.markov_params = mle_params
    
    print(f"Set interpreter params: {mle_params}")
    
    # Test simple expressions first
    print(f"\n1) Testing simple expressions:")
    
    simple_tests = [
        const(0.5),
        var('prev'),
        var('prev2'),
        op('+', const(1), const(2)),
        op('*', const(2), var('prev2')),
        op('+', op('*', const(2), var('prev2')), var('prev')),
    ]
    
    env = {'prev': 0, 'prev2': 0}
    
    for i, expr in enumerate(simple_tests):
        try:
            interpreter.reset_tracker()
            result = interpreter.evaluate(expr, env)
            steps = interpreter.step_count
            print(f"  {i}: {pretty_print_ast(expr)} = {result} ({steps} steps)")
        except Exception as e:
            print(f"  {i}: {pretty_print_ast(expr)} = ERROR: {e}")
    
    # Test markov_table function directly
    print(f"\n2) Testing markov_table function directly:")
    
    for state_idx in range(4):
        try:
            interpreter.reset_tracker()
            result = interpreter._markov_table(state_idx)
            steps = interpreter.step_count
            print(f"  markov_table({state_idx}) = {result} ({steps} steps)")
        except Exception as e:
            print(f"  markov_table({state_idx}) = ERROR: {e}")
    
    # Test the problematic expression
    print(f"\n3) Testing problematic markov_table expression:")
    
    # Build the expression step by step
    state_expr = op('+', op('*', const(2), var('prev2')), var('prev'))
    markov_expr = op('markov_table', state_expr)
    
    print(f"State expression: {pretty_print_ast(state_expr)}")
    print(f"Markov expression: {pretty_print_ast(markov_expr)}")
    
    # Test state expression first
    try:
        interpreter.reset_tracker()
        state_result = interpreter.evaluate(state_expr, env)
        steps = interpreter.step_count
        print(f"State expression result: {state_result} ({steps} steps)")
    except Exception as e:
        print(f"State expression ERROR: {e}")
        return
    
    # Test full markov expression
    try:
        interpreter.reset_tracker()
        markov_result = interpreter.evaluate(markov_expr, env)
        steps = interpreter.step_count
        print(f"Markov expression result: {markov_result} ({steps} steps)")
    except Exception as e:
        print(f"Markov expression ERROR: {e}")
        print(f"Steps before error: {interpreter.step_count}")
        
        # Check if it's a recursion issue
        if "Maximum step count exceeded" in str(e):
            print("INFINITE LOOP DETECTED!")
            print("The markov_table evaluation is stuck in a loop.")
            
            # Check the call stack or evaluation trace
            if hasattr(interpreter, 'call_stack'):
                print(f"Call stack: {interpreter.call_stack}")
    
    return True

def main():
    debug_interpreter_loop()
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
