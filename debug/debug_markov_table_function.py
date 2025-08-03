#!/usr/bin/env python3
"""Debug the markov_table function to see what it's actually returning."""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.envs import markov_k1
import itertools

def debug_markov_table_function():
    """Debug what the markov_table function actually returns."""
    print("DEBUG: markov_table Function")
    print("-" * 35)
    
    # Create interpreter with markov_table
    interpreter = LispInterpreter(max_steps=1000, timeout_seconds=10.0)
    
    # Set MLE parameters (from previous debug)
    interpreter.markov_params = {
        'p00': 0.7991,  # P(next=0 | prev2=0, prev=0)
        'p01': 0.2054,  # P(next=0 | prev2=0, prev=1)
        'p10': 0.8072,  # P(next=0 | prev2=1, prev=0)
        'p11': 0.1863,  # P(next=0 | prev2=1, prev=1)
    }
    
    print("MLE parameters set:")
    for key, val in interpreter.markov_params.items():
        print(f"  {key}: {val:.4f}")
    
    # Test markov_table function directly
    print("\nTesting markov_table function directly:")
    for state_idx in range(4):
        prob = interpreter._markov_table(state_idx)
        state_bits = (state_idx // 2, state_idx % 2)  # (prev2, prev)
        print(f"  markov_table({state_idx}) for state {state_bits}: {prob:.4f}")
    
    # Test markov_table program
    print("\nTesting markov_table program:")
    markov_prog = op('markov_table', op('+', op('*', const(2), var('prev2')), var('prev')))
    print(f"  Program: {pretty_print_ast(markov_prog)}")
    
    # Generate test data
    m1_stream = markov_k1(p_stay=0.8, seed=42)
    test_bits = list(itertools.islice(m1_stream, 20))
    
    print(f"  Test data: {test_bits}")
    
    # Test program on first few contexts
    print("\n  Program outputs:")
    k = 2
    for i in range(k, min(k + 10, len(test_bits))):
        ctx = test_bits[i-k:i]
        y_next = test_bits[i]
        
        # Set up environment
        interpreter.reset_tracker()
        env = {
            'prev2': ctx[-2] if len(ctx) >= 2 else 0,
            'prev': ctx[-1] if len(ctx) >= 1 else 0,
        }
        
        try:
            result = interpreter.evaluate(markov_prog, env)
            state = (ctx[-2], ctx[-1])
            state_idx = 2 * ctx[-2] + ctx[-1]
            
            print(f"    ctx={ctx}, state={state}, idx={state_idx}, prog_out={result:.4f}, y_next={y_next}")
            
        except Exception as e:
            print(f"    ctx={ctx}, ERROR: {e}")
    
    # Test with PredictorAdapter
    print("\n  Testing with PredictorAdapter:")
    from protosynth.predictor import PredictorAdapter
    adapter = PredictorAdapter(interpreter)
    
    for i in range(k, min(k + 5, len(test_bits))):
        ctx = test_bits[i-k:i]
        y_next = test_bits[i]
        
        try:
            prob = adapter.predict(markov_prog, ctx)
            state = (ctx[-2], ctx[-1])
            print(f"    ctx={ctx}, state={state}, adapter_out={prob:.4f}, y_next={y_next}")
            
        except Exception as e:
            print(f"    ctx={ctx}, ADAPTER ERROR: {e}")
    
    return True

def main():
    debug_markov_table_function()
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
