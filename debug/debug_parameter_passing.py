#!/usr/bin/env python3
"""Debug parameter passing to markov_table during evaluation."""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.evolve import eval_candidate, EvolutionEngine
from protosynth.envs import markov_k1
import itertools

def debug_parameter_passing():
    """Debug how parameters are passed to markov_table during evaluation."""
    print("DEBUG: Parameter Passing to markov_table")
    print("-" * 45)
    
    # Generate buffer
    m1_stream = markov_k1(p_stay=0.8, seed=42)
    buf = list(itertools.islice(m1_stream, 1000))
    
    # Create markov_table program
    markov_prog = op('markov_table', op('+', op('*', const(2), var('prev2')), var('prev')))
    print(f"Program: {pretty_print_ast(markov_prog)}")
    
    # Method 1: Direct evaluation (what eval_candidate does)
    print(f"\n1) Direct eval_candidate:")
    F_direct, metrics = eval_candidate(markov_prog, "markov_k2", buf, 2)
    print(f"   F_direct: {F_direct:.6f}")
    print(f"   Metrics: {metrics}")
    
    # Method 2: Manual evaluation with explicit parameters
    print(f"\n2) Manual evaluation with explicit MLE parameters:")
    
    # Build MLE parameters from buffer
    k = 2
    state_counts = {(a, b): {'n': 0, 'c1': 0} for a in (0, 1) for b in (0, 1)}
    
    for i in range(k, len(buf)):
        ctx = buf[i-k:i]
        y = buf[i]
        
        if len(ctx) >= 2:
            s = (ctx[-2], ctx[-1])
            state_counts[s]['n'] += 1
            if y == 1:
                state_counts[s]['c1'] += 1
    
    # Compute MLE parameters
    mle_params = {}
    for s in state_counts:
        n_s = state_counts[s]['n']
        c1_s = state_counts[s]['c1']
        
        if n_s > 0:
            # MLE with Laplace: P(next=1|s) = (c1 + 1) / (n + 2)
            p1_mle = (c1_s + 1) / (n_s + 2)
            p0_mle = 1 - p1_mle
            
            param_key = f'p{s[0]}{s[1]}'
            mle_params[param_key] = p0_mle  # Store P(next=0|s)
    
    print(f"   MLE parameters: {mle_params}")
    
    # Create interpreter with explicit parameters
    interpreter = LispInterpreter(max_steps=1000, timeout_seconds=10.0)
    interpreter.markov_params = mle_params
    
    print(f"   Interpreter params: {interpreter.markov_params}")
    
    # Test a few predictions manually
    print(f"\n   Manual predictions:")
    for i in range(k, min(k + 5, len(buf))):
        ctx = buf[i-k:i]
        y_next = buf[i]
        
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
            
            # Check what markov_table returns for this state
            table_result = interpreter._markov_table(state_idx)
            
            print(f"     ctx={ctx}, state={state}, idx={state_idx}")
            print(f"     markov_table({state_idx}) = {table_result:.4f}")
            print(f"     program result = {result:.4f}")
            print(f"     y_next = {y_next}")
            
        except Exception as e:
            print(f"     ctx={ctx}, ERROR: {e}")
    
    # Method 3: Check what happens in eval_candidate
    print(f"\n3) Tracing eval_candidate internals:")
    
    # Create evolution engine to get MLE teacher
    engine = EvolutionEngine(mu=16, lambda_=48, seed=42, k=2, N=1000, env_name="markov_k2")
    
    # Build MLE teacher
    mle_teacher = engine.build_mle_markov_candidate([buf], k=2)
    print(f"   MLE teacher program: {pretty_print_ast(mle_teacher.program)}")
    print(f"   MLE teacher fitness: {mle_teacher.fitness:.6f}")
    
    # Check if the teacher has the right parameters
    if hasattr(mle_teacher, 'interpreter') and hasattr(mle_teacher.interpreter, 'markov_params'):
        print(f"   Teacher interpreter params: {mle_teacher.interpreter.markov_params}")
    else:
        print(f"   Teacher has no interpreter or markov_params")
    
    return True

def main():
    debug_parameter_passing()
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
