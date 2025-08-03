#!/usr/bin/env python3
"""Staged parity harness to pinpoint where +0.29 bits/sym is being destroyed."""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.envs import markov_k1
from protosynth.predictor import PredictorAdapter
import itertools
import math

def H2(p, eps=1e-9):
    """Binary entropy function."""
    p = max(eps, min(1-eps, p))
    return -(p * math.log2(p) + (1-p) * math.log2(1-p))

def F_star_from_buffers(buffers, k=2):
    """Ground truth F* from exact val buffers."""
    N = 0
    ones = 0
    cnt = {(0,0): [0,0], (0,1): [0,0], (1,0): [0,0], (1,1): [0,0]}  # [n_s, c1_s]
    
    for buf in buffers:
        for i in range(k, len(buf)):
            s = (buf[i-2], buf[i-1])
            y = buf[i]
            cnt[s][0] += 1
            cnt[s][1] += y
            N += 1
            ones += y
    
    H0 = H2(ones / max(1, N))
    Hprog = 0.0
    for n, c1 in cnt.values():
        if n:
            Hprog += (n / N) * H2(c1 / n)
    
    return H0 - Hprog, cnt, N

def F_eval_direct_table(buffers, k, p_table):
    """Stage 1: Eval without interpreter/adapter (direct table)."""
    total_loss = 0
    N = 0
    ones = 0
    
    for buf in buffers:
        for i in range(k, len(buf)):
            s = (buf[i-2], buf[i-1])
            y = buf[i]
            p = max(1e-9, min(1-1e-9, p_table[s]))
            total_loss += -(y * math.log2(p) + (1-y) * math.log2(1-p))
            ones += y
            N += 1
    
    H0 = H2(ones / N)
    return H0 - (total_loss / N)

def F_eval_adapter_raw(buffers, k, interp, ast):
    """Stage 2: Eval via adapter, no penalties, no calibration."""
    total_loss = 0
    N = 0
    ones = 0
    
    adapter = PredictorAdapter(interp)
    
    for buf in buffers:
        for i in range(k, len(buf)):
            ctx = buf[i-k:i]
            y = buf[i]
            
            # Direct adapter call - no penalties, no calibration
            try:
                # Reset interpreter before each prediction
                interp.reset_tracker()
                p_raw = adapter.predict(ast, ctx)
                p = max(1e-9, min(1-1e-9, p_raw))
                total_loss += -(y * math.log2(p) + (1-y) * math.log2(1-p))
                ones += y
                N += 1
            except Exception as e:
                print(f"    ADAPTER FAILED at step {i}: ctx={ctx}, error={e}")
                return float('-inf')
    
    H0 = H2(ones / N)
    return H0 - (total_loss / N)

def F_eval_with_penalties(buffers, k, interp, ast):
    """Stage 3: Add penalties back (replace CE, don't add)."""
    total_loss = 0
    N = 0
    ones = 0
    penalty_count = 0

    adapter = PredictorAdapter(interp)
    PENALTY_BITS = 1.5

    for buf in buffers:
        for i in range(k, len(buf)):
            ctx = buf[i-k:i]
            y = buf[i]

            try:
                # Reset interpreter before each prediction (same as Stage 2)
                interp.reset_tracker()
                p_raw = adapter.predict(ast, ctx)
                p = max(1e-9, min(1-1e-9, p_raw))
                loss = -(y * math.log2(p) + (1-y) * math.log2(1-p))
                ones += y
            except Exception as e:
                penalty_count += 1
                loss = PENALTY_BITS  # REPLACES CE, doesn't add to it
                # Don't count ones for failed predictions
                if penalty_count <= 5:  # Debug first few failures
                    print(f"    PENALTY {penalty_count}: ctx={ctx}, error={e}")

            total_loss += loss
            N += 1

    H0 = H2(ones / max(1, N - penalty_count))  # Baseline from successful predictions only
    return H0 - (total_loss / N), penalty_count

def staged_parity_test():
    """Run the complete staged parity test."""
    print("STAGED PARITY HARNESS")
    print("=" * 25)
    print("Systematically isolating where +0.29 bits/sym is destroyed")
    
    # Generate exact same buffers as benchmark
    print("\nGenerating validation buffers...")
    val_buffers = []
    for seed in [42, 43, 44, 45, 46]:  # 5 buffers like ensemble
        stream = markov_k1(p_stay=0.8, seed=seed)
        buf = list(itertools.islice(stream, 8000))
        val_buffers.append(buf)
    
    print(f"Generated {len(val_buffers)} validation buffers")
    
    # Stage 0: Ground truth
    print(f"\nSTAGE 0: Ground Truth")
    print(f"-" * 20)
    F_star, counts, N_total = F_star_from_buffers(val_buffers, k=2)
    print(f"F* (theoretical maximum): {F_star:.6f}")
    print(f"Total validation samples: {N_total}")
    
    # Build MLE parameters from counts
    p_table = {}
    for s, (n, c1) in counts.items():
        if n > 0:
            # MLE: P(next=1|s) = c1/n (no Laplace for direct comparison)
            p_table[s] = c1 / n
        else:
            p_table[s] = 0.5
    
    print(f"MLE parameters: {p_table}")
    
    # Stage 1: Direct table evaluation
    print(f"\nSTAGE 1: Direct Table Evaluation")
    print(f"-" * 35)
    F_direct = F_eval_direct_table(val_buffers, 2, p_table)
    print(f"F_direct: {F_direct:.6f}")
    print(f"Difference from F*: {F_direct - F_star:.6f}")
    
    stage1_pass = abs(F_direct - F_star) < 0.01
    print(f"Stage 1: {'PASS' if stage1_pass else 'FAIL'}")
    
    if not stage1_pass:
        print("ERROR: Direct table doesn't match F*. Val slicing differs!")
        return False
    
    # Stage 2: Adapter evaluation (no penalties, no calibration)
    print(f"\nSTAGE 2: Adapter Evaluation (Raw)")
    print(f"-" * 35)
    
    # Create interpreter with MLE parameters (increase step limit)
    interpreter = LispInterpreter(max_steps=10000, timeout_seconds=10.0)
    
    # Set MLE parameters (convert to P(next=0) format for compatibility)
    mle_params = {}
    for s, p1 in p_table.items():
        param_key = f'p{s[0]}{s[1]}'
        mle_params[param_key] = 1.0 - p1  # Store P(next=0)
    
    interpreter.markov_params = mle_params
    print(f"Set interpreter params: {mle_params}")
    
    # Create markov_table program
    markov_ast = op('markov_table', op('+', op('*', const(2), var('prev2')), var('prev')))
    
    F_adapter = F_eval_adapter_raw(val_buffers, 2, interpreter, markov_ast)
    print(f"F_adapter: {F_adapter:.6f}")
    print(f"Difference from Stage 1: {F_adapter - F_direct:.6f}")
    
    stage2_pass = abs(F_adapter - F_direct) < 0.01
    print(f"Stage 2: {'PASS' if stage2_pass else 'FAIL'}")
    
    if not stage2_pass:
        print("ERROR: Adapter differs from direct table. Context binding wrong!")
        # Debug first few predictions
        print("Debugging first 10 predictions:")
        adapter = PredictorAdapter(interpreter)
        for i, buf in enumerate(val_buffers[:1]):  # Just first buffer
            for j in range(2, min(12, len(buf))):
                ctx = buf[j-2:j]
                y = buf[j]
                s = (ctx[-2], ctx[-1])
                
                try:
                    p_adapter = adapter.predict(markov_ast, ctx)
                    p_direct = p_table[s]
                    print(f"  {i}.{j}: ctx={ctx}, s={s}, y={y}, p_direct={p_direct:.4f}, p_adapter={p_adapter:.4f}")
                except Exception as e:
                    print(f"  {i}.{j}: ctx={ctx}, s={s}, y={y}, ADAPTER ERROR: {e}")
        return False
    
    # Stage 3: Add penalties back
    print(f"\nSTAGE 3: Add Penalties (Replace CE)")
    print(f"-" * 35)
    
    F_penalties, penalty_count = F_eval_with_penalties(val_buffers, 2, interpreter, markov_ast)
    print(f"F_penalties: {F_penalties:.6f}")
    print(f"Penalty count: {penalty_count}")
    print(f"Difference from Stage 2: {F_penalties - F_adapter:.6f}")
    
    stage3_pass = penalty_count == 0 and abs(F_penalties - F_adapter) < 0.01
    print(f"Stage 3: {'PASS' if stage3_pass else 'FAIL'}")
    
    if penalty_count > 0:
        print(f"ERROR: Teacher has {penalty_count} penalties! Should be 0.")
        return False
    
    # Summary
    print(f"\nPARITY TEST SUMMARY")
    print(f"=" * 20)
    print(f"F* (ground truth):     {F_star:.6f}")
    print(f"F_direct (stage 1):    {F_direct:.6f}")
    print(f"F_adapter (stage 2):   {F_adapter:.6f}")
    print(f"F_penalties (stage 3): {F_penalties:.6f}")
    
    all_pass = stage1_pass and stage2_pass and stage3_pass
    print(f"\nOverall result: {'PASS' if all_pass else 'FAIL'}")
    
    if all_pass:
        print("SUCCESS: Evaluation pipeline matches theoretical maximum!")
        print("The +0.29 bits/sym is preserved through all stages.")
    else:
        print("FAILURE: Evaluation pipeline is destroying bits/sym.")
        print("Fix the failing stage before proceeding.")
    
    return all_pass

def main():
    success = staged_parity_test()
    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
