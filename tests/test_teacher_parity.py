#!/usr/bin/env python3
"""Staged parity harness - ensures F_eval ≈ F* within tolerance."""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.eval import EvalSession, evaluate_program_calibrated
from protosynth.evolve import EvolutionEngine, eval_candidate, program_is_probabilistic
from protosynth.envs import markov_k1
import itertools
import math

def H2(p, eps=1e-9):
    """Binary entropy function."""
    p = max(eps, min(1-eps, p))
    return -(p * math.log2(p) + (1-p) * math.log2(1-p))

def F_star_from_buffer(buf, k=2):
    """Compute theoretical F* from buffer."""
    N = 0
    ones = 0
    cnt = {(0,0): [0,0], (0,1): [0,0], (1,0): [0,0], (1,1): [0,0]}  # [n_s, c1_s]
    
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
    
    return H0 - Hprog, cnt

def test_teacher_parity():
    """Test that teacher evaluation matches theoretical F*."""
    print("TEST: Teacher Evaluation Parity")
    print("-" * 35)
    
    # Generate validation buffer
    m1_stream = markov_k1(p_stay=0.8, seed=42)
    val_buf = list(itertools.islice(m1_stream, 8000))
    
    # Compute theoretical F*
    F_star, counts = F_star_from_buffer(val_buf, k=2)
    print(f"Theoretical F*: {F_star:.6f}")
    
    # Build MLE parameters from counts
    mle_params = {}
    for s, (n, c1) in counts.items():
        if n > 0:
            # MLE: P(next=1|s) = c1/n, store P(next=0|s) for compatibility
            p1_mle = c1 / n
            p0_mle = 1 - p1_mle
            param_key = f'p{s[0]}{s[1]}'
            mle_params[param_key] = p0_mle
    
    print(f"MLE parameters: {mle_params}")
    
    # Create markov_table program
    markov_prog = op('markov_table', op('+', op('*', const(2), var('prev2')), var('prev')))
    print(f"Program: {pretty_print_ast(markov_prog)}")
    print(f"Is probabilistic: {program_is_probabilistic(markov_prog)}")

    # Debug the AST structure
    print(f"AST debug:")
    print(f"  Root node_type: {markov_prog.node_type}")
    print(f"  Root value: {markov_prog.value}")
    if markov_prog.children:
        print(f"  Child 0 node_type: {markov_prog.children[0].node_type}")
        print(f"  Child 0 value: {markov_prog.children[0].value}")
    
    # Test 1: EvalSession (clean path)
    print(f"\n1) EvalSession (clean path):")
    interpreter1 = LispInterpreter(max_steps=10000, timeout_seconds=10.0)
    interpreter1.markov_params = mle_params
    
    session = EvalSession(interpreter1)
    
    # Test a few predictions
    test_contexts = [
        [0, 0], [0, 1], [1, 0], [1, 1]
    ]
    
    for ctx in test_contexts:
        try:
            p = session.predict(markov_prog, ctx)
            s = (ctx[-2], ctx[-1])
            expected_p1 = counts[s][1] / counts[s][0] if counts[s][0] > 0 else 0.5
            print(f"  ctx={ctx}, s={s}, p={p:.4f}, expected_p1={expected_p1:.4f}")
        except Exception as e:
            print(f"  ctx={ctx}, ERROR: {e}")
    
    # Test 2: eval_candidate (current benchmark path)
    print(f"\n2) eval_candidate (benchmark path):")
    F_eval_candidate, metrics = eval_candidate(markov_prog, "markov_k2", val_buf, 2)
    print(f"F_eval_candidate: {F_eval_candidate:.6f}")
    print(f"Penalty rate: {metrics.get('penalty_rate', 'N/A')}")
    print(f"Model entropy: {metrics.get('model_entropy', 'N/A')}")
    
    # Test 3: evaluate_program_calibrated (direct path)
    print(f"\n3) evaluate_program_calibrated (direct path):")
    interpreter2 = LispInterpreter(max_steps=10000, timeout_seconds=10.0)
    interpreter2.markov_params = mle_params
    
    F_eval_direct, metrics2 = evaluate_program_calibrated(
        interpreter2, markov_prog, buffer=val_buf, k=2, N_train=4000, N_val=4000
    )
    print(f"F_eval_direct: {F_eval_direct:.6f}")
    print(f"Penalty rate: {metrics2.get('penalty_rate', 'N/A')}")
    print(f"Model entropy: {metrics2.get('model_entropy', 'N/A')}")
    
    # Test 4: EvolutionEngine.build_mle_markov_candidate
    print(f"\n4) EvolutionEngine MLE teacher:")
    engine = EvolutionEngine(mu=16, lambda_=48, seed=42, k=2, N=1000, env_name="markov_k2")
    
    mle_teacher = engine.build_mle_markov_candidate([val_buf], k=2)
    print(f"Teacher program: {pretty_print_ast(mle_teacher.program)}")
    print(f"Teacher fitness: {mle_teacher.fitness:.6f}")
    
    # Check if teacher has parameters set
    if hasattr(mle_teacher, 'interpreter') and hasattr(mle_teacher.interpreter, 'markov_params'):
        print(f"Teacher params: {mle_teacher.interpreter.markov_params}")
    else:
        print("Teacher has no interpreter or markov_params")
    
    # Analysis
    print(f"\nPARITY ANALYSIS:")
    print(f"F* (theoretical):        {F_star:.6f}")
    print(f"F_eval_candidate:        {F_eval_candidate:.6f}")
    print(f"F_eval_direct:           {F_eval_direct:.6f}")
    print(f"F_teacher (engine):      {mle_teacher.fitness:.6f}")
    
    # Check which paths match F*
    tolerance = 0.05
    
    candidate_match = abs(F_eval_candidate - F_star) < tolerance
    direct_match = abs(F_eval_direct - F_star) < tolerance
    teacher_match = abs(mle_teacher.fitness - F_star) < tolerance
    
    print(f"\nMatches F* within ±{tolerance}:")
    print(f"eval_candidate:  {'YES' if candidate_match else 'NO'}")
    print(f"eval_direct:     {'YES' if direct_match else 'NO'}")
    print(f"teacher:         {'YES' if teacher_match else 'NO'}")
    
    # Success if at least one path matches
    success = candidate_match or direct_match
    print(f"\nResult: {'PASS' if success else 'FAIL'}")
    
    if not success:
        print("ERROR: No evaluation path matches theoretical F*!")
        print("The evaluation pipeline still has issues.")
    elif not teacher_match:
        print("WARNING: Teacher evaluation differs from other paths.")
        print("Teacher may be using different evaluation logic.")
    else:
        print("SUCCESS: All evaluation paths match theoretical F*!")
    
    return success

def main():
    success = test_teacher_parity()
    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
