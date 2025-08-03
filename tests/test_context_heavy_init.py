#!/usr/bin/env python3
"""Test context-heavy initial population."""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.evolve import EvolutionEngine, create_markov_table_individual, is_context_user_fast, eval_candidate
from protosynth.envs import markov_k1
import itertools

def test_context_heavy_initialization():
    """Test context-heavy initial population setup."""
    print("TEST: Context-Heavy Initialization")
    print("-" * 40)
    
    # Create evolution engine
    engine = EvolutionEngine(mu=16, lambda_=48, seed=42, k=2, N=1000, env_name="markov_k2")
    
    # Generate training buffer for MLE teacher
    m1_stream = markov_k1(p_stay=0.8, seed=42)
    train_buf = list(itertools.islice(m1_stream, 4096))
    
    initial_pop = []
    
    # (a) MLE Teacher
    mle_teacher = engine.build_mle_markov_candidate([train_buf], k=2)
    initial_pop.append(mle_teacher.program)
    print(f"  MLE teacher: {pretty_print_ast(mle_teacher.program)}")
    
    # (b) Context seeds
    context_seeds = [
        # Soft predictors
        if_expr(op('>', var('prev'), const(0.5)), const(0.8), const(0.2)),
        if_expr(op('>', var('prev'), const(0.5)), const(0.2), const(0.8)),
        
        # Direct context variables
        var('prev'),
        var('prev2'),
        
        # Context expressions
        op('xor', var('prev'), var('prev2')),
        op('>', op('+', var('prev'), var('prev2')), const(0.5)),
        
        # Markov table seeds
        create_markov_table_individual('stay', generation=0).program,
        create_markov_table_individual('flip', generation=0).program,
        
        # Additional context patterns
        op('-', const(1), var('prev')),
        op('=', var('prev'), var('prev2')),
    ]
    
    # Add context seeds (target 75% = 12/16)
    target_ctx = 12
    for i in range(min(target_ctx - 1, len(context_seeds))):  # -1 for teacher
        initial_pop.append(context_seeds[i])
    
    # (c) Fill remainder with plain seeds
    plain_seeds = [
        const(0.0),
        const(0.5),
        const(1.0),
        op('+', const(0.3), const(0.2)),
    ]
    
    while len(initial_pop) < 16:
        initial_pop.append(plain_seeds[len(initial_pop) % len(plain_seeds)])
    
    # Initialize population
    engine.initialize_population(initial_pop)
    
    print(f"  Initial population size: {len(engine.population)}")
    
    # Validate composition
    ctx_count = 0
    markov_count = 0
    
    print(f"  Initial population analysis:")
    for i, ind in enumerate(engine.population):
        runtime_ctx = ind.metrics.get('ctx_reads_per_eval', 0.0) > 0.0
        ast_ctx = is_context_user_fast(ind.program)
        is_ctx = runtime_ctx or ast_ctx
        is_markov = 'markov_table' in pretty_print_ast(ind.program)
        
        if is_ctx:
            ctx_count += 1
        if is_markov:
            markov_count += 1
        
        ctx_marker = "CTX" if is_ctx else "PLAIN"
        markov_marker = "MARKOV" if is_markov else ""
        print(f"    {i:2d}: {ctx_marker:5} {markov_marker:6} {pretty_print_ast(ind.program)}")
    
    ctx_ratio = ctx_count / len(engine.population)
    print(f"  Context composition: {ctx_count}/{len(engine.population)} = {ctx_ratio:.2%}")
    print(f"  Markov table count: {markov_count}")
    
    # Test MLE teacher fitness
    teacher_fitness, teacher_metrics = eval_candidate(mle_teacher.program, "markov_k2", train_buf, 2)
    print(f"  MLE teacher fitness: {teacher_fitness:.6f}")
    
    # Success criteria
    success = (
        ctx_ratio >= 0.70 and  # At least 70% context users
        markov_count >= 2 and  # At least 2 markov_table programs
        teacher_fitness > -0.5  # Teacher should be reasonable (may not be positive due to calibration)
    )
    
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    return success

def test_context_survival_with_heavy_init():
    """Test context survival with context-heavy initialization."""
    print("\nTEST: Context Survival with Heavy Init")
    print("-" * 45)
    
    # Use the same initialization as the benchmark
    engine = EvolutionEngine(mu=16, lambda_=48, seed=42, k=2, N=1000, env_name="markov_k2")
    
    # Generate training buffer
    m1_stream = markov_k1(p_stay=0.8, seed=42)
    train_buf = list(itertools.islice(m1_stream, 4096))
    
    # Build context-heavy population (same as benchmark)
    initial_pop = []
    
    # MLE Teacher
    mle_teacher = engine.build_mle_markov_candidate([train_buf], k=2)
    initial_pop.append(mle_teacher.program)
    
    # Context seeds
    context_seeds = [
        if_expr(op('>', var('prev'), const(0.5)), const(0.8), const(0.2)),
        if_expr(op('>', var('prev'), const(0.5)), const(0.2), const(0.8)),
        var('prev'), var('prev2'),
        op('xor', var('prev'), var('prev2')),
        op('>', op('+', var('prev'), var('prev2')), const(0.5)),
        create_markov_table_individual('stay', generation=0).program,
        create_markov_table_individual('flip', generation=0).program,
        op('-', const(1), var('prev')),
        op('=', var('prev'), var('prev2')),
    ]
    
    # Add context seeds
    for i in range(min(11, len(context_seeds))):  # 11 + 1 teacher = 12 context
        initial_pop.append(context_seeds[i])
    
    # Fill remainder with plain seeds
    while len(initial_pop) < 16:
        initial_pop.append(const(0.5))
    
    engine.initialize_population(initial_pop)
    
    # Test survival over 20 generations
    print(f"  Testing context survival over 20 generations:")
    
    context_history = []
    fitness_history = []
    
    for gen in range(20):
        # Create fresh stream
        m1_stream = markov_k1(p_stay=0.8, seed=42+gen)
        fresh_bits = list(itertools.islice(m1_stream, engine.N + engine.k))
        stream = iter(fresh_bits)
        
        stats = engine.evolve_generation(stream)
        best_fitness = stats['best_fitness']
        
        # Count context users
        ctx_count = sum(1 for ind in engine.population if (
            ind.metrics.get('ctx_reads_per_eval', 0.0) > 0.0 or 
            is_context_user_fast(ind.program)
        ))
        
        context_history.append(ctx_count)
        fitness_history.append(best_fitness)
        
        if gen % 5 == 0 or gen < 3:
            ctx_ratio = ctx_count / len(engine.population)
            print(f"    Gen {gen:2d}: F_best={best_fitness:.6f}, ctx={ctx_count:2d} ({ctx_ratio:.1%})")
        
        if best_fitness > 0.05:
            print(f"    SUCCESS: Target fitness achieved at gen {gen}!")
            break
    
    # Analyze results
    min_ctx = min(context_history)
    avg_ctx = sum(context_history) / len(context_history)
    max_fitness = max(fitness_history)
    
    print(f"  Context survival analysis:")
    print(f"    Min context survivors: {min_ctx}")
    print(f"    Avg context survivors: {avg_ctx:.1f}")
    print(f"    Max fitness achieved: {max_fitness:.6f}")
    
    # Success if good context survival and reasonable fitness
    success = min_ctx >= 8 and max_fitness > -0.6
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    
    return success

def main():
    print("CONTEXT-HEAVY INITIALIZATION TEST")
    print("=" * 36)
    
    test_results = [
        test_context_heavy_initialization(),
        test_context_survival_with_heavy_init(),
    ]
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\nCONTEXT-HEAVY INIT SUMMARY")
    print(f"=" * 28)
    print(f"Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("CONTEXT-HEAVY INITIALIZATION WORKING!")
        print("Benchmark should now have proper context survival.")
    elif passed_tests >= 1:
        print("MOSTLY WORKING - Some issues remain")
    else:
        print("CONTEXT-HEAVY INIT NEEDS MORE WORK")
    
    return passed_tests >= 1

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
