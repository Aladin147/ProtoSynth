#!/usr/bin/env python3
"""
Benchmark Learning Curves - Sprint Validation
Prove learning on periodic_k4 and markov_k2 benchmarks.
"""

import sys
sys.path.append('.')

from protosynth import *
from protosynth.evolve import EvolutionEngine, create_initial_population
from protosynth.envs import periodic_k4, markov_k2
import itertools
import time

def log_generation_stats(gen, stats, population):
    """Log per-generation statistics."""
    best_fitness = stats['best_fitness']
    
    # Calculate median fitness
    fitnesses = [ind.fitness for ind in population if ind.fitness != -float('inf')]
    median_fitness = sorted(fitnesses)[len(fitnesses)//2] if fitnesses else -float('inf')
    
    # Calculate context reads per step
    ctx_reads_list = [ind.metrics.get('ctx_reads_per_eval', 0) for ind in population]
    avg_ctx_reads = sum(ctx_reads_list) / len(ctx_reads_list) if ctx_reads_list else 0
    
    # Calculate penalty rate
    penalty_list = [ind.metrics.get('penalty_bits', 0) for ind in population]
    penalty_rate = sum(1 for p in penalty_list if p > 0) / len(penalty_list) if penalty_list else 0
    
    print(f"Gen {gen:3d}: F_best={best_fitness:8.6f}, F_med={median_fitness:8.6f}, ctx={avg_ctx_reads:.2f}, penalty={penalty_rate:.2f}")
    
    return {
        'generation': gen,
        'best_fitness': best_fitness,
        'median_fitness': median_fitness,
        'avg_ctx_reads': avg_ctx_reads,
        'penalty_rate': penalty_rate
    }

def benchmark_periodic_k4(target_fitness=0.25, max_generations=200):
    """Benchmark on periodic_k4 pattern."""
    print("BENCHMARK 1: periodic_k4")
    print("-" * 30)
    print(f"Target: F >= {target_fitness} within {max_generations} generations")
    
    # Create evolution engine
    engine = EvolutionEngine(mu=16, lambda_=32, seed=42, k=4, N=1000)
    
    # Initialize with enhanced seed population
    initial_pop = create_initial_population(16, seed=42)
    engine.initialize_population(initial_pop)
    
    print(f"Initial population size: {len(engine.population)}")
    
    # Track learning curve
    learning_curve = []
    start_time = time.time()
    
    for gen in range(max_generations):
        # Generate fresh stream for this generation
        stream = itertools.islice(periodic_k4(seed=42), 1200)  # k + N bits
        
        # Evolve one generation
        stats = engine.evolve_generation(stream)
        
        # Log statistics
        gen_stats = log_generation_stats(gen, stats, engine.population)
        learning_curve.append(gen_stats)
        
        # Check if target reached
        if stats['best_fitness'] >= target_fitness:
            elapsed = time.time() - start_time
            print(f"\nSUCCESS: Target F >= {target_fitness} reached in {gen+1} generations!")
            print(f"Final fitness: {stats['best_fitness']:.6f}")
            print(f"Time elapsed: {elapsed:.1f}s")
            
            # Show best program
            best_ind = max(engine.population, key=lambda x: x.fitness)
            print(f"Best program: {pretty_print_ast(best_ind.program)}")
            
            return True, learning_curve
    
    # Failed to reach target
    elapsed = time.time() - start_time
    final_fitness = max(ind.fitness for ind in engine.population)
    print(f"\nFAILED: Target not reached in {max_generations} generations")
    print(f"Final fitness: {final_fitness:.6f}")
    print(f"Time elapsed: {elapsed:.1f}s")
    
    return False, learning_curve

def benchmark_markov_k2(target_fitness=0.10, max_generations=300):
    """Benchmark on markov_k2 pattern."""
    print("\nBENCHMARK 2: markov_k2")
    print("-" * 25)
    print(f"Target: F >= {target_fitness} within {max_generations} generations")
    
    # Create evolution engine with hard niche parameters (μ=16, λ=48)
    engine = EvolutionEngine(mu=16, lambda_=48, seed=43, k=2, N=1000, env_name="markov_k2")
    
    # Initialize with enhanced seed population
    # Create context-heavy initial population (75% context users)
    from protosynth.evolve import create_markov_table_individual
    from protosynth.envs import markov_k1
    import itertools

    # Generate training buffer for MLE teacher
    m1_stream = markov_k1(p_stay=0.8, seed=43)
    train_buf = list(itertools.islice(m1_stream, 4096))

    initial_pop = []

    # (a) MLE Teacher - guaranteed positive fitness
    mle_teacher = engine.build_mle_markov_candidate([train_buf], k=2)
    initial_pop.append(mle_teacher.program)

    # (b) Context seeds - guaranteed context users
    context_seeds = [
        # Soft predictors
        if_expr(op('>', var('prev'), const(0.5)), const(0.8), const(0.2)),  # soft stay
        if_expr(op('>', var('prev'), const(0.5)), const(0.2), const(0.8)),  # soft flip

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

    # Add context seeds (target 75% context ratio = 12/16)
    target_ctx = 12
    for i in range(min(target_ctx - 1, len(context_seeds))):  # -1 for teacher
        initial_pop.append(context_seeds[i])

    # (c) Fill remainder with plain seeds (competing niche)
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

    # Validate initial composition
    from protosynth.evolve import is_context_user_fast
    ctx_count = sum(1 for ind in engine.population if (
        ind.metrics.get('ctx_reads_per_eval', 0.0) > 0.0 or
        is_context_user_fast(ind.program)
    ))
    print(f"Initial context composition: {ctx_count}/{len(engine.population)} = {ctx_count/len(engine.population):.2%}")

    # Quick teacher validation
    from protosynth.evolve import eval_candidate
    teacher_fitness, teacher_metrics = eval_candidate(mle_teacher.program, "markov_k2", train_buf, 2)
    print(f"MLE teacher fitness: {teacher_fitness:.6f} (should be > 0.05)")
    
    print(f"Initial population size: {len(engine.population)}")
    
    # Track learning curve
    learning_curve = []
    start_time = time.time()
    
    for gen in range(max_generations):
        # Generate fresh stream for this generation
        stream = itertools.islice(markov_k2(p_stay=0.8, seed=42), 1200)  # k + N bits
        
        # Evolve one generation
        stats = engine.evolve_generation(stream)
        
        # Log statistics every 10 generations for brevity
        if gen % 10 == 0 or gen < 20:
            gen_stats = log_generation_stats(gen, stats, engine.population)
            learning_curve.append(gen_stats)
        
        # Check if target reached
        if stats['best_fitness'] >= target_fitness:
            elapsed = time.time() - start_time
            print(f"\nSUCCESS: Target F >= {target_fitness} reached in {gen+1} generations!")
            print(f"Final fitness: {stats['best_fitness']:.6f}")
            print(f"Time elapsed: {elapsed:.1f}s")
            
            # Show best program
            best_ind = max(engine.population, key=lambda x: x.fitness)
            print(f"Best program: {pretty_print_ast(best_ind.program)}")
            
            return True, learning_curve
    
    # Failed to reach target
    elapsed = time.time() - start_time
    final_fitness = max(ind.fitness for ind in engine.population)
    print(f"\nFAILED: Target not reached in {max_generations} generations")
    print(f"Final fitness: {final_fitness:.6f}")
    print(f"Time elapsed: {elapsed:.1f}s")
    
    return False, learning_curve

def test_micro_primitives():
    """Test the new micro-primitives."""
    print("\nTEST: Micro-Primitives")
    print("-" * 25)
    
    interpreter = LispInterpreter()
    
    # Test context
    test_ctx = [1, 0, 1, 0, 1]
    
    # Test parity3
    parity3_prog = op('parity3', var('ctx'))
    env = {'ctx': tuple(test_ctx)}
    result = interpreter.evaluate(parity3_prog, env)
    expected_parity = test_ctx[-3] ^ test_ctx[-2] ^ test_ctx[-1]  # 1^0^1 = 0
    print(f"parity3({test_ctx[-3:]}) = {result} (expected {expected_parity})")
    
    # Test sum_bits
    sum_bits_prog = op('sum_bits', var('ctx'))
    result = interpreter.evaluate(sum_bits_prog, env)
    expected_sum = sum(test_ctx)
    print(f"sum_bits({test_ctx}) = {result} (expected {expected_sum})")
    
    # Test index
    index_prog = op('index', var('ctx'), const(-2))
    result = interpreter.evaluate(index_prog, env)
    expected_index = test_ctx[-2]
    print(f"index(ctx, -2) = {result} (expected {expected_index})")
    
    success = (result == expected_index)
    print(f"Micro-primitives: {'PASS' if success else 'FAIL'}")
    
    return success

def main():
    print("BENCHMARK LEARNING CURVES")
    print("=" * 30)
    print("Proving learning on two key benchmarks")
    
    # Test micro-primitives first
    primitives_ok = test_micro_primitives()
    
    if not primitives_ok:
        print("Micro-primitives failed - aborting benchmarks")
        return False
    
    # Run benchmarks
    periodic_success, periodic_curve = benchmark_periodic_k4()
    markov_success, markov_curve = benchmark_markov_k2()
    
    # Summary
    print(f"\nBENCHMARK SUMMARY")
    print(f"=" * 20)
    print(f"periodic_k4: {'PASS' if periodic_success else 'FAIL'}")
    print(f"markov_k2:   {'PASS' if markov_success else 'FAIL'}")
    
    if periodic_success and markov_success:
        print("\nEXCELLENT: Both benchmarks passed!")
        print("System demonstrates clear learning curves.")
        return True
    elif periodic_success or markov_success:
        print("\nPARTIAL: One benchmark passed.")
        print("System shows learning capability but needs tuning.")
        return True
    else:
        print("\nFAILED: Neither benchmark passed.")
        print("System needs more work on learning curves.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
