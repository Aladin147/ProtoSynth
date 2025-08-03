#!/usr/bin/env python3
"""
Module Validation Experiments

Three tight experiments to validate modular evolution:
1. Transfer & Freeze: Train modules on simple env, transfer to complex env
2. Ablation: Remove top-K modules and measure performance impact  
3. Inline vs Call: Compare modular vs inlined versions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import random
import itertools
from typing import List, Dict, Tuple
from protosynth import *
from protosynth.envs import *
from protosynth.evolve import *
from protosynth.modularity import *
from protosynth.eval import evaluate_program_on_window


def experiment_1_transfer_freeze(seed: int = 42) -> Dict:
    """
    Transfer & Freeze Experiment
    
    Train modules on periodic(k≤4), freeze top-N, then evolve fresh agent
    on markov(k≤6) with library available. Measure sample-efficiency gain.
    
    Returns:
        Dict with results including generations to reach F=0.3
    """
    print("🔄 Experiment 1: Transfer & Freeze")
    print("=" * 40)
    
    # Phase 1: Train on simple periodic patterns
    print("Phase 1: Training modules on periodic patterns...")
    
    def simple_stream_factory():
        patterns = [[1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1, 0]]
        pattern = random.choice(patterns)
        return periodic(pattern, seed=seed)
    
    # Run evolution to discover modules
    results_phase1 = run_simple_evolution(
        simple_stream_factory,
        num_generations=30,
        mu=12,
        lambda_=24,
        seed=seed
    )
    
    # Extract and freeze top modules
    population = [ind.program for ind in results_phase1['final_population']]
    validation_bits = list(itertools.islice(simple_stream_factory(), 1000))
    
    miner = SubtreeMiner(beta=0.005, min_frequency=2)
    candidates = miner.mine_and_select(population, validation_bits, n_modules=8)
    
    library = ModuleLibrary(max_modules=8)
    frozen_modules = library.register_modules(candidates)
    
    print(f"Frozen {len(frozen_modules)} modules from phase 1")
    
    # Phase 2: Evolve on complex Markov with frozen modules
    print("Phase 2: Evolving on Markov with frozen modules...")
    
    def complex_stream_factory():
        trans = {
            (0, 0): 0.8, (0, 1): 0.3,
            (1, 0): 0.7, (1, 1): 0.2
        }
        return k_order_markov(2, trans, seed=seed)
    
    # Create custom evolution with module library
    initial_programs = create_initial_population(12, seed=seed + 1)
    engine_with_modules = EvolutionEngine(mu=12, lambda_=24, seed=seed + 1)
    engine_with_modules.interpreter.module_library = library
    engine_with_modules.initialize_population(initial_programs)
    
    # Track generations to reach F=0.3
    gens_to_target_with_modules = None
    target_fitness = 0.3
    
    for gen in range(50):
        stream = complex_stream_factory()
        stats = engine_with_modules.evolve_generation(stream)
        
        if stats['best_fitness'] >= target_fitness and gens_to_target_with_modules is None:
            gens_to_target_with_modules = gen + 1
            break
        
        if gen % 10 == 0:
            print(f"  Gen {gen}: F={stats['best_fitness']:.3f}")
    
    # Phase 3: Control - evolve without modules
    print("Phase 3: Control evolution without modules...")
    
    engine_control = EvolutionEngine(mu=12, lambda_=24, seed=seed + 2)
    engine_control.initialize_population(create_initial_population(12, seed=seed + 2))
    
    gens_to_target_control = None
    
    for gen in range(50):
        stream = complex_stream_factory()
        stats = engine_control.evolve_generation(stream)
        
        if stats['best_fitness'] >= target_fitness and gens_to_target_control is None:
            gens_to_target_control = gen + 1
            break
        
        if gen % 10 == 0:
            print(f"  Control Gen {gen}: F={stats['best_fitness']:.3f}")
    
    # Calculate sample efficiency gain
    efficiency_gain = None
    if gens_to_target_control and gens_to_target_with_modules:
        efficiency_gain = gens_to_target_control / gens_to_target_with_modules
    
    results = {
        'frozen_modules': len(frozen_modules),
        'gens_to_target_with_modules': gens_to_target_with_modules,
        'gens_to_target_control': gens_to_target_control,
        'efficiency_gain': efficiency_gain,
        'target_fitness': target_fitness
    }
    
    print(f"Results: With modules: {gens_to_target_with_modules} gens, "
          f"Control: {gens_to_target_control} gens, "
          f"Gain: {efficiency_gain:.2f}x" if efficiency_gain else "Gain: N/A")
    
    return results


def experiment_2_ablation(seed: int = 42) -> Dict:
    """
    Ablation Experiment
    
    Remove top-K MDL modules one by one and re-evolve for fixed budget.
    Record ΔF and gens-to-plateau.
    
    Returns:
        Dict with ablation results
    """
    print("\n🔬 Experiment 2: Module Ablation")
    print("=" * 35)
    
    # Create a population with clear modular structure
    def create_modular_population():
        programs = []
        # Add programs that use common patterns
        for i in range(20):
            if i % 3 == 0:
                programs.append(op('+', var('x'), const(1)))  # increment pattern
            elif i % 3 == 1:
                programs.append(op('>', var('y'), const(0)))  # positive check
            else:
                programs.append(if_expr(op('>', var('z'), const(0)), 
                                      op('+', var('z'), const(1)), 
                                      const(0)))  # conditional increment
        return programs
    
    population = create_modular_population()
    validation_bits = [0, 1, 0, 1] * 250
    
    # Mine modules
    miner = SubtreeMiner(beta=0.005, min_frequency=3)
    all_candidates = miner.mine_and_select(population, validation_bits, n_modules=10)
    
    print(f"Found {len(all_candidates)} module candidates")
    
    # Test with different numbers of modules
    ablation_results = []
    
    for k in [0, 1, 2, len(all_candidates)]:
        print(f"Testing with top {k} modules...")
        
        # Create library with top k modules
        library = ModuleLibrary(max_modules=k)
        if k > 0:
            library.register_modules(all_candidates[:k])
        
        # Run short evolution
        def test_stream_factory():
            return periodic([1, 0, 1, 1], seed=seed)
        
        engine = EvolutionEngine(mu=8, lambda_=16, seed=seed + k, N=500)
        if k > 0:
            engine.interpreter.module_library = library
        
        engine.initialize_population(create_initial_population(8, seed=seed + k))
        
        best_fitness = -float('inf')
        plateau_gen = None
        
        for gen in range(20):
            stream = test_stream_factory()
            stats = engine.evolve_generation(stream)
            
            if stats['best_fitness'] > best_fitness:
                best_fitness = stats['best_fitness']
                plateau_gen = gen + 1
        
        ablation_results.append({
            'num_modules': k,
            'best_fitness': best_fitness,
            'plateau_gen': plateau_gen
        })
        
        print(f"  k={k}: F={best_fitness:.3f}, plateau at gen {plateau_gen}")
    
    # Calculate performance deltas
    baseline_fitness = ablation_results[-1]['best_fitness']  # Full modules
    
    for result in ablation_results:
        result['delta_f'] = result['best_fitness'] - baseline_fitness
    
    return {
        'ablation_results': ablation_results,
        'baseline_fitness': baseline_fitness
    }


def experiment_3_inline_vs_call(seed: int = 42) -> Dict:
    """
    Inline vs Call Experiment
    
    Inline each module into call sites and compare F under identical step limits.
    If F_inline≈F_call but AST grows, proves real modular compression.
    
    Returns:
        Dict comparing inline vs call performance
    """
    print("\n📏 Experiment 3: Inline vs Call")
    print("=" * 30)
    
    # Create a simple modular program
    library = ModuleLibrary(max_modules=4)
    
    # Register some test modules
    test_candidates = [
        SubtreeCandidate(
            subtree=op('+', var('x'), const(1)),
            frequency=3, total_nodes=3, mdl_score=0.15,
            bits_saved=0.20, size_penalty=0.05
        ),
        SubtreeCandidate(
            subtree=op('>', var('y'), const(0)),
            frequency=3, total_nodes=3, mdl_score=0.10,
            bits_saved=0.15, size_penalty=0.05
        )
    ]
    
    modules = library.register_modules(test_candidates)
    print(f"Registered {len(modules)} test modules")
    
    # Create modular program that uses modules
    modular_program = if_expr(
        library.create_module_call('mod_1', [var('x')]),  # > x 0
        library.create_module_call('mod_0', [var('x')]),  # + x 1
        const(0)
    )
    
    # Create equivalent inlined program
    inlined_program = if_expr(
        op('>', var('x'), const(0)),
        op('+', var('x'), const(1)),
        const(0)
    )
    
    # Test data
    test_bits = [1, 0, 1, 1, 0, 1] * 50
    
    # Evaluate both versions
    interpreter_modular = LispInterpreter(module_library=library)
    interpreter_inline = LispInterpreter()
    
    # Performance comparison
    fitness_modular, metrics_modular = evaluate_program_on_window(
        interpreter_modular, modular_program, test_bits, k=2
    )
    
    fitness_inline, metrics_inline = evaluate_program_on_window(
        interpreter_inline, inlined_program, test_bits, k=2
    )
    
    # Size comparison
    from protosynth.mutation import iter_nodes
    size_modular = len(list(iter_nodes(modular_program)))
    size_inline = len(list(iter_nodes(inlined_program)))
    
    # Compression ratio
    compression_ratio = size_inline / size_modular if size_modular > 0 else 1.0
    
    results = {
        'fitness_modular': fitness_modular,
        'fitness_inline': fitness_inline,
        'fitness_delta': abs(fitness_modular - fitness_inline),
        'size_modular': size_modular,
        'size_inline': size_inline,
        'compression_ratio': compression_ratio,
        'proves_compression': (abs(fitness_modular - fitness_inline) < 0.01 and 
                             compression_ratio > 1.1)
    }
    
    print(f"Modular: F={fitness_modular:.3f}, size={size_modular}")
    print(f"Inline:  F={fitness_inline:.3f}, size={size_inline}")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Proves compression: {results['proves_compression']}")
    
    return results


def run_all_experiments(seed: int = 42) -> Dict:
    """Run all three validation experiments."""
    print("🧪 Module Validation Experiments")
    print("=" * 50)
    
    start_time = time.time()
    
    results = {
        'transfer_freeze': experiment_1_transfer_freeze(seed),
        'ablation': experiment_2_ablation(seed + 100),
        'inline_vs_call': experiment_3_inline_vs_call(seed + 200)
    }
    
    total_time = time.time() - start_time
    results['total_time'] = total_time
    
    print(f"\n✅ All experiments completed in {total_time:.1f}s")
    
    return results


if __name__ == "__main__":
    results = run_all_experiments(seed=42)
    
    print("\n📊 Summary:")
    print(f"Transfer efficiency gain: {results['transfer_freeze']['efficiency_gain']:.2f}x" 
          if results['transfer_freeze']['efficiency_gain'] else "N/A")
    print(f"Ablation baseline fitness: {results['ablation']['baseline_fitness']:.3f}")
    print(f"Compression proven: {results['inline_vs_call']['proves_compression']}")
