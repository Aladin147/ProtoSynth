#!/usr/bin/env python3
"""
Final ProtoSynth Science Validation

Comprehensive validation of the complete ProtoSynth system with realistic success criteria.
Tests the actual implemented system rather than idealized versions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import itertools
import random
from typing import Dict, Any

from protosynth import *
from protosynth.curriculum_evolution import CurriculumEvolutionEngine
from protosynth.modularity import SubtreeMiner
from protosynth.diff_shrink import ASTDiffer, DeltaDebugger
from protosynth.metrics import MetricsLogger
from protosynth.repro import ReproBundle, ReproConfig


def validate_core_system() -> Dict[str, Any]:
    """Validate that the core ProtoSynth system works as designed."""
    print("🔬 Core System Validation")
    print("=" * 30)
    
    results = {}
    
    # 1. Test AST creation and manipulation
    print("  Testing AST operations...")
    try:
        prog = if_expr(op('>', var('x'), const(0)), const(1), const(0))
        interpreter = LispInterpreter()
        result = interpreter.evaluate(prog, {'x': 5})
        
        ast_success = result == 1
        results['ast_operations'] = ast_success
        print(f"    AST operations: {'✅' if ast_success else '❌'}")
    except Exception as e:
        results['ast_operations'] = False
        print(f"    AST operations: ❌ ({e})")
    
    # 2. Test mutation system
    print("  Testing mutation system...")
    try:
        from protosynth.mutation import mutate
        original = op('+', const(1), const(2))
        mutated = mutate(original, mutation_rate=0.5, rng=random.Random(42))

        mutation_success = mutated is not None
        results['mutation_system'] = mutation_success
        print(f"    Mutation system: {'✅' if mutation_success else '❌'}")
    except Exception as e:
        results['mutation_system'] = False
        print(f"    Mutation system: ❌ ({e})")
    
    # 3. Test verification system
    print("  Testing verification system...")
    try:
        from protosynth.verify import verify_ast
        valid_prog = const(42)
        is_valid, errors = verify_ast(valid_prog)
        
        verify_success = is_valid and len(errors) == 0
        results['verification_system'] = verify_success
        print(f"    Verification system: {'✅' if verify_success else '❌'}")
    except Exception as e:
        results['verification_system'] = False
        print(f"    Verification system: ❌ ({e})")
    
    # 4. Test evolution engine
    print("  Testing evolution engine...")
    try:
        from protosynth.evolve import EvolutionEngine, create_initial_population
        
        engine = EvolutionEngine(mu=4, lambda_=8, seed=42)
        initial_pop = create_initial_population(4, seed=42)
        engine.initialize_population(initial_pop)
        
        # Run one generation
        test_stream = [1, 0, 1, 0] * 25
        stats = engine.evolve_generation(test_stream)
        
        evolution_success = 'best_fitness' in stats and len(engine.population) == 4
        results['evolution_engine'] = evolution_success
        print(f"    Evolution engine: {'✅' if evolution_success else '❌'}")
    except Exception as e:
        results['evolution_engine'] = False
        print(f"    Evolution engine: ❌ ({e})")
    
    return results


def validate_emergent_modularity() -> Dict[str, Any]:
    """Validate Track A: Emergent Modularity features."""
    print("\n🧩 Emergent Modularity Validation")
    print("=" * 40)
    
    results = {}
    
    # 1. Test subtree mining
    print("  Testing subtree mining...")
    try:
        # Create population with repeated patterns
        population = [
            op('+', var('x'), const(1)),
            op('+', var('y'), const(1)),
            op('+', var('z'), const(1)),
        ]
        
        miner = SubtreeMiner(beta=0.005, min_frequency=2)
        validation_bits = [0, 1] * 50
        candidates = miner.mine_and_select(population, validation_bits, n_modules=5)
        
        mining_success = len(candidates) > 0
        results['subtree_mining'] = mining_success
        print(f"    Subtree mining: {'✅' if mining_success else '❌'} ({len(candidates)} candidates)")
    except Exception as e:
        results['subtree_mining'] = False
        print(f"    Subtree mining: ❌ ({e})")
    
    # 2. Test module library
    print("  Testing module library...")
    try:
        from protosynth.modularity import ModuleLibrary
        
        library = ModuleLibrary(max_modules=5)
        if 'candidates' in locals() and candidates:
            modules = library.register_modules(candidates)
            
            library_success = len(modules) > 0
            results['module_library'] = library_success
            print(f"    Module library: {'✅' if library_success else '❌'} ({len(modules)} modules)")
        else:
            results['module_library'] = False
            print(f"    Module library: ❌ (no candidates to test)")
    except Exception as e:
        results['module_library'] = False
        print(f"    Module library: ❌ ({e})")
    
    # 3. Test interface contracts
    print("  Testing interface contracts...")
    try:
        if 'library' in locals() and library.modules:
            module_name = list(library.modules.keys())[0]
            call = library.create_module_call(module_name, [var('test')])
            
            contract_success = call.node_type == 'call'
            results['interface_contracts'] = contract_success
            print(f"    Interface contracts: {'✅' if contract_success else '❌'}")
        else:
            results['interface_contracts'] = False
            print(f"    Interface contracts: ❌ (no modules to test)")
    except Exception as e:
        results['interface_contracts'] = False
        print(f"    Interface contracts: ❌ ({e})")
    
    return results


def validate_curriculum_exploration() -> Dict[str, Any]:
    """Validate Track B: Curriculum & Exploration features."""
    print("\n🎓 Curriculum & Exploration Validation")
    print("=" * 45)
    
    results = {}
    
    # 1. Test curriculum evolution engine
    print("  Testing curriculum evolution...")
    try:
        engine = CurriculumEvolutionEngine(
            mu=4, lambda_=8, seed=42,
            max_modules=4, archive_size=10
        )
        
        # Initialize population
        from protosynth.evolve import create_initial_population
        initial_pop = create_initial_population(4, seed=42)
        engine.evolution_engine.initialize_population(initial_pop)
        
        # Run a few generations
        for gen in range(3):
            stats = engine.evolve_generation()
        
        curriculum_success = len(engine.environments) > 1
        results['curriculum_evolution'] = curriculum_success
        print(f"    Curriculum evolution: {'✅' if curriculum_success else '❌'} ({len(engine.environments)} envs)")
    except Exception as e:
        results['curriculum_evolution'] = False
        print(f"    Curriculum evolution: ❌ ({e})")
    
    # 2. Test bandit selection
    print("  Testing bandit selection...")
    try:
        if 'engine' in locals():
            bandit_stats = engine.bandit.get_stats()
            
            bandit_success = 'selection_counts' in bandit_stats
            results['bandit_selection'] = bandit_success
            print(f"    Bandit selection: {'✅' if bandit_success else '❌'}")
        else:
            results['bandit_selection'] = False
            print(f"    Bandit selection: ❌ (no engine to test)")
    except Exception as e:
        results['bandit_selection'] = False
        print(f"    Bandit selection: ❌ ({e})")
    
    # 3. Test novelty archive
    print("  Testing novelty archive...")
    try:
        if 'engine' in locals():
            archive_stats = engine.novelty_archive.get_diversity_stats()
            
            novelty_success = 'diversity' in archive_stats
            results['novelty_archive'] = novelty_success
            print(f"    Novelty archive: {'✅' if novelty_success else '❌'}")
        else:
            results['novelty_archive'] = False
            print(f"    Novelty archive: ❌ (no engine to test)")
    except Exception as e:
        results['novelty_archive'] = False
        print(f"    Novelty archive: ❌ ({e})")
    
    return results


def validate_tooling_science() -> Dict[str, Any]:
    """Validate Track C: Tooling & Science features."""
    print("\n🔬 Tooling & Science Validation")
    print("=" * 35)
    
    results = {}
    
    # 1. Test AST diff
    print("  Testing AST diff...")
    try:
        prog1 = op('+', const(1), const(2))
        prog2 = op('*', const(1), const(2))
        
        differ = ASTDiffer()
        diffs = differ.diff(prog1, prog2)
        
        diff_success = len(diffs) > 0
        results['ast_diff'] = diff_success
        print(f"    AST diff: {'✅' if diff_success else '❌'} ({len(diffs)} diffs)")
    except Exception as e:
        results['ast_diff'] = False
        print(f"    AST diff: ❌ ({e})")
    
    # 2. Test delta debugging
    print("  Testing delta debugging...")
    try:
        complex_prog = if_expr(const(1), op('+', const(1), const(2)), const(0))
        
        interpreter = LispInterpreter()
        debugger = DeltaDebugger(interpreter)
        
        test_data = [1, 0] * 10
        original_fitness = 0.0  # Dummy fitness
        
        shrunk, stats = debugger.shrink(complex_prog, test_data, original_fitness, max_iterations=3)
        
        shrink_success = 'size_reduction' in stats
        results['delta_debugging'] = shrink_success
        print(f"    Delta debugging: {'✅' if shrink_success else '❌'}")
    except Exception as e:
        results['delta_debugging'] = False
        print(f"    Delta debugging: ❌ ({e})")
    
    # 3. Test metrics logging
    print("  Testing metrics logging...")
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = MetricsLogger(log_dir=temp_dir, experiment_name="test")
            
            # Create dummy metrics
            from protosynth.metrics import GenerationMetrics
            metrics = GenerationMetrics(
                generation=0, timestamp=time.time(),
                best_fitness=0.5, median_fitness=0.4, mean_fitness=0.3, fitness_std=0.1,
                population_size=4, avg_program_size=8.0, size_std=2.0,
                diversity_score=0.6, novelty_score=0.4, current_environment="test",
                learning_progress=0.01, num_modules=1, module_usage_rate=0.1,
                evaluation_time=0.1, generation_time=1.0,
                robustness_score=0.9, noise_level=0.0
            )
            
            logger.log_generation(metrics)
            
            metrics_success = len(logger.metrics_history) == 1
            results['metrics_logging'] = metrics_success
            print(f"    Metrics logging: {'✅' if metrics_success else '❌'}")
    except Exception as e:
        results['metrics_logging'] = False
        print(f"    Metrics logging: ❌ ({e})")
    
    # 4. Test repro bundle
    print("  Testing repro bundle...")
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = ReproBundle(temp_dir)
            
            config = ReproConfig(
                mu=4, lambda_=8, seed=42, num_generations=5,
                max_modules=4, archive_size=10, environment_names=["test"],
                max_recursion_depth=10, max_steps=100, timeout_seconds=1.0
            )
            
            # Create minimal engine
            engine = CurriculumEvolutionEngine(mu=4, lambda_=8, seed=42)
            final_stats = {'best_fitness': 0.5, 'modules_discovered': 1}
            
            bundle_path = bundle.save_run(engine, config, final_stats)
            is_valid = bundle.verify_bundle()
            
            repro_success = is_valid
            results['repro_bundle'] = repro_success
            print(f"    Repro bundle: {'✅' if repro_success else '❌'}")
    except Exception as e:
        results['repro_bundle'] = False
        print(f"    Repro bundle: ❌ ({e})")
    
    return results


def run_final_science_validation():
    """Run comprehensive final validation of ProtoSynth."""
    print("🌟 ProtoSynth Final Science Validation")
    print("=" * 45)
    print("Comprehensive validation of the complete system")
    print("=" * 50)
    
    start_time = time.time()
    
    # Run all validation suites
    core_results = validate_core_system()
    modularity_results = validate_emergent_modularity()
    curriculum_results = validate_curriculum_exploration()
    tooling_results = validate_tooling_science()
    
    # Aggregate results
    all_results = {
        'core_system': core_results,
        'emergent_modularity': modularity_results,
        'curriculum_exploration': curriculum_results,
        'tooling_science': tooling_results
    }
    
    # Calculate success metrics
    total_tests = sum(len(suite_results) for suite_results in all_results.values())
    successful_tests = sum(
        sum(1 for success in suite_results.values() if success)
        for suite_results in all_results.values()
    )
    
    success_rate = successful_tests / total_tests if total_tests > 0 else 0.0
    runtime = time.time() - start_time
    
    # Final summary
    print(f"\n🎯 Final Validation Results")
    print("=" * 30)
    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Success rate: {success_rate:.1%}")
    print(f"Runtime: {runtime:.1f}s")
    
    # Detailed breakdown
    for suite_name, suite_results in all_results.items():
        suite_success = sum(1 for success in suite_results.values() if success)
        suite_total = len(suite_results)
        suite_rate = suite_success / suite_total if suite_total > 0 else 0.0
        
        print(f"\n{suite_name.replace('_', ' ').title()}: {suite_rate:.1%} ({suite_success}/{suite_total})")
        for test_name, success in suite_results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {test_name.replace('_', ' ').title()}")
    
    # Overall assessment
    if success_rate >= 0.8:
        print(f"\n🌟 EXCELLENT: ProtoSynth system is highly functional!")
    elif success_rate >= 0.6:
        print(f"\n✅ GOOD: ProtoSynth system is largely functional with minor issues")
    elif success_rate >= 0.4:
        print(f"\n⚠️  PARTIAL: ProtoSynth system has core functionality but needs work")
    else:
        print(f"\n❌ NEEDS WORK: ProtoSynth system has significant issues")
    
    return {
        'total_tests': total_tests,
        'successful_tests': successful_tests,
        'success_rate': success_rate,
        'runtime_seconds': runtime,
        'detailed_results': all_results
    }


if __name__ == "__main__":
    summary = run_final_science_validation()
    print(f"\nOverall system validation: {summary['success_rate']:.1%} success rate")
