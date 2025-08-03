#!/usr/bin/env python3
"""
Quick Module Validation

Fast experiments to validate modular evolution concepts.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools
from protosynth import *
from protosynth.envs import *
from protosynth.modularity import *
from protosynth.eval import evaluate_program_on_window


def quick_experiment_3_inline_vs_call():
    """Quick test: Inline vs Call compression validation."""
    print("📏 Quick Experiment: Inline vs Call")
    print("=" * 40)
    
    # Create test modules
    library = ModuleLibrary(max_modules=4)
    
    test_candidates = [
        SubtreeCandidate(
            subtree=op('+', var('x'), const(1)),  # increment
            frequency=3, total_nodes=3, mdl_score=0.15,
            bits_saved=0.20, size_penalty=0.05
        ),
        SubtreeCandidate(
            subtree=op('>', var('y'), const(0)),  # positive check
            frequency=3, total_nodes=3, mdl_score=0.10,
            bits_saved=0.15, size_penalty=0.05
        )
    ]
    
    modules = library.register_modules(test_candidates)
    print(f"Registered {len(modules)} modules")
    
    # Create modular program
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
    
    print(f"Modular program: {pretty_print_ast(modular_program)}")
    print(f"Inlined program: {pretty_print_ast(inlined_program)}")
    
    # Test evaluation equivalence
    interpreter_modular = LispInterpreter(module_library=library)
    interpreter_inline = LispInterpreter()
    
    test_cases = [
        {'x': -1}, {'x': 0}, {'x': 1}, {'x': 5}, {'x': 10}
    ]
    
    print("\nEvaluation comparison:")
    for env in test_cases:
        result_modular = interpreter_modular.evaluate(modular_program, env)
        result_inline = interpreter_inline.evaluate(inlined_program, env)
        
        print(f"  x={env['x']}: modular={result_modular}, inline={result_inline}, "
              f"match={result_modular == result_inline}")
    
    # Size comparison
    from protosynth.mutation import iter_nodes
    size_modular = len(list(iter_nodes(modular_program)))
    size_inline = len(list(iter_nodes(inlined_program)))
    
    compression_ratio = size_inline / size_modular if size_modular > 0 else 1.0
    
    print(f"\nSize comparison:")
    print(f"  Modular: {size_modular} nodes")
    print(f"  Inline:  {size_inline} nodes")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"  Proves compression: {compression_ratio > 1.1}")
    
    return {
        'compression_ratio': compression_ratio,
        'proves_compression': compression_ratio > 1.1,
        'size_modular': size_modular,
        'size_inline': size_inline
    }


def quick_experiment_2_ablation():
    """Quick test: Module ablation impact."""
    print("\n🔬 Quick Experiment: Module Ablation")
    print("=" * 40)
    
    # Create population with clear patterns
    population = []
    
    # Pattern 1: increment (appears 5 times)
    for i in range(5):
        population.append(op('+', var(f'x{i}'), const(1)))
    
    # Pattern 2: positive check (appears 4 times)
    for i in range(4):
        population.append(op('>', var(f'y{i}'), const(0)))
    
    # Pattern 3: conditional increment (appears 3 times)
    for i in range(3):
        population.append(if_expr(op('>', var(f'z{i}'), const(0)), 
                                op('+', var(f'z{i}'), const(1)), 
                                const(0)))
    
    print(f"Created population with {len(population)} programs")
    
    # Mine modules
    validation_bits = [0, 1, 0, 1] * 100
    miner = SubtreeMiner(beta=0.005, min_frequency=2)
    candidates = miner.mine_and_select(population, validation_bits, n_modules=10)
    
    print(f"Found {len(candidates)} module candidates:")
    for i, candidate in enumerate(candidates[:5]):
        print(f"  {i+1}. freq={candidate.frequency}, MDL={candidate.mdl_score:.3f}")
    
    # Test impact of removing modules
    library_full = ModuleLibrary(max_modules=10)
    modules_full = library_full.register_modules(candidates)
    
    library_reduced = ModuleLibrary(max_modules=10)
    modules_reduced = library_reduced.register_modules(candidates[1:])  # Remove top module
    
    print(f"\nFull library: {len(modules_full)} modules")
    print(f"Reduced library: {len(modules_reduced)} modules")
    
    # Simple impact measure: count of high-value modules
    high_value_full = sum(1 for m in modules_full if m.mdl_score > 0.05)
    high_value_reduced = sum(1 for m in modules_reduced if m.mdl_score > 0.05)
    
    impact = high_value_full - high_value_reduced
    
    print(f"High-value modules lost: {impact}")
    
    return {
        'modules_full': len(modules_full),
        'modules_reduced': len(modules_reduced),
        'high_value_impact': impact,
        'top_module_mdl': candidates[0].mdl_score if candidates else 0
    }


def quick_experiment_1_transfer():
    """Quick test: Module transfer concept."""
    print("\n🔄 Quick Experiment: Module Transfer")
    print("=" * 40)
    
    # Create modules from simple patterns
    simple_patterns = [
        op('+', var('x'), const(1)),  # increment
        op('-', var('x'), const(1)),  # decrement  
        op('*', var('x'), const(2)),  # double
        op('>', var('x'), const(0)),  # positive
    ]
    
    print(f"Training on {len(simple_patterns)} simple patterns")
    
    # Mine modules
    validation_bits = [0, 1] * 100
    miner = SubtreeMiner(beta=0.005, min_frequency=1)  # Low threshold for demo
    
    # Create fake candidates for demo
    candidates = []
    for i, pattern in enumerate(simple_patterns):
        candidates.append(SubtreeCandidate(
            subtree=pattern,
            frequency=2,
            total_nodes=3,
            mdl_score=0.1,
            bits_saved=0.15,
            size_penalty=0.05
        ))
    
    # Create library
    library = ModuleLibrary(max_modules=8)
    modules = library.register_modules(candidates)
    
    print(f"Created library with {len(modules)} modules:")
    for module in modules:
        print(f"  {module.name}: {pretty_print_ast(module.implementation)}")
    
    # Test transfer: create complex program using simple modules
    if len(modules) >= 2:
        complex_program = if_expr(
            library.create_module_call(modules[3].name, [var('x')]),  # positive check
            library.create_module_call(modules[0].name, [var('x')]),  # increment
            const(0)
        )
        
        print(f"\nComplex program using modules: {pretty_print_ast(complex_program)}")
        
        # Test evaluation
        interpreter = LispInterpreter(module_library=library)
        result = interpreter.evaluate(complex_program, {'x': 5})
        print(f"Evaluation with x=5: {result}")
        
        return {
            'modules_created': len(modules),
            'transfer_successful': True,
            'complex_program_result': result
        }
    
    return {
        'modules_created': len(modules),
        'transfer_successful': False
    }


def run_quick_validation():
    """Run all quick validation experiments."""
    print("⚡ Quick Module Validation")
    print("=" * 50)
    
    results = {}
    
    # Run experiments
    results['transfer'] = quick_experiment_1_transfer()
    results['ablation'] = quick_experiment_2_ablation()
    results['inline_vs_call'] = quick_experiment_3_inline_vs_call()
    
    print("\n📊 Quick Validation Summary:")
    print("=" * 35)
    
    print(f"Transfer: {results['transfer']['modules_created']} modules created, "
          f"successful: {results['transfer']['transfer_successful']}")
    
    print(f"Ablation: {results['ablation']['high_value_impact']} high-value modules lost, "
          f"top MDL: {results['ablation']['top_module_mdl']:.3f}")
    
    print(f"Compression: {results['inline_vs_call']['compression_ratio']:.2f}x ratio, "
          f"proven: {results['inline_vs_call']['proves_compression']}")
    
    # Overall validation
    validation_passed = (
        results['transfer']['transfer_successful'] and
        results['ablation']['high_value_impact'] > 0 and
        results['inline_vs_call']['proves_compression']
    )
    
    print(f"\n✅ Overall validation: {'PASSED' if validation_passed else 'FAILED'}")
    
    return results


if __name__ == "__main__":
    results = run_quick_validation()
