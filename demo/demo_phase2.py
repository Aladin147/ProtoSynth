#!/usr/bin/env python3
"""
ProtoSynth Phase 2 Demo

Demonstrates the complete mutation and verification system with
configuration and logging.
"""

from protosynth import *
from protosynth.config import ProtoSynthConfig, MutationConfig, setup_logging
import logging


def demo_mutation_system():
    """Demonstrate the mutation system with logging."""
    
    print("🧬 ProtoSynth Phase 2 Demo - Mutation & Verification")
    print("=" * 60)
    
    # Setup logging to see what's happening
    setup_logging("DEBUG")
    
    # Create a test program
    program = let('x', const(10),
                  if_expr(op('>', var('x'), const(5)),
                          op('+', var('x'), const(1)),
                          const(0)))
    
    print(f"\nOriginal program: {pretty_print_ast(program)}")
    
    # Create agent with custom config
    config = ProtoSynthConfig()
    config.mutation.mutation_rate = 0.8  # High mutation rate for demo
    config.mutation.max_mutation_attempts = 5
    
    agent = SelfModifyingAgent(
        program, 
        mutation_rate=config.mutation.mutation_rate,
        max_mutation_attempts=config.mutation.max_mutation_attempts
    )
    
    print(f"Agent verification: {agent.verify()}")
    print(f"Original evaluation: {agent.evaluate()}")
    
    # Try several mutations
    print("\n🔬 Attempting mutations...")
    for i in range(5):
        try:
            mutated_agent = agent.mutate()
            mutated_code = mutated_agent.get_code_string()
            
            print(f"\nMutation {i+1} (Gen {mutated_agent.generation}):")
            print(f"  Code: {mutated_code}")
            print(f"  Verified: {mutated_agent.verify()}")
            
            try:
                result = mutated_agent.evaluate()
                print(f"  Result: {result}")
            except Exception as e:
                print(f"  Evaluation failed: {e}")
            
            # Use mutated agent for next iteration to show evolution
            agent = mutated_agent
            
        except RuntimeError as e:
            print(f"Mutation {i+1} failed: {e}")


def demo_verification_system():
    """Demonstrate the verification system."""
    
    print("\n\n🔍 Verification System Demo")
    print("=" * 40)
    
    from protosynth.verify import verify_ast
    
    # Valid ASTs
    valid_asts = [
        const(42),
        op('+', const(1), const(2)),
        let('x', const(5), var('x'))
    ]
    
    print("\n✅ Valid ASTs:")
    for ast in valid_asts:
        is_valid, errors = verify_ast(ast)
        print(f"  {pretty_print_ast(ast)}: {'PASS' if is_valid else 'FAIL'}")
        if errors:
            print(f"    Errors: {errors}")
    
    # Invalid ASTs
    invalid_asts = [
        op('+', const(1)),  # Wrong arity
        var('undefined'),   # Unbound variable
        LispNode('unknown', 'value')  # Unknown node type
    ]
    
    print("\n❌ Invalid ASTs:")
    for ast in invalid_asts:
        is_valid, errors = verify_ast(ast)
        print(f"  {pretty_print_ast(ast) if hasattr(ast, 'node_type') and ast.node_type in ['const', 'var', 'op', 'let', 'if'] else str(ast)}: {'PASS' if is_valid else 'FAIL'}")
        if errors:
            print(f"    Errors: {errors}")


def demo_config_system():
    """Demonstrate the configuration system."""
    
    print("\n\n⚙️  Configuration System Demo")
    print("=" * 40)
    
    from protosynth.config import get_config, set_config, ProtoSynthConfig, MutationConfig
    
    # Show default config
    config = get_config()
    print(f"\nDefault configuration:")
    print(f"  Mutation rate: {config.mutation.mutation_rate}")
    print(f"  Const perturb delta: {config.mutation.const_perturb_delta}")
    print(f"  Max mutation attempts: {config.mutation.max_mutation_attempts}")
    print(f"  Max AST depth: {config.verifier.max_depth}")
    print(f"  Max nodes: {config.verifier.max_nodes}")
    
    # Create custom config
    custom_config = ProtoSynthConfig()
    custom_config.mutation = MutationConfig(
        mutation_rate=0.5,
        const_perturb_delta=20,
        max_mutation_attempts=15
    )
    
    set_config(custom_config)
    
    print(f"\nCustom configuration:")
    new_config = get_config()
    print(f"  Mutation rate: {new_config.mutation.mutation_rate}")
    print(f"  Const perturb delta: {new_config.mutation.const_perturb_delta}")
    print(f"  Max mutation attempts: {new_config.mutation.max_mutation_attempts}")


if __name__ == "__main__":
    demo_mutation_system()
    demo_verification_system()
    demo_config_system()
    
    print("\n\n✨ Phase 2 Demo Complete!")
    print("ProtoSynth now has a complete mutation and verification system.")
    print("Ready for Phase 3: Fitness Evaluation & Evolution!")
