#!/usr/bin/env python3
"""
ProtoSynth Demo

This script demonstrates the basic functionality of the ProtoSynth interpreter
with simple example programs.
"""

from protosynth import (
    LispInterpreter, SelfModifyingAgent,
    const, var, let, if_expr, op, pretty_print_ast
)


def demo_basic_examples():
    """Demonstrate basic interpreter functionality with example programs."""
    
    print("🧠 ProtoSynth Demo - Basic Examples")
    print("=" * 50)
    
    interpreter = LispInterpreter()
    
    # Example 1: Constant math
    print("\n1. Constant Math: (+ 3 5)")
    ast1 = op('+', const(3), const(5))
    result1 = interpreter.evaluate(ast1)
    print(f"   AST: {pretty_print_ast(ast1)}")
    print(f"   Result: {result1}")
    
    # Example 2: Nested arithmetic
    print("\n2. Nested Arithmetic: (* (+ 2 3) (- 10 4))")
    ast2 = op('*', 
              op('+', const(2), const(3)),
              op('-', const(10), const(4)))
    result2 = interpreter.evaluate(ast2)
    print(f"   AST: {pretty_print_ast(ast2)}")
    print(f"   Result: {result2}")
    
    # Example 3: Let binding with variables
    print("\n3. Let Binding: (let x 7 (* x x))")
    ast3 = let('x', const(7), op('*', var('x'), var('x')))
    result3 = interpreter.evaluate(ast3)
    print(f"   AST: {pretty_print_ast(ast3)}")
    print(f"   Result: {result3}")
    
    # Example 4: Nested let bindings
    print("\n4. Nested Let: (let a 3 (let b 4 (+ a b)))")
    ast4 = let('a', const(3),
               let('b', const(4),
                   op('+', var('a'), var('b'))))
    result4 = interpreter.evaluate(ast4)
    print(f"   AST: {pretty_print_ast(ast4)}")
    print(f"   Result: {result4}")
    
    # Example 5: Conditional using if
    print("\n5. Conditional: (if (< 5 10) 'yes' 'no')")
    ast5 = if_expr(op('<', const(5), const(10)),
                   const('yes'),
                   const('no'))
    result5 = interpreter.evaluate(ast5)
    print(f"   AST: {pretty_print_ast(ast5)}")
    print(f"   Result: {result5}")
    
    # Example 6: Boolean operations
    print("\n6. Boolean Logic: (and true (not false))")
    ast6 = op('and', const(True), op('not', const(False)))
    result6 = interpreter.evaluate(ast6)
    print(f"   AST: {pretty_print_ast(ast6)}")
    print(f"   Result: {result6}")
    
    # Example 7: Complex conditional with let
    print("\n7. Complex: (let x 15 (if (>= x 10) (* x 2) (+ x 5)))")
    ast7 = let('x', const(15),
               if_expr(op('>=', var('x'), const(10)),
                       op('*', var('x'), const(2)),
                       op('+', var('x'), const(5))))
    result7 = interpreter.evaluate(ast7)
    print(f"   AST: {pretty_print_ast(ast7)}")
    print(f"   Result: {result7}")


def demo_self_modifying_agent():
    """Demonstrate the SelfModifyingAgent functionality."""
    
    print("\n\n🤖 ProtoSynth Demo - Self-Modifying Agent")
    print("=" * 50)
    
    # Create a simple program
    program = op('+', const(10), const(5))
    print(f"\nInitial program: {pretty_print_ast(program)}")
    
    # Create an agent with this program
    agent = SelfModifyingAgent(program)
    print(f"Agent: {agent}")
    
    # Evaluate the program
    result = agent.evaluate()
    print(f"Evaluation result: {result}")
    
    # Check verification
    is_valid = agent.verify()
    print(f"Verification passed: {is_valid}")
    
    # Get fitness score
    fitness = agent.get_fitness()
    print(f"Fitness score: {fitness:.2f}")
    
    # Create a mutated version (placeholder for now)
    mutated_agent = agent.mutate()
    print(f"Mutated agent: {mutated_agent}")
    
    # Show self-inspection capability
    interpreter = LispInterpreter()
    self_ast = interpreter.get_self_ast()
    print(f"\nInterpreter self-inspection: {pretty_print_ast(self_ast)}")


def demo_resource_limits():
    """Demonstrate resource limit enforcement."""
    
    print("\n\n⚡ ProtoSynth Demo - Resource Limits")
    print("=" * 50)
    
    # Create an interpreter with strict limits
    strict_interpreter = LispInterpreter(max_recursion_depth=3, max_steps=10, timeout_seconds=0.1)
    
    # Test recursion depth limit
    print("\n1. Testing recursion depth limit...")
    deep_ast = let('x', const(1),
                   let('y', const(2),
                       let('z', const(3),
                           let('w', const(4),  # This should exceed depth limit
                               op('+', var('x'), var('w'))))))
    
    try:
        result = strict_interpreter.evaluate(deep_ast)
        print(f"   Unexpected success: {result}")
    except RuntimeError as e:
        print(f"   ✅ Caught expected error: {e}")
    
    # Test step count limit
    print("\n2. Testing step count limit...")
    many_ops_ast = op('+',
                      op('+', op('+', const(1), const(2)), op('+', const(3), const(4))),
                      op('+', op('+', const(5), const(6)), op('+', const(7), const(8))))
    
    try:
        result = strict_interpreter.evaluate(many_ops_ast)
        print(f"   Result: {result}")
    except RuntimeError as e:
        print(f"   ✅ Caught expected error: {e}")


if __name__ == "__main__":
    demo_basic_examples()
    demo_self_modifying_agent()
    demo_resource_limits()
    
    print("\n\n✨ Demo completed successfully!")
    print("ProtoSynth Phase 1 foundation is ready for development.")
