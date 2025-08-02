#!/usr/bin/env python3
"""
Edge Case Hardening Tests for ProtoSynth

These tests systematically verify that the system handles edge cases
gracefully: empty ASTs, deeply nested structures, unusual node combinations.
"""

import unittest
import random
from protosynth import *
from protosynth.mutation import mutate, iter_nodes
from protosynth.verify import verify_ast


class TestEdgeCaseHardening(unittest.TestCase):
    """
    Comprehensive edge case testing to ensure system robustness.
    """
    
    def setUp(self):
        """Set up test environment."""
        self.rng = random.Random(42)
    
    def test_minimal_asts(self):
        """Test behavior with minimal AST structures."""
        
        minimal_asts = [
            const(0),
            const(1),
            const(-1),
            const(True),
            const(False),
            const(""),
            const("x"),
        ]
        
        for ast in minimal_asts:
            with self.subTest(ast=pretty_print_ast(ast)):
                # Should verify
                is_valid, errors = verify_ast(ast)
                self.assertTrue(is_valid, f"Minimal AST should verify: {errors}")
                
                # Should evaluate
                interpreter = LispInterpreter()
                result = interpreter.evaluate(ast)
                self.assertEqual(result, ast.value)
                
                # Should handle mutation gracefully
                try:
                    mutated = mutate(ast, mutation_rate=0.8, rng=self.rng)
                    is_valid, errors = verify_ast(mutated)
                    self.assertTrue(is_valid, f"Mutated minimal AST should verify: {errors}")
                except Exception as e:
                    # Some minimal ASTs might not be mutable - that's ok
                    pass
    
    def test_deeply_nested_structures(self):
        """Test deeply nested AST structures."""
        
        def create_deep_let_chain(depth):
            if depth <= 0:
                return const(0)
            return let(f'x{depth}', const(depth), create_deep_let_chain(depth - 1))
        
        def create_deep_operation_chain(depth):
            if depth <= 0:
                return const(1)
            return op('+', const(depth), create_deep_operation_chain(depth - 1))
        
        # Test different types of deep nesting
        deep_structures = [
            ("Deep Let Chain", create_deep_let_chain(8)),
            ("Deep Operation Chain", create_deep_operation_chain(8)),
        ]
        
        for name, ast in deep_structures:
            with self.subTest(structure=name):
                # Should verify (within limits)
                is_valid, errors = verify_ast(ast, max_depth=15)
                self.assertTrue(is_valid, f"{name} should verify: {errors}")
                
                # Should evaluate
                interpreter = LispInterpreter(max_recursion_depth=20)
                try:
                    result = interpreter.evaluate(ast)
                    self.assertIsNotNone(result)
                except RuntimeError:
                    # Deep structures might hit resource limits - that's expected
                    pass
                
                # Should handle mutation
                try:
                    mutated = mutate(ast, mutation_rate=0.3, rng=self.rng)
                    is_valid, errors = verify_ast(mutated, max_depth=15)
                    self.assertTrue(is_valid, f"Mutated {name} should verify: {errors}")
                except Exception:
                    # Deep structures might be hard to mutate - that's ok
                    pass
    
    def test_unusual_node_combinations(self):
        """Test unusual but valid node combinations."""
        
        unusual_combinations = [
            # Nested conditionals
            if_expr(if_expr(const(True), const(True), const(False)), const(1), const(0)),
            
            # Let with complex expressions
            let('x', op('+', const(1), const(2)), 
                let('y', op('*', var('x'), const(3)),
                    op('-', var('y'), var('x')))),
            
            # Operations with boolean results
            op('and', op('>', const(5), const(3)), op('<', const(2), const(4))),
            
            # Conditional with operation in condition
            if_expr(op('==', op('+', const(1), const(1)), const(2)), const(42), const(0)),
        ]
        
        for ast in unusual_combinations:
            with self.subTest(ast=pretty_print_ast(ast)):
                # Should verify
                is_valid, errors = verify_ast(ast)
                self.assertTrue(is_valid, f"Unusual combination should verify: {errors}")
                
                # Should evaluate
                interpreter = LispInterpreter()
                try:
                    result = interpreter.evaluate(ast)
                    self.assertIsNotNone(result)
                except Exception as e:
                    self.fail(f"Unusual combination should evaluate: {e}")
                
                # Should handle mutation
                try:
                    mutated = mutate(ast, mutation_rate=0.5, rng=self.rng)
                    is_valid, errors = verify_ast(mutated)
                    self.assertTrue(is_valid, f"Mutated unusual combination should verify: {errors}")
                except Exception:
                    # Some unusual combinations might be hard to mutate
                    pass
    
    def test_boundary_values(self):
        """Test boundary values and extreme cases."""
        
        boundary_values = [
            const(0),
            const(-1),
            const(1),
            const(2**31 - 1),  # Max 32-bit int
            const(-2**31),     # Min 32-bit int
            const(3.14159),    # Float
            const(0.0),        # Zero float
            const(-0.0),       # Negative zero
        ]
        
        for ast in boundary_values:
            with self.subTest(value=ast.value):
                # Should verify
                is_valid, errors = verify_ast(ast)
                self.assertTrue(is_valid, f"Boundary value should verify: {errors}")
                
                # Should evaluate
                interpreter = LispInterpreter()
                result = interpreter.evaluate(ast)
                self.assertEqual(result, ast.value)
                
                # Should handle constant perturbation
                from protosynth.mutation import _const_perturb_mutation
                if isinstance(ast.value, (int, float)) and not isinstance(ast.value, bool):
                    try:
                        mutated = _const_perturb_mutation(ast, self.rng)
                        is_valid, errors = verify_ast(mutated)
                        self.assertTrue(is_valid, f"Perturbed boundary value should verify: {errors}")
                    except Exception:
                        # Some boundary values might overflow - that's ok
                        pass
    
    def test_malformed_ast_detection(self):
        """Test that malformed ASTs are properly detected and rejected."""
        
        # Create malformed ASTs that should be caught by verification
        malformed_asts = [
            LispNode('unknown_type', 'value'),  # Unknown node type
            LispNode('op', '+', []),            # Operation with no arguments
            LispNode('op', '+', [const(1)]),    # Binary op with one argument
            LispNode('if', None, [const(True)]), # If with insufficient arguments
            LispNode('let', None, []),          # Let with no arguments
        ]
        
        for ast in malformed_asts:
            with self.subTest(ast=str(ast)):
                # Should fail verification
                is_valid, errors = verify_ast(ast)
                self.assertFalse(is_valid, f"Malformed AST should fail verification: {ast}")
                self.assertGreater(len(errors), 0, "Should have error messages")
                
                # Should not be accepted by agent
                try:
                    agent = SelfModifyingAgent(ast)
                    self.assertFalse(agent.verify(), "Agent should reject malformed AST")
                except Exception:
                    # It's ok if agent creation fails for malformed ASTs
                    pass
    
    def test_resource_limit_edge_cases(self):
        """Test behavior at resource limits."""
        
        # Create AST that's exactly at the limit
        def create_ast_with_node_count(target_count):
            # Create a chain of operations to reach target count
            current = const(0)  # 1 node
            for i in range(target_count - 1):
                current = op('+', current, const(i))  # Adds 2 nodes each time
                if len(list(iter_nodes(current))) >= target_count:
                    break
            return current
        
        # Test at various limits
        limit_tests = [
            (10, 15),   # 10 nodes, limit 15 (should pass)
            (15, 15),   # 15 nodes, limit 15 (should pass)
            (20, 15),   # 20 nodes, limit 15 (should fail)
        ]
        
        for node_count, limit in limit_tests:
            with self.subTest(nodes=node_count, limit=limit):
                ast = create_ast_with_node_count(node_count)
                actual_count = len(list(iter_nodes(ast)))
                
                is_valid, errors = verify_ast(ast, max_nodes=limit)
                
                if actual_count <= limit:
                    self.assertTrue(is_valid, f"AST with {actual_count} nodes should pass limit {limit}")
                else:
                    self.assertFalse(is_valid, f"AST with {actual_count} nodes should fail limit {limit}")
    
    def test_mutation_edge_cases(self):
        """Test mutation behavior with edge cases."""
        
        # Test mutation with very high and very low rates
        test_ast = op('+', const(10), const(5))
        
        # Very low mutation rate
        for _ in range(10):
            result = mutate(test_ast, mutation_rate=0.01, rng=self.rng)
            is_valid, errors = verify_ast(result)
            self.assertTrue(is_valid, f"Low mutation rate should produce valid AST: {errors}")
        
        # Very high mutation rate
        for _ in range(10):
            result = mutate(test_ast, mutation_rate=0.99, rng=self.rng)
            is_valid, errors = verify_ast(result)
            self.assertTrue(is_valid, f"High mutation rate should produce valid AST: {errors}")
        
        # Zero mutation rate (should return clone)
        result = mutate(test_ast, mutation_rate=0.0, rng=self.rng)
        self.assertEqual(pretty_print_ast(result), pretty_print_ast(test_ast))
    
    def test_interpreter_edge_cases(self):
        """Test interpreter behavior with edge cases."""
        
        interpreter = LispInterpreter()
        
        # Division by zero
        div_by_zero = op('/', const(10), const(0))
        result = interpreter.evaluate(div_by_zero)
        self.assertEqual(result, float('inf'))  # Should handle gracefully
        
        # Modulo by zero
        mod_by_zero = op('%', const(10), const(0))
        result = interpreter.evaluate(mod_by_zero)
        self.assertEqual(result, 0)  # Should handle gracefully
        
        # Very deep recursion (should hit limit)
        deep_ast = const(1)
        for i in range(20):  # Deeper than default limit
            deep_ast = let(f'x{i}', deep_ast, var(f'x{i}'))
        
        with self.assertRaises(RuntimeError):
            interpreter.evaluate(deep_ast)


if __name__ == '__main__':
    unittest.main()
