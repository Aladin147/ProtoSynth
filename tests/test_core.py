#!/usr/bin/env python3
"""
Unit tests for ProtoSynth core functionality.

These tests verify the basic interpreter operations, AST manipulation,
and resource control features.
"""

import unittest
import time
from protosynth import (
    LispNode, LispInterpreter, SelfModifyingAgent,
    const, var, let, if_expr, op, pretty_print_ast, clone_ast
)


class TestLispInterpreter(unittest.TestCase):
    """Test cases for the LispInterpreter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.interpreter = LispInterpreter()
    
    def test_constant_evaluation(self):
        """Test evaluation of constant values."""
        # Numbers
        self.assertEqual(self.interpreter.evaluate(const(42)), 42)
        self.assertEqual(self.interpreter.evaluate(const(3.14)), 3.14)
        
        # Booleans
        self.assertEqual(self.interpreter.evaluate(const(True)), True)
        self.assertEqual(self.interpreter.evaluate(const(False)), False)
        
        # Strings
        self.assertEqual(self.interpreter.evaluate(const("hello")), "hello")
    
    def test_arithmetic_operations(self):
        """Test basic arithmetic operations."""
        # Addition
        result = self.interpreter.evaluate(op('+', const(3), const(5)))
        self.assertEqual(result, 8)
        
        # Subtraction
        result = self.interpreter.evaluate(op('-', const(10), const(4)))
        self.assertEqual(result, 6)
        
        # Multiplication
        result = self.interpreter.evaluate(op('*', const(7), const(6)))
        self.assertEqual(result, 42)
        
        # Division
        result = self.interpreter.evaluate(op('/', const(15), const(3)))
        self.assertEqual(result, 5)
        
        # Modulo
        result = self.interpreter.evaluate(op('%', const(17), const(5)))
        self.assertEqual(result, 2)
    
    def test_comparison_operations(self):
        """Test comparison operations."""
        # Equality
        self.assertTrue(self.interpreter.evaluate(op('==', const(5), const(5))))
        self.assertFalse(self.interpreter.evaluate(op('==', const(5), const(3))))
        
        # Inequality
        self.assertTrue(self.interpreter.evaluate(op('!=', const(5), const(3))))
        self.assertFalse(self.interpreter.evaluate(op('!=', const(5), const(5))))
        
        # Less than
        self.assertTrue(self.interpreter.evaluate(op('<', const(3), const(5))))
        self.assertFalse(self.interpreter.evaluate(op('<', const(5), const(3))))
        
        # Less than or equal
        self.assertTrue(self.interpreter.evaluate(op('<=', const(3), const(5))))
        self.assertTrue(self.interpreter.evaluate(op('<=', const(5), const(5))))
        
        # Greater than
        self.assertTrue(self.interpreter.evaluate(op('>', const(5), const(3))))
        self.assertFalse(self.interpreter.evaluate(op('>', const(3), const(5))))
        
        # Greater than or equal
        self.assertTrue(self.interpreter.evaluate(op('>=', const(5), const(3))))
        self.assertTrue(self.interpreter.evaluate(op('>=', const(5), const(5))))
    
    def test_boolean_operations(self):
        """Test boolean operations."""
        # AND
        self.assertTrue(self.interpreter.evaluate(op('and', const(True), const(True))))
        self.assertFalse(self.interpreter.evaluate(op('and', const(True), const(False))))
        
        # OR
        self.assertTrue(self.interpreter.evaluate(op('or', const(True), const(False))))
        self.assertFalse(self.interpreter.evaluate(op('or', const(False), const(False))))
        
        # NOT
        self.assertFalse(self.interpreter.evaluate(op('not', const(True))))
        self.assertTrue(self.interpreter.evaluate(op('not', const(False))))
    
    def test_let_bindings(self):
        """Test let variable bindings."""
        # Simple let
        ast = let('x', const(10), var('x'))
        result = self.interpreter.evaluate(ast)
        self.assertEqual(result, 10)
        
        # Let with computation
        ast = let('x', const(5), op('*', var('x'), const(2)))
        result = self.interpreter.evaluate(ast)
        self.assertEqual(result, 10)
        
        # Nested let bindings
        ast = let('a', const(3),
                  let('b', const(4),
                      op('+', var('a'), var('b'))))
        result = self.interpreter.evaluate(ast)
        self.assertEqual(result, 7)
    
    def test_if_expressions(self):
        """Test conditional if expressions."""
        # True condition
        ast = if_expr(const(True), const('yes'), const('no'))
        result = self.interpreter.evaluate(ast)
        self.assertEqual(result, 'yes')
        
        # False condition
        ast = if_expr(const(False), const('yes'), const('no'))
        result = self.interpreter.evaluate(ast)
        self.assertEqual(result, 'no')
        
        # Computed condition
        ast = if_expr(op('<', const(5), const(10)), const(42), const(0))
        result = self.interpreter.evaluate(ast)
        self.assertEqual(result, 42)
    
    def test_resource_limits(self):
        """Test resource limit enforcement."""
        # Test recursion depth limit
        strict_interpreter = LispInterpreter(max_recursion_depth=2)
        deep_ast = let('x', const(1),
                       let('y', const(2),
                           let('z', const(3), var('z'))))  # Depth 3, should fail
        
        with self.assertRaises(RuntimeError):
            strict_interpreter.evaluate(deep_ast)
        
        # Test step count limit
        step_limited = LispInterpreter(max_steps=5)
        complex_ast = op('+',
                         op('+', const(1), const(2)),
                         op('+', const(3), const(4)))
        
        with self.assertRaises(RuntimeError):
            step_limited.evaluate(complex_ast)


class TestSerialization(unittest.TestCase):
    """Test cases for AST serialization functions."""
    
    def test_pretty_print_ast(self):
        """Test pretty printing of AST nodes."""
        # Constants
        self.assertEqual(pretty_print_ast(const(42)), "42")
        self.assertEqual(pretty_print_ast(const(True)), "true")
        self.assertEqual(pretty_print_ast(const("hello")), '"hello"')
        
        # Variables
        self.assertEqual(pretty_print_ast(var('x')), "x")
        
        # Operations
        ast = op('+', const(3), const(5))
        self.assertEqual(pretty_print_ast(ast), "(+ 3 5)")
        
        # Let expressions
        ast = let('x', const(10), var('x'))
        self.assertEqual(pretty_print_ast(ast), "(let x 10 x)")
        
        # If expressions
        ast = if_expr(const(True), const(1), const(0))
        self.assertEqual(pretty_print_ast(ast), "(if true 1 0)")
    
    def test_clone_ast(self):
        """Test deep cloning of AST nodes."""
        original = op('+', const(3), var('x'))
        cloned = clone_ast(original)
        
        # Should be equal but not the same object
        self.assertEqual(original.node_type, cloned.node_type)
        self.assertEqual(original.value, cloned.value)
        self.assertIsNot(original, cloned)
        self.assertIsNot(original.children[0], cloned.children[0])


class TestSelfModifyingAgent(unittest.TestCase):
    """Test cases for the SelfModifyingAgent class."""
    
    def test_agent_creation(self):
        """Test agent creation and basic functionality."""
        program = op('+', const(10), const(5))
        agent = SelfModifyingAgent(program)
        
        # Test evaluation
        result = agent.evaluate()
        self.assertEqual(result, 15)
        
        # Test verification
        self.assertTrue(agent.verify())
        
        # Test fitness calculation
        fitness = agent.get_fitness()
        self.assertIsInstance(fitness, float)
        self.assertGreater(fitness, 0)
        
        # Test code string representation
        code_str = agent.get_code_string()
        self.assertEqual(code_str, "(+ 10 5)")
    
    def test_agent_mutation(self):
        """Test agent mutation (placeholder functionality)."""
        program = const(42)
        agent = SelfModifyingAgent(program)
        
        mutated = agent.mutate()
        self.assertEqual(mutated.generation, agent.generation + 1)
        self.assertIsNot(agent, mutated)


if __name__ == '__main__':
    unittest.main()
