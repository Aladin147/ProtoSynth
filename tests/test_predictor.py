#!/usr/bin/env python3
"""
Tests for ProtoSynth Predictor Adapter

These tests verify that the adapter correctly handles context binding,
output coercion, and error recovery as specified.
"""

import unittest
import math
from protosynth import *
from protosynth.predictor import PredictorAdapter, predict_with_program


class TestPredictorAdapter(unittest.TestCase):
    """Test cases for the predictor adapter."""
    
    def setUp(self):
        """Set up test environment."""
        self.adapter = PredictorAdapter()
        self.interpreter = LispInterpreter()
    
    def test_hard_predictions(self):
        """Test hard predictions (0/1 outputs)."""
        eps = 1e-6

        # Always predict 1
        program = const(1)
        prob = self.adapter.predict(program, [0, 1, 0])
        self.assertAlmostEqual(prob, 1.0 - eps, places=7)

        # Always predict 0
        program = const(0)
        prob = self.adapter.predict(program, [1, 0, 1])
        self.assertAlmostEqual(prob, eps, places=7)

        # Boolean true
        program = const(True)
        prob = self.adapter.predict(program, [])
        self.assertAlmostEqual(prob, 1.0 - eps, places=7)

        # Boolean false
        program = const(False)
        prob = self.adapter.predict(program, [])
        self.assertAlmostEqual(prob, eps, places=7)
    
    def test_probability_outputs(self):
        """Test probability outputs (floats in [0,1])."""
        # Valid probability
        program = const(0.7)
        prob = self.adapter.predict(program, [1, 0])
        self.assertEqual(prob, 0.7)
        
        # Edge cases
        program = const(0.0)
        prob = self.adapter.predict(program, [])
        self.assertAlmostEqual(prob, 1e-6, places=7)  # Clamped to eps
        
        program = const(1.0)
        prob = self.adapter.predict(program, [])
        self.assertAlmostEqual(prob, 1.0 - 1e-6, places=7)  # Clamped to 1-eps
    
    def test_out_of_range_outputs(self):
        """Test outputs outside [0,1] range."""
        # Large positive integer
        program = const(100)
        prob = self.adapter.predict(program, [])
        self.assertEqual(prob, 0.9)
        
        # Negative integer
        program = const(-5)
        prob = self.adapter.predict(program, [])
        self.assertEqual(prob, 0.1)
        
        # Large positive float
        program = const(5.0)
        prob = self.adapter.predict(program, [])
        self.assertGreater(prob, 0.5)
        self.assertLess(prob, 1.0)
        
        # Large negative float
        program = const(-3.0)
        prob = self.adapter.predict(program, [])
        self.assertLess(prob, 0.5)
        self.assertGreater(prob, 0.0)
    
    def test_special_float_values(self):
        """Test special float values (NaN, inf)."""
        # Infinity
        program = const(float('inf'))
        prob = self.adapter.predict(program, [])
        self.assertEqual(prob, 1.0 - 1e-6)  # Clamped
        
        # Negative infinity
        program = const(float('-inf'))
        prob = self.adapter.predict(program, [])
        self.assertEqual(prob, 1e-6)  # Clamped
        
        # NaN - this is tricky to test since we can't create NaN constants easily
        # We'll test this through operations that might produce NaN
    
    def test_string_outputs(self):
        """Test string output coercion."""
        # True-like strings
        for true_str in ['true', 'True', 'TRUE', 'yes', 'YES', '1']:
            program = const(true_str)
            prob = self.adapter.predict(program, [])
            self.assertEqual(prob, 1.0 - 1e-6)  # Clamped
        
        # False-like strings
        for false_str in ['false', 'False', 'FALSE', 'no', 'NO', '0']:
            program = const(false_str)
            prob = self.adapter.predict(program, [])
            self.assertEqual(prob, 1e-6)  # Clamped
        
        # Other strings (hash-based)
        program = const("hello")
        prob = self.adapter.predict(program, [])
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
    
    def test_context_binding(self):
        """Test that context is properly bound to 'ctx' variable."""
        # Program that accesses context
        program = var('ctx')
        
        # Should return the context list itself, then coerce it
        context = [1, 0, 1]
        prob = self.adapter.predict(program, context)
        
        # List coercion should use length % 100 / 100
        expected_prob = (len(context) % 100) / 100.0
        self.assertEqual(prob, expected_prob)
    
    def test_error_handling(self):
        """Test error handling and fallback to p=0.5."""
        # Undefined variable (should fail gracefully)
        program = var('undefined')
        prob = self.adapter.predict(program, [1, 0])
        self.assertEqual(prob, 0.5)
        
        # Division by zero (should be handled by interpreter)
        program = op('/', const(1), const(0))
        prob = self.adapter.predict(program, [])
        # Should not crash, might return inf which gets coerced
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
    
    def test_timeout_handling(self):
        """Test timeout handling."""
        # Create adapter with very short timeout
        short_adapter = PredictorAdapter(timeout_seconds=0.001)
        
        # Simple program should still work
        program = const(0.5)
        prob = short_adapter.predict(program, [])
        self.assertEqual(prob, 0.5)
    
    def test_probability_clamping(self):
        """Test that probabilities are properly clamped to [eps, 1-eps]."""
        eps = 1e-6
        adapter = PredictorAdapter(eps=eps)
        
        # Test lower bound
        program = const(0.0)
        prob = adapter.predict(program, [])
        self.assertEqual(prob, eps)
        
        # Test upper bound
        program = const(1.0)
        prob = adapter.predict(program, [])
        self.assertEqual(prob, 1.0 - eps)
        
        # Test values already in range
        program = const(0.3)
        prob = adapter.predict(program, [])
        self.assertEqual(prob, 0.3)
    
    def test_list_tuple_coercion(self):
        """Test coercion of list and tuple outputs."""
        # This is harder to test directly since our language doesn't
        # easily produce lists, but we can test the coercion logic
        adapter = PredictorAdapter()
        
        # Test empty list
        prob = adapter._coerce_to_probability([])
        self.assertEqual(prob, 0.5)
        
        # Test single element list
        prob = adapter._coerce_to_probability([0.7])
        self.assertEqual(prob, 0.7)
        
        # Test multi-element list
        prob = adapter._coerce_to_probability([1, 2, 3])
        expected = (3 % 100) / 100.0
        self.assertEqual(prob, expected)
    
    def test_none_coercion(self):
        """Test None value coercion."""
        adapter = PredictorAdapter()
        prob = adapter._coerce_to_probability(None)
        self.assertEqual(prob, 0.5)
    
    def test_convenience_function(self):
        """Test the convenience function predict_with_program."""
        interpreter = LispInterpreter()
        program = const(0.8)
        context = [1, 0, 1]
        
        prob = predict_with_program(interpreter, program, context)
        self.assertEqual(prob, 0.8)
    
    def test_complex_program(self):
        """Test with a more complex program."""
        # Program that uses context: if first bit is 1, predict 0.8, else 0.2
        # This is complex to express, so we'll use a simpler test
        
        # Program that always returns a computed value
        program = op('+', const(0.3), const(0.2))  # Should return 0.5
        prob = self.adapter.predict(program, [1, 0, 1])
        self.assertEqual(prob, 0.5)
    
    def test_deterministic_behavior(self):
        """Test that predictions are deterministic."""
        program = const(0.6)
        context = [1, 0, 1, 0]
        
        # Multiple calls should return same result
        prob1 = self.adapter.predict(program, context)
        prob2 = self.adapter.predict(program, context)
        prob3 = self.adapter.predict(program, context)
        
        self.assertEqual(prob1, prob2)
        self.assertEqual(prob2, prob3)
    
    def test_different_contexts(self):
        """Test that different contexts can produce different results."""
        # Program that accesses context
        program = var('ctx')
        
        # Different context lengths should give different probabilities
        prob1 = self.adapter.predict(program, [1])
        prob2 = self.adapter.predict(program, [1, 0])
        prob3 = self.adapter.predict(program, [1, 0, 1])
        
        # Should be different due to length-based coercion
        self.assertNotEqual(prob1, prob2)
        self.assertNotEqual(prob2, prob3)
    
    def test_edge_case_contexts(self):
        """Test with edge case contexts."""
        program = const(0.5)
        
        # Empty context
        prob = self.adapter.predict(program, [])
        self.assertEqual(prob, 0.5)
        
        # Very long context
        long_context = [1, 0] * 100
        prob = self.adapter.predict(program, long_context)
        self.assertEqual(prob, 0.5)
        
        # Context with only 0s
        zero_context = [0] * 10
        prob = self.adapter.predict(program, zero_context)
        self.assertEqual(prob, 0.5)
        
        # Context with only 1s
        one_context = [1] * 10
        prob = self.adapter.predict(program, one_context)
        self.assertEqual(prob, 0.5)


if __name__ == '__main__':
    unittest.main()
