#!/usr/bin/env python3
"""
Tests for ProtoSynth Evaluation System

These tests verify the correctness of cross-entropy calculations,
baseline entropy, and fitness evaluation with analytic validation.
"""

import unittest
import math
import itertools
from protosynth import *
from protosynth.eval import *
from protosynth.envs import periodic, constant


class TestEvaluationSystem(unittest.TestCase):
    """Test cases for the evaluation system."""
    
    def setUp(self):
        """Set up test environment."""
        self.interpreter = LispInterpreter()
        self.eps = 1e-6
    
    def test_cross_entropy_bits(self):
        """Test cross-entropy calculation for individual predictions."""
        # Perfect prediction (y=1, p=1)
        ce = cross_entropy_bits(1, 1.0 - self.eps)
        self.assertAlmostEqual(ce, 0.0, places=5)
        
        # Perfect prediction (y=0, p=0)
        ce = cross_entropy_bits(0, self.eps)
        self.assertAlmostEqual(ce, 0.0, places=5)
        
        # Random prediction (y=1, p=0.5)
        ce = cross_entropy_bits(1, 0.5)
        self.assertAlmostEqual(ce, 1.0, places=6)
        
        # Random prediction (y=0, p=0.5)
        ce = cross_entropy_bits(0, 0.5)
        self.assertAlmostEqual(ce, 1.0, places=6)
        
        # Worst prediction (y=1, p=0)
        ce = cross_entropy_bits(1, self.eps)
        expected = -math.log2(self.eps)
        self.assertAlmostEqual(ce, expected, places=1)
        
        # Worst prediction (y=0, p=1)
        ce = cross_entropy_bits(0, 1.0 - self.eps)
        expected = -math.log2(self.eps)
        self.assertAlmostEqual(ce, expected, places=1)
    
    def test_baseline_entropy_bits(self):
        """Test baseline entropy calculation."""
        # Balanced stream (50% 1s)
        h0 = baseline_entropy_bits(0.5)
        self.assertAlmostEqual(h0, 1.0, places=6)
        
        # Constant stream (100% 1s)
        h0 = baseline_entropy_bits(1.0 - self.eps)
        self.assertLess(h0, 0.001)  # Should be very small

        # Constant stream (0% 1s)
        h0 = baseline_entropy_bits(self.eps)
        self.assertLess(h0, 0.001)  # Should be very small
        
        # Biased stream (90% 1s)
        h0 = baseline_entropy_bits(0.9)
        expected = -(0.9 * math.log2(0.9) + 0.1 * math.log2(0.1))
        self.assertAlmostEqual(h0, expected, places=6)
        
        # Biased stream (10% 1s)
        h0 = baseline_entropy_bits(0.1)
        expected = -(0.1 * math.log2(0.1) + 0.9 * math.log2(0.9))
        self.assertAlmostEqual(h0, expected, places=6)
    
    def test_perfect_predictor_fitness(self):
        """Test fitness calculation for perfect predictor."""
        # Balanced stream
        bits = [0, 1, 0, 1, 0, 1]
        fitness = perfect_predictor_fitness(bits)
        expected_h0 = baseline_entropy_bits(0.5)
        self.assertAlmostEqual(fitness, expected_h0, places=6)
        
        # Biased stream
        bits = [1, 1, 1, 0, 1, 1, 1, 1, 1, 0]  # 80% 1s
        fitness = perfect_predictor_fitness(bits)
        expected_h0 = baseline_entropy_bits(0.8)
        self.assertAlmostEqual(fitness, expected_h0, places=6)
        
        # Constant stream
        bits = [1] * 10
        fitness = perfect_predictor_fitness(bits)
        self.assertLess(fitness, 0.001)  # Should be very small
    
    def test_random_predictor_fitness(self):
        """Test fitness calculation for random predictor."""
        # Balanced stream (H0 = 1.0, Hprog = 1.0, F = 0.0)
        bits = [0, 1] * 50
        fitness = random_predictor_fitness(bits)
        self.assertAlmostEqual(fitness, 0.0, places=6)
        
        # Biased stream (H0 < 1.0, Hprog = 1.0, F < 0)
        bits = [1] * 80 + [0] * 20  # 80% 1s
        fitness = random_predictor_fitness(bits)
        expected_h0 = baseline_entropy_bits(0.8)
        expected_fitness = expected_h0 - 1.0
        self.assertAlmostEqual(fitness, expected_fitness, places=6)
        self.assertLess(fitness, 0.0)  # Should be negative
    
    def test_evaluate_program_simple(self):
        """Test program evaluation with simple programs."""
        # Always predict 1 on balanced stream
        program = const(1)
        bits = [0, 1, 0, 1, 0, 1, 0, 1]
        
        fitness, metrics = evaluate_program_on_window(self.interpreter, program, bits, k=2)
        
        # Check that evaluation completed
        self.assertIsInstance(fitness, float)
        self.assertIsInstance(metrics, dict)
        self.assertGreater(metrics['num_predictions'], 0)
        
        # Check metrics structure
        expected_keys = ['num_predictions', 'empirical_1_rate', 'baseline_entropy', 
                        'model_entropy', 'fitness', 'total_bits_saved', 'context_length']
        for key in expected_keys:
            self.assertIn(key, metrics)
    
    def test_evaluate_program_perfect_predictor(self):
        """Test evaluation with a perfect predictor on known pattern."""
        # Create a constant stream
        bits = [1] * 20
        
        # Perfect predictor always predicts 1
        program = const(1)
        
        fitness, metrics = evaluate_program_on_window(self.interpreter, program, bits, k=2)
        
        # On constant stream, perfect predictor should achieve near-optimal fitness
        self.assertGreater(fitness, 0.0)  # Should be positive
        self.assertAlmostEqual(metrics['empirical_1_rate'], 1.0, places=6)
        self.assertLess(metrics['model_entropy'], 0.1)  # Should be very low
    
    def test_evaluate_program_random_predictor(self):
        """Test evaluation with random predictor."""
        # Balanced stream
        bits = [0, 1] * 10
        
        # Random predictor
        program = const(0.5)
        
        fitness, metrics = evaluate_program_on_window(self.interpreter, program, bits, k=2)
        
        # Random predictor on balanced stream should have fitness near 0
        self.assertAlmostEqual(fitness, 0.0, places=1)
        self.assertAlmostEqual(metrics['empirical_1_rate'], 0.5, places=6)
        self.assertAlmostEqual(metrics['model_entropy'], 1.0, places=1)
    
    def test_evaluate_program_with_context(self):
        """Test that context length affects evaluation."""
        bits = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        program = const(0.5)
        
        # Different context lengths should give different numbers of predictions
        fitness1, metrics1 = evaluate_program_on_window(self.interpreter, program, bits, k=1)
        fitness2, metrics2 = evaluate_program_on_window(self.interpreter, program, bits, k=3)
        
        # Longer context means fewer predictions (need more initial bits)
        self.assertGreater(metrics1['num_predictions'], metrics2['num_predictions'])
        
        # But fitness should be similar for same program
        self.assertAlmostEqual(fitness1, fitness2, places=1)
    
    def test_evaluate_program_error_handling(self):
        """Test evaluation with programs that might fail."""
        bits = [1, 0, 1, 0, 1, 0]
        
        # Program with undefined variable
        program = var('undefined')
        
        fitness, metrics = evaluate_program_on_window(self.interpreter, program, bits, k=2)
        
        # Should not crash, should fallback to p=0.5
        self.assertIsInstance(fitness, float)
        self.assertGreater(metrics['num_predictions'], 0)
    
    def test_evaluate_program_empty_stream(self):
        """Test evaluation with insufficient data."""
        bits = [1, 0]  # Too short for k=2
        program = const(0.5)
        
        fitness, metrics = evaluate_program_on_window(self.interpreter, program, bits, k=2)
        
        # Should handle gracefully
        self.assertEqual(fitness, -float('inf'))
        self.assertEqual(metrics['num_predictions'], 0)
    
    def test_fitness_ordering(self):
        """Test that fitness correctly orders predictors."""
        bits = [1, 1, 1, 0, 1, 1, 0, 1, 1, 1]  # 80% 1s
        
        # Perfect predictor (always predicts correctly)
        # This is hard to implement, so use a very good predictor
        good_program = const(0.9)  # Biased toward 1s
        
        # Random predictor
        random_program = const(0.5)
        
        # Bad predictor (biased wrong way)
        bad_program = const(0.1)  # Biased toward 0s
        
        fitness_good, _ = evaluate_program_on_window(self.interpreter, good_program, bits, k=2)
        fitness_random, _ = evaluate_program_on_window(self.interpreter, random_program, bits, k=2)
        fitness_bad, _ = evaluate_program_on_window(self.interpreter, bad_program, bits, k=2)
        
        # Good predictor should have highest fitness
        self.assertGreater(fitness_good, fitness_random)
        self.assertGreater(fitness_random, fitness_bad)
    
    def test_evaluation_deterministic(self):
        """Test that evaluation is deterministic."""
        bits = [1, 0, 1, 1, 0, 0, 1, 0]
        program = const(0.6)
        
        fitness1, metrics1 = evaluate_program_on_window(self.interpreter, program, bits, k=2)
        fitness2, metrics2 = evaluate_program_on_window(self.interpreter, program, bits, k=2)
        
        self.assertEqual(fitness1, fitness2)
        self.assertEqual(metrics1['num_predictions'], metrics2['num_predictions'])
        self.assertEqual(metrics1['model_entropy'], metrics2['model_entropy'])
    
    def test_evaluation_with_stream_generator(self):
        """Test evaluation with stream generators."""
        # Use periodic stream
        stream = periodic([1, 0, 1])
        program = const(0.5)
        
        fitness, metrics = evaluate_program(self.interpreter, program, stream, k=2, N=10)
        
        self.assertIsInstance(fitness, float)
        self.assertEqual(metrics['num_predictions'], 10)
        self.assertEqual(metrics['context_length'], 2)
    
    def test_metrics_completeness(self):
        """Test that all expected metrics are returned."""
        bits = [1, 0, 1, 0, 1, 0]
        program = const(0.5)
        
        fitness, metrics = evaluate_program_on_window(self.interpreter, program, bits, k=2)
        
        expected_metrics = [
            'num_predictions', 'empirical_1_rate', 'baseline_entropy',
            'model_entropy', 'fitness', 'total_bits_saved', 'context_length',
            'avg_prediction', 'prediction_variance'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics, f"Missing metric: {metric}")
            self.assertIsInstance(metrics[metric], (int, float), f"Invalid type for {metric}")
    
    def test_edge_case_probabilities(self):
        """Test evaluation with edge case probabilities."""
        bits = [1, 0, 1, 0]
        
        # Test with extreme probabilities
        extreme_programs = [
            const(0.0),    # Always predict 0
            const(1.0),    # Always predict 1
            const(0.001),  # Very low
            const(0.999),  # Very high
        ]
        
        for program in extreme_programs:
            with self.subTest(program=program.value):
                fitness, metrics = evaluate_program_on_window(self.interpreter, program, bits, k=1)
                
                # Should not crash and should produce valid metrics
                self.assertIsInstance(fitness, float)
                self.assertFalse(math.isnan(fitness))
                self.assertGreater(metrics['num_predictions'], 0)


if __name__ == '__main__':
    unittest.main()
