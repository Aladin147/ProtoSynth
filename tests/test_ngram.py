#!/usr/bin/env python3
"""
Tests for ProtoSynth N-gram Baseline Predictor

These tests verify the correctness of the n-gram predictor implementation,
including training, prediction, and evaluation functionality.
"""

import unittest
import math
from protosynth.ngram import NGramPredictor, compare_ngram_orders


class TestNGramPredictor(unittest.TestCase):
    """Test cases for the N-gram predictor."""
    
    def setUp(self):
        """Set up test environment."""
        self.predictor = NGramPredictor(k=2, alpha=1.0)
    
    def test_initialization(self):
        """Test predictor initialization."""
        predictor = NGramPredictor(k=3, alpha=0.5)
        
        self.assertEqual(predictor.k, 3)
        self.assertEqual(predictor.alpha, 0.5)
        self.assertFalse(predictor.is_trained)
        self.assertEqual(len(predictor.context_counts), 0)
        self.assertEqual(len(predictor.outcome_counts), 0)
    
    def test_training_simple(self):
        """Test training on simple sequences."""
        # Simple alternating pattern
        bits = [0, 1, 0, 1, 0, 1, 0, 1]
        
        self.predictor.train(bits)
        
        self.assertTrue(self.predictor.is_trained)
        self.assertGreater(len(self.predictor.context_counts), 0)
        self.assertGreater(len(self.predictor.outcome_counts), 0)
    
    def test_training_counts(self):
        """Test that training produces correct counts."""
        # Pattern: 0,1,0,1,0,1
        # Contexts (k=2): (0,1)->0, (1,0)->1, (0,1)->0, (1,0)->1
        bits = [0, 1, 0, 1, 0, 1]
        
        self.predictor.train(bits)
        
        # Check context counts
        self.assertEqual(self.predictor.context_counts[(0, 1)], 2)
        self.assertEqual(self.predictor.context_counts[(1, 0)], 2)
        
        # Check outcome counts
        self.assertEqual(self.predictor.outcome_counts[((0, 1), 0)], 2)
        self.assertEqual(self.predictor.outcome_counts[((1, 0), 1)], 2)
    
    def test_prediction_without_training(self):
        """Test prediction before training."""
        prob = self.predictor.predict([0, 1])
        self.assertEqual(prob, 0.5)  # Should return default
    
    def test_prediction_with_smoothing(self):
        """Test prediction with add-α smoothing."""
        # Train on simple pattern
        bits = [0, 1, 0, 1, 0, 1]
        self.predictor.train(bits)
        
        # Test known context
        prob = self.predictor.predict([0, 1])
        # count(01,0) = 2, count(01) = 2, α = 1
        # P(0|01) = (2+1)/(2+2) = 3/4 = 0.75
        # P(1|01) = (0+1)/(2+2) = 1/4 = 0.25
        expected = 0.25
        self.assertAlmostEqual(prob, expected, places=6)
        
        # Test unknown context (should use pure smoothing)
        prob = self.predictor.predict([1, 1])
        # count(11,1) = 0, count(11) = 0, α = 1
        # P(1|11) = (0+1)/(0+2) = 1/2 = 0.5
        expected = 0.5
        self.assertAlmostEqual(prob, expected, places=6)
    
    def test_prediction_context_length(self):
        """Test prediction with different context lengths."""
        bits = [0, 1, 0, 1, 0, 1]
        self.predictor.train(bits)
        
        # Too short context (should pad with zeros)
        prob1 = self.predictor.predict([1])
        prob2 = self.predictor.predict([0, 1])
        # Both should be equivalent to context [0, 1]
        self.assertEqual(prob1, prob2)
        
        # Too long context (should truncate)
        prob3 = self.predictor.predict([1, 0, 1])
        prob4 = self.predictor.predict([0, 1])
        # Should be equivalent to context [0, 1]
        self.assertEqual(prob3, prob4)
    
    def test_evaluation_simple(self):
        """Test evaluation on simple data."""
        # Create training and test data
        train_bits = [0, 1, 0, 1, 0, 1] * 5
        test_bits = [0, 1, 0, 1, 0, 1] * 3
        
        self.predictor.train(train_bits)
        fitness, metrics = self.predictor.evaluate_on_stream(test_bits)
        
        # Check that evaluation completed
        self.assertIsInstance(fitness, float)
        self.assertIsInstance(metrics, dict)
        self.assertGreater(metrics['num_predictions'], 0)
        
        # Check metrics structure
        expected_keys = ['num_predictions', 'empirical_1_rate', 'baseline_entropy',
                        'model_entropy', 'fitness', 'total_bits_saved', 'context_length',
                        'smoothing_alpha', 'num_unique_contexts', 'avg_prediction']
        for key in expected_keys:
            self.assertIn(key, metrics)
    
    def test_evaluation_perfect_prediction(self):
        """Test evaluation on perfectly predictable sequence."""
        # Perfectly periodic sequence that n-gram can learn
        pattern = [0, 1, 0, 1]
        train_bits = pattern * 10  # 40 bits
        test_bits = pattern * 5    # 20 bits

        self.predictor.train(train_bits)
        fitness, metrics = self.predictor.evaluate_on_stream(test_bits)

        # Should achieve reasonable fitness on learnable pattern
        self.assertGreater(fitness, -0.5)  # Should not be terrible
        self.assertLess(metrics['model_entropy'], 1.5)  # Should be reasonable
    
    def test_evaluation_random_sequence(self):
        """Test evaluation on random-like sequence."""
        # Balanced random-ish sequence
        bits = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0] * 5
        
        self.predictor.train(bits[:30])
        fitness, metrics = self.predictor.evaluate_on_stream(bits[30:])
        
        # Should have reasonable performance
        self.assertIsInstance(fitness, float)
        self.assertFalse(math.isnan(fitness))
    
    def test_evaluation_insufficient_data(self):
        """Test evaluation with insufficient data."""
        # Too short for k=2
        bits = [0, 1]
        
        self.predictor.train([0, 1, 0, 1, 0])  # Train on something
        fitness, metrics = self.predictor.evaluate_on_stream(bits)
        
        self.assertEqual(fitness, -float('inf'))
        self.assertEqual(metrics['num_predictions'], 0)
    
    def test_different_smoothing_values(self):
        """Test different smoothing parameters."""
        bits = [0, 1, 0, 1, 0, 1, 0, 1]

        reference_prob = None

        # Test different α values
        for alpha in [0.1, 0.5, 1.0, 2.0]:
            predictor = NGramPredictor(k=2, alpha=alpha)
            predictor.train(bits)

            prob = predictor.predict([0, 1])

            # Should be valid probability
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

            # Store reference for comparison
            if alpha == 1.0:
                reference_prob = prob

        # Test that different α gives different results
        predictor_diff = NGramPredictor(k=2, alpha=0.1)
        predictor_diff.train(bits)
        prob_diff = predictor_diff.predict([0, 1])
        self.assertNotEqual(prob_diff, reference_prob)
    
    def test_different_context_lengths(self):
        """Test different context lengths."""
        bits = [0, 1, 0, 1, 0, 1, 0, 1] * 3
        
        for k in [1, 2, 3, 4]:
            if len(bits) <= k:
                continue
                
            predictor = NGramPredictor(k=k, alpha=1.0)
            predictor.train(bits)
            
            fitness, metrics = predictor.evaluate_on_stream(bits)
            
            self.assertEqual(metrics['context_length'], k)
            self.assertIsInstance(fitness, float)
    
    def test_model_info(self):
        """Test model information retrieval."""
        info = self.predictor.get_model_info()
        
        expected_keys = ['k', 'alpha', 'is_trained', 'num_contexts', 'total_observations']
        for key in expected_keys:
            self.assertIn(key, info)
        
        self.assertEqual(info['k'], 2)
        self.assertEqual(info['alpha'], 1.0)
        self.assertFalse(info['is_trained'])
        
        # After training
        bits = [0, 1, 0, 1, 0, 1]
        self.predictor.train(bits)
        
        info = self.predictor.get_model_info()
        self.assertTrue(info['is_trained'])
        self.assertGreater(info['num_contexts'], 0)
        self.assertGreater(info['total_observations'], 0)
    
    def test_deterministic_behavior(self):
        """Test that predictor behavior is deterministic."""
        bits = [0, 1, 1, 0, 1, 0, 0, 1]
        
        # Train two identical predictors
        pred1 = NGramPredictor(k=2, alpha=1.0)
        pred2 = NGramPredictor(k=2, alpha=1.0)
        
        pred1.train(bits)
        pred2.train(bits)
        
        # Should give identical predictions
        for context in [[0, 0], [0, 1], [1, 0], [1, 1]]:
            prob1 = pred1.predict(context)
            prob2 = pred2.predict(context)
            self.assertEqual(prob1, prob2)
    
    def test_compare_ngram_orders(self):
        """Test the comparison function for different n-gram orders."""
        # Create longer test data with some pattern
        bits = ([0, 1, 0, 1] * 25) + ([1, 1, 0, 0] * 25)  # 200 bits total

        results = compare_ngram_orders(bits, max_k=3, alpha=1.0)

        # Should have results for different k values
        self.assertGreater(len(results), 0)

        for k, (fitness, metrics) in results.items():
            self.assertIsInstance(k, int)
            self.assertIsInstance(fitness, float)
            self.assertIsInstance(metrics, dict)
            self.assertEqual(metrics['context_length'], k)
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Empty training data
        predictor = NGramPredictor(k=2, alpha=1.0)
        predictor.train([])
        self.assertFalse(predictor.is_trained)
        
        # Training data too short
        predictor.train([0, 1])
        self.assertFalse(predictor.is_trained)
        
        # Evaluation without training
        with self.assertRaises(ValueError):
            predictor.evaluate_on_stream([0, 1, 0, 1])
    
    def test_performance_on_known_patterns(self):
        """Test performance on patterns where we know the optimal result."""
        # Perfectly periodic pattern
        pattern = [0, 1, 0, 1]
        bits = pattern * 20
        
        predictor = NGramPredictor(k=2, alpha=0.1)  # Low smoothing
        predictor.train(bits[:40])
        
        fitness, metrics = predictor.evaluate_on_stream(bits[40:])
        
        # Should achieve good fitness on periodic pattern
        self.assertGreater(fitness, 0.1)
        
        # Should predict the pattern well
        self.assertLess(metrics['model_entropy'], 0.5)


if __name__ == '__main__':
    unittest.main()
