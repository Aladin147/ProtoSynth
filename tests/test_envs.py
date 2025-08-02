#!/usr/bin/env python3
"""
Tests for ProtoSynth Stream Environments

These tests verify that stream generators produce correct, deterministic
sequences and handle edge cases properly.
"""

import unittest
import itertools
from protosynth.envs import *


class TestStreamEnvironments(unittest.TestCase):
    """Test cases for stream environment generators."""
    
    def test_periodic_basic(self):
        """Test basic periodic pattern generation."""
        # Simple alternating pattern
        gen = periodic([1, 0])
        result = list(itertools.islice(gen, 6))
        self.assertEqual(result, [1, 0, 1, 0, 1, 0])
        
        # Single bit pattern
        gen = periodic([1])
        result = list(itertools.islice(gen, 5))
        self.assertEqual(result, [1, 1, 1, 1, 1])
        
        # Complex pattern
        gen = periodic([1, 1, 0, 1])
        result = list(itertools.islice(gen, 8))
        self.assertEqual(result, [1, 1, 0, 1, 1, 1, 0, 1])
    
    def test_periodic_string_input(self):
        """Test periodic with string input."""
        gen = periodic("101")
        result = list(itertools.islice(gen, 6))
        self.assertEqual(result, [1, 0, 1, 1, 0, 1])
    
    def test_periodic_deterministic(self):
        """Test that periodic is deterministic with same seed."""
        gen1 = periodic([1, 0, 1], seed=42)
        gen2 = periodic([1, 0, 1], seed=42)
        
        result1 = list(itertools.islice(gen1, 10))
        result2 = list(itertools.islice(gen2, 10))
        
        self.assertEqual(result1, result2)
    
    def test_periodic_edge_cases(self):
        """Test periodic with edge cases."""
        # Empty pattern should raise error
        with self.assertRaises(ValueError):
            list(itertools.islice(periodic([]), 1))

        # Invalid bits should raise error
        with self.assertRaises(ValueError):
            list(itertools.islice(periodic([0, 1, 2]), 1))
    
    def test_k_order_markov_basic(self):
        """Test basic k-order Markov generation."""
        # 1st order: deterministic alternating
        trans = {(0,): 1.0, (1,): 0.0}
        gen = k_order_markov(1, trans, seed=42)
        result = list(itertools.islice(gen, 8))
        # Should start with context (0,), emit 1, then alternate
        expected = [1, 0, 1, 0, 1, 0, 1, 0]
        self.assertEqual(result, expected)
    
    def test_k_order_markov_second_order(self):
        """Test 2nd order Markov (XOR pattern)."""
        # XOR of previous two bits
        trans = {
            (0, 0): 0.0,  # 00 -> 0
            (0, 1): 1.0,  # 01 -> 1  
            (1, 0): 1.0,  # 10 -> 1
            (1, 1): 0.0,  # 11 -> 0
        }
        gen = k_order_markov(2, trans, seed=42)
        result = list(itertools.islice(gen, 10))
        
        # Verify XOR property: each bit should be XOR of previous two
        for i in range(2, len(result)):
            expected_bit = result[i-2] ^ result[i-1]
            self.assertEqual(result[i], expected_bit)
    
    def test_k_order_markov_deterministic(self):
        """Test Markov determinism with same seed."""
        trans = {(0,): 0.7, (1,): 0.3}
        
        gen1 = k_order_markov(1, trans, seed=123)
        gen2 = k_order_markov(1, trans, seed=123)
        
        result1 = list(itertools.islice(gen1, 20))
        result2 = list(itertools.islice(gen2, 20))
        
        self.assertEqual(result1, result2)
    
    def test_k_order_markov_zero_order(self):
        """Test 0-order Markov (independent bits)."""
        trans = {(): 0.0}  # Always emit 0
        gen = k_order_markov(0, trans, seed=42)
        result = list(itertools.islice(gen, 5))
        self.assertEqual(result, [0, 0, 0, 0, 0])
    
    def test_arith_prog_basic(self):
        """Test arithmetic progression generator."""
        # a=1, d=3, mod=8: sequence 1,4,7,2,5,0,3,6,1,4,...
        # LSBs: 1,0,1,0,1,0,1,0,1,0,...
        gen = arith_prog(1, 3, 8)
        result = list(itertools.islice(gen, 10))
        expected = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        self.assertEqual(result, expected)
    
    def test_arith_prog_different_params(self):
        """Test arithmetic progression with different parameters."""
        # a=0, d=1, mod=4: sequence 0,1,2,3,0,1,2,3,...
        # LSBs: 0,1,0,1,0,1,0,1,...
        gen = arith_prog(0, 1, 4)
        result = list(itertools.islice(gen, 8))
        expected = [0, 1, 0, 1, 0, 1, 0, 1]
        self.assertEqual(result, expected)
    
    def test_arith_prog_deterministic(self):
        """Test arithmetic progression determinism."""
        gen1 = arith_prog(2, 5, 7, seed=42)
        gen2 = arith_prog(2, 5, 7, seed=99)  # Seed doesn't affect deterministic sequence
        
        result1 = list(itertools.islice(gen1, 10))
        result2 = list(itertools.islice(gen2, 10))
        
        self.assertEqual(result1, result2)
    
    def test_arith_prog_edge_cases(self):
        """Test arithmetic progression edge cases."""
        # Zero modulus should raise error
        with self.assertRaises(ValueError):
            list(itertools.islice(arith_prog(1, 1, 0), 1))

        # Negative modulus should raise error
        with self.assertRaises(ValueError):
            list(itertools.islice(arith_prog(1, 1, -1), 1))
    
    def test_noisy_basic(self):
        """Test noisy stream generation."""
        # No noise (p_flip=0) should preserve original
        base = periodic([1, 0])
        gen = noisy(base, 0.0, seed=42)
        result = list(itertools.islice(gen, 6))
        expected = [1, 0, 1, 0, 1, 0]
        self.assertEqual(result, expected)
        
        # Full noise (p_flip=1) should flip all bits
        base = periodic([1, 0])
        gen = noisy(base, 1.0, seed=42)
        result = list(itertools.islice(gen, 6))
        expected = [0, 1, 0, 1, 0, 1]
        self.assertEqual(result, expected)
    
    def test_noisy_deterministic(self):
        """Test noisy stream determinism with same seed."""
        base1 = periodic([1, 1, 0])
        base2 = periodic([1, 1, 0])
        
        gen1 = noisy(base1, 0.3, seed=123)
        gen2 = noisy(base2, 0.3, seed=123)
        
        result1 = list(itertools.islice(gen1, 20))
        result2 = list(itertools.islice(gen2, 20))
        
        self.assertEqual(result1, result2)
    
    def test_noisy_edge_cases(self):
        """Test noisy stream edge cases."""
        base = periodic([1])

        # Invalid probability should raise error
        with self.assertRaises(ValueError):
            list(itertools.islice(noisy(base, -0.1), 1))

        base = periodic([1])  # Reset generator
        with self.assertRaises(ValueError):
            list(itertools.islice(noisy(base, 1.1), 1))
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        # Alternating
        gen = alternating()
        result = list(itertools.islice(gen, 6))
        self.assertEqual(result, [0, 1, 0, 1, 0, 1])
        
        # Constant 0
        gen = constant(0)
        result = list(itertools.islice(gen, 5))
        self.assertEqual(result, [0, 0, 0, 0, 0])
        
        # Constant 1
        gen = constant(1)
        result = list(itertools.islice(gen, 5))
        self.assertEqual(result, [1, 1, 1, 1, 1])
        
        # Invalid constant
        with self.assertRaises(ValueError):
            constant(2)
    
    def test_random_bits(self):
        """Test random bit generation."""
        # Deterministic with seed
        gen1 = random_bits(0.5, seed=42)
        gen2 = random_bits(0.5, seed=42)
        
        result1 = list(itertools.islice(gen1, 20))
        result2 = list(itertools.islice(gen2, 20))
        
        self.assertEqual(result1, result2)
        
        # Extreme probabilities
        gen_zero = random_bits(0.0, seed=42)
        result_zero = list(itertools.islice(gen_zero, 10))
        self.assertEqual(result_zero, [0] * 10)
        
        gen_one = random_bits(1.0, seed=42)
        result_one = list(itertools.islice(gen_one, 10))
        self.assertEqual(result_one, [1] * 10)
    
    def test_create_environment_factory(self):
        """Test environment factory function."""
        # Periodic
        env = create_environment("periodic", pattern=[1, 0, 1], seed=42)
        result = list(itertools.islice(env, 6))
        self.assertEqual(result, [1, 0, 1, 1, 0, 1])
        
        # Markov
        env = create_environment("markov", k=1, trans={(0,): 1.0, (1,): 0.0}, seed=42)
        result = list(itertools.islice(env, 6))
        self.assertEqual(result, [1, 0, 1, 0, 1, 0])
        
        # Arithmetic
        env = create_environment("arith", a=1, d=3, mod=8, seed=42)
        result = list(itertools.islice(env, 6))
        self.assertEqual(result, [1, 0, 1, 0, 1, 0])
        
        # Random
        env = create_environment("random", p=0.0, seed=42)
        result = list(itertools.islice(env, 5))
        self.assertEqual(result, [0, 0, 0, 0, 0])
        
        # Unknown type
        with self.assertRaises(ValueError):
            create_environment("unknown")


if __name__ == '__main__':
    unittest.main()
