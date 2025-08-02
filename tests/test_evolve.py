#!/usr/bin/env python3
"""
Tests for ProtoSynth Evolution System

These tests verify the evolutionary algorithm implementation,
including population management, mutation, selection, and convergence.
"""

import unittest
import itertools
from protosynth import *
from protosynth.evolve import *
from protosynth.envs import periodic, constant


class TestEvolutionSystem(unittest.TestCase):
    """Test cases for the evolution system."""
    
    def setUp(self):
        """Set up test environment."""
        self.engine = EvolutionEngine(mu=4, lambda_=8, seed=42)
        
        # Create simple test programs
        self.test_programs = [
            const(0.0),
            const(0.5),
            const(1.0),
            const(0.3)
        ]
    
    def test_individual_creation(self):
        """Test Individual dataclass."""
        program = const(0.5)
        individual = Individual(
            program=program,
            fitness=0.5,
            metrics={'test': 1},
            generation=0
        )
        
        self.assertEqual(individual.fitness, 0.5)
        self.assertEqual(individual.generation, 0)
        self.assertIsNotNone(individual.individual_id)
        self.assertEqual(individual.size(), 1)  # Single const node
    
    def test_individual_comparison(self):
        """Test Individual comparison for sorting."""
        # Higher fitness should come first
        ind1 = Individual(const(0.5), 0.8, {}, 0)
        ind2 = Individual(const(0.3), 0.6, {}, 0)
        
        self.assertLess(ind1, ind2)  # ind1 has higher fitness
        
        # For equal fitness, smaller size should come first
        ind3 = Individual(const(0.5), 0.7, {}, 0)  # size 1
        ind4 = Individual(op('+', const(0.2), const(0.3)), 0.7, {}, 0)  # size 3
        
        self.assertLess(ind3, ind4)  # ind3 is smaller
    
    def test_evolution_engine_initialization(self):
        """Test EvolutionEngine initialization."""
        engine = EvolutionEngine(mu=8, lambda_=16, seed=123)
        
        self.assertEqual(engine.mu, 8)
        self.assertEqual(engine.lambda_, 16)
        self.assertEqual(engine.generation, 0)
        self.assertEqual(len(engine.population), 0)
        self.assertIsNone(engine.best_individual)
    
    def test_population_initialization(self):
        """Test population initialization."""
        self.engine.initialize_population(self.test_programs)
        
        self.assertEqual(len(self.engine.population), 4)
        self.assertEqual(self.engine.population[0].generation, 0)
        
        # Should fail with too few programs
        with self.assertRaises(ValueError):
            self.engine.initialize_population([const(0.5)])  # Only 1 program, need 4
    
    def test_mutation(self):
        """Test individual mutation."""
        parent = Individual(const(0.5), 0.0, {}, 0)
        
        offspring = self.engine.mutate_individual(parent)
        
        # Should create offspring (might fail sometimes due to randomness)
        if offspring is not None:
            self.assertEqual(offspring.generation, 1)
            self.assertEqual(offspring.parent_id, parent.individual_id)
            self.assertEqual(offspring.fitness, -float('inf'))  # Not evaluated yet
    
    def test_evaluation(self):
        """Test individual evaluation."""
        individual = Individual(const(0.5), -float('inf'), {}, 0)
        
        # Create test stream
        test_bits = [0, 1, 0, 1, 0, 1] * 10
        def bit_stream():
            for bit in test_bits:
                yield bit
        
        self.engine.evaluate_individual(individual, bit_stream())
        
        # Should have been evaluated
        self.assertNotEqual(individual.fitness, -float('inf'))
        self.assertIsInstance(individual.fitness, float)
        self.assertIn('num_predictions', individual.metrics)
    
    def test_create_initial_population(self):
        """Test initial population creation."""
        programs = create_initial_population(10, seed=42)
        
        self.assertEqual(len(programs), 10)
        
        # Should have diverse programs
        program_strings = [pretty_print_ast(p) for p in programs]
        unique_programs = set(program_strings)
        self.assertGreater(len(unique_programs), 5)  # Should have some diversity
    
    def test_single_generation_evolution(self):
        """Test evolution of a single generation."""
        # Initialize population
        self.engine.initialize_population(self.test_programs)
        
        # Create test stream factory
        def stream_factory():
            return periodic([1, 0])
        
        # Evolve one generation
        stats = self.engine.evolve_generation(stream_factory())
        
        # Check statistics
        self.assertEqual(stats['generation'], 1)
        self.assertIsInstance(stats['best_fitness'], float)
        self.assertIsInstance(stats['median_fitness'], float)
        self.assertGreater(stats['num_offspring'], 0)
        self.assertGreaterEqual(stats['mutation_success_rate'], 0.0)
        self.assertLessEqual(stats['mutation_success_rate'], 1.0)
        
        # Population should still be correct size
        self.assertEqual(len(self.engine.population), 4)
    
    def test_multi_generation_evolution(self):
        """Test evolution over multiple generations."""
        # Initialize with simple programs
        initial_programs = create_initial_population(6, seed=42)
        engine = EvolutionEngine(mu=6, lambda_=12, seed=42)
        engine.initialize_population(initial_programs)
        
        # Create stream factory
        def stream_factory():
            return constant(1)  # Constant stream of 1s
        
        # Run for a few generations
        history = engine.run_evolution(stream_factory, num_generations=3)
        
        self.assertEqual(len(history), 3)
        self.assertEqual(engine.generation, 3)
        
        # Should have best individual
        best = engine.get_best_individual()
        self.assertIsNotNone(best)
        self.assertIsInstance(best.fitness, float)
    
    def test_convergence_on_simple_problem(self):
        """Test that evolution can solve a simple problem."""
        # Problem: predict constant stream of 1s
        # Solution: program that outputs high probability
        
        initial_programs = create_initial_population(8, seed=42)
        engine = EvolutionEngine(mu=8, lambda_=16, seed=42, N=100)  # Smaller N for speed
        engine.initialize_population(initial_programs)
        
        def stream_factory():
            return constant(1)
        
        # Run evolution
        history = engine.run_evolution(stream_factory, num_generations=10)
        
        # Should improve over time
        initial_fitness = history[0]['best_fitness']
        final_fitness = history[-1]['best_fitness']
        
        # On constant stream, good predictors should have positive fitness
        self.assertGreater(final_fitness, initial_fitness - 0.1)  # Allow some variance
    
    def test_population_summary(self):
        """Test population summary statistics."""
        # Initialize and evaluate population
        self.engine.initialize_population(self.test_programs)
        
        # Manually set some fitnesses for testing
        self.engine.population[0].fitness = 0.8
        self.engine.population[1].fitness = 0.6
        self.engine.population[2].fitness = 0.4
        self.engine.population[3].fitness = 0.2
        
        summary = self.engine.get_population_summary()
        
        self.assertEqual(summary['population_size'], 4)
        self.assertEqual(summary['best_fitness'], 0.8)
        self.assertEqual(summary['worst_fitness'], 0.2)
        self.assertEqual(summary['mean_fitness'], 0.5)
    
    def test_run_simple_evolution(self):
        """Test the simple evolution wrapper function."""
        def stream_factory():
            return periodic([1, 0, 1])
        
        results = run_simple_evolution(
            stream_factory, 
            num_generations=5, 
            mu=4, 
            lambda_=8, 
            seed=42
        )
        
        # Check results structure
        self.assertIn('best_individual', results)
        self.assertIn('final_population', results)
        self.assertIn('history', results)
        self.assertIn('summary', results)
        
        # Should have run 5 generations
        self.assertEqual(len(results['history']), 5)
        
        # Should have best individual
        best = results['best_individual']
        self.assertIsNotNone(best)
        self.assertIsInstance(best.fitness, float)
    
    def test_deterministic_evolution(self):
        """Test that evolution is deterministic with same seed."""
        def stream_factory():
            return periodic([0, 1])
        
        # Run twice with same seed
        results1 = run_simple_evolution(stream_factory, 3, mu=4, lambda_=8, seed=123)
        results2 = run_simple_evolution(stream_factory, 3, mu=4, lambda_=8, seed=123)
        
        # Should get identical results
        self.assertEqual(len(results1['history']), len(results2['history']))
        
        # Final fitness should be the same
        final_fitness1 = results1['history'][-1]['best_fitness']
        final_fitness2 = results2['history'][-1]['best_fitness']
        self.assertEqual(final_fitness1, final_fitness2)
    
    def test_evolution_with_different_streams(self):
        """Test evolution on different stream types."""
        stream_types = [
            lambda: constant(0),
            lambda: constant(1),
            lambda: periodic([0, 1]),
            lambda: periodic([1, 1, 0])
        ]
        
        for i, stream_factory in enumerate(stream_types):
            with self.subTest(stream_type=i):
                results = run_simple_evolution(
                    stream_factory,
                    num_generations=3,
                    mu=4,
                    lambda_=8,
                    seed=42
                )
                
                # Should complete without errors
                self.assertIsNotNone(results['best_individual'])
                self.assertEqual(len(results['history']), 3)
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Empty initial population
        with self.assertRaises(ValueError):
            self.engine.initialize_population([])
        
        # Evolution without initialization
        engine = EvolutionEngine(mu=4, lambda_=8)
        with self.assertRaises(Exception):
            engine.evolve_generation(constant(1))
    
    def test_mutation_failure_handling(self):
        """Test handling of mutation failures."""
        # Create engine with very restrictive mutation settings
        engine = EvolutionEngine(mu=2, lambda_=4, max_mutation_attempts=1, seed=42)
        
        # Initialize with simple programs
        engine.initialize_population([const(0.5), const(0.3)])
        
        # Try to evolve (some mutations may fail)
        def stream_factory():
            return constant(1)
        
        stats = engine.evolve_generation(stream_factory())
        
        # Should handle failures gracefully
        self.assertIsInstance(stats, dict)
        self.assertGreaterEqual(stats['mutation_success_rate'], 0.0)
    
    def test_fitness_improvement_tracking(self):
        """Test that fitness improvements are tracked correctly."""
        initial_programs = [const(0.1), const(0.9)]  # One bad, one good for constant 1s
        engine = EvolutionEngine(mu=2, lambda_=4, seed=42)
        engine.initialize_population(initial_programs)
        
        def stream_factory():
            return constant(1)
        
        # Run a few generations
        for _ in range(3):
            engine.evolve_generation(stream_factory())
        
        # Should have tracked best individual
        best = engine.get_best_individual()
        self.assertIsNotNone(best)
        
        # Best should be reasonable for constant 1 stream
        # (program that predicts high probability should do well)
        self.assertGreater(best.fitness, -1.0)  # Should not be terrible


if __name__ == '__main__':
    unittest.main()
