#!/usr/bin/env python3
"""
Comprehensive Integration Tests for ProtoSynth

These tests verify the complete mutation-verification-agent pipeline
works correctly end-to-end, catching integration issues that unit
tests might miss.
"""

import unittest
import random
import time
from protosynth import *
from protosynth.config import ProtoSynthConfig, setup_logging


class TestEndToEndIntegration(unittest.TestCase):
    """
    End-to-end integration tests for the complete ProtoSynth system.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Use consistent random seed for reproducible tests
        self.rng = random.Random(12345)
        
        # Setup logging for debugging
        setup_logging("WARNING")  # Reduce noise in tests
    
    def test_complete_agent_lifecycle(self):
        """Test complete agent lifecycle: creation, mutation, verification, evaluation."""
        
        # Create initial program
        program = let('x', const(10),
                     if_expr(op('>', var('x'), const(5)),
                            op('+', var('x'), const(1)),
                            const(0)))
        
        # Create agent
        agent = SelfModifyingAgent(program, mutation_rate=0.5)
        
        # Verify initial state
        self.assertTrue(agent.verify())
        self.assertIsNotNone(agent.evaluate())
        self.assertEqual(agent.generation, 0)
        
        # Test mutation chain
        current_agent = agent
        for generation in range(1, 6):
            try:
                mutated_agent = current_agent.mutate()
                
                # Verify mutated agent
                self.assertEqual(mutated_agent.generation, generation)
                self.assertTrue(mutated_agent.verify())
                self.assertIsNotNone(mutated_agent.evaluate())
                
                # Ensure it's a different object
                self.assertIsNot(mutated_agent, current_agent)
                
                current_agent = mutated_agent
                
            except RuntimeError as e:
                # Some mutations might fail - that's ok
                self.assertIn("Failed to generate valid mutation", str(e))
                break
    
    def test_mutation_verification_integration(self):
        """Test that mutation and verification systems work together correctly."""
        
        from protosynth.mutation import mutate
        from protosynth.verify import verify_ast
        
        test_programs = [
            const(42),
            op('+', const(1), const(2)),
            let('x', const(5), var('x')),
            if_expr(const(True), const(1), const(0)),
            let('a', const(10),
                let('b', const(20),
                    op('*', var('a'), var('b'))))
        ]
        
        for program in test_programs:
            with self.subTest(program=pretty_print_ast(program)):
                # Original should verify
                is_valid, errors = verify_ast(program)
                self.assertTrue(is_valid, f"Original program should verify: {errors}")
                
                # Apply multiple mutations
                current_ast = program
                for i in range(10):
                    mutated_ast = mutate(current_ast, mutation_rate=0.3, rng=self.rng)
                    
                    # Every mutation should verify
                    is_valid, errors = verify_ast(mutated_ast)
                    self.assertTrue(is_valid, 
                                  f"Mutation {i+1} should verify: {errors}\n"
                                  f"Original: {pretty_print_ast(current_ast)}\n"
                                  f"Mutated: {pretty_print_ast(mutated_ast)}")
                    
                    current_ast = mutated_ast
    
    def test_interpreter_mutation_compatibility(self):
        """Test that mutated programs remain evaluable by the interpreter."""
        
        interpreter = LispInterpreter()
        
        # Start with evaluable programs
        evaluable_programs = [
            const(42),
            op('+', const(10), const(5)),
            let('x', const(7), op('*', var('x'), const(2))),
            if_expr(op('<', const(3), const(5)), const(100), const(200))
        ]
        
        for program in evaluable_programs:
            with self.subTest(program=pretty_print_ast(program)):
                # Original should evaluate
                original_result = interpreter.evaluate(program)
                self.assertIsNotNone(original_result)
                
                # Create agent and mutate
                agent = SelfModifyingAgent(program, mutation_rate=0.4)
                
                for i in range(5):
                    try:
                        mutated_agent = agent.mutate()
                        
                        # Mutated program should also evaluate
                        mutated_result = mutated_agent.evaluate()
                        self.assertIsNotNone(mutated_result)
                        
                        agent = mutated_agent
                        
                    except RuntimeError:
                        # Some mutations might fail - that's acceptable
                        break
    
    def test_config_system_integration(self):
        """Test that configuration system integrates properly with all components."""
        
        from protosynth.config import ProtoSynthConfig, MutationConfig, VerifierConfig
        
        # Create custom configuration
        config = ProtoSynthConfig()
        config.mutation.mutation_rate = 0.8
        config.mutation.max_mutation_attempts = 15
        config.verifier.max_depth = 15
        config.verifier.max_nodes = 200
        
        # Create agent with custom config
        program = op('+', const(10), const(5))
        agent = SelfModifyingAgent(
            program,
            mutation_rate=config.mutation.mutation_rate,
            max_mutation_attempts=config.mutation.max_mutation_attempts
        )
        
        # Test that config values are respected
        self.assertEqual(agent.mutation_rate, 0.8)
        self.assertEqual(agent.max_mutation_attempts, 15)
        
        # Test mutation with high rate
        mutated_agent = agent.mutate()
        self.assertIsNotNone(mutated_agent)
        self.assertTrue(mutated_agent.verify())
    
    def test_error_recovery_and_robustness(self):
        """Test system robustness and error recovery."""
        
        # Test with potentially problematic ASTs
        edge_case_programs = [
            const(0),  # Minimal program
            op('+', const(1)),  # Invalid arity (should be caught by verifier)
            var('undefined'),  # Unbound variable (should be caught by verifier)
        ]
        
        for program in edge_case_programs:
            with self.subTest(program=str(program)):
                try:
                    agent = SelfModifyingAgent(program)
                    
                    # If agent creation succeeds, it should verify
                    if agent.verify():
                        # Try mutation
                        try:
                            mutated_agent = agent.mutate()
                            self.assertTrue(mutated_agent.verify())
                        except RuntimeError:
                            # Mutation failure is acceptable for edge cases
                            pass
                    
                except Exception as e:
                    # Some edge cases might fail agent creation - that's ok
                    self.assertIsInstance(e, (ValueError, RuntimeError))
    
    def test_performance_integration(self):
        """Test that the integrated system performs adequately."""
        
        # Create a moderately complex program
        complex_program = let('x', const(10),
                             let('y', const(20),
                                 let('z', const(30),
                                     if_expr(op('>', var('x'), const(5)),
                                            op('+', op('*', var('x'), var('y')), var('z')),
                                            const(0)))))
        
        agent = SelfModifyingAgent(complex_program, mutation_rate=0.3)
        
        # Time a sequence of mutations
        start_time = time.time()
        
        for i in range(20):
            try:
                agent = agent.mutate()
            except RuntimeError:
                break
        
        end_time = time.time()
        
        # Should complete in reasonable time (< 5 seconds for 20 mutations)
        self.assertLess(end_time - start_time, 5.0, 
                       "20 mutations should complete in under 5 seconds")
    
    def test_deterministic_behavior(self):
        """Test that the system behaves deterministically with same seeds."""
        
        program = op('+', const(10), const(5))
        
        # Create two identical agents with same seed
        agent1 = SelfModifyingAgent(program, mutation_rate=0.5)
        agent1.rng = random.Random(42)
        
        agent2 = SelfModifyingAgent(program, mutation_rate=0.5)
        agent2.rng = random.Random(42)
        
        # Apply same mutations
        for i in range(3):
            try:
                mutated1 = agent1.mutate()
                mutated2 = agent2.mutate()
                
                # Should produce identical results
                self.assertEqual(mutated1.get_code_string(), mutated2.get_code_string())
                
                agent1 = mutated1
                agent2 = mutated2
                
            except RuntimeError:
                # Both should fail the same way
                with self.assertRaises(RuntimeError):
                    agent2.mutate()
                break


if __name__ == '__main__':
    unittest.main()
