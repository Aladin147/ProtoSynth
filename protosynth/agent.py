"""
Self-Modifying Agent for ProtoSynth

This module implements the SelfModifyingAgent class, which represents
an autonomous entity that can inspect, modify, and evaluate its own code.
"""

from typing import Any, Optional
import random
import logging
from .core import LispNode, LispInterpreter, clone_ast, pretty_print_ast
from .mutation import mutate
from .verify import verify_ast

logger = logging.getLogger(__name__)


class SelfModifyingAgent:
    """
    A self-modifying agent that holds code (an AST) and can mutate,
    verify, and evaluate itself.
    
    This is the core entity in the ProtoSynth architecture that embodies
    the self-modifying behavior through mutation and fitness evaluation.
    """
    
    def __init__(self, initial_ast: LispNode, interpreter: Optional[LispInterpreter] = None,
                 mutation_rate: float = 0.15, max_mutation_attempts: int = 10):
        """
        Initialize the agent with an initial AST.

        Args:
            initial_ast: The starting program as an AST
            interpreter: Optional custom interpreter (defaults to standard one)
            mutation_rate: Probability of applying each mutation type (default: 0.15)
            max_mutation_attempts: Maximum attempts to generate valid mutation (default: 10)
        """
        self.ast = clone_ast(initial_ast)  # Always work with a copy
        self.interpreter = interpreter or LispInterpreter()
        self.mutation_rate = mutation_rate
        self.max_mutation_attempts = max_mutation_attempts
        self.generation = 0
        self.fitness_history = []
        self.rng = random.Random()  # Each agent has its own RNG
    
    def get_ast(self) -> LispNode:
        """
        Get the current AST of this agent.
        
        Returns:
            A deep copy of the current AST
        """
        return clone_ast(self.ast)
    
    def mutate(self) -> 'SelfModifyingAgent':
        """
        Create a mutated version of this agent.

        Applies mutations to the AST and verifies the result. If verification
        fails, retries up to max_mutation_attempts times.

        Returns:
            A new SelfModifyingAgent with a mutated and verified AST

        Raises:
            RuntimeError: If unable to generate valid mutation after max attempts
        """
        for attempt in range(self.max_mutation_attempts):
            try:
                # Apply mutation
                mutated_ast = mutate(self.ast, self.mutation_rate, self.rng)

                # Verify the mutated AST
                is_valid, errors = verify_ast(mutated_ast)

                if is_valid:
                    logger.info(f"Mutation successful on attempt {attempt + 1}")
                    # Create new agent with mutated AST
                    mutated_agent = SelfModifyingAgent(
                        mutated_ast,
                        self.interpreter,
                        self.mutation_rate,
                        self.max_mutation_attempts
                    )
                    mutated_agent.generation = self.generation + 1
                    mutated_agent.rng = random.Random(self.rng.randint(0, 2**32-1))  # New seed
                    return mutated_agent

                # If verification failed, try again
                logger.debug(f"Mutation attempt {attempt + 1} failed verification: {errors}")
                continue

            except Exception as e:
                # If mutation failed, try again
                continue

        # If all attempts failed, raise error
        raise RuntimeError(f"Failed to generate valid mutation after {self.max_mutation_attempts} attempts")
    
    def verify(self) -> bool:
        """
        Verify that the current AST is syntactically valid and safe to execute.

        Returns:
            True if the AST passes verification, False otherwise
        """
        is_valid, errors = verify_ast(self.ast)
        return is_valid
    
    def evaluate(self, input_data: Any = None) -> Any:
        """
        Evaluate the current AST using the interpreter.
        
        Args:
            input_data: Optional input data for the program
            
        Returns:
            The result of evaluating the AST
            
        Raises:
            RuntimeError: If evaluation fails or exceeds resource limits
        """
        try:
            # For now, just evaluate the AST directly
            # In the future, this might incorporate input_data
            return self.interpreter.evaluate(self.ast)
        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {e}")
    
    def get_fitness(self, environment_data: Any = None) -> float:
        """
        Calculate the fitness score for this agent.
        
        For now, this is a placeholder that returns a simple score.
        In Phase 3, this will implement compression-driven evaluation.
        
        Args:
            environment_data: Optional environment data for fitness calculation
            
        Returns:
            A fitness score (higher is better)
        """
        try:
            # Placeholder fitness: inverse of AST complexity
            ast_str = pretty_print_ast(self.ast)
            complexity = len(ast_str)
            
            # Simple fitness: prefer simpler programs that don't crash
            result = self.evaluate()
            if result is not None:
                fitness = 1000.0 / (complexity + 1)
            else:
                fitness = 0.0
                
            self.fitness_history.append(fitness)
            return fitness
            
        except Exception:
            # Failed evaluation gets zero fitness
            fitness = 0.0
            self.fitness_history.append(fitness)
            return fitness
    
    def get_code_string(self) -> str:
        """
        Get a human-readable string representation of the current code.
        
        Returns:
            The AST rendered as Lisp-style code
        """
        return pretty_print_ast(self.ast)
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"SelfModifyingAgent(gen={self.generation}, code='{self.get_code_string()}')"
