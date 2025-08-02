"""
Self-Modifying Agent for ProtoSynth

This module implements the SelfModifyingAgent class, which represents
an autonomous entity that can inspect, modify, and evaluate its own code.
"""

from typing import Any, Optional
from .core import LispNode, LispInterpreter, clone_ast, pretty_print_ast


class SelfModifyingAgent:
    """
    A self-modifying agent that holds code (an AST) and can mutate,
    verify, and evaluate itself.
    
    This is the core entity in the ProtoSynth architecture that embodies
    the self-modifying behavior through mutation and fitness evaluation.
    """
    
    def __init__(self, initial_ast: LispNode, interpreter: Optional[LispInterpreter] = None):
        """
        Initialize the agent with an initial AST.
        
        Args:
            initial_ast: The starting program as an AST
            interpreter: Optional custom interpreter (defaults to standard one)
        """
        self.ast = clone_ast(initial_ast)  # Always work with a copy
        self.interpreter = interpreter or LispInterpreter()
        self.generation = 0
        self.fitness_history = []
    
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
        
        For now, this is a placeholder that returns a copy of the current agent.
        In Phase 2, this will implement actual mutation operations.
        
        Returns:
            A new SelfModifyingAgent with a mutated AST
        """
        # Placeholder: just return a copy for now
        mutated_agent = SelfModifyingAgent(self.ast, self.interpreter)
        mutated_agent.generation = self.generation + 1
        return mutated_agent
    
    def verify(self) -> bool:
        """
        Verify that the current AST is syntactically valid and safe to execute.
        
        For now, this is a placeholder that always returns True.
        In Phase 2, this will implement proper verification logic.
        
        Returns:
            True if the AST passes verification, False otherwise
        """
        # Placeholder: basic checks
        try:
            # Check if we can pretty-print the AST (basic syntax check)
            pretty_print_ast(self.ast)
            return True
        except Exception:
            return False
    
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
