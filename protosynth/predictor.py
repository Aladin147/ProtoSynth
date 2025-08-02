"""
ProtoSynth Program→Predictor Adapter

This module provides the interface between evolved AST programs and
the prediction task. It handles context binding, output coercion,
and error recovery.
"""

import logging
from typing import List, Union, Any, Optional
from .core import LispNode, LispInterpreter, const

logger = logging.getLogger(__name__)

# Small epsilon for probability clamping
EPS = 1e-6


class PredictorAdapter:
    """
    Adapter that converts AST programs into predictors for binary sequences.
    
    The adapter handles:
    - Context binding (prebound variable 'ctx' with last k bits)
    - Output coercion (convert various types to probabilities)
    - Error handling (timeouts, exceptions → p=0.5)
    - Probability clamping to [ε, 1-ε]
    """
    
    def __init__(self, interpreter: Optional[LispInterpreter] = None, 
                 timeout_seconds: float = 0.1, eps: float = EPS):
        """
        Initialize the predictor adapter.
        
        Args:
            interpreter: Lisp interpreter to use (creates default if None)
            timeout_seconds: Maximum time allowed for prediction
            eps: Epsilon for probability clamping
        """
        self.interpreter = interpreter or LispInterpreter(timeout_seconds=timeout_seconds)
        self.eps = eps
        self.timeout_seconds = timeout_seconds
    
    def predict(self, program: LispNode, context: List[int]) -> float:
        """
        Use a program to predict the next bit given context.
        
        Args:
            program: AST program to evaluate
            context: List of previous k bits
            
        Returns:
            float: Probability of next bit being 1, clamped to [eps, 1-eps]
        """
        try:
            # Create environment with prebound 'ctx' variable
            environment = {'ctx': context}
            
            # Evaluate the program
            result = self.interpreter.evaluate(program, environment)
            
            # Coerce result to probability
            probability = self._coerce_to_probability(result)
            
            # Clamp to valid range
            clamped_prob = max(self.eps, min(1.0 - self.eps, probability))
            
            logger.debug(f"Prediction: ctx={context} -> raw={result} -> p={clamped_prob:.4f}")
            
            return clamped_prob
            
        except Exception as e:
            logger.debug(f"Prediction failed: {e}, returning p=0.5")
            return 0.5
    
    def _coerce_to_probability(self, value: Any) -> float:
        """
        Coerce various output types to probability values.
        
        Args:
            value: Raw output from program evaluation
            
        Returns:
            float: Probability value
            
        Raises:
            ValueError: If value cannot be coerced to probability
        """
        # Handle None
        if value is None:
            return 0.5
        
        # Handle boolean (hard prediction)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        
        # Handle integer
        if isinstance(value, int):
            if value == 0:
                return 0.0
            elif value == 1:
                return 1.0
            elif value > 1:
                # Large integers → high probability
                return 0.9
            else:  # value < 0
                # Negative integers → low probability
                return 0.1
        
        # Handle float
        if isinstance(value, float):
            # Check for special values
            if value != value:  # NaN
                return 0.5
            if value == float('inf'):
                return 1.0
            if value == float('-inf'):
                return 0.0
            
            # If already in [0, 1], use as-is
            if 0.0 <= value <= 1.0:
                return value
            
            # If outside [0, 1], apply sigmoid-like mapping
            if value > 1.0:
                return 0.5 + 0.4 * (1.0 - 1.0 / (1.0 + value - 1.0))
            else:  # value < 0.0
                return 0.5 - 0.4 * (1.0 - 1.0 / (1.0 - value + 1.0))
        
        # Handle string (convert to hash-based probability)
        if isinstance(value, str):
            if value.lower() in ['true', 'yes', '1']:
                return 1.0
            elif value.lower() in ['false', 'no', '0']:
                return 0.0
            else:
                # Use hash for consistent but arbitrary mapping
                hash_val = hash(value) % 1000
                return hash_val / 1000.0
        
        # Handle list/tuple (use length or first element)
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return 0.5
            elif len(value) == 1:
                return self._coerce_to_probability(value[0])
            else:
                # Use length modulo for probability
                return (len(value) % 100) / 100.0
        
        # Fallback for unknown types
        logger.warning(f"Unknown type for coercion: {type(value)}, value: {value}")
        return 0.5


def create_context_program(program: LispNode, context_var: str = 'ctx') -> LispNode:
    """
    Create a program that expects a context variable to be prebound.
    
    This is a helper function for creating programs that work with the
    predictor adapter's context binding convention.
    
    Args:
        program: Base program AST
        context_var: Name of context variable (default: 'ctx')
        
    Returns:
        LispNode: Program that can access context via the specified variable
    """
    # For now, just return the program as-is since the adapter
    # handles context binding in the environment
    return program


def predict_with_program(interpreter: LispInterpreter, program: LispNode, 
                        context: List[int]) -> float:
    """
    Convenience function to predict with a program.
    
    Args:
        interpreter: Lisp interpreter
        program: AST program
        context: Context bits
        
    Returns:
        float: Prediction probability
    """
    adapter = PredictorAdapter(interpreter)
    return adapter.predict(program, context)


# Example programs for testing
def create_example_programs() -> dict:
    """Create example programs for testing the predictor adapter."""
    
    examples = {}
    
    # Always predict 1
    examples['always_one'] = const(1)
    
    # Always predict 0  
    examples['always_zero'] = const(0)
    
    # Random (0.5 probability)
    examples['random'] = const(0.5)
    
    # Use context length
    from .core import op, var
    examples['context_length'] = op('/', 
                                   op('+', var('ctx'), const([1])),  # This will fail gracefully
                                   const(10))
    
    # Simple pattern: predict 1 if context has more 1s than 0s
    # This is complex to express in our current language, so use a simpler version
    examples['majority'] = const(0.7)  # Placeholder
    
    return examples


# Test utilities
def test_predictor_adapter():
    """Test the predictor adapter with various inputs."""
    
    adapter = PredictorAdapter()
    
    # Test programs
    programs = create_example_programs()
    
    # Test contexts
    contexts = [
        [],
        [0],
        [1],
        [0, 1],
        [1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0]
    ]
    
    print("Testing Predictor Adapter")
    print("=" * 40)
    
    for prog_name, program in programs.items():
        print(f"\nProgram: {prog_name}")
        for ctx in contexts:
            try:
                prob = adapter.predict(program, ctx)
                print(f"  ctx={ctx} -> p={prob:.3f}")
            except Exception as e:
                print(f"  ctx={ctx} -> ERROR: {e}")


if __name__ == "__main__":
    test_predictor_adapter()
