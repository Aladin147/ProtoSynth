"""
ProtoSynth Evaluation System

This module implements the fitness evaluation system based on cross-entropy
and compression-driven metrics. The core idea is that better predictors
achieve lower cross-entropy, which translates to better compression.
"""

import math
import logging
from typing import List, Tuple, Iterator, Optional
from .predictor import PredictorAdapter
from .core import LispNode, LispInterpreter

logger = logging.getLogger(__name__)

# Small epsilon for numerical stability
EPS = 1e-6


def cross_entropy_bits(y: int, p: float) -> float:
    """
    Calculate cross-entropy for a single prediction in bits.
    
    Args:
        y: True bit value (0 or 1)
        p: Predicted probability of bit being 1
        
    Returns:
        float: Cross-entropy in bits
        
    Formula: -[y * log2(p) + (1-y) * log2(1-p)]
    """
    # Clamp probability to avoid log(0)
    p = max(EPS, min(1.0 - EPS, p))
    
    if y == 1:
        return -math.log2(p)
    else:
        return -math.log2(1.0 - p)


def baseline_entropy_bits(q: float) -> float:
    """
    Calculate baseline entropy for a stream with empirical 1-rate q.
    
    Args:
        q: Empirical probability of 1s in the stream
        
    Returns:
        float: Baseline entropy in bits per symbol
        
    Formula: H0 = -[q * log2(q) + (1-q) * log2(1-q)]
    """
    # Clamp to avoid log(0)
    q = max(EPS, min(1.0 - EPS, q))
    
    return -(q * math.log2(q) + (1.0 - q) * math.log2(1.0 - q))


def evaluate_program(interpreter: LispInterpreter, program: LispNode, 
                    stream: Iterator[int], k: int = 4, N: int = 2048) -> Tuple[float, dict]:
    """
    Evaluate a program's prediction performance on a bit stream.
    
    Args:
        interpreter: Lisp interpreter for program evaluation
        program: AST program to evaluate
        stream: Bit stream iterator
        k: Context length (number of previous bits to use)
        N: Number of symbols to evaluate on
        
    Returns:
        Tuple of (fitness, metrics_dict) where:
        - fitness: F = H0 - Hprog (bits saved per symbol)
        - metrics_dict: Additional metrics for analysis
    """
    adapter = PredictorAdapter(interpreter)
    
    # Collect stream data
    buf = []
    predictions = []
    targets = []
    
    logger.debug(f"Evaluating program on {N} symbols with context length {k}")
    
    for bit in stream:
        buf.append(bit)
        
        # Need at least k+1 bits to make a prediction
        if len(buf) <= k:
            continue
        
        # Extract context and target
        context = buf[-(k+1):-1]  # Last k bits
        target = buf[-1]          # Current bit to predict
        
        # Make prediction
        try:
            prediction = adapter.predict(program, context)
            predictions.append(prediction)
            targets.append(target)
            
            logger.debug(f"ctx={context} -> p={prediction:.3f}, y={target}")
            
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            predictions.append(0.5)  # Fallback
            targets.append(target)
        
        # Stop when we have enough predictions
        if len(predictions) >= N:
            break
    
    # Check if we have any predictions
    if not predictions:
        logger.warning("No predictions made, returning -inf fitness")
        return -float('inf'), {
            'num_predictions': 0,
            'baseline_entropy': 0.0,
            'model_entropy': 0.0,
            'total_bits_saved': 0.0
        }
    
    # Calculate empirical 1-rate
    q = sum(targets) / len(targets)
    
    # Calculate baseline entropy
    H0 = baseline_entropy_bits(q)
    
    # Calculate model cross-entropy
    cross_entropies = [cross_entropy_bits(y, p) for y, p in zip(targets, predictions)]
    Hprog = sum(cross_entropies) / len(cross_entropies)
    
    # Calculate fitness (bits saved per symbol)
    fitness = H0 - Hprog
    
    # Calculate total bits saved
    total_bits_saved = len(predictions) * fitness
    
    # Prepare metrics
    metrics = {
        'num_predictions': len(predictions),
        'empirical_1_rate': q,
        'baseline_entropy': H0,
        'model_entropy': Hprog,
        'fitness': fitness,
        'total_bits_saved': total_bits_saved,
        'context_length': k,
        'avg_prediction': sum(predictions) / len(predictions),
        'prediction_variance': sum((p - sum(predictions)/len(predictions))**2 for p in predictions) / len(predictions)
    }
    
    logger.info(f"Evaluation complete: F={fitness:.4f}, H0={H0:.4f}, Hprog={Hprog:.4f}")
    
    return fitness, metrics


def evaluate_program_on_window(interpreter: LispInterpreter, program: LispNode,
                              bits: List[int], k: int = 4) -> Tuple[float, dict]:
    """
    Evaluate a program on a fixed window of bits.
    
    This is a convenience function for evaluation on pre-collected data.
    
    Args:
        interpreter: Lisp interpreter
        program: AST program
        bits: List of bits to evaluate on
        k: Context length
        
    Returns:
        Tuple of (fitness, metrics)
    """
    def bit_stream():
        for bit in bits:
            yield bit
    
    return evaluate_program(interpreter, program, bit_stream(), k, len(bits) - k)


def perfect_predictor_fitness(bits: List[int]) -> float:
    """
    Calculate the fitness of a perfect predictor on given bits.
    
    A perfect predictor always predicts the correct bit with probability 1.0,
    so its cross-entropy is 0, and fitness equals baseline entropy.
    
    Args:
        bits: List of bits
        
    Returns:
        float: Perfect predictor fitness (equals baseline entropy)
    """
    if not bits:
        return 0.0
    
    q = sum(bits) / len(bits)
    return baseline_entropy_bits(q)


def random_predictor_fitness(bits: List[int]) -> float:
    """
    Calculate the fitness of a random predictor (always predicts p=0.5).
    
    Args:
        bits: List of bits
        
    Returns:
        float: Random predictor fitness
    """
    if not bits:
        return 0.0
    
    q = sum(bits) / len(bits)
    H0 = baseline_entropy_bits(q)
    
    # Random predictor has cross-entropy of 1.0 bit per symbol
    Hprog = 1.0
    
    return H0 - Hprog


def validate_evaluation_system():
    """
    Validate the evaluation system with analytic checks.
    
    This function performs sanity checks to ensure the evaluation
    system behaves correctly in known cases.
    """
    print("🧪 Validating Evaluation System")
    print("=" * 40)
    
    # Test 1: Perfect predictor on constant stream
    print("\n1. Perfect predictor on constant stream")
    constant_bits = [1] * 100
    perfect_fitness = perfect_predictor_fitness(constant_bits)
    print(f"   Perfect fitness: {perfect_fitness:.6f}")
    print(f"   Expected: ~0 (constant stream has low entropy)")
    
    # Test 2: Random predictor on balanced stream
    print("\n2. Random predictor on balanced stream")
    balanced_bits = [0, 1] * 50
    random_fitness = random_predictor_fitness(balanced_bits)
    print(f"   Random fitness: {random_fitness:.6f}")
    print(f"   Expected: ~0 (balanced stream, random predictor)")
    
    # Test 3: Cross-entropy calculation
    print("\n3. Cross-entropy calculations")
    # Perfect prediction
    ce_perfect = cross_entropy_bits(1, 1.0 - EPS)
    print(f"   Perfect prediction CE: {ce_perfect:.6f}")
    print(f"   Expected: ~0")
    
    # Worst prediction
    ce_worst = cross_entropy_bits(1, EPS)
    print(f"   Worst prediction CE: {ce_worst:.6f}")
    print(f"   Expected: ~{-math.log2(EPS):.1f}")
    
    # Random prediction
    ce_random = cross_entropy_bits(1, 0.5)
    print(f"   Random prediction CE: {ce_random:.6f}")
    print(f"   Expected: 1.0")
    
    # Test 4: Baseline entropy
    print("\n4. Baseline entropy")
    h0_balanced = baseline_entropy_bits(0.5)
    print(f"   Balanced stream H0: {h0_balanced:.6f}")
    print(f"   Expected: 1.0")
    
    h0_biased = baseline_entropy_bits(0.9)
    print(f"   Biased stream (90% 1s) H0: {h0_biased:.6f}")
    print(f"   Expected: ~0.47")
    
    print("\n✅ Validation complete!")


# Test utilities
def create_test_programs():
    """Create test programs for evaluation."""
    from .core import const, var, op, if_expr
    
    programs = {
        'always_zero': const(0),
        'always_one': const(1),
        'random': const(0.5),
        'biased_high': const(0.8),
        'biased_low': const(0.2),
    }
    
    return programs


def benchmark_evaluation_speed():
    """Benchmark evaluation speed."""
    import time
    from .envs import periodic
    
    print("\n⚡ Evaluation Speed Benchmark")
    print("=" * 40)
    
    interpreter = LispInterpreter()
    program = const(0.5)  # Simple program
    
    # Create test stream
    stream_gen = periodic([1, 0, 1, 1, 0])
    
    # Benchmark different N values
    for N in [100, 1000, 5000]:
        start_time = time.time()
        
        fitness, metrics = evaluate_program(interpreter, program, stream_gen, k=4, N=N)
        
        end_time = time.time()
        eval_time = end_time - start_time
        
        print(f"N={N:4d}: {eval_time*1000:.1f}ms ({N/eval_time:.0f} predictions/sec)")
        
        # Reset stream
        stream_gen = periodic([1, 0, 1, 1, 0])


if __name__ == "__main__":
    validate_evaluation_system()
    benchmark_evaluation_speed()
