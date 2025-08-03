"""
ProtoSynth Evaluation System

This module implements the fitness evaluation system based on cross-entropy
and compression-driven metrics. The core idea is that better predictors
achieve lower cross-entropy, which translates to better compression.
"""

import math
import logging
from typing import List, Tuple, Iterator, Optional, Dict, Any
from .predictor import PredictorAdapter
from .core import LispNode, LispInterpreter

logger = logging.getLogger(__name__)

# Shared evaluation adapter to ensure consistency
EVAL_ADAPTER = None  # Will be set to the predictor function

# Small epsilon for numerical stability (reduced for less punitive CE)
EPS = 1e-12


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


def calibrate_delta(pred_bits: List[int], y_bits: List[int], eps: float = 1e-12) -> float:
    """
    Calibrate binary predictions to probabilities using MLE.

    Args:
        pred_bits: Binary predictions (0 or 1)
        y_bits: True binary labels
        eps: Minimum/maximum delta to avoid extreme probabilities

    Returns:
        Optimal delta for mapping: p = (1-2δ)*b + δ
    """
    T = len(y_bits)
    if T == 0:
        return 0.5

    errs = sum(int(pb != yt) for pb, yt in zip(pred_bits, y_bits))
    d = errs / max(1, T)
    return max(eps, min(0.5 - eps, d))

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
    # Reset interpreter state for fresh evaluation
    interpreter.reset_tracker(max_steps=max(1000, N * 2), timeout=10.0)

    adapter = PredictorAdapter(interpreter)

    # Set global adapter for consistency checking
    global EVAL_ADAPTER
    if EVAL_ADAPTER is None:
        EVAL_ADAPTER = adapter.predict

    # Assert we're using the same adapter everywhere
    assert adapter.predict.__name__ == EVAL_ADAPTER.__name__, f"Adapter mismatch in evaluation path"
    
    # Collect stream data
    buf = []
    predictions = []
    targets = []
    total_loss = 0.0  # Track additional penalties
    total_ctx_reads = 0  # Track total context reads across all predictions
    exc_counts = {}  # Track exception types

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
            # Reset interpreter before each prediction to prevent step accumulation
            interpreter.reset_tracker()
            prediction = adapter.predict(program, context)
            predictions.append(prediction)
            targets.append(target)
            logger.debug(f"ctx={context} -> p={prediction:.3f}, y={target}")

            # Accumulate context reads from this prediction
            if hasattr(adapter.interpreter, 'ctx_reads'):
                total_ctx_reads += adapter.interpreter.ctx_reads

        except Exception as e:
            # Count exception types for debugging
            exc_type = type(e).__name__
            exc_counts[exc_type] = exc_counts.get(exc_type, 0) + 1

            logger.debug(f"Prediction failed: {exc_type}: {e}")
            # Failed prediction - strict penalty (no silent fallback)
            predictions.append(0.5)  # Use baseline for cross-entropy calculation
            targets.append(target)
            total_loss += 1.5  # Penalty for failure
        
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

    # Add penalty for failed predictions
    penalty_per_symbol = total_loss / len(predictions) if len(predictions) > 0 else 0.0
    Hprog += penalty_per_symbol

    # Calculate base fitness (bits saved per symbol)
    base_fitness = H0 - Hprog

    # Add context bonus: small reward for non-constant behavior
    context_bonus = 0.0
    if len(predictions) > 1:
        # Calculate variance of predictions (proxy for using context)
        mean_pred = sum(predictions) / len(predictions)
        variance = sum((p - mean_pred)**2 for p in predictions) / len(predictions)
        # Small bonus: 0.01 * variance (max ~0.0025 for binary predictions)
        context_bonus = 0.01 * variance

    # Final fitness with context bonus
    fitness = base_fitness + context_bonus

    # Calculate total bits saved
    total_bits_saved = len(predictions) * fitness
    
    # Log exception counts for debugging
    if exc_counts:
        logger.debug(f"Eval exceptions: {exc_counts}")

    # Prepare metrics
    metrics = {
        'num_predictions': len(predictions),
        'empirical_1_rate': q,
        'baseline_entropy': H0,
        'model_entropy': Hprog,
        'base_fitness': base_fitness,
        'context_bonus': context_bonus,
        'fitness': fitness,
        'total_bits_saved': total_bits_saved,
        'context_length': k,
        'avg_prediction': sum(predictions) / len(predictions),
        'prediction_variance': sum((p - sum(predictions)/len(predictions))**2 for p in predictions) / len(predictions),
        'ctx_reads': total_ctx_reads,
        'ctx_reads_per_eval': total_ctx_reads / len(predictions) if len(predictions) > 0 else 0.0,
        'penalty_bits': total_loss,
        'penalty_per_symbol': penalty_per_symbol,
        'exception_counts': exc_counts
    }
    
    logger.info(f"Evaluation complete: F={fitness:.4f}, H0={H0:.4f}, Hprog={Hprog:.4f}")
    
    return fitness, metrics


def evaluate_program_calibrated(interpreter: LispInterpreter, program: LispNode,
                               buffer: List[int], k: int, N_train: int = 2048,
                               N_val: int = 2048) -> Tuple[float, Dict]:
    """
    Evaluate program with state-conditioned probability calibration for binary predictors.

    Args:
        interpreter: Lisp interpreter
        program: AST program to evaluate
        buffer: Full bit buffer (train + val)
        k: Context length
        N_train: Training samples for calibration
        N_val: Validation samples for final evaluation

    Returns:
        Tuple of (fitness, metrics) on validation set
    """
    # Reset interpreter state
    interpreter.reset_tracker()

    # Check if this is a markov_table program that needs MLE parameters
    from .evolve import program_is_probabilistic
    if program_is_probabilistic(program):
        # Fit MLE parameters on the training portion
        _fit_mle_parameters(interpreter, buffer, k, N_train)

    adapter = PredictorAdapter(interpreter)

    # Train slice - collect per-state errors for state-conditioned calibration
    stats = {(a, b): {"n": 0, "err": 0} for a in (0, 1) for b in (0, 1)}

    for i in range(k, k + N_train):
        if i >= len(buffer):
            break

        ctx = buffer[i-k:i]
        y = buffer[i]

        try:
            # Reset interpreter before each prediction to prevent step accumulation
            interpreter.reset_tracker()
            p_raw = adapter.predict(program, ctx)
            b = 1 if p_raw >= 0.5 else 0

            # Get Markov state (last 2 bits of context)
            if len(ctx) >= 2:
                s = (ctx[-2], ctx[-1])
            else:
                s = (0, ctx[-1]) if len(ctx) >= 1 else (0, 0)

            stats[s]["n"] += 1
            stats[s]["err"] += int(b != y)

        except Exception:
            # Default to state (0,0) for failed predictions
            s = (0, 0)
            stats[s]["n"] += 1
            stats[s]["err"] += 1

    # Calculate per-state deltas
    delta = {}
    for s in stats:
        n = stats[s]["n"]
        err = stats[s]["err"]
        delta[s] = max(1e-3, min(0.5 - 1e-3, err / max(1, n)))

    def map_prob(b: int, s: tuple) -> float:
        """Map binary prediction to calibrated probability per state."""
        d = delta.get(s, 0.5)  # Default to 0.5 if state not seen
        return (1 - d) if b == 1 else d

    # Validation slice with calibrated probabilities
    val_losses = []
    ones = 0

    # Context reads will be tracked as binary indicator

    for i in range(k + N_train, k + N_train + N_val):
        if i >= len(buffer):
            break

        ctx = buffer[i-k:i]
        y = buffer[i]

        try:
            # Reset interpreter before each prediction to prevent step accumulation
            interpreter.reset_tracker()
            p_raw = adapter.predict(program, ctx)

            # Import probabilistic program detection
            from .evolve import program_is_probabilistic

            # Check if this is a probabilistic program
            is_prob_program = program_is_probabilistic(program)

            if is_prob_program:
                # Probabilistic programs output probabilities - use as-is, NO calibration
                p = max(1e-6, min(1.0 - 1e-6, p_raw))
            elif abs(p_raw - round(p_raw)) < 1e-6:  # Binary output (0 or 1)
                # Apply state-conditioned calibration to binary predictions
                b = 1 if p_raw >= 0.5 else 0

                # Get Markov state for calibration
                if len(ctx) >= 2:
                    s = (ctx[-2], ctx[-1])
                else:
                    s = (0, ctx[-1]) if len(ctx) >= 1 else (0, 0)

                p = map_prob(b, s)
            else:
                # Non-binary, non-probabilistic - clamp to valid probability
                p = max(1e-6, min(1.0 - 1e-6, p_raw))

            val_losses.append(cross_entropy_bits(y, p))
            ones += y

        except Exception:
            val_losses.append(1.5)  # Penalty

    # Context reads tracking (binary: did we access any context variables?)

    if not val_losses:
        return -float('inf'), {'error': 'No validation data'}

    # Calculate fitness
    q = ones / len(val_losses)
    H0 = baseline_entropy_bits(q)
    Hprog = sum(val_losses) / len(val_losses)
    fitness = H0 - Hprog

    metrics = {
        'num_predictions': len(val_losses),
        'empirical_1_rate': q,
        'baseline_entropy': H0,
        'model_entropy': Hprog,
        'fitness': fitness,
        'delta_calibration': delta,
        'train_samples': sum(stats[s]["n"] for s in stats),
        'val_samples': len(val_losses),
        'ctx_reads_per_eval': 1.0 if interpreter.ctx_reads > 0 else 0.0,  # Binary: did we access context?
    }

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
    
    from .core import const
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


class NGramPredictor:
    """Simple n-gram predictor for baseline comparisons."""

    def __init__(self, order: int = 3, alpha: float = 0.1):
        """
        Initialize n-gram predictor.

        Args:
            order: N-gram order (context length)
            alpha: Smoothing parameter
        """
        self.order = order
        self.alpha = alpha
        self.counts = {}
        self.context_counts = {}

    def fit(self, sequence: List[int]):
        """Fit the n-gram model to a sequence."""
        self.counts.clear()
        self.context_counts.clear()

        for i in range(len(sequence)):
            # Get context
            context = tuple(sequence[max(0, i-self.order):i])

            # Count context
            if context not in self.context_counts:
                self.context_counts[context] = 0
            self.context_counts[context] += 1

            # Count context + next symbol
            if i < len(sequence):
                next_symbol = sequence[i]
                key = (context, next_symbol)

                if key not in self.counts:
                    self.counts[key] = 0
                self.counts[key] += 1

    def predict_proba(self, context: List[int], next_symbol: int) -> float:
        """Predict probability of next symbol given context."""
        context_tuple = tuple(context[-self.order:]) if len(context) >= self.order else tuple(context)

        key = (context_tuple, next_symbol)

        # Add-alpha smoothing
        count = self.counts.get(key, 0)
        context_count = self.context_counts.get(context_tuple, 0)

        # Smoothed probability
        prob = (count + self.alpha) / (context_count + 2 * self.alpha)

        return prob


def _fit_mle_parameters(interpreter: LispInterpreter, buffer: List[int], k: int, N_train: int):
    """Fit MLE parameters for markov_table on training data."""
    # Collect per-state counts on training portion
    state_counts = {(a, b): {'n': 0, 'c1': 0} for a in (0, 1) for b in (0, 1)}

    for i in range(k, min(k + N_train, len(buffer))):
        ctx = buffer[i-k:i]
        y = buffer[i]

        if len(ctx) >= 2:
            s = (ctx[-2], ctx[-1])
            state_counts[s]['n'] += 1
            if y == 1:
                state_counts[s]['c1'] += 1

    # Compute MLE parameters with Laplace smoothing
    mle_params = {}
    for s in state_counts:
        n_s = state_counts[s]['n']
        c1_s = state_counts[s]['c1']

        # MLE with Laplace: P(next=1|s) = (c1 + 1) / (n + 2)
        p1_mle = (c1_s + 1) / (n_s + 2)
        p0_mle = 1 - p1_mle

        param_key = f'p{s[0]}{s[1]}'
        mle_params[param_key] = p0_mle  # Store P(next=0|s) for compatibility

    # Set parameters in interpreter
    interpreter.markov_params = mle_params


class EvalSession:
    """Context manager for consistent evaluation with proper state management."""

    def __init__(self, interpreter: LispInterpreter):
        self.interpreter = interpreter
        self.adapter = PredictorAdapter(interpreter)

    def predict(self, ast: LispNode, ctx: List[int]) -> float:
        """Make a prediction with guaranteed clean state."""
        # Reset interpreter state - CRITICAL for preventing step accumulation
        self.interpreter.reset_tracker()

        # Assert tracker is reset (guardrail)
        assert self.interpreter.step_count == 0, f"Tracker not reset: steps={self.interpreter.step_count}"

        # Use adapter for consistent prediction
        return self.adapter.predict(ast, ctx)


if __name__ == "__main__":
    validate_evaluation_system()
    benchmark_evaluation_speed()
