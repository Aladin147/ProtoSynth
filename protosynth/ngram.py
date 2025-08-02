"""
ProtoSynth N-gram Baseline Predictor

This module implements a classical n-gram predictor with add-α smoothing
as a baseline reference model for comparison with evolved programs.
"""

import logging
from typing import List, Dict, Tuple
from collections import defaultdict
from .eval import baseline_entropy_bits, cross_entropy_bits

logger = logging.getLogger(__name__)


class NGramPredictor:
    """
    N-gram predictor with add-α (Laplace) smoothing.
    
    This serves as a baseline reference model that uses classical
    statistical methods to predict the next bit based on context.
    """
    
    def __init__(self, k: int = 4, alpha: float = 1.0):
        """
        Initialize the n-gram predictor.
        
        Args:
            k: Context length (order of the n-gram model)
            alpha: Smoothing parameter (add-α smoothing)
        """
        self.k = k
        self.alpha = alpha
        
        # Context counts: context_tuple -> total_count
        self.context_counts = defaultdict(int)
        
        # Outcome counts: (context_tuple, outcome) -> count
        self.outcome_counts = defaultdict(int)
        
        # Flag to track if model has been trained
        self.is_trained = False
        
        logger.debug(f"Initialized {k}-gram predictor with α={alpha}")
    
    def train(self, bits: List[int]) -> None:
        """
        Train the n-gram model on a sequence of bits.
        
        Args:
            bits: Training sequence of 0s and 1s
        """
        if len(bits) <= self.k:
            logger.warning(f"Training sequence too short: {len(bits)} <= {self.k}")
            return
        
        # Clear previous training
        self.context_counts.clear()
        self.outcome_counts.clear()
        
        # Scan through the sequence
        for i in range(self.k, len(bits)):
            # Extract context and outcome
            context = tuple(bits[i-self.k:i])
            outcome = bits[i]
            
            # Update counts
            self.context_counts[context] += 1
            self.outcome_counts[(context, outcome)] += 1
        
        self.is_trained = True
        
        num_contexts = len(self.context_counts)
        total_observations = sum(self.context_counts.values())
        
        logger.info(f"Trained on {total_observations} observations, "
                   f"{num_contexts} unique contexts")
    
    def predict(self, context: List[int]) -> float:
        """
        Predict probability of next bit being 1 given context.
        
        Args:
            context: List of previous k bits
            
        Returns:
            float: Probability of next bit being 1
        """
        if not self.is_trained:
            logger.warning("Model not trained, returning 0.5")
            return 0.5
        
        # Ensure context has correct length
        if len(context) != self.k:
            if len(context) < self.k:
                # Pad with zeros if too short
                context = [0] * (self.k - len(context)) + context
            else:
                # Truncate if too long
                context = context[-self.k:]
        
        context_tuple = tuple(context)
        
        # Get counts with smoothing
        count_context = self.context_counts[context_tuple]
        count_1 = self.outcome_counts[(context_tuple, 1)]
        count_0 = self.outcome_counts[(context_tuple, 0)]
        
        # Apply add-α smoothing
        # P(1|context) = (count(context,1) + α) / (count(context) + 2α)
        numerator = count_1 + self.alpha
        denominator = count_context + 2 * self.alpha
        
        probability = numerator / denominator
        
        logger.debug(f"Predict: ctx={context} -> counts=({count_0},{count_1}) -> p={probability:.3f}")
        
        return probability
    
    def evaluate_on_stream(self, bits: List[int]) -> Tuple[float, Dict]:
        """
        Evaluate the n-gram model on a test sequence.
        
        Args:
            bits: Test sequence of bits
            
        Returns:
            Tuple of (fitness, metrics_dict)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        if len(bits) <= self.k:
            return -float('inf'), {'num_predictions': 0}
        
        predictions = []
        targets = []
        
        # Make predictions
        for i in range(self.k, len(bits)):
            context = bits[i-self.k:i]
            target = bits[i]
            
            prediction = self.predict(context)
            predictions.append(prediction)
            targets.append(target)
        
        # Calculate metrics
        if not predictions:
            return -float('inf'), {'num_predictions': 0}
        
        # Empirical 1-rate
        q = sum(targets) / len(targets)
        
        # Baseline entropy
        H0 = baseline_entropy_bits(q)
        
        # Model cross-entropy
        cross_entropies = [cross_entropy_bits(y, p) for y, p in zip(targets, predictions)]
        Hprog = sum(cross_entropies) / len(cross_entropies)
        
        # Fitness
        fitness = H0 - Hprog
        
        # Metrics
        metrics = {
            'num_predictions': len(predictions),
            'empirical_1_rate': q,
            'baseline_entropy': H0,
            'model_entropy': Hprog,
            'fitness': fitness,
            'total_bits_saved': len(predictions) * fitness,
            'context_length': self.k,
            'smoothing_alpha': self.alpha,
            'num_unique_contexts': len(self.context_counts),
            'avg_prediction': sum(predictions) / len(predictions)
        }
        
        logger.info(f"N-gram evaluation: F={fitness:.4f}, H0={H0:.4f}, Hprog={Hprog:.4f}")
        
        return fitness, metrics
    
    def get_model_info(self) -> Dict:
        """Get information about the trained model."""
        return {
            'k': self.k,
            'alpha': self.alpha,
            'is_trained': self.is_trained,
            'num_contexts': len(self.context_counts),
            'total_observations': sum(self.context_counts.values()) if self.is_trained else 0
        }


def compare_ngram_orders(bits: List[int], max_k: int = 6, alpha: float = 1.0) -> Dict[int, Tuple[float, Dict]]:
    """
    Compare n-gram predictors of different orders on the same data.
    
    Args:
        bits: Bit sequence for evaluation
        max_k: Maximum context length to test
        alpha: Smoothing parameter
        
    Returns:
        Dict mapping k -> (fitness, metrics)
    """
    if len(bits) < max_k + 100:
        raise ValueError("Bit sequence too short for comparison")
    
    # Split into train/test
    split_point = len(bits) // 2
    train_bits = bits[:split_point]
    test_bits = bits[split_point:]
    
    results = {}
    
    print(f"🔍 Comparing N-gram orders (α={alpha})")
    print("=" * 40)
    
    for k in range(1, max_k + 1):
        if len(train_bits) <= k or len(test_bits) <= k:
            continue
        
        # Train model
        model = NGramPredictor(k=k, alpha=alpha)
        model.train(train_bits)
        
        # Evaluate on test set
        fitness, metrics = model.evaluate_on_stream(test_bits)
        results[k] = (fitness, metrics)
        
        print(f"k={k}: F={fitness:.4f}, H_prog={metrics['model_entropy']:.4f}, "
              f"contexts={metrics['num_unique_contexts']}")
    
    return results


def benchmark_ngram_performance():
    """Benchmark n-gram predictor performance."""
    import time
    from .envs import periodic
    
    print("\n⚡ N-gram Performance Benchmark")
    print("=" * 40)
    
    # Create test data
    test_bits = []
    stream = periodic([1, 0, 1, 1, 0])
    for i, bit in enumerate(stream):
        test_bits.append(bit)
        if i >= 10000:
            break
    
    # Test different model sizes
    for k in [2, 4, 6]:
        model = NGramPredictor(k=k, alpha=1.0)
        
        # Time training
        start_time = time.time()
        model.train(test_bits[:5000])
        train_time = time.time() - start_time
        
        # Time evaluation
        start_time = time.time()
        fitness, metrics = model.evaluate_on_stream(test_bits[5000:])
        eval_time = time.time() - start_time
        
        predictions_per_sec = metrics['num_predictions'] / eval_time if eval_time > 0 else 0
        
        print(f"k={k}: train={train_time*1000:.1f}ms, eval={eval_time*1000:.1f}ms, "
              f"{predictions_per_sec:.0f} pred/sec")


if __name__ == "__main__":
    # Demo the n-gram predictor
    print("🧮 N-gram Predictor Demo")
    print("=" * 30)
    
    # Create simple test data
    test_data = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 10
    
    # Train and evaluate
    model = NGramPredictor(k=2, alpha=1.0)
    model.train(test_data[:50])
    
    fitness, metrics = model.evaluate_on_stream(test_data[50:])
    
    print(f"Fitness: {fitness:.4f}")
    print(f"Model entropy: {metrics['model_entropy']:.4f}")
    print(f"Unique contexts: {metrics['num_unique_contexts']}")
    
    benchmark_ngram_performance()
