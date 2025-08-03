"""
ProtoSynth Curriculum & Exploration

Implements auto-paced curriculum learning with:
- Learning-progress bandit for environment selection
- Robustness guards with noise schedules
- Novelty search lite with behavior signatures
"""

import random
import hashlib
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None
from typing import List, Dict, Tuple, Callable, Optional, Any
from dataclasses import dataclass
from collections import deque, defaultdict

from .core import LispNode, LispInterpreter
from .envs import periodic, k_order_markov, noisy
from .eval import evaluate_program_on_window


@dataclass
class EnvironmentSpec:
    """Specification for a training environment."""
    name: str
    factory: Callable[[], Any]  # Returns iterator
    difficulty: float  # 0.0 = easy, 1.0 = hard
    description: str


@dataclass
class BehaviorSignature:
    """Compact signature of program behavior for novelty detection."""
    ops_histogram: Dict[str, int]
    depth: int
    prediction_trace_hash: str
    ast_features: Dict[str, float]
    
    def distance(self, other: 'BehaviorSignature') -> float:
        """Calculate distance between behavior signatures."""
        # Hamming distance on prediction traces
        trace_dist = 0.0
        if self.prediction_trace_hash != other.prediction_trace_hash:
            # Simple hash-based distance (0 or 1)
            trace_dist = 1.0
        
        # L1 distance on operation histograms
        all_ops = set(self.ops_histogram.keys()) | set(other.ops_histogram.keys())
        ops_dist = sum(abs(self.ops_histogram.get(op, 0) - other.ops_histogram.get(op, 0))
                      for op in all_ops)
        ops_dist = ops_dist / max(1, len(all_ops))  # Normalize
        
        # Depth difference
        depth_dist = abs(self.depth - other.depth) / 20.0  # Normalize by max expected depth
        
        # Combine distances
        return 0.5 * trace_dist + 0.3 * ops_dist + 0.2 * depth_dist


class LearningProgressBandit:
    """
    ε-greedy bandit for environment selection based on learning progress.
    
    Reward = slope of fitness over last W generations.
    """
    
    def __init__(self, environments: List[EnvironmentSpec], window_size: int = 8, epsilon: float = 0.1):
        """
        Initialize the bandit.
        
        Args:
            environments: Available environments
            window_size: Window for computing learning progress
            epsilon: Exploration probability
        """
        self.environments = environments
        self.window_size = window_size
        self.epsilon = epsilon
        
        # Track fitness history per environment
        self.fitness_history: Dict[str, deque] = {
            env.name: deque(maxlen=window_size) for env in environments
        }
        
        # Track selection counts
        self.selection_counts: Dict[str, int] = {env.name: 0 for env in environments}
        
        # Current environment
        self.current_env: Optional[EnvironmentSpec] = None
    
    def select_environment(self) -> EnvironmentSpec:
        """Select environment using ε-greedy strategy."""
        if random.random() < self.epsilon:
            # Explore: random selection
            env = random.choice(self.environments)
        else:
            # Exploit: select environment with highest learning progress
            env = self._select_best_environment()
        
        self.current_env = env
        self.selection_counts[env.name] += 1
        return env
    
    def update_fitness(self, env_name: str, fitness: float):
        """Update fitness history for an environment."""
        self.fitness_history[env_name].append(fitness)
    
    def _select_best_environment(self) -> EnvironmentSpec:
        """Select environment with highest learning progress."""
        best_env = None
        best_progress = -float('inf')
        
        for env in self.environments:
            progress = self._compute_learning_progress(env.name)
            if progress > best_progress:
                best_progress = progress
                best_env = env
        
        return best_env or self.environments[0]
    
    def _compute_learning_progress(self, env_name: str) -> float:
        """Compute learning progress (slope) for an environment."""
        history = list(self.fitness_history[env_name])
        
        if len(history) < 2:
            return 0.0
        
        # Simple linear regression slope
        n = len(history)
        x = list(range(n))
        y = history
        
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return slope
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bandit statistics."""
        return {
            'selection_counts': dict(self.selection_counts),
            'learning_progress': {
                env.name: self._compute_learning_progress(env.name)
                for env in self.environments
            },
            'current_env': self.current_env.name if self.current_env else None
        }


class NoveltyArchive:
    """
    Archive for novelty search with behavior signatures.
    
    Maintains top-novel and top-fitness programs.
    """
    
    def __init__(self, archive_size: int = 50):
        """
        Initialize the archive.
        
        Args:
            archive_size: Maximum number of programs to store
        """
        self.archive_size = archive_size
        self.programs: List[Tuple[LispNode, float, BehaviorSignature]] = []  # (program, fitness, signature)
    
    def add_program(self, program: LispNode, fitness: float, signature: BehaviorSignature):
        """Add a program to the archive."""
        # Calculate novelty score
        novelty = self._compute_novelty(signature)
        
        # Combined score: fitness + novelty
        combined_score = fitness + 0.1 * novelty  # Weight novelty lower than fitness
        
        # Add to archive
        self.programs.append((program, combined_score, signature))
        
        # Keep only top programs
        self.programs.sort(key=lambda x: x[1], reverse=True)
        self.programs = self.programs[:self.archive_size]
    
    def _compute_novelty(self, signature: BehaviorSignature) -> float:
        """Compute novelty score based on distance to archive."""
        if not self.programs:
            return 1.0  # Maximum novelty for first program
        
        # Find k-nearest neighbors (k=5)
        k = min(5, len(self.programs))
        distances = []
        
        for _, _, archived_signature in self.programs:
            dist = signature.distance(archived_signature)
            distances.append(dist)
        
        distances.sort()
        
        # Average distance to k nearest neighbors
        novelty = sum(distances[:k]) / k
        return novelty
    
    def get_diversity_stats(self) -> Dict[str, float]:
        """Get diversity statistics."""
        if len(self.programs) < 2:
            return {'diversity': 0.0, 'archive_size': len(self.programs)}
        
        # Compute pairwise distances
        total_distance = 0.0
        count = 0
        
        for i in range(len(self.programs)):
            for j in range(i + 1, len(self.programs)):
                dist = self.programs[i][2].distance(self.programs[j][2])
                total_distance += dist
                count += 1
        
        avg_diversity = total_distance / count if count > 0 else 0.0
        
        return {
            'diversity': avg_diversity,
            'archive_size': len(self.programs)
        }


def create_behavior_signature(program: LispNode, test_bits: List[int], 
                            interpreter: LispInterpreter) -> BehaviorSignature:
    """
    Create a behavior signature for a program.
    
    Args:
        program: Program to analyze
        test_bits: Test sequence for prediction trace
        interpreter: Interpreter for evaluation
        
    Returns:
        BehaviorSignature
    """
    from .mutation import iter_nodes
    
    # Operation histogram
    ops_histogram = defaultdict(int)
    depth = 0
    
    for _, _, node in iter_nodes(program):
        if node.node_type == 'op':
            ops_histogram[node.value] += 1
        
        # Estimate depth (simplified)
        if hasattr(node, 'children') and node.children:
            depth = max(depth, len(node.children))
    
    # Prediction trace hash (first 64 predictions)
    try:
        predictions = []
        for i in range(min(64, len(test_bits) - 1)):
            context = test_bits[max(0, i-2):i+1]  # Simple context
            env = {'context': context}
            
            try:
                pred = interpreter.evaluate(program, env)
                predictions.append(1 if pred > 0.5 else 0)
            except:
                predictions.append(0)  # Default prediction on error
        
        # Hash the prediction sequence
        pred_str = ''.join(map(str, predictions))
        trace_hash = hashlib.md5(pred_str.encode()).hexdigest()[:16]
        
    except:
        trace_hash = "error"
    
    # AST features
    ast_features = {
        'total_nodes': len(list(iter_nodes(program))),
        'op_diversity': len(ops_histogram),
        'max_depth': depth
    }
    
    return BehaviorSignature(
        ops_histogram=dict(ops_histogram),
        depth=depth,
        prediction_trace_hash=trace_hash,
        ast_features=ast_features
    )


def create_default_environments() -> List[EnvironmentSpec]:
    """Create default curriculum environments."""
    return [
        EnvironmentSpec(
            name="periodic_simple",
            factory=lambda: periodic([1, 0], seed=42),
            difficulty=0.1,
            description="Simple alternating pattern"
        ),
        EnvironmentSpec(
            name="periodic_complex",
            factory=lambda: periodic([1, 0, 1, 1], seed=42),
            difficulty=0.3,
            description="Complex periodic pattern"
        ),
        EnvironmentSpec(
            name="markov_simple",
            factory=lambda: k_order_markov(1, {(0,): 0.7, (1,): 0.3}, seed=42),
            difficulty=0.5,
            description="Simple Markov chain"
        ),
        EnvironmentSpec(
            name="markov_complex",
            factory=lambda: k_order_markov(2, {
                (0, 0): 0.8, (0, 1): 0.3,
                (1, 0): 0.7, (1, 1): 0.2
            }, seed=42),
            difficulty=0.7,
            description="Complex Markov chain"
        ),
        EnvironmentSpec(
            name="noisy_periodic",
            factory=lambda: noisy(periodic([1, 0, 1], seed=42), p_flip=0.05),
            difficulty=0.6,
            description="Noisy periodic pattern"
        ),
        EnvironmentSpec(
            name="noisy_markov",
            factory=lambda: noisy(k_order_markov(1, {(0,): 0.6, (1,): 0.4}, seed=42), p_flip=0.1),
            difficulty=0.9,
            description="Noisy Markov chain"
        )
    ]
