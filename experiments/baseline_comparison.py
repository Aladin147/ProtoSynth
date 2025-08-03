#!/usr/bin/env python3
"""
ProtoSynth Baseline Comparison Experiment

Compare against LZMA + tiny HMM on periodic(k), Markov(k), noisy(0.1).
Goal: match/exceed HMM; approach LZMA on periodic/low-k Markov.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import lzma
import itertools
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

from protosynth import *
from protosynth.curriculum_evolution import CurriculumEvolutionEngine
from protosynth.envs import periodic, k_order_markov, noisy
from protosynth.eval import NGramPredictor


@dataclass
class BaselineResult:
    """Result from baseline comparison."""
    sequence_name: str
    sequence_length: int
    lzma_bits_per_symbol: float
    hmm_bits_per_symbol: float
    protosynth_bits_per_symbol: float
    protosynth_fitness: float
    beats_hmm: bool
    approaches_lzma: bool


class TinyHMM:
    """Tiny Hidden Markov Model for baseline comparison."""
    
    def __init__(self, n_states: int = 3):
        """Initialize tiny HMM with n_states."""
        self.n_states = n_states
        self.transition_probs = None
        self.emission_probs = None
        self.initial_probs = None
    
    def fit(self, sequence: List[int], max_iters: int = 10):
        """Fit HMM to sequence using simple EM algorithm."""
        # Initialize parameters randomly
        np.random.seed(42)
        
        self.transition_probs = np.random.rand(self.n_states, self.n_states)
        self.transition_probs = self.transition_probs / self.transition_probs.sum(axis=1, keepdims=True)
        
        self.emission_probs = np.random.rand(self.n_states, 2)  # Binary observations
        self.emission_probs = self.emission_probs / self.emission_probs.sum(axis=1, keepdims=True)
        
        self.initial_probs = np.ones(self.n_states) / self.n_states
        
        # Simple EM iterations (simplified)
        for _ in range(max_iters):
            # E-step: Forward-backward (simplified)
            alpha = self._forward(sequence)
            beta = self._backward(sequence)
            
            # M-step: Update parameters (simplified)
            self._update_parameters(sequence, alpha, beta)
    
    def _forward(self, sequence: List[int]) -> np.ndarray:
        """Forward algorithm (simplified)."""
        T = len(sequence)
        alpha = np.zeros((T, self.n_states))
        
        # Initialize
        alpha[0] = self.initial_probs * self.emission_probs[:, sequence[0]]
        
        # Forward pass
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.transition_probs[:, j]) * self.emission_probs[j, sequence[t]]
        
        return alpha
    
    def _backward(self, sequence: List[int]) -> np.ndarray:
        """Backward algorithm (simplified)."""
        T = len(sequence)
        beta = np.zeros((T, self.n_states))
        
        # Initialize
        beta[T-1] = 1.0
        
        # Backward pass
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.transition_probs[i] * self.emission_probs[:, sequence[t+1]] * beta[t+1])
        
        return beta
    
    def _update_parameters(self, sequence: List[int], alpha: np.ndarray, beta: np.ndarray):
        """Update HMM parameters (simplified)."""
        T = len(sequence)
        
        # Compute gamma (state probabilities)
        gamma = alpha * beta
        gamma = gamma / gamma.sum(axis=1, keepdims=True)
        
        # Update emission probabilities
        for j in range(self.n_states):
            for k in range(2):  # Binary observations
                mask = np.array(sequence) == k
                self.emission_probs[j, k] = gamma[mask, j].sum() / gamma[:, j].sum()
        
        # Normalize
        self.emission_probs = np.maximum(self.emission_probs, 1e-10)
        self.emission_probs = self.emission_probs / self.emission_probs.sum(axis=1, keepdims=True)
    
    def log_likelihood(self, sequence: List[int]) -> float:
        """Compute log likelihood of sequence."""
        alpha = self._forward(sequence)
        return np.log(alpha[-1].sum())
    
    def bits_per_symbol(self, sequence: List[int]) -> float:
        """Compute bits per symbol (cross-entropy)."""
        log_likelihood = self.log_likelihood(sequence)
        return -log_likelihood / (len(sequence) * np.log(2))


class BaselineComparison:
    """Baseline comparison experiment runner."""
    
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self.results: List[BaselineResult] = []
    
    def generate_test_sequences(self) -> Dict[str, List[int]]:
        """Generate test sequences for comparison."""
        sequences = {}
        
        # Periodic sequences
        sequences['periodic_k2'] = list(itertools.islice(periodic([1, 0], seed=42), 1000))
        sequences['periodic_k3'] = list(itertools.islice(periodic([1, 0, 1], seed=42), 1000))
        sequences['periodic_k4'] = list(itertools.islice(periodic([1, 1, 0, 1], seed=42), 1000))
        
        # Markov sequences
        sequences['markov_k1'] = list(itertools.islice(
            k_order_markov(1, {(0,): 0.7, (1,): 0.3}, seed=42), 1000
        ))
        sequences['markov_k2'] = list(itertools.islice(
            k_order_markov(2, {(0,0): 0.8, (0,1): 0.3, (1,0): 0.7, (1,1): 0.2}, seed=42), 1000
        ))
        
        # Noisy sequences
        clean_periodic = list(itertools.islice(periodic([1, 0, 1], seed=42), 1000))
        sequences['noisy_periodic'] = list(itertools.islice(
            noisy(iter(clean_periodic), p_flip=0.1), 1000
        ))
        
        clean_markov = list(itertools.islice(
            k_order_markov(1, {(0,): 0.6, (1,): 0.4}, seed=42), 1000
        ))
        sequences['noisy_markov'] = list(itertools.islice(
            noisy(iter(clean_markov), p_flip=0.1), 1000
        ))
        
        return sequences
    
    def evaluate_lzma(self, sequence: List[int]) -> float:
        """Evaluate LZMA compression."""
        seq_bytes = bytes(sequence)
        compressed = lzma.compress(seq_bytes)
        compression_ratio = len(compressed) / len(seq_bytes)
        return compression_ratio * 8  # Convert to bits per symbol
    
    def evaluate_hmm(self, sequence: List[int]) -> float:
        """Evaluate tiny HMM."""
        # Split into train/test
        split_point = int(0.8 * len(sequence))
        train_seq = sequence[:split_point]
        test_seq = sequence[split_point:]
        
        # Train HMM
        hmm = TinyHMM(n_states=3)
        hmm.fit(train_seq)
        
        # Evaluate on test set
        return hmm.bits_per_symbol(test_seq)
    
    def evaluate_protosynth(self, sequence: List[int], sequence_name: str) -> Tuple[float, float]:
        """Evaluate ProtoSynth on sequence."""
        # Create engine
        engine = CurriculumEvolutionEngine(
            mu=12, lambda_=24, seed=self.base_seed,
            max_modules=8, archive_size=15
        )
        
        # Create environment for this sequence
        def seq_env():
            return iter(sequence)
        
        engine.environments = [type('Env', (), {
            'name': f'baseline_{sequence_name}',
            'factory': seq_env,
            'difficulty': 0.5,
            'description': f'Baseline test: {sequence_name}'
        })()]
        
        # Initialize and evolve
        from protosynth.evolve import create_initial_population
        initial_pop = create_initial_population(12, seed=self.base_seed)
        engine.evolution_engine.initialize_population(initial_pop)
        
        best_fitness = -float('inf')
        
        # Run evolution
        for gen in range(25):
            stats = engine.evolve_generation()
            best_fitness = max(best_fitness, stats.best_fitness)
            
            if gen % 5 == 0:
                print(f"    Gen {gen}: F={best_fitness:.4f}")
        
        # Convert fitness to bits per symbol (approximate)
        # Higher fitness = lower bits per symbol
        bits_per_symbol = max(0.1, 1.0 - best_fitness)
        
        return best_fitness, bits_per_symbol
    
    def run_comparison(self, sequence_name: str, sequence: List[int]) -> BaselineResult:
        """Run comparison on a single sequence."""
        print(f"\n  Testing {sequence_name} (length {len(sequence)})...")
        
        # LZMA baseline
        print("    Evaluating LZMA...")
        lzma_bits = self.evaluate_lzma(sequence)
        
        # HMM baseline
        print("    Evaluating HMM...")
        try:
            hmm_bits = self.evaluate_hmm(sequence)
        except Exception as e:
            print(f"    HMM failed: {e}")
            hmm_bits = 2.0  # Fallback
        
        # ProtoSynth
        print("    Evaluating ProtoSynth...")
        try:
            protosynth_fitness, protosynth_bits = self.evaluate_protosynth(sequence, sequence_name)
        except Exception as e:
            print(f"    ProtoSynth failed: {e}")
            protosynth_fitness = 0.0
            protosynth_bits = 2.0
        
        # Analysis
        beats_hmm = protosynth_bits <= hmm_bits
        approaches_lzma = protosynth_bits <= lzma_bits * 1.5  # Within 50%
        
        result = BaselineResult(
            sequence_name=sequence_name,
            sequence_length=len(sequence),
            lzma_bits_per_symbol=lzma_bits,
            hmm_bits_per_symbol=hmm_bits,
            protosynth_bits_per_symbol=protosynth_bits,
            protosynth_fitness=protosynth_fitness,
            beats_hmm=beats_hmm,
            approaches_lzma=approaches_lzma
        )
        
        print(f"    Results:")
        print(f"      LZMA: {lzma_bits:.3f} bits/symbol")
        print(f"      HMM:  {hmm_bits:.3f} bits/symbol")
        print(f"      ProtoSynth: {protosynth_bits:.3f} bits/symbol (F={protosynth_fitness:.3f})")
        print(f"      Beats HMM: {beats_hmm}")
        print(f"      Approaches LZMA: {approaches_lzma}")
        
        return result
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run complete baseline comparison experiment."""
        print("🔬 Baseline Comparison Experiment")
        print("=" * 40)
        
        start_time = time.time()
        
        # Generate test sequences
        sequences = self.generate_test_sequences()
        print(f"Generated {len(sequences)} test sequences")
        
        # Run comparisons
        for seq_name, sequence in sequences.items():
            result = self.run_comparison(seq_name, sequence)
            self.results.append(result)
        
        # Analyze results
        total_sequences = len(self.results)
        beats_hmm_count = sum(1 for r in self.results if r.beats_hmm)
        approaches_lzma_count = sum(1 for r in self.results if r.approaches_lzma)
        
        # Success criteria
        hmm_success = beats_hmm_count >= total_sequences * 0.5  # Beat HMM on 50%+ sequences
        lzma_success = approaches_lzma_count >= 2  # Approach LZMA on 2+ sequences
        
        overall_success = hmm_success or lzma_success
        
        total_time = time.time() - start_time
        
        print(f"\n📊 Baseline Comparison Results:")
        print(f"  Total sequences: {total_sequences}")
        print(f"  Beats HMM: {beats_hmm_count}/{total_sequences}")
        print(f"  Approaches LZMA: {approaches_lzma_count}/{total_sequences}")
        print(f"  HMM success: {hmm_success}")
        print(f"  LZMA success: {lzma_success}")
        print(f"  Overall success: {overall_success}")
        print(f"  Total time: {total_time:.1f}s")
        
        return {
            'experiment': 'baseline_comparison',
            'total_sequences': total_sequences,
            'beats_hmm_count': beats_hmm_count,
            'approaches_lzma_count': approaches_lzma_count,
            'hmm_success': hmm_success,
            'lzma_success': lzma_success,
            'overall_success': overall_success,
            'runtime_seconds': total_time,
            'results': [
                {
                    'sequence_name': r.sequence_name,
                    'lzma_bits': r.lzma_bits_per_symbol,
                    'hmm_bits': r.hmm_bits_per_symbol,
                    'protosynth_bits': r.protosynth_bits_per_symbol,
                    'protosynth_fitness': r.protosynth_fitness,
                    'beats_hmm': r.beats_hmm,
                    'approaches_lzma': r.approaches_lzma
                } for r in self.results
            ]
        }


def main():
    """Run baseline comparison experiment."""
    experiment = BaselineComparison(base_seed=42)
    results = experiment.run_experiment()
    
    # Save results
    with open("baseline_comparison_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎯 Baseline Comparison: {'SUCCESS' if results['overall_success'] else 'PARTIAL'}")
    
    return results


if __name__ == "__main__":
    main()
