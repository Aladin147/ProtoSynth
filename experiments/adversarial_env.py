#!/usr/bin/env python3
"""
ProtoSynth Adversarial Environment Experiment

Teacher mutates transition table under small budget; alternate 1:3 (env:agent) steps.
Target: agent recovers ≥60% of baseline F after 50 gens.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import random
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

from protosynth import *
from protosynth.curriculum_evolution import CurriculumEvolutionEngine
from protosynth.envs import k_order_markov


@dataclass
class AdversarialResult:
    """Result from adversarial environment experiment."""
    baseline_fitness: float
    final_fitness: float
    recovery_rate: float
    generations_survived: int
    env_mutations: int
    success: bool


class AdversarialEnvironment:
    """Adversarial environment that mutates its transition table."""
    
    def __init__(self, initial_transitions: Dict[Tuple[int, ...], float], 
                 mutation_budget: float = 0.1, seed: int = 42):
        """
        Initialize adversarial environment.
        
        Args:
            initial_transitions: Initial transition probabilities
            mutation_budget: Maximum change per mutation (0.0 to 1.0)
            seed: Random seed
        """
        self.initial_transitions = initial_transitions.copy()
        self.current_transitions = initial_transitions.copy()
        self.mutation_budget = mutation_budget
        self.rng = random.Random(seed)
        self.mutation_count = 0
    
    def mutate_environment(self, agent_fitness: float) -> float:
        """
        Mutate environment to maximize agent's cross-entropy (minimize fitness).
        
        Args:
            agent_fitness: Current agent fitness
            
        Returns:
            New environment difficulty score
        """
        self.mutation_count += 1
        
        # Choose a random transition to mutate
        transitions = list(self.current_transitions.keys())
        if not transitions:
            return 0.0
        
        target_transition = self.rng.choice(transitions)
        current_prob = self.current_transitions[target_transition]
        
        # Mutate towards making the environment harder
        # If agent is doing well, make environment more unpredictable
        if agent_fitness > 0.0:
            # Make transition more random (closer to 0.5)
            if current_prob < 0.5:
                delta = min(self.mutation_budget, 0.5 - current_prob)
            else:
                delta = -min(self.mutation_budget, current_prob - 0.5)
        else:
            # Random mutation if agent is already struggling
            delta = self.rng.uniform(-self.mutation_budget, self.mutation_budget)
        
        # Apply mutation with bounds checking
        new_prob = max(0.01, min(0.99, current_prob + delta))
        self.current_transitions[target_transition] = new_prob
        
        # Normalize complementary transitions if needed
        self._normalize_transitions()
        
        # Return difficulty score (higher = more different from initial)
        difficulty = self._compute_difficulty()
        
        print(f"    Env mutation {self.mutation_count}: "
              f"{target_transition} {current_prob:.3f} → {new_prob:.3f}, "
              f"difficulty={difficulty:.3f}")
        
        return difficulty
    
    def _normalize_transitions(self):
        """Ensure transition probabilities are valid."""
        # For binary outcomes, ensure complementary probabilities sum to 1
        # This is a simplified normalization
        for context in set(k[:-1] if len(k) > 1 else () for k in self.current_transitions.keys()):
            # Find all transitions from this context
            context_transitions = [k for k in self.current_transitions.keys() 
                                 if (len(k) > 1 and k[:-1] == context) or (len(k) == 1 and context == ())]
            
            if len(context_transitions) == 2:
                # Binary case - normalize to sum to 1
                total = sum(self.current_transitions[k] for k in context_transitions)
                if total > 0:
                    for k in context_transitions:
                        self.current_transitions[k] /= total
    
    def _compute_difficulty(self) -> float:
        """Compute how much the environment has changed from initial."""
        if not self.initial_transitions:
            return 0.0
        
        total_change = 0.0
        for key in self.initial_transitions:
            if key in self.current_transitions:
                change = abs(self.current_transitions[key] - self.initial_transitions[key])
                total_change += change
        
        return total_change / len(self.initial_transitions)
    
    def create_sequence_generator(self, length: int = 1000):
        """Create sequence generator with current transitions."""
        return k_order_markov(
            order=len(list(self.current_transitions.keys())[0]) - 1,
            transitions=self.current_transitions,
            seed=self.rng.randint(0, 10000)
        )
    
    def reset(self):
        """Reset environment to initial state."""
        self.current_transitions = self.initial_transitions.copy()
        self.mutation_count = 0


class AdversarialExperiment:
    """Adversarial environment experiment runner."""
    
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self.results: List[AdversarialResult] = []
    
    def measure_baseline_fitness(self, initial_transitions: Dict[Tuple[int, ...], float]) -> float:
        """Measure baseline fitness on static environment."""
        print("  Measuring baseline fitness...")
        
        # Create baseline engine
        engine = CurriculumEvolutionEngine(
            mu=12, lambda_=24, seed=self.base_seed,
            max_modules=8, archive_size=15
        )
        
        # Create static environment
        def baseline_env():
            return k_order_markov(
                order=len(list(initial_transitions.keys())[0]) - 1,
                transitions=initial_transitions,
                seed=42
            )
        
        engine.environments = [type('Env', (), {
            'name': 'baseline_static',
            'factory': baseline_env,
            'difficulty': 0.3,
            'description': 'Baseline static environment'
        })()]
        
        # Initialize and evolve
        from protosynth.evolve import create_initial_population
        initial_pop = create_initial_population(12, seed=self.base_seed)
        engine.evolution_engine.initialize_population(initial_pop)
        
        best_fitness = -float('inf')
        
        # Evolve for baseline measurement
        for gen in range(25):
            stats = engine.evolve_generation()
            best_fitness = max(best_fitness, stats.best_fitness)
            
            if gen % 5 == 0:
                print(f"    Baseline gen {gen}: F={best_fitness:.4f}")
        
        print(f"  Baseline fitness: {best_fitness:.4f}")
        return best_fitness
    
    def run_adversarial_evolution(self, initial_transitions: Dict[Tuple[int, ...], float],
                                 baseline_fitness: float) -> AdversarialResult:
        """Run evolution against adversarial environment."""
        print("  Running adversarial evolution...")
        
        # Create adversarial environment
        adv_env = AdversarialEnvironment(
            initial_transitions=initial_transitions,
            mutation_budget=0.1,  # 10% max change per mutation
            seed=self.base_seed + 1
        )
        
        # Create agent engine
        agent_engine = CurriculumEvolutionEngine(
            mu=12, lambda_=24, seed=self.base_seed + 2,
            max_modules=8, archive_size=15
        )
        
        # Initialize agent population
        from protosynth.evolve import create_initial_population
        initial_pop = create_initial_population(12, seed=self.base_seed + 2)
        agent_engine.evolution_engine.initialize_population(initial_pop)
        
        # Adversarial evolution loop
        agent_fitness_history = []
        generations_survived = 0
        
        for gen in range(50):
            # Create current environment
            def current_env():
                return adv_env.create_sequence_generator(length=500)
            
            agent_engine.environments = [type('Env', (), {
                'name': f'adversarial_gen_{gen}',
                'factory': current_env,
                'difficulty': 0.5,
                'description': f'Adversarial environment gen {gen}'
            })()]
            
            # Agent evolution step (3 generations per environment mutation)
            best_gen_fitness = -float('inf')
            
            for agent_step in range(3):
                stats = agent_engine.evolve_generation()
                best_gen_fitness = max(best_gen_fitness, stats.best_fitness)
                generations_survived += 1
            
            agent_fitness_history.append(best_gen_fitness)
            
            # Environment mutation step (every 3 agent generations)
            if gen % 1 == 0:  # Mutate environment every generation
                env_difficulty = adv_env.mutate_environment(best_gen_fitness)
            
            if gen % 5 == 0:
                print(f"    Adversarial gen {gen}: Agent F={best_gen_fitness:.4f}, "
                      f"Env mutations={adv_env.mutation_count}")
            
            # Early stopping if agent completely fails
            if best_gen_fitness < -2.0:
                print(f"    Agent failed at generation {gen}")
                break
        
        # Analyze results
        final_fitness = agent_fitness_history[-1] if agent_fitness_history else -float('inf')
        recovery_rate = final_fitness / baseline_fitness if baseline_fitness > 0 else 0.0
        
        # Success criteria: recover ≥60% of baseline fitness
        success = recovery_rate >= 0.6
        
        result = AdversarialResult(
            baseline_fitness=baseline_fitness,
            final_fitness=final_fitness,
            recovery_rate=recovery_rate,
            generations_survived=generations_survived,
            env_mutations=adv_env.mutation_count,
            success=success
        )
        
        print(f"  Adversarial result: Final F={final_fitness:.4f}, "
              f"Recovery={recovery_rate:.1%}, Success={success}")
        
        return result
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run complete adversarial environment experiment."""
        print("🔬 Adversarial Environment Experiment")
        print("=" * 40)
        
        start_time = time.time()
        
        # Define test environments
        test_environments = [
            # Simple Markov chain
            {
                'name': 'markov_k1_simple',
                'transitions': {(0,): 0.7, (1,): 0.3}
            },
            # More complex Markov chain
            {
                'name': 'markov_k2_complex',
                'transitions': {(0,0): 0.8, (0,1): 0.2, (1,0): 0.4, (1,1): 0.6}
            }
        ]
        
        # Run experiments
        for env_config in test_environments:
            print(f"\nTesting {env_config['name']}...")
            
            # Measure baseline
            baseline_fitness = self.measure_baseline_fitness(env_config['transitions'])
            
            # Run adversarial evolution
            result = self.run_adversarial_evolution(env_config['transitions'], baseline_fitness)
            self.results.append(result)
        
        # Analyze overall results
        total_experiments = len(self.results)
        successful_experiments = sum(1 for r in self.results if r.success)
        
        avg_recovery_rate = sum(r.recovery_rate for r in self.results) / total_experiments
        avg_generations_survived = sum(r.generations_survived for r in self.results) / total_experiments
        
        overall_success = successful_experiments >= total_experiments * 0.5  # 50% success rate
        
        total_time = time.time() - start_time
        
        print(f"\n📊 Adversarial Environment Results:")
        print(f"  Total experiments: {total_experiments}")
        print(f"  Successful: {successful_experiments}")
        print(f"  Success rate: {successful_experiments/total_experiments:.1%}")
        print(f"  Average recovery rate: {avg_recovery_rate:.1%}")
        print(f"  Average generations survived: {avg_generations_survived:.1f}")
        print(f"  Overall success: {overall_success}")
        print(f"  Total time: {total_time:.1f}s")
        
        return {
            'experiment': 'adversarial_environment',
            'total_experiments': total_experiments,
            'successful_experiments': successful_experiments,
            'success_rate': successful_experiments / total_experiments,
            'average_recovery_rate': avg_recovery_rate,
            'average_generations_survived': avg_generations_survived,
            'overall_success': overall_success,
            'runtime_seconds': total_time,
            'results': [
                {
                    'baseline_fitness': r.baseline_fitness,
                    'final_fitness': r.final_fitness,
                    'recovery_rate': r.recovery_rate,
                    'generations_survived': r.generations_survived,
                    'env_mutations': r.env_mutations,
                    'success': r.success
                } for r in self.results
            ]
        }


def main():
    """Run adversarial environment experiment."""
    experiment = AdversarialExperiment(base_seed=42)
    results = experiment.run_experiment()
    
    # Save results
    with open("adversarial_env_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎯 Adversarial Environment: {'SUCCESS' if results['overall_success'] else 'PARTIAL'}")
    
    return results


if __name__ == "__main__":
    main()
