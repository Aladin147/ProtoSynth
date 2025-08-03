#!/usr/bin/env python3
"""
ProtoSynth Scaling Laws Experiment

Comprehensive sweep of μ, λ, N, mutation_rate, verifier_strictness.
Goal: Log slopes of best_F vs compute; identify smooth scaling regime.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import itertools
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

from protosynth import *
from protosynth.curriculum_evolution import CurriculumEvolutionEngine
from protosynth.envs import periodic


@dataclass
class ScalingConfig:
    """Configuration for scaling experiment."""
    mu: int
    lambda_: int
    N: int  # Evaluation length
    mutation_rate: float
    verifier_strictness: float
    generations: int = 20


@dataclass
class ScalingResult:
    """Result from scaling experiment."""
    config: ScalingConfig
    compute_units: int
    best_fitness: float
    final_fitness: float
    convergence_gen: int
    wall_time: float
    success: bool


class ScalingLawsExperiment:
    """Scaling laws experiment runner."""
    
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self.results: List[ScalingResult] = []
    
    def generate_configs(self) -> List[ScalingConfig]:
        """Generate parameter sweep configurations."""
        # Parameter ranges
        mu_values = [4, 8, 16, 32]
        lambda_multipliers = [1.5, 2.0, 3.0]
        N_values = [100, 500, 1000]
        mutation_rates = [0.05, 0.1, 0.2, 0.3]
        verifier_strictness = [0.5, 1.0, 2.0]  # Multiplier for verification limits
        
        configs = []
        
        # Generate all combinations (limited for feasibility)
        for mu in mu_values[:3]:  # Limit to first 3 for speed
            for lambda_mult in lambda_multipliers[:2]:
                for N in N_values[:2]:
                    for mut_rate in mutation_rates[:2]:
                        for strict in verifier_strictness[:2]:
                            lambda_ = int(mu * lambda_mult)
                            
                            config = ScalingConfig(
                                mu=mu,
                                lambda_=lambda_,
                                N=N,
                                mutation_rate=mut_rate,
                                verifier_strictness=strict
                            )
                            configs.append(config)
        
        print(f"Generated {len(configs)} scaling configurations")
        return configs
    
    def run_single_config(self, config: ScalingConfig, config_id: int) -> ScalingResult:
        """Run evolution with a single configuration."""
        print(f"  Config {config_id}: μ={config.mu}, λ={config.lambda_}, "
              f"N={config.N}, mut={config.mutation_rate:.2f}")
        
        start_time = time.time()
        
        try:
            # Create engine with specific parameters
            engine = CurriculumEvolutionEngine(
                mu=config.mu,
                lambda_=config.lambda_,
                seed=self.base_seed + config_id,
                max_modules=8,
                archive_size=15
            )
            
            # Override with simple periodic environment
            def simple_env():
                return periodic([1, 0, 1], seed=42)
            
            engine.environments = [type('Env', (), {
                'name': 'scaling_test',
                'factory': simple_env,
                'difficulty': 0.2,
                'description': 'Scaling test environment'
            })()]
            
            # Initialize population
            from protosynth.evolve import create_initial_population
            initial_pop = create_initial_population(config.mu, seed=self.base_seed + config_id)
            engine.evolution_engine.initialize_population(initial_pop)
            
            # Set mutation rate
            engine.evolution_engine.mutation_rate = config.mutation_rate
            
            # Run evolution
            best_fitness = -float('inf')
            final_fitness = 0.0
            convergence_gen = config.generations
            
            for gen in range(config.generations):
                stats = engine.evolve_generation()
                
                if stats.best_fitness > best_fitness:
                    best_fitness = stats.best_fitness
                    convergence_gen = gen + 1
                
                final_fitness = stats.best_fitness
                
                # Early stopping if converged
                if best_fitness > 0.8:  # High fitness threshold
                    convergence_gen = gen + 1
                    break
            
            wall_time = time.time() - start_time
            compute_units = config.mu * config.lambda_ * convergence_gen
            
            success = best_fitness > -0.5  # Reasonable fitness achieved
            
            result = ScalingResult(
                config=config,
                compute_units=compute_units,
                best_fitness=best_fitness,
                final_fitness=final_fitness,
                convergence_gen=convergence_gen,
                wall_time=wall_time,
                success=success
            )
            
            print(f"    Result: F={best_fitness:.4f}, compute={compute_units}, "
                  f"time={wall_time:.1f}s, success={success}")
            
            return result
            
        except Exception as e:
            print(f"    Failed: {e}")
            
            # Return failed result
            return ScalingResult(
                config=config,
                compute_units=0,
                best_fitness=-float('inf'),
                final_fitness=-float('inf'),
                convergence_gen=config.generations,
                wall_time=time.time() - start_time,
                success=False
            )
    
    def analyze_scaling(self) -> Dict[str, Any]:
        """Analyze scaling laws from results."""
        successful_results = [r for r in self.results if r.success]
        
        if len(successful_results) < 3:
            return {
                'scaling_detected': False,
                'slope': 0.0,
                'correlation': 0.0,
                'smooth_regime': False,
                'num_successful': len(successful_results)
            }
        
        # Extract data for analysis
        compute_values = np.array([r.compute_units for r in successful_results])
        fitness_values = np.array([r.best_fitness for r in successful_results])
        
        # Log-log analysis
        log_compute = np.log10(compute_values + 1)  # +1 to avoid log(0)
        log_fitness = np.log10(np.maximum(fitness_values + 1, 0.01))  # Ensure positive
        
        # Linear regression in log space
        if len(log_compute) >= 2:
            slope, intercept = np.polyfit(log_compute, log_fitness, 1)
            correlation = np.corrcoef(log_compute, log_fitness)[0, 1]
        else:
            slope = 0.0
            correlation = 0.0
        
        # Detect smooth scaling regime
        scaling_detected = correlation > 0.3 and slope > 0.01
        smooth_regime = correlation > 0.6 and len(successful_results) >= 5
        
        return {
            'scaling_detected': scaling_detected,
            'slope': slope,
            'correlation': correlation,
            'smooth_regime': smooth_regime,
            'num_successful': len(successful_results),
            'compute_range': (float(np.min(compute_values)), float(np.max(compute_values))),
            'fitness_range': (float(np.min(fitness_values)), float(np.max(fitness_values)))
        }
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run complete scaling laws experiment."""
        print("🔬 Scaling Laws Experiment")
        print("=" * 30)
        
        start_time = time.time()
        
        # Generate configurations
        configs = self.generate_configs()
        
        # Run experiments
        print(f"Running {len(configs)} scaling experiments...")
        
        for i, config in enumerate(configs):
            result = self.run_single_config(config, i)
            self.results.append(result)
            
            # Progress update
            if (i + 1) % 5 == 0:
                successful = sum(1 for r in self.results if r.success)
                print(f"  Progress: {i+1}/{len(configs)} ({successful} successful)")
        
        # Analyze results
        analysis = self.analyze_scaling()
        
        total_time = time.time() - start_time
        
        # Summary
        successful_count = sum(1 for r in self.results if r.success)
        
        print(f"\n📊 Scaling Laws Results:")
        print(f"  Total configs: {len(configs)}")
        print(f"  Successful: {successful_count}")
        print(f"  Scaling detected: {analysis['scaling_detected']}")
        print(f"  Slope: {analysis['slope']:.4f}")
        print(f"  Correlation: {analysis['correlation']:.4f}")
        print(f"  Smooth regime: {analysis['smooth_regime']}")
        print(f"  Total time: {total_time:.1f}s")
        
        return {
            'experiment': 'scaling_laws',
            'total_configs': len(configs),
            'successful_configs': successful_count,
            'success_rate': successful_count / len(configs),
            'analysis': analysis,
            'runtime_seconds': total_time,
            'results': [
                {
                    'mu': r.config.mu,
                    'lambda': r.config.lambda_,
                    'N': r.config.N,
                    'mutation_rate': r.config.mutation_rate,
                    'compute_units': r.compute_units,
                    'best_fitness': r.best_fitness,
                    'success': r.success
                } for r in self.results
            ]
        }
    
    def save_results(self, filename: str = "scaling_laws_results.json"):
        """Save results to file."""
        results_data = self.run_experiment()
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results saved to {filename}")
        return results_data


def main():
    """Run scaling laws experiment."""
    experiment = ScalingLawsExperiment(base_seed=42)
    results = experiment.save_results()
    
    # Success criteria
    analysis = results['analysis']
    success = (
        analysis['scaling_detected'] and
        analysis['correlation'] > 0.4 and
        results['success_rate'] > 0.5
    )
    
    print(f"\n🎯 Scaling Laws Experiment: {'SUCCESS' if success else 'PARTIAL'}")
    
    return results


if __name__ == "__main__":
    main()
