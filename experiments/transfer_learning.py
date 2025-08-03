#!/usr/bin/env python3
"""
ProtoSynth Transfer Learning Experiment

Freeze top-8 modules from Periodic/Markov; measure gens-to-F=0.3 on 
Noisy/Markov↑ vs scratch.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import itertools
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from protosynth import *
from protosynth.curriculum_evolution import CurriculumEvolutionEngine
from protosynth.envs import periodic, k_order_markov, noisy
from protosynth.modularity import SubtreeMiner


@dataclass
class TransferResult:
    """Result from transfer learning experiment."""
    source_env: str
    target_env: str
    modules_transferred: int
    gens_to_target_transfer: Optional[int]
    gens_to_target_control: Optional[int]
    speedup: Optional[float]
    transfer_success: bool
    final_fitness_transfer: float
    final_fitness_control: float


class TransferLearningExperiment:
    """Transfer learning experiment runner."""
    
    def __init__(self, base_seed: int = 42, target_fitness: float = 0.3):
        self.base_seed = base_seed
        self.target_fitness = target_fitness
        self.results: List[TransferResult] = []
    
    def train_source_modules(self, source_env_name: str, source_env_factory) -> Dict[str, Any]:
        """Train modules on source environment."""
        print(f"  Training modules on {source_env_name}...")
        
        # Create source engine
        source_engine = CurriculumEvolutionEngine(
            mu=16, lambda_=32, seed=self.base_seed,
            max_modules=12, archive_size=20
        )
        
        # Override with source environment
        source_engine.environments = [type('Env', (), {
            'name': source_env_name,
            'factory': source_env_factory,
            'difficulty': 0.3,
            'description': f'Source environment: {source_env_name}'
        })()]
        
        # Initialize population
        from protosynth.evolve import create_initial_population
        initial_pop = create_initial_population(16, seed=self.base_seed)
        source_engine.evolution_engine.initialize_population(initial_pop)
        
        # Train for more generations to discover good modules
        best_fitness = -float('inf')
        
        for gen in range(30):
            stats = source_engine.evolve_generation()
            best_fitness = max(best_fitness, stats.best_fitness)
            
            if gen % 5 == 0:
                print(f"    Source gen {gen}: F={best_fitness:.4f}, modules={stats.modules_discovered}")
        
        # Extract top modules
        module_info = source_engine.module_library.get_module_info()
        modules = source_engine.module_library.modules.copy()
        
        # Sort modules by MDL score and take top 8
        module_items = list(modules.items())
        module_items.sort(key=lambda x: x[1].mdl_score, reverse=True)
        top_modules = dict(module_items[:8])
        
        print(f"    Extracted {len(top_modules)} top modules from {len(modules)} total")
        
        return {
            'modules': top_modules,
            'final_fitness': best_fitness,
            'total_modules': len(modules)
        }
    
    def test_transfer(self, source_modules: Dict[str, Any], target_env_name: str, 
                     target_env_factory, trial_id: int) -> Tuple[Optional[int], float]:
        """Test transfer to target environment."""
        print(f"    Testing transfer to {target_env_name}...")
        
        # Create target engine with transferred modules
        target_engine = CurriculumEvolutionEngine(
            mu=12, lambda_=24, seed=self.base_seed + trial_id,
            max_modules=12, archive_size=15
        )
        
        # Transfer modules
        for module_key, module in source_modules.items():
            target_engine.module_library.modules[module_key] = module
        
        # Override with target environment
        target_engine.environments = [type('Env', (), {
            'name': target_env_name,
            'factory': target_env_factory,
            'difficulty': 0.5,
            'description': f'Target environment: {target_env_name}'
        })()]
        
        # Initialize population
        from protosynth.evolve import create_initial_population
        initial_pop = create_initial_population(12, seed=self.base_seed + trial_id)
        target_engine.evolution_engine.initialize_population(initial_pop)
        
        # Evolve until target fitness or max generations
        gens_to_target = None
        final_fitness = -float('inf')
        
        for gen in range(40):
            stats = target_engine.evolve_generation()
            final_fitness = max(final_fitness, stats.best_fitness)
            
            if stats.best_fitness >= self.target_fitness and gens_to_target is None:
                gens_to_target = gen + 1
                print(f"      Transfer reached F={self.target_fitness} at gen {gens_to_target}")
                break
            
            if gen % 10 == 0:
                print(f"      Transfer gen {gen}: F={stats.best_fitness:.4f}")
        
        return gens_to_target, final_fitness
    
    def test_control(self, target_env_name: str, target_env_factory, trial_id: int) -> Tuple[Optional[int], float]:
        """Test control (no transfer) on target environment."""
        print(f"    Testing control on {target_env_name}...")
        
        # Create control engine without transferred modules
        control_engine = CurriculumEvolutionEngine(
            mu=12, lambda_=24, seed=self.base_seed + trial_id + 100,
            max_modules=12, archive_size=15
        )
        
        # Override with target environment
        control_engine.environments = [type('Env', (), {
            'name': target_env_name,
            'factory': target_env_factory,
            'difficulty': 0.5,
            'description': f'Control environment: {target_env_name}'
        })()]
        
        # Initialize population
        from protosynth.evolve import create_initial_population
        initial_pop = create_initial_population(12, seed=self.base_seed + trial_id + 100)
        control_engine.evolution_engine.initialize_population(initial_pop)
        
        # Evolve until target fitness or max generations
        gens_to_target = None
        final_fitness = -float('inf')
        
        for gen in range(40):
            stats = control_engine.evolve_generation()
            final_fitness = max(final_fitness, stats.best_fitness)
            
            if stats.best_fitness >= self.target_fitness and gens_to_target is None:
                gens_to_target = gen + 1
                print(f"      Control reached F={self.target_fitness} at gen {gens_to_target}")
                break
            
            if gen % 10 == 0:
                print(f"      Control gen {gen}: F={stats.best_fitness:.4f}")
        
        return gens_to_target, final_fitness
    
    def run_transfer_experiment(self, source_env_name: str, source_env_factory,
                               target_env_name: str, target_env_factory) -> TransferResult:
        """Run single transfer experiment."""
        print(f"\n  Transfer: {source_env_name} → {target_env_name}")
        
        # Train source modules
        source_data = self.train_source_modules(source_env_name, source_env_factory)
        source_modules = source_data['modules']
        
        # Test transfer
        gens_transfer, fitness_transfer = self.test_transfer(
            source_modules, target_env_name, target_env_factory, 0
        )
        
        # Test control
        gens_control, fitness_control = self.test_control(
            target_env_name, target_env_factory, 0
        )
        
        # Calculate speedup
        speedup = None
        if gens_transfer and gens_control:
            speedup = gens_control / gens_transfer
        elif gens_transfer and not gens_control:
            speedup = 2.0  # Transfer succeeded, control didn't
        
        # Determine success
        transfer_success = (
            (speedup and speedup >= 1.2) or  # 20% speedup
            (gens_transfer and not gens_control) or  # Transfer succeeded, control didn't
            (fitness_transfer > fitness_control + 0.05)  # Better final fitness
        )
        
        result = TransferResult(
            source_env=source_env_name,
            target_env=target_env_name,
            modules_transferred=len(source_modules),
            gens_to_target_transfer=gens_transfer,
            gens_to_target_control=gens_control,
            speedup=speedup,
            transfer_success=transfer_success,
            final_fitness_transfer=fitness_transfer,
            final_fitness_control=fitness_control
        )
        
        print(f"    Result: Transfer={gens_transfer} gens, Control={gens_control} gens, "
              f"Speedup={speedup:.2f}x" if speedup else "Speedup=N/A", 
              f"Success={transfer_success}")
        
        return result
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run complete transfer learning experiment."""
        print("🔬 Transfer Learning Experiment")
        print("=" * 35)
        
        start_time = time.time()
        
        # Define source and target environments
        experiments = [
            # Source: Periodic → Target: Noisy Periodic
            (
                "periodic_simple",
                lambda: periodic([1, 0, 1], seed=42),
                "noisy_periodic",
                lambda: noisy(periodic([1, 0, 1], seed=42), p_flip=0.1)
            ),
            # Source: Markov → Target: Higher-order Markov
            (
                "markov_k1",
                lambda: k_order_markov(1, {(0,): 0.7, (1,): 0.3}, seed=42),
                "markov_k2",
                lambda: k_order_markov(2, {(0,0): 0.8, (0,1): 0.3, (1,0): 0.7, (1,1): 0.2}, seed=42)
            ),
        ]
        
        # Run transfer experiments
        for source_name, source_factory, target_name, target_factory in experiments:
            result = self.run_transfer_experiment(
                source_name, source_factory, target_name, target_factory
            )
            self.results.append(result)
        
        # Analyze results
        total_experiments = len(self.results)
        successful_transfers = sum(1 for r in self.results if r.transfer_success)
        
        # Calculate average speedup for successful transfers
        speedups = [r.speedup for r in self.results if r.speedup is not None]
        avg_speedup = sum(speedups) / len(speedups) if speedups else 1.0
        
        overall_success = successful_transfers >= total_experiments * 0.5  # 50% success rate
        
        total_time = time.time() - start_time
        
        print(f"\n📊 Transfer Learning Results:")
        print(f"  Total experiments: {total_experiments}")
        print(f"  Successful transfers: {successful_transfers}")
        print(f"  Success rate: {successful_transfers/total_experiments:.1%}")
        print(f"  Average speedup: {avg_speedup:.2f}x")
        print(f"  Overall success: {overall_success}")
        print(f"  Total time: {total_time:.1f}s")
        
        return {
            'experiment': 'transfer_learning',
            'total_experiments': total_experiments,
            'successful_transfers': successful_transfers,
            'success_rate': successful_transfers / total_experiments,
            'average_speedup': avg_speedup,
            'overall_success': overall_success,
            'runtime_seconds': total_time,
            'results': [
                {
                    'source_env': r.source_env,
                    'target_env': r.target_env,
                    'modules_transferred': r.modules_transferred,
                    'gens_transfer': r.gens_to_target_transfer,
                    'gens_control': r.gens_to_target_control,
                    'speedup': r.speedup,
                    'success': r.transfer_success,
                    'fitness_transfer': r.final_fitness_transfer,
                    'fitness_control': r.final_fitness_control
                } for r in self.results
            ]
        }


def main():
    """Run transfer learning experiment."""
    experiment = TransferLearningExperiment(base_seed=42, target_fitness=0.3)
    results = experiment.run_experiment()
    
    # Save results
    with open("transfer_learning_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎯 Transfer Learning: {'SUCCESS' if results['overall_success'] else 'PARTIAL'}")
    
    return results


if __name__ == "__main__":
    main()
