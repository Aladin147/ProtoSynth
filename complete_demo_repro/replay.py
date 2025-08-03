#!/usr/bin/env python3
"""
ProtoSynth Reproducibility Script

This script exactly reproduces the evolution run from the saved bundle.
Generated automatically by ProtoSynth repro system.
"""

import sys
import json
import pickle
from pathlib import Path

# Add ProtoSynth to path (adjust as needed)
sys.path.append('..')
sys.path.append('../..')

from protosynth.curriculum_evolution import CurriculumEvolutionEngine
from protosynth.repro import ReproConfig, verify_reproduction
from protosynth.eval import evaluate_program_on_window


def load_config():
    """Load configuration from bundle."""
    with open('config.json', 'r') as f:
        config_dict = json.load(f)
    
    return ReproConfig(**config_dict)


def replay_evolution():
    """Replay the evolution run exactly."""
    print("Replaying ProtoSynth evolution run...")
    
    # Load configuration
    config = load_config()
    print(f"Loaded config: seed={config.seed}, generations={config.num_generations}")
    
    # Create engine with exact same parameters
    engine = CurriculumEvolutionEngine(
        mu=config.mu,
        lambda_=config.lambda_,
        seed=config.seed,
        max_modules=config.max_modules,
        archive_size=config.archive_size
    )
    
    # Run evolution
    print("Running evolution...")
    stats_list = engine.run_curriculum(num_generations=config.num_generations)
    
    # Get final results
    final_stats = stats_list[-1] if stats_list else None
    
    if final_stats:
        print(f"Replay completed:")
        print(f"  Final fitness: {final_stats.best_fitness:.6f}")
        print(f"  Final modules: {final_stats.modules_discovered}")
        print(f"  Final environment: {final_stats.current_env}")
    
    return engine, final_stats


def verify_reproduction():
    """Verify that reproduction matches original within tolerance."""
    print("\nVerifying reproduction accuracy...")
    
    # Load original metadata
    with open('metadata.json', 'r') as f:
        original_metadata = json.load(f)
    
    original_fitness = original_metadata['final_fitness']
    
    # Run replay
    engine, final_stats = replay_evolution()
    
    if final_stats:
        replay_fitness = final_stats.best_fitness
        fitness_diff = abs(replay_fitness - original_fitness)
        
        print(f"Original fitness: {original_fitness:.6f}")
        print(f"Replay fitness:   {replay_fitness:.6f}")
        print(f"Difference:       {fitness_diff:.6f}")
        
        # Check tolerance (±0.01)
        tolerance = 0.01
        if fitness_diff <= tolerance:
            print(f"Reproduction VERIFIED (within +/-{tolerance})")
            return True
        else:
            print(f"Reproduction FAILED (exceeds +/-{tolerance})")
            return False
    else:
        print("Replay failed to produce results")
        return False


if __name__ == "__main__":
    success = verify_reproduction()
    sys.exit(0 if success else 1)
