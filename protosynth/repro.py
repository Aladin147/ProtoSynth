"""
ProtoSynth Reproducibility Bundle

Implements comprehensive reproducibility system:
- Save seeds, configs, AST, modules in one folder
- replay.py script for exact reproduction
- Verification that replay reproduces F within ±0.01
"""

import json
import pickle
import shutil
import time
import os
import sys
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

from .core import LispNode, LispInterpreter
from .modularity import ModuleLibrary, Module
from .curriculum_evolution import CurriculumEvolutionEngine
from .core import pretty_print_ast


@dataclass
class ReproConfig:
    """Configuration for reproducible runs."""
    # Evolution parameters
    mu: int
    lambda_: int
    seed: int
    num_generations: int
    
    # Module parameters
    max_modules: int
    archive_size: int
    
    # Environment parameters
    environment_names: List[str]
    
    # Interpreter parameters
    max_recursion_depth: int
    max_steps: int
    timeout_seconds: float
    
    # Other parameters
    mutation_rate: float = 0.1
    crossover_rate: float = 0.2


class ReproBundle:
    """
    Comprehensive reproducibility bundle.
    
    Saves all necessary information to exactly reproduce an evolution run.
    """
    
    def __init__(self, bundle_dir: str):
        """
        Initialize reproducibility bundle.
        
        Args:
            bundle_dir: Directory to store bundle files
        """
        self.bundle_dir = Path(bundle_dir)
        self.bundle_dir.mkdir(exist_ok=True)
        
        # File paths
        self.config_path = self.bundle_dir / "config.json"
        self.modules_path = self.bundle_dir / "modules.pkl"
        self.best_program_path = self.bundle_dir / "best_program.lisp"
        self.best_program_pkl_path = self.bundle_dir / "best_program.pkl"
        self.metadata_path = self.bundle_dir / "metadata.json"
        self.replay_script_path = self.bundle_dir / "replay.py"
        
        print(f"📦 Repro bundle initialized: {self.bundle_dir}")
    
    def save_run(self, engine: CurriculumEvolutionEngine, config: ReproConfig, 
                final_stats: Dict[str, Any]) -> str:
        """
        Save a complete evolution run for reproducibility.
        
        Args:
            engine: Evolution engine with final state
            config: Configuration used for the run
            final_stats: Final statistics from the run
            
        Returns:
            Path to the bundle directory
        """
        print(f"💾 Saving reproducibility bundle...")
        
        # Save configuration
        with open(self.config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        
        # Save module library
        with open(self.modules_path, 'wb') as f:
            pickle.dump(engine.module_library, f)
        
        # Save best program
        if engine.evolution_engine.population:
            best_individual = max(engine.evolution_engine.population, key=lambda x: x.fitness)
            best_program = best_individual.program
            
            # Save as human-readable Lisp
            with open(self.best_program_path, 'w') as f:
                f.write(pretty_print_ast(best_program))
            
            # Save as pickle for exact reproduction
            with open(self.best_program_pkl_path, 'wb') as f:
                pickle.dump(best_program, f)
        
        # Save metadata
        metadata = {
            'creation_timestamp': time.time(),
            'protosynth_version': '1.0.0',
            'final_stats': final_stats,
            'bundle_format_version': '1.0',
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}",
            'total_generations': len(engine.stats_history),
            'final_fitness': final_stats.get('best_fitness', 0.0) if hasattr(final_stats, 'get') else getattr(final_stats, 'best_fitness', 0.0)
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Generate replay script
        self._generate_replay_script(config)
        
        print(f"✅ Bundle saved to {self.bundle_dir}")
        return str(self.bundle_dir)
    
    def _generate_replay_script(self, config: ReproConfig):
        """Generate replay.py script for exact reproduction."""
        script_content = f'''#!/usr/bin/env python3
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
    print(f"Loaded config: seed={{config.seed}}, generations={{config.num_generations}}")
    
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
        print(f"  Final fitness: {{final_stats.best_fitness:.6f}}")
        print(f"  Final modules: {{final_stats.modules_discovered}}")
        print(f"  Final environment: {{final_stats.current_env}}")
    
    return engine, final_stats


def verify_reproduction():
    """Verify that reproduction matches original within tolerance."""
    print("\\nVerifying reproduction accuracy...")
    
    # Load original metadata
    with open('metadata.json', 'r') as f:
        original_metadata = json.load(f)
    
    original_fitness = original_metadata['final_fitness']
    
    # Run replay
    engine, final_stats = replay_evolution()
    
    if final_stats:
        replay_fitness = final_stats.best_fitness
        fitness_diff = abs(replay_fitness - original_fitness)
        
        print(f"Original fitness: {{original_fitness:.6f}}")
        print(f"Replay fitness:   {{replay_fitness:.6f}}")
        print(f"Difference:       {{fitness_diff:.6f}}")
        
        # Check tolerance (±0.01)
        tolerance = 0.01
        if fitness_diff <= tolerance:
            print(f"Reproduction VERIFIED (within +/-{{tolerance}})")
            return True
        else:
            print(f"Reproduction FAILED (exceeds +/-{{tolerance}})")
            return False
    else:
        print("Replay failed to produce results")
        return False


if __name__ == "__main__":
    success = verify_reproduction()
    sys.exit(0 if success else 1)
'''
        
        with open(self.replay_script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(self.replay_script_path, 0o755)
        
        print(f"📝 Replay script generated: {self.replay_script_path}")
    
    def load_bundle(self) -> Dict[str, Any]:
        """
        Load a reproducibility bundle.
        
        Returns:
            Dictionary with loaded bundle contents
        """
        if not self.bundle_dir.exists():
            raise FileNotFoundError(f"Bundle directory not found: {self.bundle_dir}")
        
        bundle = {}
        
        # Load configuration
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                bundle['config'] = ReproConfig(**json.load(f))
        
        # Load modules
        if self.modules_path.exists():
            with open(self.modules_path, 'rb') as f:
                bundle['modules'] = pickle.load(f)
        
        # Load best program
        if self.best_program_pkl_path.exists():
            with open(self.best_program_pkl_path, 'rb') as f:
                bundle['best_program'] = pickle.load(f)
        
        # Load metadata
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                bundle['metadata'] = json.load(f)
        
        return bundle
    
    def verify_bundle(self) -> bool:
        """
        Verify that bundle is complete and valid.
        
        Returns:
            True if bundle is valid
        """
        required_files = [
            self.config_path,
            self.metadata_path,
            self.replay_script_path
        ]
        
        missing_files = [f for f in required_files if not f.exists()]
        
        if missing_files:
            print(f"❌ Bundle incomplete. Missing files: {missing_files}")
            return False
        
        # Verify config format
        try:
            with open(self.config_path, 'r') as f:
                config_dict = json.load(f)
                ReproConfig(**config_dict)  # Validate structure
        except Exception as e:
            print(f"❌ Invalid config format: {e}")
            return False
        
        print("✅ Bundle verification passed")
        return True


def create_demo_bundle():
    """Create a demo reproducibility bundle."""
    print("📦 Creating Demo Reproducibility Bundle")
    print("=" * 45)
    
    # Create demo configuration
    config = ReproConfig(
        mu=8,
        lambda_=16,
        seed=42,
        num_generations=10,
        max_modules=8,
        archive_size=20,
        environment_names=["periodic_simple", "markov_simple"],
        max_recursion_depth=10,
        max_steps=100,
        timeout_seconds=1.0
    )
    
    # Create and run evolution engine
    engine = CurriculumEvolutionEngine(
        mu=config.mu,
        lambda_=config.lambda_,
        seed=config.seed,
        max_modules=config.max_modules,
        archive_size=config.archive_size
    )
    
    print("Running demo evolution...")
    stats_list = engine.run_curriculum(num_generations=config.num_generations)
    
    final_stats = stats_list[-1] if stats_list else None
    
    # Create bundle
    bundle = ReproBundle("demo_repro_bundle")
    bundle_path = bundle.save_run(engine, config, final_stats)
    
    # Verify bundle
    is_valid = bundle.verify_bundle()
    
    print(f"\\n📊 Demo Bundle Summary:")
    print(f"  Bundle path: {bundle_path}")
    print(f"  Valid: {is_valid}")
    if final_stats:
        print(f"  Final fitness: {final_stats.best_fitness:.6f}")
        print(f"  Modules discovered: {final_stats.modules_discovered}")
    
    print(f"\\n🔄 To reproduce this run:")
    print(f"  cd {bundle_path}")
    print(f"  python replay.py")
    
    return bundle


if __name__ == "__main__":
    create_demo_bundle()
