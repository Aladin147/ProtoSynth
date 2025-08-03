"""
ProtoSynth Metrics Dashboard

Implements comprehensive metrics logging and visualization:
- Per-generation metrics logging (CSV/JSON)
- Real-time plotting and visualization
- Plateau detection for curriculum step-ups
- Performance analytics
"""

import json
import csv
import time
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    plt = None
    np = None
    plt = None
    np = None


@dataclass
class GenerationMetrics:
    """Metrics for a single generation."""
    generation: int
    timestamp: float
    
    # Fitness metrics
    best_fitness: float
    median_fitness: float
    mean_fitness: float
    fitness_std: float
    
    # Population metrics
    population_size: int
    avg_program_size: float
    size_std: float
    
    # Diversity metrics
    diversity_score: float
    novelty_score: float
    
    # Environment metrics
    current_environment: str
    learning_progress: float
    
    # Module metrics
    num_modules: int
    module_usage_rate: float
    
    # Performance metrics
    evaluation_time: float
    generation_time: float
    
    # Robustness metrics
    robustness_score: float
    noise_level: float


class MetricsLogger:
    """
    Comprehensive metrics logging system.
    
    Logs metrics to CSV and JSON formats with real-time plotting.
    """
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = "protosynth_run"):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory for log files
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        
        # Create log directory
        self.log_dir.mkdir(exist_ok=True)
        
        # File paths
        self.csv_path = self.log_dir / f"{experiment_name}_metrics.csv"
        self.json_path = self.log_dir / f"{experiment_name}_metrics.json"
        
        # Metrics storage
        self.metrics_history: List[GenerationMetrics] = []
        
        # Plateau detection
        self.plateau_window = 10
        self.plateau_threshold = 0.01
        
        # Initialize CSV file
        self._init_csv()
        
        print(f"📊 Metrics logger initialized: {self.log_dir}")
    
    def _init_csv(self):
        """Initialize CSV file with headers."""
        if not self.csv_path.exists():
            # Get field names from dataclass
            fields = list(GenerationMetrics.__dataclass_fields__.keys())
            
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(fields)
    
    def log_generation(self, metrics: GenerationMetrics):
        """
        Log metrics for a generation.
        
        Args:
            metrics: Generation metrics to log
        """
        self.metrics_history.append(metrics)
        
        # Write to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(asdict(metrics).values())
        
        # Update JSON file
        self._update_json()
        
        # Check for plateaus
        if len(self.metrics_history) >= self.plateau_window:
            plateau_detected = self._detect_plateau()
            if plateau_detected:
                print(f"🔄 Plateau detected at generation {metrics.generation}")
    
    def _update_json(self):
        """Update JSON file with all metrics."""
        data = {
            'experiment_name': self.experiment_name,
            'timestamp': time.time(),
            'total_generations': len(self.metrics_history),
            'metrics': [asdict(m) for m in self.metrics_history]
        }
        
        with open(self.json_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _detect_plateau(self) -> bool:
        """Detect if fitness has plateaued."""
        if len(self.metrics_history) < self.plateau_window:
            return False
        
        recent_fitness = [m.best_fitness for m in self.metrics_history[-self.plateau_window:]]
        
        # Check if fitness variance is below threshold
        if HAS_PLOTTING and np is not None:
            fitness_std = np.std(recent_fitness)
        else:
            # Fallback calculation without numpy
            mean_fitness = sum(recent_fitness) / len(recent_fitness)
            variance = sum((f - mean_fitness) ** 2 for f in recent_fitness) / len(recent_fitness)
            fitness_std = variance ** 0.5

        return fitness_std < self.plateau_threshold
    
    def plot_metrics(self, save_plots: bool = True, show_plots: bool = False):
        """
        Generate comprehensive plots of metrics.

        Args:
            save_plots: Whether to save plots to files
            show_plots: Whether to display plots
        """
        if not HAS_PLOTTING or plt is None:
            print("Plotting not available (matplotlib not installed)")
            return

        if not self.metrics_history:
            print("No metrics to plot")
            return
        
        # Extract data
        generations = [m.generation for m in self.metrics_history]
        best_fitness = [m.best_fitness for m in self.metrics_history]
        median_fitness = [m.median_fitness for m in self.metrics_history]
        diversity = [m.diversity_score for m in self.metrics_history]
        program_size = [m.avg_program_size for m in self.metrics_history]
        num_modules = [m.num_modules for m in self.metrics_history]
        robustness = [m.robustness_score for m in self.metrics_history]
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'ProtoSynth Metrics: {self.experiment_name}', fontsize=16)
        
        # Fitness evolution
        axes[0, 0].plot(generations, best_fitness, 'b-', label='Best', linewidth=2)
        axes[0, 0].plot(generations, median_fitness, 'g--', label='Median', alpha=0.7)
        axes[0, 0].set_title('Fitness Evolution')
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Fitness')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Diversity
        axes[0, 1].plot(generations, diversity, 'r-', linewidth=2)
        axes[0, 1].set_title('Population Diversity')
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('Diversity Score')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Program size
        axes[0, 2].plot(generations, program_size, 'm-', linewidth=2)
        axes[0, 2].set_title('Average Program Size')
        axes[0, 2].set_xlabel('Generation')
        axes[0, 2].set_ylabel('Nodes')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Module discovery
        axes[1, 0].plot(generations, num_modules, 'c-', linewidth=2)
        axes[1, 0].set_title('Module Discovery')
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Number of Modules')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Robustness
        axes[1, 1].plot(generations, robustness, 'orange', linewidth=2)
        axes[1, 1].set_title('Robustness Score')
        axes[1, 1].set_xlabel('Generation')
        axes[1, 1].set_ylabel('Robustness')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Environment progression
        environments = [m.current_environment for m in self.metrics_history]
        env_changes = []
        current_env = None
        
        for i, env in enumerate(environments):
            if env != current_env:
                env_changes.append((i, env))
                current_env = env
        
        axes[1, 2].set_title('Environment Progression')
        axes[1, 2].set_xlabel('Generation')
        
        # Color-code environment changes
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
        for i, (gen, env) in enumerate(env_changes):
            color = colors[i % len(colors)]
            next_gen = env_changes[i + 1][0] if i + 1 < len(env_changes) else len(generations)
            axes[1, 2].axvspan(gen, next_gen, alpha=0.3, color=color, label=env)
        
        axes[1, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.log_dir / f"{self.experiment_name}_plots.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📈 Plots saved to {plot_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the run."""
        if not self.metrics_history:
            return {}
        
        final_metrics = self.metrics_history[-1]
        
        # Fitness progression
        initial_fitness = self.metrics_history[0].best_fitness
        final_fitness = final_metrics.best_fitness
        fitness_improvement = final_fitness - initial_fitness
        
        # Peak fitness
        peak_fitness = max(m.best_fitness for m in self.metrics_history)
        peak_generation = next(i for i, m in enumerate(self.metrics_history) 
                              if m.best_fitness == peak_fitness)
        
        # Environment usage
        environments = [m.current_environment for m in self.metrics_history]
        env_counts = {}
        for env in environments:
            env_counts[env] = env_counts.get(env, 0) + 1
        
        # Module progression
        final_modules = final_metrics.num_modules
        max_modules = max(m.num_modules for m in self.metrics_history)
        
        return {
            'total_generations': len(self.metrics_history),
            'runtime_hours': (final_metrics.timestamp - self.metrics_history[0].timestamp) / 3600,
            'fitness_progression': {
                'initial': initial_fitness,
                'final': final_fitness,
                'peak': peak_fitness,
                'peak_generation': peak_generation,
                'improvement': fitness_improvement
            },
            'environment_usage': env_counts,
            'module_discovery': {
                'final_count': final_modules,
                'max_count': max_modules
            },
            'final_diversity': final_metrics.diversity_score,
            'final_robustness': final_metrics.robustness_score
        }
    
    def export_for_analysis(self, output_path: Optional[str] = None) -> str:
        """
        Export metrics in analysis-friendly format.
        
        Args:
            output_path: Path for export file
            
        Returns:
            Path to exported file
        """
        if output_path is None:
            output_path = self.log_dir / f"{self.experiment_name}_analysis.json"
        
        analysis_data = {
            'metadata': {
                'experiment_name': self.experiment_name,
                'export_timestamp': time.time(),
                'total_generations': len(self.metrics_history)
            },
            'summary': self.get_summary_stats(),
            'raw_metrics': [asdict(m) for m in self.metrics_history]
        }
        
        with open(output_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"📊 Analysis data exported to {output_path}")
        return str(output_path)


def demo_metrics_dashboard():
    """Demonstrate metrics dashboard functionality."""
    print("📊 Metrics Dashboard Demo")
    print("=" * 30)
    
    # Create logger
    logger = MetricsLogger(log_dir="demo_logs", experiment_name="demo_run")
    
    # Simulate some generations
    print("Simulating evolution run...")
    
    for gen in range(20):
        # Simulate metrics (with fallback for missing numpy)
        import random

        def randn():
            return random.gauss(0, 1) if not HAS_PLOTTING or np is None else np.random.randn()

        metrics = GenerationMetrics(
            generation=gen,
            timestamp=time.time(),
            best_fitness=0.1 + 0.02 * gen + 0.01 * randn(),
            median_fitness=0.05 + 0.015 * gen + 0.005 * randn(),
            mean_fitness=0.03 + 0.01 * gen + 0.005 * randn(),
            fitness_std=0.02 + 0.001 * randn(),
            population_size=20,
            avg_program_size=8 + 2 * randn(),
            size_std=2.0,
            diversity_score=0.5 + 0.1 * randn(),
            novelty_score=0.3 + 0.1 * randn(),
            current_environment=["periodic_simple", "periodic_complex", "markov_simple"][gen // 7],
            learning_progress=0.01 * randn(),
            num_modules=min(gen // 3, 8),
            module_usage_rate=min(0.1 * gen, 0.8),
            evaluation_time=0.1 + 0.01 * randn(),
            generation_time=1.0 + 0.1 * randn(),
            robustness_score=0.8 + 0.1 * randn(),
            noise_level=min(0.01 * gen, 0.1)
        )
        
        logger.log_generation(metrics)
        
        if gen % 5 == 0:
            print(f"  Gen {gen}: F={metrics.best_fitness:.3f}, "
                  f"env={metrics.current_environment}, "
                  f"modules={metrics.num_modules}")
    
    # Generate plots
    print("\nGenerating plots...")
    logger.plot_metrics(save_plots=True, show_plots=False)
    
    # Get summary
    summary = logger.get_summary_stats()
    print(f"\nRun Summary:")
    print(f"  Fitness improvement: {summary['fitness_progression']['improvement']:.3f}")
    print(f"  Peak fitness: {summary['fitness_progression']['peak']:.3f}")
    print(f"  Final modules: {summary['module_discovery']['final_count']}")
    print(f"  Environment usage: {summary['environment_usage']}")
    
    # Export for analysis
    export_path = logger.export_for_analysis()
    
    return logger


if __name__ == "__main__":
    demo_metrics_dashboard()
