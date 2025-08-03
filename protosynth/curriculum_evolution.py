"""
Curriculum-Driven Evolution Engine

Integrates curriculum learning, novelty search, and robustness testing.
"""

import random
import itertools
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from .core import LispNode, LispInterpreter
from .evolve import EvolutionEngine, Individual
from .modularity import ModuleLibrary
from .curriculum import (
    LearningProgressBandit, NoveltyArchive, EnvironmentSpec,
    create_behavior_signature, create_default_environments
)
from .eval import evaluate_program_on_window, evaluate_program_calibrated
from .envs import noisy
from .predictor import PredictorAdapter


# Canonical meta for envs (extend if you have more)
ENV_META = {
    "periodic_k4": {"kind": "periodic", "k": 4},
    "periodic_k3": {"kind": "periodic", "k": 3},
    "periodic_k2": {"kind": "periodic", "k": 2},
    "markov_k1":   {"kind": "markov",   "k": 1},
    "markov_k2":   {"kind": "markov",   "k": 2},
}

def _env_spec(env_name: str):
    meta = ENV_META.get(env_name)
    if not meta:
        # sensible default
        meta = {"kind": "periodic", "k": 2}
    return meta["kind"], meta["k"]

def _materialize_stream(stream_factory, k: int, N: int, seed: int = None):
    """Materialize a stream into a buffer."""
    if seed is not None:
        random.seed(seed)
    stream = stream_factory()
    return list(itertools.islice(stream, N + k))


@dataclass
class CurriculumStats:
    """Statistics for curriculum evolution."""
    generation: int
    current_env: str
    best_fitness: float
    diversity: float
    learning_progress: float
    robustness_score: float
    modules_discovered: int


class CurriculumEvolutionEngine:
    """
    Evolution engine with curriculum learning and novelty search.
    
    Features:
    - Auto-paced curriculum via learning-progress bandit
    - Novelty search with behavior signatures
    - Robustness testing with noise schedules
    - Module discovery and reuse
    """
    
    def __init__(self, 
                 mu: int = 20,
                 lambda_: int = 40,
                 seed: int = 42,
                 max_modules: int = 32,
                 archive_size: int = 50,
                 environments: Optional[List[EnvironmentSpec]] = None):
        """
        Initialize curriculum evolution engine.
        
        Args:
            mu: Population size
            lambda_: Offspring size
            seed: Random seed
            max_modules: Maximum modules in library
            archive_size: Novelty archive size
            environments: Custom environments (uses defaults if None)
        """
        self.mu = mu
        self.lambda_ = lambda_
        self.seed = seed
        
        # Initialize components
        self.module_library = ModuleLibrary(max_modules=max_modules)
        self.interpreter = LispInterpreter(module_library=self.module_library)
        
        # Evolution engine with appropriate parameters for curriculum learning
        self.evolution_engine = EvolutionEngine(
            mu=mu, lambda_=lambda_, seed=seed,
            k=2,  # Shorter context for faster learning
            N=800  # Match our stream length (1000 - k - buffer)
        )
        self.evolution_engine.interpreter = self.interpreter
        
        # Curriculum components
        self.environments = environments or create_default_environments()
        self.bandit = LearningProgressBandit(self.environments)
        self.novelty_archive = NoveltyArchive(archive_size=archive_size)
        
        # Tracking
        self.generation = 0
        self.stats_history: List[CurriculumStats] = []
        
        # Robustness testing
        self.noise_schedule = self._create_noise_schedule()
        
        random.seed(seed)
    
    def _create_noise_schedule(self) -> Dict[int, float]:
        """Create noise schedule: p_flip increases over time."""
        schedule = {}
        for gen in range(200):  # 200 generations
            if gen < 50:
                schedule[gen] = 0.0  # No noise initially
            elif gen < 100:
                schedule[gen] = 0.02  # Light noise
            elif gen < 150:
                schedule[gen] = 0.05  # Medium noise
            else:
                schedule[gen] = 0.1   # Heavy noise
        return schedule

    def _evaluate_population_unified(self, env_name: str, N_train: int = 2048, N_val: int = 2048, ensemble: int = 1):
        """Evaluate population using unified evaluation pipeline."""
        kind, k = _env_spec(env_name)

        # Get environment factory
        env_spec = next((e for e in self.environments if e.name == env_name), None)
        if not env_spec:
            raise ValueError(f"Environment {env_name} not found")

        # Materialize validation buffer ONCE for this generation (no generator reuse)
        buffer = _materialize_stream(env_spec.factory, k=k, N=N_val, seed=random.randint(0, 2**31 - 1))

        # Evaluate every candidate via the SAME pipeline (uses calibrated evaluation for markov)
        for cand in self.evolution_engine.population:
            # Use the curriculum engine's interpreter
            interp = self.interpreter

            if kind == "markov":
                # Use calibrated evaluation for markov chains
                F, metrics = evaluate_program_calibrated(
                    interpreter=interp,
                    program=cand.program,
                    buffer=buffer,
                    k=k,
                    N_train=N_train,
                    N_val=N_val
                )
            else:
                # Use standard evaluation for periodic patterns
                F, metrics = evaluate_program_on_window(
                    interpreter=interp,
                    program=cand.program,
                    bits=buffer,
                    k=k
                )

            cand.fitness = F
            if hasattr(cand, 'metrics'):
                cand.metrics = metrics

        # Guardrails so it can't regress
        assert all(hasattr(c, "fitness") for c in self.evolution_engine.population)

        # Log for debugging
        best_fitness = max(c.fitness for c in self.evolution_engine.population)
        print(f"[curriculum] env={env_name} k={k} bestF={best_fitness:.3f}")

    def _perform_selection_and_mutation(self):
        """Perform selection and mutation to create next generation."""
        # Generate offspring through mutation
        offspring = []
        attempts = 0
        max_attempts = self.lambda_ * 3

        while len(offspring) < self.lambda_ and attempts < max_attempts:
            # Select parent randomly
            parent = random.choice(self.evolution_engine.population)

            # Mutate (simplified - just copy for now)
            from .evolve import Individual
            child = Individual(
                program=parent.program,  # TODO: Add actual mutation
                fitness=0.0,
                metrics={},
                generation=self.generation + 1
            )
            offspring.append(child)
            attempts += 1

        # Combine parents and offspring
        combined_population = self.evolution_engine.population + offspring

        # Select top mu individuals
        combined_population.sort(key=lambda x: x.fitness, reverse=True)
        self.evolution_engine.population = combined_population[:self.mu]

    def evolve_generation(self) -> CurriculumStats:
        """Evolve one generation with curriculum learning."""
        # Select environment using bandit
        env_spec = self.bandit.select_environment()

        # Evaluate with unified path
        self._evaluate_population_unified(env_spec.name, N_train=2048, N_val=2048, ensemble=1)

        # Continue with selection (get stats from population)
        fitnesses = [c.fitness for c in self.evolution_engine.population]
        evolution_stats = {
            'best_fitness': max(fitnesses) if fitnesses else float('-inf'),
            'mean_fitness': sum(fitnesses) / len(fitnesses) if fitnesses else 0.0
        }

        # Perform selection and mutation for next generation
        self._perform_selection_and_mutation()

        # Update bandit with learning progress
        self.bandit.update_fitness(env_spec.name, evolution_stats['best_fitness'])

        # Create training stream for novelty archive
        training_stream = list(itertools.islice(env_spec.factory(), 1000))

        # Update novelty archive
        self._update_novelty_archive(training_stream)
        
        # Test robustness
        robustness_score = self._test_robustness(env_spec)
        
        # Update module library
        self._update_modules()
        
        # Collect stats
        stats = CurriculumStats(
            generation=self.generation,
            current_env=env_spec.name,
            best_fitness=evolution_stats['best_fitness'],
            diversity=self.novelty_archive.get_diversity_stats()['diversity'],
            learning_progress=self.bandit._compute_learning_progress(env_spec.name),
            robustness_score=robustness_score,
            modules_discovered=len(self.module_library.modules)
        )
        
        self.stats_history.append(stats)
        self.generation += 1
        self.module_library.advance_generation()
        
        return stats
    
    def _update_novelty_archive(self, test_bits: List[int]):
        """Update novelty archive with current population."""
        for individual in self.evolution_engine.population:
            # Create behavior signature
            signature = create_behavior_signature(
                individual.program, test_bits, self.interpreter
            )
            
            # Add to archive
            self.novelty_archive.add_program(
                individual.program, individual.fitness, signature
            )
    
    def _test_robustness(self, env_spec: EnvironmentSpec) -> float:
        """Test robustness of best individual under noise."""
        if not self.evolution_engine.population:
            return 0.0
        
        # Get current noise level
        noise_level = self.noise_schedule.get(self.generation, 0.1)
        
        if noise_level == 0.0:
            return 1.0  # Perfect robustness with no noise
        
        # Get best individual
        best_individual = max(self.evolution_engine.population, key=lambda x: x.fitness)
        
        # Test on noisy version of environment
        try:
            clean_stream = list(itertools.islice(env_spec.factory(), 500))
            noisy_stream = list(itertools.islice(
                noisy(iter(clean_stream), p_flip=noise_level), 500
            ))
            
            # Evaluate on both
            clean_fitness, _ = evaluate_program_on_window(
                self.interpreter, best_individual.program, clean_stream, k=2
            )
            
            noisy_fitness, _ = evaluate_program_on_window(
                self.interpreter, best_individual.program, noisy_stream, k=2
            )
            
            # Robustness = retention of performance
            if clean_fitness > 0:
                robustness = max(0.0, noisy_fitness / clean_fitness)
            else:
                robustness = 1.0 if noisy_fitness >= 0 else 0.0
            
            return robustness
            
        except Exception:
            return 0.0
    
    def _update_modules(self):
        """Update module library with current population."""
        if self.generation % 10 == 0:  # Update modules every 10 generations
            from .modularity import SubtreeMiner
            
            # Extract programs from population
            programs = [ind.program for ind in self.evolution_engine.population]
            
            # Mine new modules
            miner = SubtreeMiner(beta=0.005, min_frequency=2)
            validation_bits = [0, 1] * 200  # Simple validation sequence
            
            try:
                candidates = miner.mine_and_select(programs, validation_bits, n_modules=5)
                
                if candidates:
                    new_modules = self.module_library.register_modules(candidates)
                    if new_modules:
                        print(f"Gen {self.generation}: Discovered {len(new_modules)} new modules")
                        
                        # Update credit scores
                        fitness_scores = [ind.fitness for ind in self.evolution_engine.population]
                        self.module_library.update_credit_scores(programs, fitness_scores)
                        
            except Exception as e:
                print(f"Module mining failed: {e}")
    
    def get_curriculum_progression(self) -> Dict[str, Any]:
        """Get curriculum progression statistics."""
        if not self.stats_history:
            return {}
        
        # Environment progression
        env_sequence = [stat.current_env for stat in self.stats_history[-20:]]
        env_counts = {}
        for env in env_sequence:
            env_counts[env] = env_counts.get(env, 0) + 1
        
        # Learning trends
        recent_fitness = [stat.best_fitness for stat in self.stats_history[-10:]]
        recent_diversity = [stat.diversity for stat in self.stats_history[-10:]]
        
        # Robustness trend
        recent_robustness = [stat.robustness_score for stat in self.stats_history[-10:]]
        
        return {
            'generation': self.generation,
            'environment_distribution': env_counts,
            'recent_fitness_trend': recent_fitness,
            'recent_diversity_trend': recent_diversity,
            'recent_robustness_trend': recent_robustness,
            'total_modules': len(self.module_library.modules),
            'bandit_stats': self.bandit.get_stats(),
            'archive_stats': self.novelty_archive.get_diversity_stats()
        }
    
    def run_curriculum(self, num_generations: int = 100) -> List[CurriculumStats]:
        """
        Run curriculum evolution for specified generations.
        
        Args:
            num_generations: Number of generations to run
            
        Returns:
            List of statistics for each generation
        """
        print(f"🎓 Starting curriculum evolution for {num_generations} generations")
        print(f"Available environments: {[env.name for env in self.environments]}")
        
        # Initialize population
        from .evolve import create_initial_population
        initial_pop = create_initial_population(self.mu, seed=self.seed)
        self.evolution_engine.initialize_population(initial_pop)
        
        stats_list = []
        
        for gen in range(num_generations):
            stats = self.evolve_generation()
            stats_list.append(stats)
            
            # Progress reporting
            if gen % 10 == 0 or gen == num_generations - 1:
                print(f"Gen {gen:3d}: env={stats.current_env:15s} "
                      f"F={stats.best_fitness:.3f} "
                      f"div={stats.diversity:.3f} "
                      f"rob={stats.robustness_score:.3f} "
                      f"mods={stats.modules_discovered}")
        
        # Final report
        progression = self.get_curriculum_progression()
        print(f"\n📊 Curriculum Summary:")
        print(f"Final fitness: {stats_list[-1].best_fitness:.3f}")
        print(f"Final diversity: {stats_list[-1].diversity:.3f}")
        print(f"Modules discovered: {stats_list[-1].modules_discovered}")
        print(f"Environment distribution: {progression['environment_distribution']}")
        
        return stats_list


def demo_curriculum_evolution():
    """Demonstrate curriculum evolution."""
    print("🎓 Curriculum Evolution Demo")
    print("=" * 40)
    
    # Create curriculum engine
    engine = CurriculumEvolutionEngine(
        mu=12, lambda_=24, seed=42,
        max_modules=16, archive_size=30
    )
    
    # Run short curriculum
    stats = engine.run_curriculum(num_generations=30)
    
    # Show progression
    progression = engine.get_curriculum_progression()
    
    print(f"\nProgression analysis:")
    print(f"Environment usage: {progression['environment_distribution']}")
    print(f"Bandit stats: {progression['bandit_stats']}")
    
    return engine, stats


if __name__ == "__main__":
    demo_curriculum_evolution()
