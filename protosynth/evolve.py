"""
ProtoSynth Evolution System

This module implements the core evolutionary loop using (μ+λ) Evolution Strategy.
Programs evolve to become better predictors through mutation and selection.
"""

import logging
import random
import time
from typing import List, Tuple, Dict, Iterator, Optional
from dataclasses import dataclass
from .core import LispNode, LispInterpreter
from .agent import SelfModifyingAgent
from .eval import evaluate_program
from .mutation import iter_nodes

logger = logging.getLogger(__name__)


@dataclass
class Individual:
    """Represents an individual in the population."""
    program: LispNode
    fitness: float
    metrics: Dict
    generation: int
    parent_id: Optional[int] = None
    individual_id: Optional[int] = None
    
    def __post_init__(self):
        if self.individual_id is None:
            self.individual_id = id(self)
    
    def size(self) -> int:
        """Return the size of the program (number of nodes)."""
        return len(list(iter_nodes(self.program)))
    
    def __lt__(self, other):
        """Comparison for sorting. Higher fitness first, then smaller size."""
        if abs(self.fitness - other.fitness) < 1e-9:
            return self.size() < other.size()  # Smaller is better for ties
        return self.fitness > other.fitness  # Higher fitness is better


class EvolutionEngine:
    """
    (μ+λ) Evolution Strategy for evolving prediction programs.
    
    The algorithm:
    1. Maintain μ elite individuals
    2. Generate λ offspring through mutation
    3. Evaluate all μ+λ individuals
    4. Select top μ for next generation
    """
    
    def __init__(self, mu: int = 16, lambda_: int = 64, 
                 mutation_rate: float = 0.10, max_mutation_attempts: int = 20,
                 k: int = 6, N: int = 4096, seed: int = None):
        """
        Initialize the evolution engine.
        
        Args:
            mu: Number of elites to keep
            lambda_: Number of offspring to generate
            mutation_rate: Probability of mutation per node
            max_mutation_attempts: Max attempts for valid mutation
            k: Context length for prediction
            N: Number of symbols to evaluate on
            seed: Random seed for reproducibility
        """
        self.mu = mu
        self.lambda_ = lambda_
        self.mutation_rate = mutation_rate
        self.max_mutation_attempts = max_mutation_attempts
        self.k = k
        self.N = N
        
        # Set up random number generator
        self.rng = random.Random(seed)
        
        # Evolution state
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.history: List[Dict] = []
        
        # Statistics
        self.total_evaluations = 0
        self.total_mutations = 0
        self.total_mutation_failures = 0
        
        # Interpreter for evaluation
        self.interpreter = LispInterpreter()
        
        logger.info(f"Initialized evolution engine: μ={mu}, λ={lambda_}, "
                   f"mutation_rate={mutation_rate}, k={k}, N={N}")
    
    def initialize_population(self, initial_programs: List[LispNode]) -> None:
        """
        Initialize the population with seed programs.
        
        Args:
            initial_programs: List of initial AST programs
        """
        if len(initial_programs) < self.mu:
            raise ValueError(f"Need at least {self.mu} initial programs, got {len(initial_programs)}")
        
        self.population = []
        
        # Take first μ programs as initial population
        for i, program in enumerate(initial_programs[:self.mu]):
            individual = Individual(
                program=program,
                fitness=-float('inf'),  # Will be evaluated
                metrics={},
                generation=0,
                individual_id=i
            )
            self.population.append(individual)
        
        logger.info(f"Initialized population with {len(self.population)} individuals")
    
    def mutate_individual(self, parent: Individual) -> Optional[Individual]:
        """
        Create a mutated offspring from a parent.
        
        Args:
            parent: Parent individual
            
        Returns:
            Mutated offspring or None if mutation failed
        """
        agent = SelfModifyingAgent(parent.program,
                                 mutation_rate=self.mutation_rate,
                                 max_mutation_attempts=self.max_mutation_attempts)

        # Set the agent's RNG to our evolution RNG for reproducibility
        agent.rng = self.rng
        
        try:
            mutated_agent = agent.mutate()
            self.total_mutations += 1
            
            offspring = Individual(
                program=mutated_agent.get_ast(),
                fitness=-float('inf'),  # Will be evaluated
                metrics={},
                generation=self.generation + 1,
                parent_id=parent.individual_id
            )
            
            return offspring
            
        except Exception as e:
            self.total_mutation_failures += 1
            logger.debug(f"Mutation failed: {e}")
            return None
    
    def evaluate_individual(self, individual: Individual, stream: Iterator[int]) -> None:
        """
        Evaluate an individual's fitness on a stream.
        
        Args:
            individual: Individual to evaluate
            stream: Bit stream for evaluation
        """
        try:
            fitness, metrics = evaluate_program(
                self.interpreter, individual.program, stream, 
                k=self.k, N=self.N
            )
            
            individual.fitness = fitness
            individual.metrics = metrics
            self.total_evaluations += 1
            
            logger.debug(f"Evaluated individual {individual.individual_id}: F={fitness:.4f}")
            
        except Exception as e:
            logger.warning(f"Evaluation failed for individual {individual.individual_id}: {e}")
            individual.fitness = -float('inf')
            individual.metrics = {'error': str(e)}
    
    def evolve_generation(self, stream: Iterator[int]) -> Dict:
        """
        Evolve one generation.
        
        Args:
            stream: Bit stream for evaluation
            
        Returns:
            Generation statistics
        """
        start_time = time.time()
        
        # Generate offspring
        offspring = []
        attempts = 0
        max_attempts = self.lambda_ * 3  # Allow some failures
        
        while len(offspring) < self.lambda_ and attempts < max_attempts:
            # Select parent (uniform random from current population)
            parent = self.rng.choice(self.population)
            
            # Mutate
            child = self.mutate_individual(parent)
            if child is not None:
                offspring.append(child)
            
            attempts += 1
        
        logger.info(f"Generated {len(offspring)} offspring from {attempts} attempts")
        
        # Combine parents and offspring
        combined_population = self.population + offspring
        
        # Collect stream data once for all evaluations
        stream_bits = []
        for i, bit in enumerate(stream):
            stream_bits.append(bit)
            if i >= self.N + self.k:  # Collect enough for evaluation
                break

        # Evaluate all individuals that need evaluation
        for individual in combined_population:
            if individual.fitness == -float('inf'):
                def bit_stream():
                    for bit in stream_bits:
                        yield bit
                self.evaluate_individual(individual, bit_stream())
        
        # Sort by fitness (descending) and size (ascending for ties)
        combined_population.sort()
        
        # Select top μ for next generation
        self.population = combined_population[:self.mu]
        self.generation += 1
        
        # Update best individual
        if self.population[0].fitness > (self.best_individual.fitness if self.best_individual else -float('inf')):
            self.best_individual = self.population[0]
        
        # Calculate statistics
        fitnesses = [ind.fitness for ind in self.population if ind.fitness != -float('inf')]
        sizes = [ind.size() for ind in self.population]
        
        generation_time = time.time() - start_time
        
        stats = {
            'generation': self.generation,
            'best_fitness': max(fitnesses) if fitnesses else -float('inf'),
            'median_fitness': sorted(fitnesses)[len(fitnesses)//2] if fitnesses else -float('inf'),
            'mean_fitness': sum(fitnesses) / len(fitnesses) if fitnesses else -float('inf'),
            'best_size': min(sizes),
            'median_size': sorted(sizes)[len(sizes)//2],
            'mean_size': sum(sizes) / len(sizes),
            'num_offspring': len(offspring),
            'mutation_success_rate': len(offspring) / attempts if attempts > 0 else 0.0,
            'generation_time': generation_time,
            'total_evaluations': self.total_evaluations,
            'total_mutations': self.total_mutations,
            'total_mutation_failures': self.total_mutation_failures
        }
        
        self.history.append(stats)
        
        logger.info(f"Generation {self.generation}: "
                   f"best_F={stats['best_fitness']:.4f}, "
                   f"median_F={stats['median_fitness']:.4f}, "
                   f"best_size={stats['best_size']}")
        
        return stats
    
    def run_evolution(self, stream_factory, num_generations: int, 
                     progress_callback=None) -> List[Dict]:
        """
        Run evolution for multiple generations.
        
        Args:
            stream_factory: Function that returns a fresh bit stream
            num_generations: Number of generations to run
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of generation statistics
        """
        logger.info(f"Starting evolution for {num_generations} generations")
        
        for gen in range(num_generations):
            # Get fresh stream for this generation
            stream = stream_factory()
            
            # Evolve one generation
            stats = self.evolve_generation(stream)
            
            # Progress callback
            if progress_callback:
                progress_callback(gen, stats)
            
            # Early stopping if no progress
            if len(self.history) >= 10:
                recent_best = [h['best_fitness'] for h in self.history[-10:]]
                if all(f == recent_best[0] for f in recent_best):
                    logger.info(f"Early stopping at generation {self.generation} (no progress)")
                    break
        
        logger.info(f"Evolution complete after {self.generation} generations")
        return self.history
    
    def get_best_individual(self) -> Optional[Individual]:
        """Get the best individual found so far."""
        return self.best_individual
    
    def get_population_summary(self) -> Dict:
        """Get summary statistics of current population."""
        if not self.population:
            return {}
        
        fitnesses = [ind.fitness for ind in self.population if ind.fitness != -float('inf')]
        sizes = [ind.size() for ind in self.population]
        
        return {
            'population_size': len(self.population),
            'generation': self.generation,
            'best_fitness': max(fitnesses) if fitnesses else -float('inf'),
            'worst_fitness': min(fitnesses) if fitnesses else -float('inf'),
            'mean_fitness': sum(fitnesses) / len(fitnesses) if fitnesses else -float('inf'),
            'fitness_std': (sum((f - sum(fitnesses)/len(fitnesses))**2 for f in fitnesses) / len(fitnesses))**0.5 if len(fitnesses) > 1 else 0.0,
            'mean_size': sum(sizes) / len(sizes),
            'size_range': (min(sizes), max(sizes))
        }


def create_initial_population(size: int, seed: int = None) -> List[LispNode]:
    """
    Create an initial population of diverse programs.

    Args:
        size: Number of programs to create
        seed: Random seed

    Returns:
        List of initial AST programs
    """
    from .core import const, var, op, if_expr, let

    rng = random.Random(seed)
    programs = []

    # Basic predictors
    basic_programs = [
        const(0.0),      # Always predict 0
        const(1.0),      # Always predict 1
        const(0.5),      # Random predictor
        const(0.3),      # Biased low
        const(0.7),      # Biased high
    ]

    programs.extend(basic_programs)

    # Generate more diverse programs
    while len(programs) < size:
        program_type = rng.choice(['const', 'simple_op', 'conditional'])

        if program_type == 'const':
            # Random constant
            value = rng.uniform(0.0, 1.0)
            programs.append(const(value))

        elif program_type == 'simple_op':
            # Simple arithmetic
            left = const(rng.uniform(0.0, 1.0))
            right = const(rng.uniform(0.0, 1.0))
            op_type = rng.choice(['+', '*'])
            programs.append(op(op_type, left, right))

        elif program_type == 'conditional':
            # Simple conditional
            condition = const(rng.choice([True, False]))
            then_val = const(rng.uniform(0.0, 1.0))
            else_val = const(rng.uniform(0.0, 1.0))
            programs.append(if_expr(condition, then_val, else_val))

    return programs[:size]


def evaluate_program_on_stream_window(interpreter: LispInterpreter, program: LispNode,
                                    bits: List[int], k: int = 4, N: int = 2048) -> Tuple[float, Dict]:
    """
    Evaluate a program on a fixed window of bits.

    This is a helper for evolution that works with pre-collected bit sequences.

    Args:
        interpreter: Lisp interpreter
        program: AST program to evaluate
        bits: List of bits
        k: Context length
        N: Maximum number of predictions

    Returns:
        Tuple of (fitness, metrics)
    """
    def bit_stream():
        for bit in bits:
            yield bit

    return evaluate_program(interpreter, program, bit_stream(), k, min(N, len(bits) - k))


def run_simple_evolution(stream_factory, num_generations: int = 50,
                        mu: int = 16, lambda_: int = 64, seed: int = None) -> Dict:
    """
    Run a simple evolution experiment.

    Args:
        stream_factory: Function that returns bit streams
        num_generations: Number of generations
        mu: Population size
        lambda_: Offspring count
        seed: Random seed

    Returns:
        Results dictionary with best individual and statistics
    """
    # Create initial population
    initial_programs = create_initial_population(mu, seed)

    # Initialize evolution engine
    engine = EvolutionEngine(mu=mu, lambda_=lambda_, seed=seed)
    engine.initialize_population(initial_programs)

    # Run evolution
    def progress_callback(gen, stats):
        if gen % 10 == 0:
            print(f"Gen {gen:3d}: F_best={stats['best_fitness']:.4f}, "
                  f"F_med={stats['median_fitness']:.4f}, "
                  f"size={stats['best_size']}")

    history = engine.run_evolution(stream_factory, num_generations, progress_callback)

    # Return results
    best = engine.get_best_individual()

    return {
        'best_individual': best,
        'final_population': engine.population,
        'history': history,
        'summary': engine.get_population_summary(),
        'total_evaluations': engine.total_evaluations,
        'total_mutations': engine.total_mutations
    }


class OverfittingGuardEngine(EvolutionEngine):
    """
    Evolution engine with overfitting protection using train/validation split.

    This engine evaluates individuals on training data for ranking but uses
    validation data for final selection to prevent overfitting.
    """

    def __init__(self, train_val_split: float = 0.5, **kwargs):
        """
        Initialize overfitting guard engine.

        Args:
            train_val_split: Fraction of data to use for training (rest for validation)
            **kwargs: Arguments passed to base EvolutionEngine
        """
        super().__init__(**kwargs)
        self.train_val_split = train_val_split
        self.validation_history = []

    def evaluate_individual_with_split(self, individual: Individual,
                                     train_stream: Iterator[int],
                                     val_stream: Iterator[int]) -> None:
        """
        Evaluate individual on both training and validation streams.

        Args:
            individual: Individual to evaluate
            train_stream: Training bit stream
            val_stream: Validation bit stream
        """
        try:
            # Evaluate on training data
            train_fitness, train_metrics = evaluate_program(
                self.interpreter, individual.program, train_stream,
                k=self.k, N=self.N
            )

            # Evaluate on validation data
            val_fitness, val_metrics = evaluate_program(
                self.interpreter, individual.program, val_stream,
                k=self.k, N=self.N
            )

            # Store both fitnesses
            individual.fitness = train_fitness  # Used for ranking
            individual.metrics = {
                'train_fitness': train_fitness,
                'val_fitness': val_fitness,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'overfitting_gap': train_fitness - val_fitness
            }

            self.total_evaluations += 2  # Count both evaluations

        except Exception as e:
            logger.warning(f"Split evaluation failed for individual {individual.individual_id}: {e}")
            individual.fitness = -float('inf')
            individual.metrics = {'error': str(e)}

    def evolve_generation_with_guard(self, train_stream_factory, val_stream_factory) -> Dict:
        """
        Evolve one generation with overfitting protection.

        Args:
            train_stream_factory: Function returning training streams
            val_stream_factory: Function returning validation streams

        Returns:
            Generation statistics including validation metrics
        """
        start_time = time.time()

        # Generate offspring (same as base class)
        offspring = []
        attempts = 0
        max_attempts = self.lambda_ * 3

        while len(offspring) < self.lambda_ and attempts < max_attempts:
            parent = self.rng.choice(self.population)
            child = self.mutate_individual(parent)
            if child is not None:
                offspring.append(child)
            attempts += 1

        # Combine parents and offspring
        combined_population = self.population + offspring

        # Collect stream data
        train_bits = []
        train_stream = train_stream_factory()
        for i, bit in enumerate(train_stream):
            train_bits.append(bit)
            if i >= self.N + self.k:
                break

        val_bits = []
        val_stream = val_stream_factory()
        for i, bit in enumerate(val_stream):
            val_bits.append(bit)
            if i >= self.N + self.k:
                break

        # Evaluate all individuals on both train and val
        for individual in combined_population:
            if individual.fitness == -float('inf'):
                def train_stream():
                    for bit in train_bits:
                        yield bit

                def val_stream():
                    for bit in val_bits:
                        yield bit

                self.evaluate_individual_with_split(individual, train_stream(), val_stream())

        # Rank by training fitness but select by validation fitness
        # First, sort by training fitness to get top candidates
        combined_population.sort(key=lambda x: x.fitness, reverse=True)

        # Take top 2*μ candidates based on training fitness
        candidates = combined_population[:2 * self.mu]

        # Among candidates, select top μ based on validation fitness
        candidates.sort(key=lambda x: x.metrics.get('val_fitness', -float('inf')), reverse=True)
        self.population = candidates[:self.mu]

        self.generation += 1

        # Update best individual based on validation fitness
        val_fitnesses = [ind.metrics.get('val_fitness', -float('inf')) for ind in self.population]
        best_val_idx = val_fitnesses.index(max(val_fitnesses))

        if max(val_fitnesses) > (self.best_individual.metrics.get('val_fitness', -float('inf')) if self.best_individual else -float('inf')):
            self.best_individual = self.population[best_val_idx]

        # Calculate statistics
        train_fitnesses = [ind.fitness for ind in self.population if ind.fitness != -float('inf')]
        val_fitnesses = [ind.metrics.get('val_fitness', -float('inf')) for ind in self.population if ind.metrics.get('val_fitness', -float('inf')) != -float('inf')]
        overfitting_gaps = [ind.metrics.get('overfitting_gap', 0) for ind in self.population if 'overfitting_gap' in ind.metrics]

        generation_time = time.time() - start_time

        stats = {
            'generation': self.generation,
            'best_train_fitness': max(train_fitnesses) if train_fitnesses else -float('inf'),
            'best_val_fitness': max(val_fitnesses) if val_fitnesses else -float('inf'),
            'median_train_fitness': sorted(train_fitnesses)[len(train_fitnesses)//2] if train_fitnesses else -float('inf'),
            'median_val_fitness': sorted(val_fitnesses)[len(val_fitnesses)//2] if val_fitnesses else -float('inf'),
            'mean_overfitting_gap': sum(overfitting_gaps) / len(overfitting_gaps) if overfitting_gaps else 0.0,
            'max_overfitting_gap': max(overfitting_gaps) if overfitting_gaps else 0.0,
            'num_offspring': len(offspring),
            'generation_time': generation_time
        }

        self.history.append(stats)
        self.validation_history.append({
            'generation': self.generation,
            'val_fitness': stats['best_val_fitness'],
            'train_fitness': stats['best_train_fitness'],
            'overfitting_gap': stats['best_train_fitness'] - stats['best_val_fitness']
        })

        logger.info(f"Generation {self.generation}: "
                   f"train_F={stats['best_train_fitness']:.4f}, "
                   f"val_F={stats['best_val_fitness']:.4f}, "
                   f"gap={stats['mean_overfitting_gap']:.4f}")

        return stats


def split_stream_data(bits: List[int], train_fraction: float = 0.5) -> Tuple[List[int], List[int]]:
    """
    Split bit sequence into training and validation sets.

    Args:
        bits: Full bit sequence
        train_fraction: Fraction to use for training

    Returns:
        Tuple of (train_bits, val_bits)
    """
    split_point = int(len(bits) * train_fraction)
    return bits[:split_point], bits[split_point:]


def run_evolution_with_overfitting_guard(stream_factory, num_generations: int = 50,
                                       mu: int = 16, lambda_: int = 64,
                                       train_val_split: float = 0.5, seed: int = None) -> Dict:
    """
    Run evolution with overfitting protection using train/val split.

    Args:
        stream_factory: Function that returns bit streams
        num_generations: Number of generations
        mu: Population size
        lambda_: Offspring count
        train_val_split: Fraction of data for training
        seed: Random seed

    Returns:
        Results with overfitting metrics
    """
    from .envs import periodic

    # Create initial population
    initial_programs = create_initial_population(mu, seed)

    # Initialize overfitting guard engine
    engine = OverfittingGuardEngine(train_val_split=train_val_split, mu=mu, lambda_=lambda_, seed=seed)
    engine.initialize_population(initial_programs)

    # Collect full dataset once
    full_bits = []
    stream = stream_factory()
    for i, bit in enumerate(stream):
        full_bits.append(bit)
        if i >= 10000:  # Collect enough data
            break

    # Split into train/val
    train_bits, val_bits = split_stream_data(full_bits, train_val_split)

    def train_stream_factory():
        for bit in train_bits:
            yield bit

    def val_stream_factory():
        for bit in val_bits:
            yield bit

    # Run evolution with overfitting guard
    history = []
    for gen in range(num_generations):
        stats = engine.evolve_generation_with_guard(train_stream_factory, val_stream_factory)
        history.append(stats)

        if gen % 10 == 0:
            print(f"Gen {gen:3d}: train_F={stats['best_train_fitness']:.4f}, "
                  f"val_F={stats['best_val_fitness']:.4f}, "
                  f"gap={stats['mean_overfitting_gap']:.4f}")

    best = engine.get_best_individual()

    return {
        'best_individual': best,
        'final_population': engine.population,
        'history': history,
        'validation_history': engine.validation_history,
        'summary': engine.get_population_summary(),
        'total_evaluations': engine.total_evaluations,
        'train_bits_count': len(train_bits),
        'val_bits_count': len(val_bits)
    }
