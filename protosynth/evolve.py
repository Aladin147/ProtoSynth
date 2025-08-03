"""
ProtoSynth Evolution System

This module implements the core evolutionary loop using (μ+λ) Evolution Strategy.
Programs evolve to become better predictors through mutation and selection.
"""

import logging
import random
import time
import itertools
from typing import List, Tuple, Dict, Iterator, Optional
from dataclasses import dataclass
from collections import deque
import numpy as np
from .core import LispNode, LispInterpreter
from .agent import SelfModifyingAgent
from .eval import evaluate_program, evaluate_program_calibrated
from .core import pretty_print_ast, op, const, var
from .mutation import iter_nodes

logger = logging.getLogger(__name__)


class CtxStats:
    """Track context user statistics for adaptive K_ctx decay."""

    def __init__(self, W: int = 10):
        self.bestF = deque(maxlen=W)
        self.ctx_ratio = deque(maxlen=W)
        self.avgF_ctx = deque(maxlen=W)
        self.avgF_plain = deque(maxlen=W)
        self.ready_gen: Optional[int] = None  # Track when gates first became ready

    def push(self, bestF: float, survivors: List['Individual']) -> None:
        """Update statistics with current generation data."""
        self.bestF.append(bestF)

        # Use combined context detection (runtime + AST-based)
        ctx = []
        plain = []

        for s in survivors:
            runtime_ctx = s.metrics.get('ctx_reads_per_eval', 0) > 0
            ast_ctx = is_context_user_fast(s.program)

            if runtime_ctx or ast_ctx:
                ctx.append(s)
            else:
                plain.append(s)

        self.ctx_ratio.append(len(ctx) / max(1, len(survivors)))

        if ctx:
            self.avgF_ctx.append(sum(c.fitness for c in ctx) / len(ctx))
        else:
            self.avgF_ctx.append(-1e9)

        if plain:
            self.avgF_plain.append(sum(p.fitness for p in plain) / len(plain))
        else:
            self.avgF_plain.append(-1e9)

    def ready(self) -> bool:
        """Check if context users are ready to compete (stable non-negative + context share)."""
        if not self.bestF:
            return False

        # F gate: best_F ≥ 0.0 for 10 consecutive gens (stable non-negative)
        avg_bestF = sum(self.bestF) / len(self.bestF)
        f_gate = avg_bestF >= 0.0 and len(self.bestF) >= 10 and all(f >= 0.0 for f in self.bestF)

        # Context share gate: ctx_ratio ≥ 0.30
        avg_ctx_ratio = sum(self.ctx_ratio) / len(self.ctx_ratio)
        ctx_gate = avg_ctx_ratio >= 0.30

        # Parity gate: avg_F_ctx ≥ avg_F_plain - 0.02
        avg_ctx_fitness = sum(self.avgF_ctx) / len(self.avgF_ctx)
        avg_plain_fitness = sum(self.avgF_plain) / len(self.avgF_plain)
        parity_gate = avg_ctx_fitness >= avg_plain_fitness - 0.02

        logger.debug(f"Gates: F={f_gate} ({avg_bestF:.3f}), ctx={ctx_gate} ({avg_ctx_ratio:.3f}), parity={parity_gate} ({avg_ctx_fitness:.3f} vs {avg_plain_fitness:.3f})")

        return f_gate and ctx_gate and parity_gate


def adaptive_K_ctx(gen: int, mu: int, stats: CtxStats, base: int = 8, decay_gens: int = 100) -> int:
    """
    Gate-only adaptive K_ctx that freezes until context users are truly ready.

    Args:
        gen: Current generation
        mu: Population size
        stats: Context statistics tracker
        base: Base context niche size
        decay_gens: Generations to decay over once ready

    Returns:
        Number of context users to protect
    """
    # Keep K_ctx = max(base, mu//2) UNTIL all gates hold for W=10 consecutive gens
    if not stats.ready():
        return max(base, mu // 2)

    # Only start decay after gates are ready
    # For simplicity, start decay immediately when ready (could add consecutive gate tracking)
    if stats.ready_gen is None:
        stats.ready_gen = gen  # Mark when we first became ready

    gens_since_ready = gen - stats.ready_gen

    if gens_since_ready >= decay_gens:
        return 0

    # Linear decay over decay_gens generations
    frac = gens_since_ready / decay_gens
    return int(round((max(base, mu // 2)) * (1.0 - frac)))


def is_context_user_fast(ast: LispNode) -> bool:
    """
    Quick AST-based context user detection.
    Returns True if program calls markov_table with non-constant arg or uses context variables.
    """
    if ast.node_type == "call" and ast.value == "markov_table":
        # markov_table with non-constant arg is a context user
        return not (len(ast.children) == 1 and ast.children[0].node_type == "const")

    if ast.node_type == "var" and ast.value in ['prev', 'prev2', 'prev3', 'prev4', 'prev5', 'prev6', 'ctx']:
        return True

    # Recursively check children
    return any(is_context_user_fast(ch) for ch in ast.children)


def program_is_probabilistic(ast: LispNode) -> bool:
    """
    Detect if program outputs probabilities (should skip calibration).
    Returns True for programs that call probabilistic modules.
    """
    # Declare prob outputs for these modules
    PROB_MODULES = {"markov_table", "soft_prev", "soft_flip"}

    def walk(node):
        # Check for both "call" and "op" node types (markov_table uses "op")
        if (node.node_type in ["call", "op"]) and node.value in PROB_MODULES:
            return True
        return any(walk(ch) for ch in node.children)

    return walk(ast)


def eval_candidate(ast: LispNode, env_name: str, buf: List[int], k: int,
                  ensemble_buffers: Optional[List[List[int]]] = None) -> Tuple[float, Dict]:
    """
    Evaluate candidate with environment-specific evaluation strategy and ensemble averaging.

    Args:
        ast: Program AST to evaluate
        env_name: Environment name (e.g., "markov_k2", "periodic_k4")
        buf: Primary bit buffer for evaluation
        k: Context length
        ensemble_buffers: Additional buffers for ensemble evaluation (Markov only)

    Returns:
        Tuple of (fitness, metrics)
    """
    if "markov" in env_name and ensemble_buffers:
        # Ensemble evaluation for variance reduction
        all_buffers = [buf] + ensemble_buffers
        fitnesses = []
        all_metrics = []

        for buffer in all_buffers:
            # Create fresh interpreter for each evaluation
            interp = LispInterpreter(max_steps=max(1000, len(buffer) * 2), timeout_seconds=10.0)

            # Use larger windows for variance reduction
            N_total = len(buffer) - k
            N_train = min(8192, N_total // 2)
            N_val = min(8192, N_total - N_train)

            fitness, metrics = evaluate_program_calibrated(
                interp, ast, buffer=buffer, k=k, N_train=N_train, N_val=N_val
            )
            fitnesses.append(fitness)
            all_metrics.append(metrics)

        # Average fitness across ensemble
        avg_fitness = sum(fitnesses) / len(fitnesses)

        # Combine metrics (use first buffer's metrics as base)
        combined_metrics = all_metrics[0].copy()
        combined_metrics['ensemble_fitnesses'] = fitnesses
        combined_metrics['ensemble_std'] = (sum((f - avg_fitness)**2 for f in fitnesses) / len(fitnesses))**0.5

        return avg_fitness, combined_metrics

    else:
        # Single buffer evaluation
        # Create fresh interpreter for each evaluation
        interp = LispInterpreter(max_steps=max(1000, len(buf) * 2), timeout_seconds=10.0)

        if "markov" in env_name:
            # Markov environments need calibration for binary predictors
            N_total = len(buf) - k
            N_train = min(8192, N_total // 2)  # Increased for variance reduction
            N_val = min(8192, N_total - N_train)

            return evaluate_program_calibrated(
                interp, ast, buffer=buf, k=k, N_train=N_train, N_val=N_val
            )
        else:
            # Other environments use standard evaluation
            stream = iter(buf)
            return evaluate_program(interp, ast, stream, k=k, N=len(buf) - k)


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
        """Comparison for sorting. Higher fitness first, then smaller size, then higher ctx_reads."""
        if abs(self.fitness - other.fitness) < 1e-9:
            # Tie-break by smaller AST size
            if self.size() == other.size():
                # Further tie-break by higher context reads
                self_ctx_reads = self.metrics.get('ctx_reads_per_eval', 0)
                other_ctx_reads = other.metrics.get('ctx_reads_per_eval', 0)
                return self_ctx_reads > other_ctx_reads  # Higher ctx_reads is better
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
                 k: int = 6, N: int = 4096, seed: Optional[int] = None,
                 env_name: str = "default", warmup_gens: int = 5):
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
        self.env_name = env_name
        self.warmup_gens = warmup_gens
        
        # Set up random number generator
        self.rng = random.Random(seed)
        
        # Evolution state
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.history: List[Dict] = []

        # Context statistics tracker for adaptive K_ctx
        self.ctx_stats = CtxStats(W=10)

        # Pinned teacher tracking
        self.pinned_teacher: Optional[Individual] = None
        self.teacher_good_gens = 0  # Consecutive gens with best_F >= 0.05
        
        # Statistics
        self.total_evaluations = 0
        self.total_mutations = 0
        self.total_mutation_failures = 0
        
        # Interpreter for evaluation with proper limits
        max_steps = max(1000, N * 2)  # Scale with sequence length
        timeout_seconds = 10.0  # Longer timeout for evolution
        self.interpreter = LispInterpreter(
            max_steps=max_steps,
            timeout_seconds=timeout_seconds
        )
        
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
    
    def evaluate_individual(self, individual: Individual, stream: Iterator[int],
                           ensemble_buffers: Optional[List[List[int]]] = None) -> None:
        """
        Evaluate an individual's fitness on a stream.
        
        Args:
            individual: Individual to evaluate
            stream: Bit stream for evaluation
        """
        try:
            # Use environment-specific evaluation
            # Convert stream to buffer for eval_candidate
            buf = []
            for i, bit in enumerate(stream):
                buf.append(bit)
                if i >= self.N + self.k:
                    break

            fitness, metrics = eval_candidate(
                individual.program, self.env_name, buf, self.k, ensemble_buffers
            )

            individual.fitness = fitness
            individual.metrics = metrics
            self.total_evaluations += 1

            # Debug logging
            logger.debug(f"Evaluated individual {individual.individual_id}: F={fitness:.6f}")
            logger.debug(f"  Program: {individual.program}")
            logger.debug(f"  Metrics: {metrics}")

        except Exception as e:
            logger.warning(f"Evaluation failed for individual {individual.individual_id}: {e}")
            individual.fitness = -float('inf')
            individual.metrics = {'error': str(e)}

    def refit_markov_table(self, individual: Individual, buffer: List[int]) -> None:
        """
        Lamarckian refit of markov_table parameters using MLE on training data.

        Args:
            individual: Individual with markov_table program
            buffer: Training data buffer
        """
        # Check if program contains markov_table
        program_str = pretty_print_ast(individual.program)
        if 'markov_table' not in program_str:
            return

        # Collect per-state counts on training slice
        k = 2  # markov_table requires k=2 for prev2, prev
        N_train = min(1000, len(buffer) - k)

        counts = {(a, b): {'n': 0, 'c1': 0} for a in (0, 1) for b in (0, 1)}

        for i in range(k, k + N_train):
            if i >= len(buffer):
                break

            ctx = buffer[i-k:i]
            y = buffer[i]

            if len(ctx) >= 2:
                s = (ctx[-2], ctx[-1])  # (prev2, prev)
                counts[s]['n'] += 1
                if y == 1:
                    counts[s]['c1'] += 1

        # MLE with Laplace smoothing: p_s = (c1_s + 1) / (n_s + 2)
        new_params = {}
        for s in counts:
            n_s = counts[s]['n']
            c1_s = counts[s]['c1']
            p_s = (c1_s + 1) / (n_s + 2)  # Laplace smoothing
            p_s = max(1e-3, min(1 - 1e-3, p_s))  # Clamp to [ε, 1-ε]

            # Map state tuple to parameter name
            param_key = f'p{s[0]}{s[1]}'
            new_params[param_key] = p_s

        # Store parameters in individual's metrics
        individual.metrics['markov_params'] = new_params

        logger.debug(f"Refit markov_table for individual {individual.individual_id}: {new_params}")

    def rank_key(self, individual: Individual) -> tuple:
        """
        Ranking key for selection: primary=fitness (desc), secondary=ctx_reads (desc), tertiary=size (asc).
        """
        ctx_reads = individual.metrics.get('ctx_reads_per_eval', 0.0)
        return (-individual.fitness, -ctx_reads, individual.size(), individual.individual_id)

    def select_with_ctx_quota(self, population: List[Individual], mu: int, K_ctx: int) -> List[Individual]:
        """
        Robust selection that guarantees K_ctx context users survive.

        Args:
            population: Combined population (parents + offspring)
            mu: Number of survivors
            K_ctx: Minimum number of context users to keep

        Returns:
            List of exactly mu survivors with at least K_ctx context users
        """
        # Partition by context usage (runtime + AST-based detection)
        ctx_pool = []
        plain_pool = []

        for ind in population:
            runtime_ctx = ind.metrics.get('ctx_reads_per_eval', 0.0) > 0.0
            ast_ctx = is_context_user_fast(ind.program)

            if runtime_ctx or ast_ctx:
                ctx_pool.append(ind)
            else:
                plain_pool.append(ind)

        # Sort each pool by ranking key
        ctx_sorted = sorted(ctx_pool, key=self.rank_key)
        plain_sorted = sorted(plain_pool, key=self.rank_key)
        all_sorted = sorted(population, key=self.rank_key)

        # Pin context survivors first (HARD minimum)
        k = min(K_ctx, len(ctx_sorted))
        pinned_ctx = ctx_sorted[:k]

        # Fill the rest from global pool excluding pinned
        pinned_ids = {ind.individual_id for ind in pinned_ctx}
        remainder = [ind for ind in all_sorted if ind.individual_id not in pinned_ids][:mu - len(pinned_ctx)]

        survivors = pinned_ctx + remainder

        # HARD invariants - fail loudly if quota violated (use combined detection)
        ctx_count = sum(1 for ind in survivors if (ind.metrics.get('ctx_reads_per_eval', 0.0) > 0.0 or is_context_user_fast(ind.program)))
        assert ctx_count >= k, f"Context quota violated: kept {ctx_count} but quota was {k}"
        assert len(survivors) == mu, f"Wrong survivor count: {len(survivors)} != {mu}"

        logger.info(f"Gen {self.generation}: Hard quota enforced - {ctx_count}/{mu} context survivors (quota={K_ctx})")

        return survivors

    def build_mle_markov_candidate(self, buffers: List[List[int]], k: int) -> Individual:
        """
        Build an MLE Markov table candidate that should have positive fitness.

        Args:
            buffers: Training buffers for MLE fitting
            k: Context length (should be 2 for markov_k2)

        Returns:
            Individual with MLE-fitted markov_table program
        """
        # Collect counts across all buffers
        total_counts = {(a, b): {'n': 0, 'c1': 0} for a in (0, 1) for b in (0, 1)}

        for buffer in buffers:
            N_train = min(2000, len(buffer) - k)

            for i in range(k, k + N_train):
                if i >= len(buffer):
                    break

                ctx = buffer[i-k:i]
                y = buffer[i]

                if len(ctx) >= 2:
                    s = (ctx[-2], ctx[-1])  # (prev2, prev)
                    total_counts[s]['n'] += 1
                    if y == 1:
                        total_counts[s]['c1'] += 1

        # MLE with Laplace smoothing: p_s = (c1_s + 1) / (n_s + 2)
        mle_params = {}
        for s in total_counts:
            n_s = total_counts[s]['n']
            c1_s = total_counts[s]['c1']
            p_s = (c1_s + 1) / (n_s + 2)  # Laplace smoothing
            p_s = max(1e-3, min(1 - 1e-3, p_s))  # Clamp to [ε, 1-ε]

            # Map state tuple to parameter name
            param_key = f'p{s[0]}{s[1]}'
            mle_params[param_key] = p_s

        # Create markov_table program
        program = op('markov_table', op('+', op('*', const(2), var('prev2')), var('prev')))

        # Create individual with MLE parameters
        individual = Individual(
            program=program,
            fitness=-float('inf'),  # Will be evaluated
            metrics={'markov_params': mle_params},
            generation=self.generation
        )

        logger.info(f"Gen {self.generation}: Built MLE teacher with params {mle_params}")

        return individual

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
            # Select parent with forced context breeding for Markov environments
            if "markov" in self.env_name and self.generation < 50:
                # Force ≥50% breeding from context users for first 50 generations
                ctx_parents = [ind for ind in self.population if ind.metrics.get('ctx_reads_per_eval', 0) > 0]
                plain_parents = [ind for ind in self.population if ind.metrics.get('ctx_reads_per_eval', 0) == 0]

                # Ensure at least half the offspring come from context users
                if len(ctx_parents) > 0 and len(offspring) < self.lambda_ // 2:
                    # Prioritize context users for first half of offspring
                    parent = self.rng.choice(ctx_parents)
                    logger.debug(f"Gen {self.generation}: Selected context parent for breeding")
                elif len(ctx_parents) > 0 and self.rng.random() < 0.5:
                    # 50% chance of context parent for second half
                    parent = self.rng.choice(ctx_parents)
                else:
                    # Select from full population
                    parent = self.rng.choice(self.population)
            else:
                # Normal parent selection
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
        for bit in stream:
            stream_bits.append(bit)
            if len(stream_bits) >= self.N + self.k + 10:  # Collect enough for evaluation + buffer
                break

        logger.debug(f"Collected {len(stream_bits)} bits for evaluation (need {self.N + self.k})")

        # Generate ensemble buffers for variance reduction (Markov only)
        ensemble_buffers = None
        if "markov" in self.env_name:
            ensemble_buffers = []
            for i in range(4):  # M=5 total (1 primary + 4 ensemble)
                # Generate additional buffer with different seed
                from .envs import markov_k1
                ensemble_stream = markov_k1(p_stay=0.8, seed=42 + self.generation * 10 + i + 1)
                ensemble_buf = []
                for bit in ensemble_stream:
                    ensemble_buf.append(bit)
                    if len(ensemble_buf) >= len(stream_bits):
                        break
                ensemble_buffers.append(ensemble_buf)

        # Lamarckian refit for markov_table programs (after mutation, before evaluation)
        if "markov" in self.env_name:
            for individual in combined_population:
                self.refit_markov_table(individual, stream_bits)

            # Anchor: Add MLE teacher candidate for guaranteed positive fitness
            if ensemble_buffers:
                all_buffers = [stream_bits] + ensemble_buffers
            else:
                all_buffers = [stream_bits]

            # Build fresh MLE teacher
            mle_teacher = self.build_mle_markov_candidate(all_buffers, self.k)

            # Sanity tripwire: teacher should have reasonable fitness
            # Set MLE parameters before evaluation
            if 'markov_params' in mle_teacher.metrics:
                # Create interpreter with MLE parameters for teacher evaluation
                teacher_interpreter = LispInterpreter(max_steps=10000, timeout_seconds=10.0)
                teacher_interpreter.markov_params = mle_teacher.metrics['markov_params']

                # Use evaluate_program_calibrated directly with the configured interpreter
                from .eval import evaluate_program_calibrated
                teacher_fitness, teacher_metrics = evaluate_program_calibrated(
                    teacher_interpreter, mle_teacher.program, buffer=stream_bits, k=self.k,
                    N_train=min(4000, len(stream_bits)//2), N_val=min(4000, len(stream_bits)//2)
                )
            else:
                # Fallback to regular eval_candidate
                teacher_fitness, teacher_metrics = eval_candidate(
                    mle_teacher.program, self.env_name, stream_bits, self.k, ensemble_buffers
                )

            mle_teacher.fitness = teacher_fitness
            mle_teacher.metrics.update(teacher_metrics)

            logger.debug(f"Gen {self.generation}: MLE teacher F={teacher_fitness:.6f}")

            # Pin teacher if not already pinned or if current teacher is better
            if self.pinned_teacher is None or teacher_fitness > self.pinned_teacher.fitness:
                self.pinned_teacher = mle_teacher
                logger.info(f"Gen {self.generation}: Pinned new teacher F={teacher_fitness:.6f}")

            # Add pinned teacher to population (guaranteed parent)
            combined_population.append(self.pinned_teacher)

        # Evaluate all individuals that need evaluation
        for individual in combined_population:
            if individual.fitness == -float('inf'):
                def bit_stream():
                    for bit in stream_bits:
                        yield bit
                self.evaluate_individual(individual, bit_stream(), ensemble_buffers)
        
        # Apply lexicase selection every 5 generations to break plateaus
        if self.generation % 5 == 4:  # Every 5th generation (0-indexed)
            self.population = self._lexicase_selection(combined_population, self.mu)
        else:
            if "markov" in self.env_name:
                # Use gate-only adaptive K_ctx (no time-based decay)
                K_ctx = adaptive_K_ctx(self.generation, self.mu, self.ctx_stats, base=8, decay_gens=100)

                # Use robust selection with hard quota enforcement
                self.population = self.select_with_ctx_quota(combined_population, self.mu, K_ctx)

                # Update context statistics for adaptive decay
                best_fitness = max(ind.fitness for ind in self.population)
                self.ctx_stats.push(best_fitness, self.population)

                # Update teacher pinning status
                best_fitness = max(ind.fitness for ind in self.population)
                if best_fitness >= 0.05:
                    self.teacher_good_gens += 1
                else:
                    self.teacher_good_gens = 0

                # Unpin teacher after 10 consecutive good generations
                if self.teacher_good_gens >= 10 and self.pinned_teacher is not None:
                    logger.info(f"Gen {self.generation}: Unpinning teacher after {self.teacher_good_gens} good gens")
                    self.pinned_teacher = None
                    self.teacher_good_gens = 0

                # Log adaptive statistics
                ctx_count = sum(1 for ind in self.population if (
                    ind.metrics.get('ctx_reads_per_eval', 0) > 0 or is_context_user_fast(ind.program)
                ))
                ready_status = "READY" if self.ctx_stats.ready() else "PROTECTED"
                teacher_status = "PINNED" if self.pinned_teacher is not None else "FREE"
                logger.info(f"Gen {self.generation}: K_ctx={K_ctx}, ctx={ctx_count}, {ready_status}, teacher={teacher_status}")

            else:
                # Normal selection for non-Markov environments
                combined_population.sort(key=self.rank_key)
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

    def _lexicase_selection(self, population: List[Individual], num_select: int) -> List[Individual]:
        """
        Lexicase selection: select individuals based on per-context performance.

        This helps break fitness plateaus by selecting individuals that excel
        on specific contexts, even if their average fitness is lower.
        """
        if not population:
            return []

        # Extract per-context errors from metrics (if available)
        selected = []
        remaining_population = population.copy()

        for _ in range(num_select):
            if not remaining_population:
                break

            # For simplicity, use fitness-based selection with some randomness
            # In a full implementation, this would use per-context error vectors

            # Select top candidates (top 50% by fitness)
            remaining_population.sort(key=lambda x: x.fitness, reverse=True)
            top_half = remaining_population[:max(1, len(remaining_population) // 2)]

            # Randomly select from top half to add diversity
            chosen = self.rng.choice(top_half)
            selected.append(chosen)
            remaining_population.remove(chosen)

        return selected
    
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
            'mean_fitness': round(sum(fitnesses) / len(fitnesses), 12) if fitnesses else -float('inf'),
            'fitness_std': (sum((f - sum(fitnesses)/len(fitnesses))**2 for f in fitnesses) / len(fitnesses))**0.5 if len(fitnesses) > 1 else 0.0,
            'mean_size': sum(sizes) / len(sizes),
            'size_range': (min(sizes), max(sizes))
        }


def create_initial_population(size: int, seed: Optional[int] = None) -> List[LispNode]:
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

    # Context-aware seed programs (high priority)
    context_programs = [
        var('prev'),                                    # Return previous bit (k>=1)
        var('prev2'),                                   # Return bit from 2 steps ago (k>=2)
        var('prev3'),                                   # Return bit from 3 steps ago (k>=3)
        var('prev4'),                                   # Return bit from 4 steps ago (k>=4)
        op('xor', var('prev'), var('prev2')),          # XOR of last two bits (k>=2)
        op('>', op('+', var('prev'), var('prev2')), const(0.5)),  # Majority of last 2 (k>=2)
        op('=', var('prev'), var('prev2')),            # Are last two bits equal? (k>=2)
        op('-', const(1), var('prev')),                # Flip previous bit (k>=1)
        # Soft predictors for Markov chains (critical for stochastic environments)
        if_expr(op('>', var('prev'), const(0.5)), const(0.8), const(0.2)),  # Soft stay: if prev>0.5 then 0.8 else 0.2
        if_expr(op('>', var('prev'), const(0.5)), const(0.2), const(0.8)),  # Soft flip: if prev>0.5 then 0.2 else 0.8

        # Parametric Markov table seeds (critical for Markov learning)
        op('markov_table', op('+', op('*', const(2), var('prev2')), var('prev'))),  # state2 = 2*prev2 + prev
        # Enhanced seeds for better pattern recognition
        op('>', op('+', op('+', var('prev'), var('prev2')), var('prev3')), const(1.5)),  # Majority of last 3 (k>=3)
        op('xor', op('xor', var('prev'), var('prev2')), var('prev3')),  # Parity of last 3 (k>=3)
        op('parity3', var('ctx')),                     # Parity3 micro-primitive (k>=3)
        op('>', op('sum_bits', var('ctx')), const(1.5)),  # Sum-based majority (k>=1)
        op('index', var('ctx'), const(-3)),            # Access 3rd bit back (k>=3)
        op('index', var('ctx'), const(-4)),            # Access 4th bit back (k>=4)
    ]

    # Note: Context filtering will be done at runtime based on k parameter

    # Basic predictors (lower priority)
    basic_programs = [
        const(0.5),      # Random predictor (baseline)
        const(0.0),      # Always predict 0
        const(1.0),      # Always predict 1
        const(0.3),      # Biased low
        const(0.7),      # Biased high
    ]

    # Add context programs first (they're more likely to be useful)
    programs.extend(context_programs)
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


def create_markov_table_individual(bias_type: str, generation: int = 0) -> Individual:
    """
    Create an individual with a markov_table program and pre-configured parameters.

    Args:
        bias_type: 'stay' for stay-biased, 'flip' for flip-biased
        generation: Generation number

    Returns:
        Individual with markov_table program and configured parameters
    """
    # Create markov_table program
    program = op('markov_table', op('+', op('*', const(2), var('prev2')), var('prev')))

    # Configure parameters based on bias type
    if bias_type == 'stay':
        # Stay-biased: p00=p11≈0.8, p01=p10≈0.2
        markov_params = {'p00': 0.8, 'p01': 0.2, 'p10': 0.2, 'p11': 0.8}
    else:  # flip
        # Flip-biased: p00=p11≈0.2, p01=p10≈0.8
        markov_params = {'p00': 0.2, 'p01': 0.8, 'p10': 0.8, 'p11': 0.2}

    # Create individual
    individual = Individual(
        program=program,
        fitness=-float('inf'),
        metrics={'markov_params': markov_params},
        generation=generation
    )

    return individual


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
                        mu: int = 16, lambda_: int = 64, seed: Optional[int] = None) -> Dict:
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
        try:
            if callable(train_stream_factory):
                train_stream = train_stream_factory()
            else:
                train_stream = iter(train_stream_factory)  # Ensure it's iterable

            for i, bit in enumerate(train_stream):
                train_bits.append(bit)
                if i >= self.N + self.k:
                    break
        except Exception as e:
            logger.warning(f"Failed to collect training stream: {e}")
            train_bits = [0, 1] * (self.N + self.k)  # Fallback

        val_bits = []
        try:
            if callable(val_stream_factory):
                val_stream = val_stream_factory()
            else:
                val_stream = iter(val_stream_factory)  # Ensure it's iterable

            for i, bit in enumerate(val_stream):
                val_bits.append(bit)
                if i >= self.N + self.k:
                    break
        except Exception as e:
            logger.warning(f"Failed to collect validation stream: {e}")
            val_bits = [1, 0] * (self.N + self.k)  # Fallback

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
                                       train_val_split: float = 0.5, seed: Optional[int] = None) -> Dict:
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
