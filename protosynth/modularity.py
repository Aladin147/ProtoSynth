"""
ProtoSynth Emergent Modularity System

This module implements subtree mining, moduleization, and reuse mechanisms
for discovering and exploiting modular structure in evolved programs.
"""

import logging
import hashlib
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
from .core import LispNode, LispInterpreter
from .mutation import iter_nodes
from .eval import evaluate_program_on_window

logger = logging.getLogger(__name__)


@dataclass
class SubtreeCandidate:
    """Represents a candidate subtree for moduleization."""
    subtree: LispNode
    frequency: int
    total_nodes: int
    mdl_score: float
    bits_saved: float
    size_penalty: float
    
    def __hash__(self):
        return hash(self.subtree_hash())
    
    def subtree_hash(self) -> str:
        """Generate a hash for the subtree structure."""
        return subtree_to_hash(self.subtree)


def subtree_to_hash(node: LispNode) -> str:
    """Convert a subtree to a canonical hash string."""
    if node.node_type == 'const':
        # Normalize constants to avoid over-specificity
        if isinstance(node.value, (int, float)):
            # Round to 2 decimal places for grouping
            normalized = round(float(node.value), 2)
            return f"const:{normalized}"
        else:
            return f"const:{node.value}"
    elif node.node_type == 'var':
        # Use generic variable names for pattern matching
        return f"var:X"
    elif node.node_type in ['op', 'if', 'let']:
        # Recursively hash children
        child_hashes = []
        if hasattr(node, 'children') and node.children:
            child_hashes = [subtree_to_hash(child) for child in node.children]
        elif hasattr(node, 'value') and isinstance(node.value, list):
            child_hashes = [subtree_to_hash(child) for child in node.value]

        children_str = ",".join(child_hashes)
        return f"{node.node_type}:{node.value if not isinstance(node.value, list) else ''}({children_str})"
    else:
        return f"{node.node_type}:{node.value}"


def extract_all_subtrees(ast: LispNode, min_size: int = 2, max_size: int = 8) -> List[LispNode]:
    """
    Extract all subtrees from an AST within size bounds.

    Args:
        ast: Root AST node
        min_size: Minimum subtree size (number of nodes)
        max_size: Maximum subtree size (number of nodes)

    Returns:
        List of subtree nodes
    """
    subtrees = []

    def extract_recursive(node: LispNode):
        # Count nodes in this subtree
        node_count = len(list(iter_nodes(node)))

        # Add this subtree if it's in the size range
        if min_size <= node_count <= max_size:
            subtrees.append(node)

        # Recursively extract from children
        if hasattr(node, 'children') and node.children:
            for child in node.children:
                if isinstance(child, LispNode):
                    extract_recursive(child)

    extract_recursive(ast)
    return subtrees


class SubtreeMiner:
    """
    Mines frequent subtrees from a population and scores them using MDL.
    
    The MDL (Minimum Description Length) principle scores subtrees by:
    Δ(MDL) = bits_saved - β * nodes
    
    Where:
    - bits_saved: Compression benefit from reusing the subtree
    - β: Size penalty parameter (default 0.005)
    - nodes: Number of nodes in the subtree
    """
    
    def __init__(self, beta: float = 0.0035, min_frequency: int = 2,
                 min_size: int = 2, max_size: int = 8):
        """
        Initialize the subtree miner.

        Args:
            beta: Size penalty parameter for MDL scoring (reduced for better discovery)
            min_frequency: Minimum frequency for a subtree to be considered (reduced for testing)
            min_size: Minimum subtree size (nodes)
            max_size: Maximum subtree size (nodes)
        """
        self.beta = beta
        self.min_frequency = min_frequency
        self.min_size = min_size
        self.max_size = max_size
        
        # Mining state
        self.subtree_counts = Counter()
        self.subtree_examples = {}  # hash -> actual LispNode
        self.total_programs = 0
        
        logger.info(f"Initialized SubtreeMiner: β={beta}, min_freq={min_frequency}")
    
    def add_program(self, ast: LispNode) -> None:
        """Add a program to the mining dataset."""
        self.total_programs += 1
        
        # Extract all subtrees
        subtrees = extract_all_subtrees(ast, self.min_size, self.max_size)
        
        # Count subtree patterns
        for subtree in subtrees:
            subtree_hash = subtree_to_hash(subtree)
            self.subtree_counts[subtree_hash] += 1
            
            # Store an example of this subtree pattern
            if subtree_hash not in self.subtree_examples:
                self.subtree_examples[subtree_hash] = subtree
    
    def mine_candidates(self, validation_bits: List[int]) -> List[SubtreeCandidate]:
        """
        Mine subtree candidates and score them using MDL.
        
        Args:
            validation_bits: Validation data for measuring compression benefit
            
        Returns:
            List of scored subtree candidates
        """
        candidates = []
        interpreter = LispInterpreter()
        
        logger.info(f"Mining from {len(self.subtree_counts)} unique subtree patterns")
        
        for subtree_hash, frequency in self.subtree_counts.items():
            if frequency < self.min_frequency:
                continue
            
            subtree = self.subtree_examples[subtree_hash]
            
            # Calculate subtree properties
            total_nodes = len(list(iter_nodes(subtree)))
            
            # Estimate compression benefit
            bits_saved = self.estimate_compression_benefit(
                subtree, frequency, validation_bits, interpreter
            )
            
            # Calculate MDL score
            size_penalty = self.beta * total_nodes
            mdl_score = bits_saved - size_penalty
            
            candidate = SubtreeCandidate(
                subtree=subtree,
                frequency=frequency,
                total_nodes=total_nodes,
                mdl_score=mdl_score,
                bits_saved=bits_saved,
                size_penalty=size_penalty
            )
            
            candidates.append(candidate)
            
            logger.debug(f"Candidate: freq={frequency}, nodes={total_nodes}, "
                        f"bits_saved={bits_saved:.3f}, MDL={mdl_score:.3f}")
        
        # Sort by MDL score (descending)
        candidates.sort(key=lambda c: c.mdl_score, reverse=True)
        
        logger.info(f"Found {len(candidates)} candidates, "
                   f"{sum(1 for c in candidates if c.mdl_score > 0)} with positive MDL")
        
        return candidates
    
    def estimate_compression_benefit(self, subtree: LispNode, frequency: int,
                                   validation_bits: List[int], 
                                   interpreter: LispInterpreter) -> float:
        """
        Estimate the compression benefit of a subtree.
        
        This is a simplified estimate based on:
        1. How often the subtree appears (frequency)
        2. How well it predicts on validation data
        3. Size savings from reuse
        
        Args:
            subtree: The subtree to evaluate
            frequency: How often it appears in the population
            validation_bits: Validation data
            interpreter: Interpreter for evaluation
            
        Returns:
            Estimated bits saved through compression
        """
        try:
            # Try to evaluate the subtree as a standalone predictor
            fitness, metrics = evaluate_program_on_window(
                interpreter, subtree, validation_bits, k=4
            )
            
            # If it's a good predictor, it has high reuse value
            predictor_quality = max(0, fitness)  # Clamp to positive
            
            # Estimate compression benefit
            # Higher frequency and better prediction quality = more bits saved
            base_benefit = frequency * 0.1  # Base benefit from reuse
            quality_bonus = predictor_quality * frequency * 0.05
            
            total_benefit = base_benefit + quality_bonus
            
            return total_benefit
            
        except Exception as e:
            logger.debug(f"Failed to evaluate subtree: {e}")
            # Fallback: just use frequency-based estimate
            return frequency * 0.05
    
    def get_top_modules(self, n: int = 10) -> List[SubtreeCandidate]:
        """Get the top n module candidates by MDL score."""
        # This requires mining to have been done first
        if not hasattr(self, '_last_candidates'):
            raise ValueError("Must call mine_candidates() first")
        
        positive_candidates = [c for c in self._last_candidates if c.mdl_score > 0]
        return positive_candidates[:n]
    
    def mine_and_select(self, population: List[LispNode], 
                       validation_bits: List[int], n_modules: int = 10) -> List[SubtreeCandidate]:
        """
        Complete mining pipeline: add population, mine, and select top modules.
        
        Args:
            population: List of AST programs to mine from
            validation_bits: Validation data for scoring
            n_modules: Number of top modules to return
            
        Returns:
            List of top module candidates
        """
        # Clear previous state
        self.subtree_counts.clear()
        self.subtree_examples.clear()
        self.total_programs = 0
        
        # Add all programs to mining dataset
        for ast in population:
            self.add_program(ast)
        
        # Mine candidates
        candidates = self.mine_candidates(validation_bits)
        self._last_candidates = candidates
        
        # Return top modules
        return self.get_top_modules(n_modules)


def demo_subtree_mining():
    """Demonstrate subtree mining functionality."""
    from .core import const, var, op, let, if_expr
    
    print("🔍 Subtree Mining Demo")
    print("=" * 30)
    
    # Create a population with repeated patterns
    population = [
        # Pattern 1: (+ x 1) appears multiple times
        op('+', var('x'), const(1)),
        let('x', const(5), op('+', var('x'), const(1))),
        if_expr(const(True), op('+', var('y'), const(1)), const(0)),
        
        # Pattern 2: (> x 0) appears multiple times  
        op('>', var('x'), const(0)),
        if_expr(op('>', var('z'), const(0)), const(1), const(0)),
        let('x', const(10), op('>', var('x'), const(0))),
        
        # Some unique programs
        const(42),
        op('*', const(2), const(3)),
    ]
    
    # Create validation data
    validation_bits = [0, 1, 0, 1] * 50
    
    # Mine subtrees
    miner = SubtreeMiner(beta=0.005, min_frequency=2)
    top_modules = miner.mine_and_select(population, validation_bits, n_modules=5)
    
    print(f"Found {len(top_modules)} module candidates:")
    for i, module in enumerate(top_modules):
        print(f"  {i+1}. Freq={module.frequency}, Nodes={module.total_nodes}, "
              f"MDL={module.mdl_score:.3f}")
        print(f"     Pattern: {subtree_to_hash(module.subtree)}")
    
    return top_modules


@dataclass(frozen=True)
class Refinement:
    """Minimal type refinement for module contracts."""
    type: str  # "int" | "float" | "bit" | "any"
    range: Tuple[float, float] = (-float('inf'), float('inf'))  # e.g., (0.0, 1.0)


@dataclass(frozen=True)
class ModuleSignature:
    """Complete module signature with versioning."""
    name: str
    major: int
    minor: int
    arity: int
    inputs: Tuple[Refinement, ...]
    output: Refinement
    contract_hash: str

    @classmethod
    def create(cls, name: str, major: int, minor: int, arity: int,
              inputs: List[Refinement], output: Refinement) -> 'ModuleSignature':
        """Create a module signature with computed hash."""
        import hashlib

        # Compute contract hash from structure
        contract_data = f"{arity}:{inputs}:{output}"
        contract_hash = hashlib.md5(contract_data.encode()).hexdigest()[:8]

        return cls(
            name=name,
            major=major,
            minor=minor,
            arity=arity,
            inputs=tuple(inputs),
            output=output,
            contract_hash=contract_hash
        )


@dataclass
class ModuleContract:
    """Legacy contract class - kept for compatibility."""
    arity: int
    return_type: str  # 'float', 'int', 'bool', 'any'
    return_range: Optional[Tuple[float, float]] = None  # For numeric types
    arg_types: Optional[List[str]] = None  # Types for each argument

    def validate_call(self, args: List[LispNode]) -> Tuple[bool, List[str]]:
        """
        Validate a module call against this contract.

        Args:
            args: Arguments to the module call

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check arity
        if len(args) != self.arity:
            errors.append(f"Expected {self.arity} arguments, got {len(args)}")

        # Check argument types if specified
        if self.arg_types and len(args) == self.arity:
            for i, (arg, expected_type) in enumerate(zip(args, self.arg_types)):
                if not self._check_arg_type(arg, expected_type):
                    errors.append(f"Argument {i} should be {expected_type}, got {arg.node_type}")

        return len(errors) == 0, errors

    def validate_return_value(self, value) -> Tuple[bool, List[str]]:
        """
        Validate a return value against this contract.

        Args:
            value: The returned value

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check return type
        if self.return_type == 'float':
            if not isinstance(value, (int, float)):
                errors.append(f"Expected float return, got {type(value)}")
            elif self.return_range:
                min_val, max_val = self.return_range
                if not (min_val <= value <= max_val):
                    errors.append(f"Return value {value} outside range [{min_val}, {max_val}]")
        elif self.return_type == 'int':
            if not isinstance(value, int):
                errors.append(f"Expected int return, got {type(value)}")
        elif self.return_type == 'bool':
            if not isinstance(value, bool):
                errors.append(f"Expected bool return, got {type(value)}")
        # 'any' type accepts anything

        return len(errors) == 0, errors

    def _check_arg_type(self, arg: LispNode, expected_type: str) -> bool:
        """Check if an argument matches the expected type."""
        if expected_type == 'any':
            return True
        elif expected_type == 'const':
            return arg.node_type == 'const'
        elif expected_type == 'var':
            return arg.node_type == 'var'
        elif expected_type == 'numeric':
            return (arg.node_type == 'const' and
                   isinstance(arg.value, (int, float)) and
                   not isinstance(arg.value, bool))
        else:
            return arg.node_type == expected_type


@dataclass
class Module:
    """Represents a reusable module with a name and implementation."""
    name: str
    implementation: LispNode
    arity: int
    mdl_score: float
    frequency: int
    contract: Optional[ModuleContract] = None
    signature: Optional[ModuleSignature] = None
    version: str = "1.0.0"
    credit_score: float = 0.0  # Rolling Shapley-lite score
    usage_count: int = 0
    last_used_gen: int = 0

    def __hash__(self):
        return hash(f"{self.name}@{self.version}")

    def get_canonical_form(self) -> str:
        """Get canonicalized representation for deduplication."""
        return canonicalize_subtree(self.implementation)

    def get_version_parts(self) -> Tuple[int, int]:
        """Parse version string into (major, minor) tuple."""
        try:
            parts = self.version.split('.')
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
            return major, minor
        except (ValueError, IndexError):
            return 1, 0


def canonicalize_subtree(node: LispNode) -> str:
    """
    Create canonical representation of subtree for deduplication.

    Features:
    - Alpha-renaming of variables (X, Y, Z, ...)
    - Sorted argument order for commutative operations
    - Normalized constants (binned to reduce near-duplicates)
    - Generic variable names

    Args:
        node: Subtree to canonicalize

    Returns:
        Canonical string representation
    """
    # First pass: collect all variables for alpha-renaming
    var_mapping = {}
    var_counter = 0

    def collect_vars(n: LispNode):
        nonlocal var_counter
        if n.node_type == 'var' and n.value not in var_mapping:
            var_mapping[n.value] = chr(ord('X') + var_counter)
            var_counter += 1

        if hasattr(n, 'children') and n.children:
            for child in n.children:
                if isinstance(child, LispNode):
                    collect_vars(child)

    collect_vars(node)

    # Second pass: canonicalize with alpha-renaming
    def canonicalize_recursive(n: LispNode) -> str:
        if n.node_type == 'const':
            if isinstance(n.value, (int, float)):
                # Bin constants to reduce near-duplicates
                val = float(n.value)
                if val == 0.0:
                    binned = 0.0
                elif val == 1.0:
                    binned = 1.0
                elif val < 0:
                    binned = -1.0
                else:
                    # Round to nearest 0.5
                    binned = round(val * 2) / 2
                return f"const:{binned}"
            else:
                return f"const:{n.value}"

        elif n.node_type == 'var':
            # Use alpha-renamed variable
            return f"var:{var_mapping.get(n.value, 'X')}"

        elif n.node_type == 'op':
            op_name = n.value

            # Canonicalize children
            child_reprs = []
            if hasattr(n, 'children') and n.children:
                child_reprs = [canonicalize_recursive(child) for child in n.children if isinstance(child, LispNode)]

            # Sort arguments for commutative operations
            if op_name in ['+', '*', 'and', 'or', '==', '!=']:
                child_reprs.sort()

            children_str = ",".join(child_reprs)
            return f"op:{op_name}({children_str})"

        elif n.node_type in ['if', 'let', 'call']:
            # Preserve order for control structures and calls
            child_reprs = []
            if hasattr(n, 'children') and n.children:
                child_reprs = [canonicalize_recursive(child) for child in n.children if isinstance(child, LispNode)]

            children_str = ",".join(child_reprs)
            return f"{n.node_type}:{n.value or ''}({children_str})"

        else:
            return f"{n.node_type}:{n.value or ''}"

    return canonicalize_recursive(node)


# Global module registry
ModuleRegistry: Dict[str, Dict[int, List[ModuleSignature]]] = {}


def register_signature(signature: ModuleSignature) -> None:
    """Register a module signature in the global registry."""
    if signature.name not in ModuleRegistry:
        ModuleRegistry[signature.name] = {}

    if signature.major not in ModuleRegistry[signature.name]:
        ModuleRegistry[signature.name][signature.major] = []

    # Insert in sorted order by minor version
    major_versions = ModuleRegistry[signature.name][signature.major]

    # Remove existing signature with same minor version
    major_versions[:] = [s for s in major_versions if s.minor != signature.minor]

    # Insert new signature
    major_versions.append(signature)
    major_versions.sort(key=lambda s: s.minor)


def resolve_signature(name: str, major: int) -> Optional[ModuleSignature]:
    """Resolve module signature - picks latest minor with same major."""
    if name not in ModuleRegistry:
        return None

    if major not in ModuleRegistry[name]:
        return None

    major_versions = ModuleRegistry[name][major]
    if not major_versions:
        return None

    # Return latest minor version
    return major_versions[-1]


class ModuleLibrary:
    """
    Enhanced module library with credit assignment, versioning, and canonicalization.

    Features:
    - Credit assignment via stochastic masking
    - Module versioning with semantic versioning
    - Canonicalization to reduce near-duplicates
    - LRU-based garbage collection
    """

    def __init__(self, max_modules: int = 32, credit_tau: float = 0.02):
        """
        Initialize the module library.

        Args:
            max_modules: Maximum number of modules to store
            credit_tau: Credit threshold for keeping module versions
        """
        self.max_modules = max_modules
        self.credit_tau = credit_tau
        self.modules: Dict[str, Module] = {}  # name@version -> Module
        self.canonical_cache: Dict[str, str] = {}  # canonical -> name@version
        self.call_count = 0
        self.current_generation = 0

        logger.info(f"Initialized ModuleLibrary with capacity {max_modules}, tau={credit_tau}")

    def register_modules(self, candidates: List[SubtreeCandidate]) -> List[Module]:
        """
        Register modules from subtree candidates with deduplication and versioning.

        Args:
            candidates: List of subtree candidates to convert to modules

        Returns:
            List of registered modules
        """
        registered = []
        module_counter = 0

        for candidate in candidates:
            if candidate.mdl_score <= 0:
                continue  # Only register positive MDL modules

            # Check for near-duplicates using canonicalization
            canonical = canonicalize_subtree(candidate.subtree)

            if canonical in self.canonical_cache:
                # Update existing module if this one is better
                existing_key = self.canonical_cache[canonical]
                existing_module = self.modules[existing_key]

                if candidate.mdl_score > existing_module.mdl_score:
                    # Create new version
                    version_parts = existing_module.version.split('.')
                    new_version = f"{version_parts[0]}.{int(version_parts[1]) + 1}.0"

                    new_module = self._create_module_from_candidate(
                        candidate, existing_module.name, new_version
                    )

                    new_key = f"{new_module.name}@{new_module.version}"
                    self.modules[new_key] = new_module
                    self.canonical_cache[canonical] = new_key
                    registered.append(new_module)

                    logger.info(f"Updated module {new_module.name} to v{new_version}")
                continue

            # Create new module
            if len(self.modules) >= self.max_modules:
                self._garbage_collect()

            module_name = f"mod_{module_counter}"
            module_counter += 1

            module = self._create_module_from_candidate(candidate, module_name, "1.0.0")

            module_key = f"{module.name}@{module.version}"
            self.modules[module_key] = module
            self.canonical_cache[canonical] = module_key
            registered.append(module)

            logger.info(f"Registered new module {module.name}@{module.version}: "
                       f"arity={module.arity}, MDL={candidate.mdl_score:.3f}")

        return registered

    def _create_module_from_candidate(self, candidate: SubtreeCandidate,
                                    name: str, version: str) -> Module:
        """Create a Module from a SubtreeCandidate."""
        arity = self._calculate_arity(candidate.subtree)
        contract = self._infer_contract(candidate.subtree, arity)

        return Module(
            name=name,
            implementation=candidate.subtree,
            arity=arity,
            mdl_score=candidate.mdl_score,
            frequency=candidate.frequency,
            contract=contract,
            version=version,
            credit_score=candidate.mdl_score,  # Initialize with MDL score
            usage_count=0,
            last_used_gen=self.current_generation
        )

    def _garbage_collect(self):
        """Remove least valuable modules when at capacity."""
        if len(self.modules) < self.max_modules:
            return

        # Sort by credit score (ascending) and last used generation
        modules_by_value = sorted(
            self.modules.items(),
            key=lambda x: (x[1].credit_score, -x[1].last_used_gen)
        )

        # Remove lowest value modules
        to_remove = len(self.modules) - self.max_modules + 1

        for i in range(to_remove):
            key, module = modules_by_value[i]

            # Remove from canonical cache
            canonical = module.get_canonical_form()
            if canonical in self.canonical_cache and self.canonical_cache[canonical] == key:
                del self.canonical_cache[canonical]

            del self.modules[key]
            logger.info(f"GC removed module {module.name}@{module.version} "
                       f"(credit={module.credit_score:.3f})")

    def update_credit_scores(self, population: List[LispNode], fitness_scores: List[float],
                           mask_probability: float = 0.05):
        """
        Update module credit scores using stochastic masking.

        For each module, mask it out in 5% of evaluations and measure impact.
        This provides a Shapley-lite attribution of bits saved per module.

        Args:
            population: Current population of programs
            fitness_scores: Fitness scores for the population
            mask_probability: Probability of masking each module
        """
        if not self.modules or not population:
            return

        import random

        # For each module, estimate its contribution
        for _, module in self.modules.items():
            if random.random() > mask_probability:
                continue  # Skip this module this time

            # Count programs that use this module
            programs_using_module = []
            baseline_fitness = []

            for i, program in enumerate(population):
                if self._program_uses_module(program, module.name):
                    programs_using_module.append(program)
                    baseline_fitness.append(fitness_scores[i])

            if not programs_using_module:
                continue

            # Estimate fitness without this module (simplified)
            # In practice, you'd re-evaluate with module masked out
            avg_baseline = sum(baseline_fitness) / len(baseline_fitness)

            # Simple heuristic: modules used in high-fitness programs get more credit
            credit_delta = (avg_baseline - 0.5) * 0.1  # Scale factor

            # Update rolling credit score
            decay = 0.9
            module.credit_score = decay * module.credit_score + (1 - decay) * credit_delta
            module.last_used_gen = self.current_generation

            logger.debug(f"Updated credit for {module.name}: {module.credit_score:.3f}")

    def _program_uses_module(self, program: LispNode, module_name: str) -> bool:
        """Check if a program uses a specific module."""
        from protosynth.mutation import iter_nodes

        for _, _, node in iter_nodes(program):
            if node.node_type == 'call' and node.value == module_name:
                return True
        return False

    def advance_generation(self):
        """Advance the generation counter for credit tracking."""
        self.current_generation += 1

    def _calculate_arity(self, subtree: LispNode) -> int:
        """Calculate the arity (number of unique variables) in a subtree."""
        variables = set()

        def collect_vars(node: LispNode):
            if node.node_type == 'var':
                variables.add(node.value)
            elif hasattr(node, 'children') and node.children:
                for child in node.children:
                    if isinstance(child, LispNode):
                        collect_vars(child)

        collect_vars(subtree)
        return len(variables)

    def _infer_contract(self, subtree: LispNode, arity: int) -> ModuleContract:
        """
        Infer a contract from a subtree's structure.

        Args:
            subtree: The subtree to analyze
            arity: Number of arguments (variables)

        Returns:
            Inferred ModuleContract
        """
        # Analyze the subtree to infer return type and constraints
        return_type = self._infer_return_type(subtree)
        return_range = self._infer_return_range(subtree, return_type)

        # For now, accept any argument types
        arg_types = ['any'] * arity

        return ModuleContract(
            arity=arity,
            return_type=return_type,
            return_range=return_range,
            arg_types=arg_types
        )

    def _infer_return_type(self, node: LispNode) -> str:
        """Infer the return type of a subtree."""
        if node.node_type == 'const':
            if isinstance(node.value, bool):
                return 'bool'
            elif isinstance(node.value, int):
                return 'int'
            elif isinstance(node.value, float):
                return 'float'
            else:
                return 'any'
        elif node.node_type == 'op':
            op_name = node.value
            if op_name in ['>', '<', '>=', '<=', '==', '!=', 'and', 'or']:
                return 'bool'
            elif op_name in ['+', '-', '*', '/', '%']:
                return 'float'  # Conservative: assume float for arithmetic
            else:
                return 'any'
        elif node.node_type == 'if':
            # Return type is the union of then/else branches
            # For simplicity, just return 'any'
            return 'any'
        else:
            return 'any'

    def _infer_return_range(self, node: LispNode, return_type: str) -> Optional[Tuple[float, float]]:
        """Infer the return range for numeric types."""
        # For boolean types, range is 0-1 when converted to float
        if return_type == 'bool':
            return (0.0, 1.0)

        if return_type not in ['int', 'float']:
            return None

        # Simple heuristics for common patterns
        if node.node_type == 'op' and node.value in ['>', '<', '>=', '<=', '==', '!=']:
            # Comparison operations return 0 or 1 (as float)
            return (0.0, 1.0)
        elif node.node_type == 'const' and isinstance(node.value, (int, float)):
            # Constant value
            val = float(node.value)
            return (val, val)
        else:
            # Unknown range
            return None

    def create_module_call(self, module_name: str, args: List[LispNode],
                          version: Optional[str] = None) -> LispNode:
        """
        Create a module call node with contract validation.

        Args:
            module_name: Name of the module to call
            args: Arguments to pass to the module
            version: Specific version to call (defaults to latest)

        Returns:
            LispNode representing the module call

        Raises:
            ValueError: If module not found or contract validation fails
        """
        # Find the module (with version handling)
        module = self._find_module(module_name, version)
        if module is None:
            available = [k for k in self.modules.keys() if k.startswith(module_name)]
            raise ValueError(f"Module {module_name} not found. Available: {available}")

        # Validate against contract
        if module.contract:
            is_valid, errors = module.contract.validate_call(args)
            if not is_valid:
                raise ValueError(f"Contract violation for {module_name}: {'; '.join(errors)}")
        else:
            # Fallback to basic arity check
            if len(args) != module.arity:
                raise ValueError(f"Module {module_name} expects {module.arity} args, got {len(args)}")

        # Update usage tracking
        module.usage_count += 1
        module.last_used_gen = self.current_generation

        # Create a special 'call' node
        return LispNode('call', module_name, args)

    def _find_module(self, module_name: str, version: Optional[str] = None) -> Optional[Module]:
        """Find a module by name and optional version."""
        if version:
            # Look for specific version
            key = f"{module_name}@{version}"
            return self.modules.get(key)
        else:
            # Find latest version - try both versioned and unversioned keys
            # First try exact match (unversioned)
            if module_name in self.modules:
                return self.modules[module_name]

            # Then try versioned matches
            matching_modules = [
                (key, module) for key, module in self.modules.items()
                if key.startswith(f"{module_name}@") or key == module_name
            ]

            if not matching_modules:
                return None

            # Sort by version (simple string sort works for semantic versioning)
            matching_modules.sort(key=lambda x: x[0], reverse=True)
            return matching_modules[0][1]

    def expand_module_call(self, call_node: LispNode) -> LispNode:
        """
        Expand a module call by substituting arguments into the implementation.

        Args:
            call_node: Module call node to expand

        Returns:
            Expanded AST with arguments substituted
        """
        if call_node.node_type != 'call':
            raise ValueError("Not a module call node")

        module_name = call_node.value
        args = call_node.children

        # Find the module (with version handling)
        module = self._find_module(module_name)
        if module is None:
            available = [k for k in self.modules.keys() if k.startswith(module_name)]
            raise ValueError(f"Module {module_name} not found. Available: {available}")

        # Create variable substitution mapping
        var_mapping = {}
        var_names = self._get_variable_names(module.implementation)

        for i, var_name in enumerate(sorted(var_names)):
            if i < len(args):
                var_mapping[var_name] = args[i]

        # Substitute variables in the implementation
        return self._substitute_variables(module.implementation, var_mapping)

    def _get_variable_names(self, subtree: LispNode) -> Set[str]:
        """Get all variable names in a subtree."""
        variables = set()

        def collect_vars(node: LispNode):
            if node.node_type == 'var':
                variables.add(node.value)
            elif hasattr(node, 'children') and node.children:
                for child in node.children:
                    if isinstance(child, LispNode):
                        collect_vars(child)

        collect_vars(subtree)
        return variables

    def _substitute_variables(self, node: LispNode, var_mapping: Dict[str, LispNode]) -> LispNode:
        """Substitute variables in an AST with given mappings."""
        if node.node_type == 'var' and node.value in var_mapping:
            return var_mapping[node.value]
        elif hasattr(node, 'children') and node.children:
            new_children = []
            for child in node.children:
                if isinstance(child, LispNode):
                    new_children.append(self._substitute_variables(child, var_mapping))
                else:
                    new_children.append(child)
            return LispNode(node.node_type, node.value, new_children)
        else:
            return LispNode(node.node_type, node.value, node.children[:] if node.children else [])

    def get_module_info(self) -> Dict:
        """Get information about the module library."""
        return {
            'num_modules': len(self.modules),
            'max_modules': self.max_modules,
            'modules': {name: {
                'arity': mod.arity,
                'mdl_score': mod.mdl_score,
                'frequency': mod.frequency
            } for name, mod in self.modules.items()}
        }


def demo_moduleization():
    """Demonstrate moduleization and reuse functionality."""
    from .core import const, var, op, let, if_expr

    print("\n🔧 Moduleization & Reuse Demo")
    print("=" * 35)

    # Create population with repeated patterns
    population = [
        op('+', var('x'), const(1)),  # Pattern 1: increment
        op('+', var('y'), const(1)),
        op('+', var('z'), const(1)),
        op('>', var('a'), const(0)),  # Pattern 2: positive check
        op('>', var('b'), const(0)),
        op('>', var('c'), const(0)),
    ]

    # Mine subtrees
    miner = SubtreeMiner(beta=0.005, min_frequency=2)
    validation_bits = [0, 1] * 50
    candidates = miner.mine_and_select(population, validation_bits, n_modules=5)

    print(f"Found {len(candidates)} module candidates")

    # Create module library
    library = ModuleLibrary(max_modules=10)
    modules = library.register_modules(candidates)

    print(f"Registered {len(modules)} modules:")
    for module in modules:
        print(f"  {module.name}: arity={module.arity}, MDL={module.mdl_score:.3f}")

    # Demonstrate module calls
    if modules:
        module = modules[0]
        print(f"\nDemonstrating calls to {module.name}:")

        # Create some calls
        try:
            call1 = library.create_module_call(module.name, [var('input')])
            print(f"  Call: {call1}")

            expanded = library.expand_module_call(call1)
            print(f"  Expanded: {expanded}")

        except Exception as e:
            print(f"  Error: {e}")

    return library


if __name__ == "__main__":
    demo_subtree_mining()
    demo_moduleization()
