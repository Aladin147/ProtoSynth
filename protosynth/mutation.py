"""
ProtoSynth Mutation Engine

This module implements the mutation system for self-modifying ASTs,
including AST traversal, mutation operations, and mutation pipelines.
"""

from typing import Iterator, Tuple, Optional, Any, Callable, Dict
from enum import Enum
import random
import logging
from .core import LispNode, clone_ast

logger = logging.getLogger(__name__)


def iter_nodes(ast: LispNode) -> Iterator[Tuple[Optional[LispNode], Optional[int], LispNode]]:
    """
    Generator that yields (parent, child_idx, node) for every node in the AST.
    
    This performs a depth-first traversal of the AST, yielding each node along
    with its parent and the index within the parent's children list.
    
    Args:
        ast: The root AST node to traverse
        
    Yields:
        Tuple of (parent, child_idx, node) where:
        - parent: The parent node (None for root)
        - child_idx: Index in parent.children (None for root)  
        - node: The current node being visited
        
    Example:
        >>> ast = op('+', const(1), const(2))
        >>> list(iter_nodes(ast))
        [(None, None, op_node), (op_node, 0, const_1), (op_node, 1, const_2)]
    """
    def _traverse(node: LispNode, parent: Optional[LispNode] = None, child_idx: Optional[int] = None):
        """Internal recursive traversal function."""
        # Yield current node
        yield (parent, child_idx, node)
        
        # Traverse children if they exist
        if hasattr(node, 'children') and node.children:
            for idx, child in enumerate(node.children):
                # Recursively traverse each child
                yield from _traverse(child, node, idx)
    
    # Start traversal from root
    yield from _traverse(ast)


class MutationType(Enum):
    """Enumeration of available mutation types."""
    OP_SWAP = "op_swap"
    CONST_PERTURB = "const_perturb"
    VAR_RENAME = "var_rename"
    SUBTREE_INSERT = "subtree_insert"
    SUBTREE_DELETE = "subtree_delete"


# Mutation function implementations
def _op_swap_mutation(root: LispNode, rng: random.Random) -> LispNode:
    """
    Randomly pick an operation node and replace its operator with another
    from the allowed set, respecting arity constraints.

    Args:
        root: The AST to mutate
        rng: Random number generator

    Returns:
        A cloned AST with one operator potentially swapped

    Raises:
        ValueError: If no suitable operation nodes found for mutation
    """
    # Get all operation nodes
    op_nodes = [(parent, child_idx, node) for parent, child_idx, node in iter_nodes(root)
                if node.node_type == 'op']

    if not op_nodes:
        raise ValueError("No suitable operation nodes found for operator swap")

    # Clone the entire AST first
    mutated = clone_ast(root)

    # Find the corresponding node in the cloned AST
    cloned_op_nodes = [(parent, child_idx, node) for parent, child_idx, node in iter_nodes(mutated)
                       if node.node_type == 'op']

    # Pick a random operation node to mutate
    target_idx = rng.randint(0, len(cloned_op_nodes) - 1)
    parent, child_idx, target_node = cloned_op_nodes[target_idx]

    # Get available operators grouped by arity
    from .core import LispInterpreter
    interpreter = LispInterpreter()

    # Group operators by arity (number of arguments they expect)
    arity_groups = {
        1: ['not'],  # Unary operators
        2: ['+', '-', '*', '/', '%', '==', '!=', '<', '<=', '>', '>=', 'and', 'or']  # Binary operators
    }

    current_arity = len(target_node.children)

    if current_arity not in arity_groups:
        raise ValueError(f"No operators available for arity {current_arity}")

    available_ops = arity_groups[current_arity]

    # Remove current operator to ensure we actually change something
    available_ops = [op for op in available_ops if op != target_node.value]

    if not available_ops:
        # If no alternatives available, try a different node or fail gracefully
        # Remove this node from consideration and try again
        remaining_nodes = [(p, ci, n) for p, ci, n in cloned_op_nodes
                          if n is not target_node and len(n.children) in arity_groups
                          and len([op for op in arity_groups[len(n.children)] if op != n.value]) > 0]

        if not remaining_nodes:
            raise ValueError(f"No suitable operation nodes found for mutation")

        # Try again with a different node
        target_idx = rng.randint(0, len(remaining_nodes) - 1)
        parent, child_idx, target_node = remaining_nodes[target_idx]
        current_arity = len(target_node.children)
        available_ops = [op for op in arity_groups[current_arity] if op != target_node.value]

    # Pick a new operator
    new_operator = rng.choice(available_ops)
    target_node.value = new_operator

    return mutated


def _const_perturb_mutation(root: LispNode, rng: random.Random, delta: int = 10) -> LispNode:
    """
    Pick a numeric constant and add/subtract a random value within delta range.

    Args:
        root: The AST to mutate
        rng: Random number generator
        delta: Maximum perturbation amount (default: 10)

    Returns:
        A cloned AST with one numeric constant perturbed

    Raises:
        ValueError: If no suitable numeric constants found for mutation
    """
    # Get all numeric constant nodes
    numeric_const_nodes = []
    for parent, child_idx, node in iter_nodes(root):
        if (node.node_type == 'const' and
            isinstance(node.value, (int, float)) and
            not isinstance(node.value, bool)):  # Exclude booleans
            numeric_const_nodes.append((parent, child_idx, node))

    if not numeric_const_nodes:
        raise ValueError("No suitable numeric constants found for perturbation")

    # Clone the entire AST first
    mutated = clone_ast(root)

    # Find the corresponding nodes in the cloned AST
    cloned_numeric_nodes = []
    for parent, child_idx, node in iter_nodes(mutated):
        if (node.node_type == 'const' and
            isinstance(node.value, (int, float)) and
            not isinstance(node.value, bool)):
            cloned_numeric_nodes.append((parent, child_idx, node))

    # Pick a random numeric constant to mutate
    target_idx = rng.randint(0, len(cloned_numeric_nodes) - 1)
    parent, child_idx, target_node = cloned_numeric_nodes[target_idx]

    # Generate perturbation
    perturbation = rng.randint(-delta, delta)

    # Apply perturbation while preserving type
    if isinstance(target_node.value, int):
        target_node.value = int(target_node.value + perturbation)
    else:  # float
        target_node.value = float(target_node.value + perturbation)

    return mutated


def _var_rename_mutation(root: LispNode, rng: random.Random) -> LispNode:
    """
    Choose a let binding, generate a fresh variable name, and update
    all in-scope variable references.

    Args:
        root: The AST to mutate
        rng: Random number generator

    Returns:
        A cloned AST with one variable renamed throughout its scope

    Raises:
        ValueError: If no suitable let bindings found for mutation
    """
    # Find all let bindings
    let_bindings = []
    for parent, child_idx, node in iter_nodes(root):
        if node.node_type == 'let' and len(node.children) >= 3:
            let_bindings.append((parent, child_idx, node))

    if not let_bindings:
        raise ValueError("No suitable let bindings found for variable rename")

    # Clone the entire AST first
    mutated = clone_ast(root)

    # Find corresponding let bindings in cloned AST
    cloned_let_bindings = []
    for parent, child_idx, node in iter_nodes(mutated):
        if node.node_type == 'let' and len(node.children) >= 3:
            cloned_let_bindings.append((parent, child_idx, node))

    # Pick a random let binding to rename
    target_idx = rng.randint(0, len(cloned_let_bindings) - 1)
    parent, child_idx, let_node = cloned_let_bindings[target_idx]

    # Get the variable name being bound
    if let_node.children[0].node_type != 'var':
        raise ValueError("Invalid let binding structure")

    old_name = let_node.children[0].value

    # Generate a fresh variable name
    existing_names = set()
    for _, _, node in iter_nodes(mutated):
        if node.node_type == 'var':
            existing_names.add(node.value)
        elif node.node_type == 'let' and len(node.children) >= 1 and node.children[0].node_type == 'var':
            existing_names.add(node.children[0].value)

    # Generate new name that doesn't conflict
    counter = 1
    new_name = f"v{counter}"
    while new_name in existing_names:
        counter += 1
        new_name = f"v{counter}"

    # Rename the variable in the let binding
    let_node.children[0].value = new_name

    # Rename all variable references in the let body (simplified scope handling)
    # This is a simplified implementation - full scope analysis would be more complex
    def rename_vars_in_subtree(node):
        if node.node_type == 'var' and node.value == old_name:
            node.value = new_name
        for child in node.children:
            rename_vars_in_subtree(child)

    # Rename in the let body (third child)
    if len(let_node.children) >= 3:
        rename_vars_in_subtree(let_node.children[2])

    return mutated


def _is_safe_replacement_position(parent: LispNode, child_idx: int) -> bool:
    """
    Check if a position is safe for arbitrary node replacement.

    Some positions have structural constraints:
    - First child of 'let' must be a variable (binding name)
    - Second child of 'let' should not reference the variable being bound
    """
    if parent.node_type == 'let':
        if child_idx == 0:  # Variable name position
            return False  # Must remain a variable
        elif child_idx == 1:  # Value expression position
            # The value expression should not reference the variable being bound
            # This is a complex check, so for now we'll be conservative
            return False  # Avoid replacing value expressions in let bindings

    return True


def _subtree_insert_mutation(root: LispNode, rng: random.Random) -> LispNode:
    """
    Clone a random subtree and splice it into another location.

    Args:
        root: The AST to mutate
        rng: Random number generator

    Returns:
        A cloned AST with a subtree inserted at a new location

    Raises:
        ValueError: If no suitable locations found for insertion
    """
    # Find all nodes that could be duplicated
    all_nodes = list(iter_nodes(root))
    if len(all_nodes) < 2:
        raise ValueError("AST too small for subtree insertion")

    # Clone the AST
    mutated = clone_ast(root)
    cloned_nodes = list(iter_nodes(mutated))

    # Pick a source subtree to clone (not the root)
    source_candidates = [(p, ci, n) for p, ci, n in cloned_nodes if p is not None]
    if not source_candidates:
        raise ValueError("No suitable source subtrees found")

    source_parent, source_idx, source_node = rng.choice(source_candidates)
    source_clone = clone_ast(source_node)

    # Find safe insertion points that don't violate structural constraints
    target_candidates = [(p, ci, n) for p, ci, n in cloned_nodes
                        if (p is not None and
                            n is not source_node and
                            _is_safe_replacement_position(p, ci))]

    if not target_candidates:
        raise ValueError("No suitable insertion points found")

    target_parent, target_idx, target_node = rng.choice(target_candidates)

    # Replace the target with our cloned source
    target_parent.children[target_idx] = source_clone

    return mutated


def _subtree_delete_mutation(root: LispNode, rng: random.Random) -> LispNode:
    """
    Remove a child node and replace it with a default constant.

    Args:
        root: The AST to mutate
        rng: Random number generator

    Returns:
        A cloned AST with a subtree deleted and replaced

    Raises:
        ValueError: If no suitable nodes found for deletion
    """
    # Find all non-root nodes that can be safely deleted (respecting structural constraints)
    all_nodes = [(p, ci, n) for p, ci, n in iter_nodes(root)
                 if p is not None and _is_safe_replacement_position(p, ci)]

    if not all_nodes:
        raise ValueError("No suitable nodes found for deletion")

    # Clone the AST
    mutated = clone_ast(root)
    cloned_nodes = [(p, ci, n) for p, ci, n in iter_nodes(mutated)
                    if p is not None and _is_safe_replacement_position(p, ci)]

    # Pick a node to delete
    target_parent, target_idx, target_node = rng.choice(cloned_nodes)

    # Replace with a simple default constant
    default_values = [0, 1, True, False]
    default_value = rng.choice(default_values)

    from .core import LispNode
    replacement = LispNode('const', default_value)
    target_parent.children[target_idx] = replacement

    return mutated


# Mutation registry mapping types to functions
MUTATION_REGISTRY: Dict[MutationType, Callable[[LispNode, random.Random], LispNode]] = {
    MutationType.OP_SWAP: _op_swap_mutation,
    MutationType.CONST_PERTURB: _const_perturb_mutation,
    MutationType.VAR_RENAME: _var_rename_mutation,
    MutationType.SUBTREE_INSERT: _subtree_insert_mutation,
    MutationType.SUBTREE_DELETE: _subtree_delete_mutation,
}


def _is_mutation_applicable(ast: LispNode, mutation_type: MutationType) -> bool:
    """
    Quick check if a mutation type is applicable to an AST without actually running it.

    This is more efficient than running the mutation to test applicability.
    """
    if mutation_type == MutationType.OP_SWAP:
        # Need at least one operation node
        return any(node.node_type == 'op' for _, _, node in iter_nodes(ast))

    elif mutation_type == MutationType.CONST_PERTURB:
        # Need at least one numeric constant
        return any(node.node_type == 'const' and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
                  for _, _, node in iter_nodes(ast))

    elif mutation_type == MutationType.VAR_RENAME:
        # Need at least one let binding
        return any(node.node_type == 'let' for _, _, node in iter_nodes(ast))

    elif mutation_type in [MutationType.SUBTREE_INSERT, MutationType.SUBTREE_DELETE]:
        # Need at least 2 nodes and some safe replacement positions
        nodes = list(iter_nodes(ast))
        if len(nodes) < 2:
            return False
        return any(_is_safe_replacement_position(parent, child_idx)
                  for parent, child_idx, node in nodes if parent is not None)

    return True  # Default to applicable


def mutate(ast: LispNode, mutation_rate: float = 0.15, rng: Optional[random.Random] = None,
           max_attempts: int = 3) -> LispNode:
    """
    Apply mutations to an AST with given probability.

    This function attempts each type of mutation with the given probability.
    If multiple mutations are applicable, one is chosen randomly.

    Args:
        ast: The AST to mutate
        mutation_rate: Probability of applying each mutation type (default: 0.15)
        rng: Random number generator (creates new one if None)
        max_attempts: Maximum attempts if mutations fail (default: 3)

    Returns:
        A mutated clone of the AST, or the original AST if no mutations applied

    Raises:
        ValueError: If the AST is invalid
    """
    if rng is None:
        rng = random.Random()

    # Validate input AST
    if not isinstance(ast, LispNode):
        raise ValueError("Input must be a LispNode")

    # Collect potentially applicable mutations
    candidate_mutations = []

    for mutation_type, mutation_func in MUTATION_REGISTRY.items():
        if rng.random() < mutation_rate and _is_mutation_applicable(ast, mutation_type):
            candidate_mutations.append((mutation_type, mutation_func))

    if not candidate_mutations:
        logger.debug("No mutations selected or applicable")
        return clone_ast(ast)

    # Try mutations with retry logic
    for attempt in range(max_attempts):
        # Pick one mutation to apply
        mutation_type, mutation_func = rng.choice(candidate_mutations)

        logger.debug(f"Applying mutation: {mutation_type.value} (attempt {attempt + 1})")

        try:
            result = mutation_func(ast, rng)
            logger.debug(f"Mutation {mutation_type.value} succeeded")
            return result

        except (ValueError, RuntimeError, TypeError) as e:
            logger.debug(f"Mutation {mutation_type.value} failed: {e}")

            # Remove this mutation from candidates for next attempt
            candidate_mutations = [(mt, mf) for mt, mf in candidate_mutations if mt != mutation_type]

            if not candidate_mutations:
                logger.debug("No more candidate mutations available")
                break

        except Exception as e:
            logger.warning(f"Unexpected error in mutation {mutation_type.value}: {e}")
            # Remove this mutation and continue
            candidate_mutations = [(mt, mf) for mt, mf in candidate_mutations if mt != mutation_type]

            if not candidate_mutations:
                break

    logger.debug("All mutation attempts failed, returning original")
    return clone_ast(ast)
