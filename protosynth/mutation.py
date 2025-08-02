"""
ProtoSynth Mutation Engine

This module implements the mutation system for self-modifying ASTs,
including AST traversal, mutation operations, and mutation pipelines.
"""

from typing import Iterator, Tuple, Optional, Any, Callable, Dict
from enum import Enum
import random
from .core import LispNode, clone_ast


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
    # For now, implement a simple version that duplicates a subtree
    # More sophisticated versions could insert into operations that accept variable arity

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

    # Find insertion points (operations that could accept an additional argument)
    # For simplicity, we'll replace an existing node rather than truly inserting
    target_candidates = [(p, ci, n) for p, ci, n in cloned_nodes
                        if p is not None and n is not source_node]

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
    # Find all non-root nodes that can be deleted
    all_nodes = [(p, ci, n) for p, ci, n in iter_nodes(root) if p is not None]

    if not all_nodes:
        raise ValueError("No suitable nodes found for deletion")

    # Clone the AST
    mutated = clone_ast(root)
    cloned_nodes = [(p, ci, n) for p, ci, n in iter_nodes(mutated) if p is not None]

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


def mutate(ast: LispNode, mutation_rate: float = 0.15, rng: Optional[random.Random] = None) -> LispNode:
    """
    Apply mutations to an AST with given probability.

    This function attempts each type of mutation with the given probability.
    If multiple mutations are applicable, one is chosen randomly.

    Args:
        ast: The AST to mutate
        mutation_rate: Probability of applying each mutation type (default: 0.15)
        rng: Random number generator (creates new one if None)

    Returns:
        A mutated clone of the AST, or the original AST if no mutations applied

    Raises:
        ValueError: If the AST is invalid or no mutations are possible
    """
    if rng is None:
        rng = random.Random()

    # Collect applicable mutations
    applicable_mutations = []

    for mutation_type, mutation_func in MUTATION_REGISTRY.items():
        if rng.random() < mutation_rate:
            try:
                # Test if this mutation is applicable
                mutation_func(ast, rng)
                applicable_mutations.append((mutation_type, mutation_func))
            except ValueError:
                # This mutation type is not applicable to this AST
                continue

    if not applicable_mutations:
        # No mutations applicable or selected, return clone of original
        return clone_ast(ast)

    # Pick one mutation to apply
    mutation_type, mutation_func = rng.choice(applicable_mutations)

    try:
        return mutation_func(ast, rng)
    except ValueError as e:
        # If the chosen mutation fails, return original
        return clone_ast(ast)
