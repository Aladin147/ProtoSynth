"""
ProtoSynth Verification System

This module implements syntactic and semantic verification for ASTs,
ensuring they are safe to execute and structurally valid.
"""

from typing import List, Tuple, Set, Dict, Any
from .core import LispNode, LispInterpreter
from .mutation import iter_nodes


def check_arity(ast: LispNode) -> Tuple[bool, List[str]]:
    """
    Check that all operations have the correct number of arguments and valid node types.

    Args:
        ast: The AST to check

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Define expected arities for operations
    expected_arities = {
        # Unary operations
        'not': 1,

        # Binary operations
        '+': 2, '-': 2, '*': 2, '/': 2, '%': 2,
        '==': 2, '!=': 2, '<': 2, '<=': 2, '>': 2, '>=': 2,
        'and': 2, 'or': 2,
    }

    # Valid node types
    valid_node_types = {'const', 'var', 'let', 'if', 'op'}

    for parent, child_idx, node in iter_nodes(ast):
        # Check for valid node types
        if node.node_type not in valid_node_types:
            errors.append(f"Unknown node type: {node.node_type}")
            continue

        if node.node_type == 'op':
            op_name = node.value
            actual_arity = len(node.children)

            if op_name not in expected_arities:
                errors.append(f"Unknown operation: {op_name}")
                continue

            expected_arity = expected_arities[op_name]
            if actual_arity != expected_arity:
                errors.append(f"Operation '{op_name}' expects {expected_arity} arguments, got {actual_arity}")

        elif node.node_type == 'if':
            # If expressions should have exactly 3 children
            if len(node.children) != 3:
                errors.append(f"If expression expects 3 arguments (condition, then, else), got {len(node.children)}")

        elif node.node_type == 'let':
            # Let expressions should have exactly 3 children
            if len(node.children) != 3:
                errors.append(f"Let expression expects 3 arguments (name, value, body), got {len(node.children)}")
            elif len(node.children) > 0 and node.children[0].node_type != 'var':
                errors.append(f"Let expression first argument must be a variable, got {node.children[0].node_type}")

    return len(errors) == 0, errors


def check_free_vars(ast: LispNode) -> Tuple[bool, List[str]]:
    """
    Check that all variable references are bound by let expressions.
    
    Args:
        ast: The AST to check
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    def check_scope(node: LispNode, bound_vars: Set[str]) -> None:
        """Recursively check variable bindings in scope."""
        
        if node.node_type == 'var':
            if node.value not in bound_vars:
                errors.append(f"Unbound variable: {node.value}")
        
        elif node.node_type == 'let':
            if len(node.children) >= 3:
                # Get the variable being bound
                var_node = node.children[0]
                if var_node.node_type == 'var':
                    var_name = var_node.value
                    
                    # Check the value expression with current scope
                    check_scope(node.children[1], bound_vars)
                    
                    # Check the body with extended scope
                    extended_scope = bound_vars | {var_name}
                    check_scope(node.children[2], extended_scope)
                    return  # Don't process children again
        
        # Process all children with current scope
        for child in node.children:
            check_scope(child, bound_vars)
    
    check_scope(ast, set())
    return len(errors) == 0, errors


def check_depth(ast: LispNode, max_depth: int = 10) -> Tuple[bool, List[str]]:
    """
    Check that the AST depth doesn't exceed the maximum allowed.
    
    Args:
        ast: The AST to check
        max_depth: Maximum allowed depth
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    def get_depth(node: LispNode) -> int:
        """Calculate the maximum depth of the AST."""
        if not node.children:
            return 1
        return 1 + max(get_depth(child) for child in node.children)
    
    actual_depth = get_depth(ast)
    
    if actual_depth > max_depth:
        errors.append(f"AST depth {actual_depth} exceeds maximum allowed depth {max_depth}")
    
    return len(errors) == 0, errors


def check_node_count(ast: LispNode, max_nodes: int = 100) -> Tuple[bool, List[str]]:
    """
    Check that the AST doesn't have too many nodes (resource hint).
    
    Args:
        ast: The AST to check
        max_nodes: Maximum allowed number of nodes
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    node_count = len(list(iter_nodes(ast)))
    
    if node_count > max_nodes:
        errors.append(f"AST has {node_count} nodes, exceeds maximum allowed {max_nodes}")
    
    return len(errors) == 0, errors


def verify_ast(ast: LispNode, max_depth: int = 10, max_nodes: int = 100) -> Tuple[bool, List[str]]:
    """
    Perform comprehensive verification of an AST.
    
    This function runs all verification checks and returns the combined results.
    
    Args:
        ast: The AST to verify
        max_depth: Maximum allowed depth
        max_nodes: Maximum allowed number of nodes
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    all_errors = []
    
    # Check basic structure
    if not isinstance(ast, LispNode):
        return False, ["AST root must be a LispNode"]
    
    # Run all verification checks
    checks = [
        check_arity(ast),
        check_free_vars(ast),
        check_depth(ast, max_depth),
        check_node_count(ast, max_nodes),
    ]
    
    for is_valid, errors in checks:
        if not is_valid:
            all_errors.extend(errors)
    
    return len(all_errors) == 0, all_errors
