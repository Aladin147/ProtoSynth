"""
ProtoSynth Core Interpreter

This module implements the foundational Lisp-like interpreter for ProtoSynth,
a self-modifying AI architecture that uses compression-driven evaluation.
"""

from typing import Any, Dict, List, Union, Optional
import time


class LispNode:
    """
    Base class for all AST nodes in the ProtoSynth Lisp-like language.
    
    Each node represents either:
    - A constant value (numbers, booleans, strings)
    - A variable reference
    - An operation with arguments
    - A control structure (if, let)
    """
    
    def __init__(self, node_type: str, value: Any = None, children: Optional[List['LispNode']] = None):
        self.node_type = node_type
        self.value = value
        self.children = children or []
    
    def __repr__(self):
        if self.children:
            children_repr = ', '.join(repr(child) for child in self.children)
            return f"LispNode({self.node_type}, {self.value}, [{children_repr}])"
        else:
            return f"LispNode({self.node_type}, {self.value})"


class LispInterpreter:
    """
    Resource-constrained interpreter for the ProtoSynth Lisp-like language.
    
    Features:
    - Depth-limited recursive evaluation
    - Operation count tracking
    - Timeout enforcement
    - Support for basic arithmetic, boolean operations, and control flow
    """
    
    def __init__(self, max_recursion_depth: int = 10, max_steps: int = 100, timeout_seconds: float = 1.0):
        self.max_recursion_depth = max_recursion_depth
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds
        self.step_count = 0
        self.start_time = None
        
        # Built-in operations
        self.operations = {
            # Arithmetic operations
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: a / b if b != 0 else float('inf'),
            '%': lambda a, b: a % b if b != 0 else 0,
            
            # Comparison operations
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
            '<': lambda a, b: a < b,
            '<=': lambda a, b: a <= b,
            '>': lambda a, b: a > b,
            '>=': lambda a, b: a >= b,
            
            # Boolean operations
            'and': lambda a, b: a and b,
            'or': lambda a, b: a or b,
            'not': lambda a: not a,
        }
    
    def evaluate(self, node: LispNode, environment: Optional[Dict[str, Any]] = None, depth: int = 0) -> Any:
        """
        Evaluate a LispNode in the given environment.
        
        Args:
            node: The AST node to evaluate
            environment: Variable bindings (defaults to empty dict)
            depth: Current recursion depth
            
        Returns:
            The evaluated result
            
        Raises:
            RuntimeError: If resource limits are exceeded
        """
        if environment is None:
            environment = {}
            self.step_count = 0
            self.start_time = time.time()
        
        # Check resource limits
        self._check_limits(depth)
        
        self.step_count += 1
        
        # Handle different node types
        if node.node_type == 'const':
            return node.value
        
        elif node.node_type == 'var':
            if node.value not in environment:
                raise NameError(f"Undefined variable: {node.value}")
            return environment[node.value]
        
        elif node.node_type == 'let':
            # let name val body
            if len(node.children) != 3:
                raise ValueError("let requires exactly 3 arguments: name, value, body")
            
            name_node, val_node, body_node = node.children
            if name_node.node_type != 'var':
                raise ValueError("let name must be a variable")
            
            name = name_node.value
            val = self.evaluate(val_node, environment, depth + 1)
            
            # Create new environment with the binding
            new_env = environment.copy()
            new_env[name] = val
            
            return self.evaluate(body_node, new_env, depth + 1)
        
        elif node.node_type == 'if':
            # if cond then else
            if len(node.children) != 3:
                raise ValueError("if requires exactly 3 arguments: condition, then, else")
            
            cond_node, then_node, else_node = node.children
            condition = self.evaluate(cond_node, environment, depth + 1)
            
            if condition:
                return self.evaluate(then_node, environment, depth + 1)
            else:
                return self.evaluate(else_node, environment, depth + 1)
        
        elif node.node_type == 'op':
            # op name args
            op_name = node.value
            if op_name not in self.operations:
                raise ValueError(f"Unknown operation: {op_name}")
            
            # Evaluate all arguments
            args = [self.evaluate(child, environment, depth + 1) for child in node.children]
            
            # Apply the operation
            op_func = self.operations[op_name]
            try:
                return op_func(*args)
            except TypeError as e:
                raise ValueError(f"Invalid arguments for operation {op_name}: {e}")
        
        else:
            raise ValueError(f"Unknown node type: {node.node_type}")
    
    def _check_limits(self, depth: int):
        """Check if resource limits have been exceeded."""
        if depth > self.max_recursion_depth:
            raise RuntimeError(f"Maximum recursion depth exceeded: {depth} > {self.max_recursion_depth}")
        
        if self.step_count > self.max_steps:
            raise RuntimeError(f"Maximum step count exceeded: {self.step_count} > {self.max_steps}")
        
        if self.start_time and time.time() - self.start_time > self.timeout_seconds:
            raise RuntimeError(f"Timeout exceeded: {time.time() - self.start_time:.2f}s > {self.timeout_seconds}s")
    
    def get_self_ast(self) -> LispNode:
        """
        Return an AST representation of this interpreter's configuration.

        This enables self-inspection by providing access to the interpreter's
        current settings and capabilities as an AST structure.

        Returns:
            An AST representing the interpreter's configuration
        """
        # Create an AST that represents the interpreter's configuration
        config_ast = LispNode('op', 'interpreter-config', [
            const(self.max_recursion_depth),
            const(self.max_steps),
            const(self.timeout_seconds),
            const(len(self.operations))  # Number of available operations
        ])

        return config_ast


# Helper functions for creating AST nodes
def const(value: Any) -> LispNode:
    """Create a constant node."""
    return LispNode('const', value)

def var(name: str) -> LispNode:
    """Create a variable reference node."""
    return LispNode('var', name)

def let(name: str, value: LispNode, body: LispNode) -> LispNode:
    """Create a let binding node."""
    return LispNode('let', None, [var(name), value, body])

def if_expr(condition: LispNode, then_expr: LispNode, else_expr: LispNode) -> LispNode:
    """Create an if expression node."""
    return LispNode('if', None, [condition, then_expr, else_expr])

def op(operation: str, *args: LispNode) -> LispNode:
    """Create an operation node."""
    return LispNode('op', operation, list(args))


# Serialization functions
def pretty_print_ast(node: LispNode) -> str:
    """
    Render an AST node back into human-readable Lisp-style code.

    Args:
        node: The AST node to render

    Returns:
        A string representation in Lisp syntax
    """
    if node.node_type == 'const':
        # Handle different constant types
        if isinstance(node.value, str):
            return f'"{node.value}"'
        elif isinstance(node.value, bool):
            return 'true' if node.value else 'false'
        else:
            return str(node.value)

    elif node.node_type == 'var':
        return str(node.value)

    elif node.node_type == 'let':
        if len(node.children) != 3:
            return f"(let INVALID-ARGS)"

        name_node, val_node, body_node = node.children
        name = pretty_print_ast(name_node)
        val = pretty_print_ast(val_node)
        body = pretty_print_ast(body_node)
        return f"(let {name} {val} {body})"

    elif node.node_type == 'if':
        if len(node.children) != 3:
            return f"(if INVALID-ARGS)"

        cond, then_expr, else_expr = node.children
        cond_str = pretty_print_ast(cond)
        then_str = pretty_print_ast(then_expr)
        else_str = pretty_print_ast(else_expr)
        return f"(if {cond_str} {then_str} {else_str})"

    elif node.node_type == 'op':
        op_name = node.value
        args_str = ' '.join(pretty_print_ast(child) for child in node.children)
        return f"({op_name} {args_str})" if args_str else f"({op_name})"

    else:
        return f"(UNKNOWN-{node.node_type} {node.value})"


def clone_ast(node: LispNode) -> LispNode:
    """
    Create a deep copy of an AST node.

    This is essential for safe mutation operations, ensuring that
    the original AST remains unchanged when mutations are applied.

    Args:
        node: The AST node to clone

    Returns:
        A deep copy of the node and all its children
    """
    # Clone the children recursively
    cloned_children = [clone_ast(child) for child in node.children]

    # Create a new node with the same type and value, but cloned children
    return LispNode(node.node_type, node.value, cloned_children)
