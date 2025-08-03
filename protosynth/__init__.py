"""
ProtoSynth: A self-modifying, non-gradient-based AI architecture.

This package implements a Lisp-like interpreter that can modify its own code
through mutation and evaluate fitness using compression-driven metrics.
"""

from .core import LispNode, LispInterpreter, const, var, let, if_expr, op, pretty_print_ast, clone_ast
from .agent import SelfModifyingAgent
from .mutation import mutate, MutationType
from .verify import verify_ast

__version__ = "0.1.0"
__all__ = [
    "LispNode", "LispInterpreter", "const", "var", "let", "if_expr", "op",
    "pretty_print_ast", "clone_ast", "SelfModifyingAgent", "mutate", "MutationType", "verify_ast"
]
