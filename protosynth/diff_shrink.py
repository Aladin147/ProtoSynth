"""
ProtoSynth Diff & Shrink Tools

Implements:
- AST diff viewer for comparing programs
- Delta-debugging shrinker for minimizing programs while preserving fitness
"""

import copy
from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

from .core import LispNode, LispInterpreter
from .mutation import iter_nodes
from .eval import evaluate_program_on_window


class DiffType(Enum):
    """Types of differences between ASTs."""
    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"
    MOVED = "moved"


@dataclass
class ASTDiff:
    """Represents a difference between two ASTs."""
    diff_type: DiffType
    path: List[int]  # Path to the node in the tree
    old_node: Optional[LispNode]
    new_node: Optional[LispNode]
    description: str


class ASTDiffer:
    """
    AST diff viewer for comparing programs.
    
    Provides detailed comparison of AST structures with visual output.
    """
    
    def __init__(self):
        """Initialize the AST differ."""
        pass
    
    def diff(self, ast1: LispNode, ast2: LispNode) -> List[ASTDiff]:
        """
        Compare two ASTs and return list of differences.
        
        Args:
            ast1: First AST (old)
            ast2: Second AST (new)
            
        Returns:
            List of differences
        """
        diffs = []
        
        # Simple structural comparison
        nodes1 = list(self._enumerate_nodes(ast1))
        nodes2 = list(self._enumerate_nodes(ast2))
        
        # Create mappings for comparison (use tuple for hashable paths)
        map1 = {tuple(path): node for path, node in nodes1}
        map2 = {tuple(path): node for path, node in nodes2}

        all_paths = set(map1.keys()) | set(map2.keys())
        
        for path in sorted(all_paths):
            node1 = map1.get(path)
            node2 = map2.get(path)
            
            if node1 is None:
                # Node added
                diffs.append(ASTDiff(
                    diff_type=DiffType.ADDED,
                    path=list(path),
                    old_node=None,
                    new_node=node2,
                    description=f"Added {self._node_summary(node2)} at {self._path_str(path)}"
                ))
            elif node2 is None:
                # Node removed
                diffs.append(ASTDiff(
                    diff_type=DiffType.REMOVED,
                    path=list(path),
                    old_node=node1,
                    new_node=None,
                    description=f"Removed {self._node_summary(node1)} at {self._path_str(path)}"
                ))
            elif not self._nodes_equal(node1, node2):
                # Node changed
                diffs.append(ASTDiff(
                    diff_type=DiffType.CHANGED,
                    path=list(path),
                    old_node=node1,
                    new_node=node2,
                    description=f"Changed {self._node_summary(node1)} → {self._node_summary(node2)} at {self._path_str(path)}"
                ))
        
        return diffs
    
    def _enumerate_nodes(self, ast: LispNode, path: List[int] = None) -> List[Tuple[List[int], LispNode]]:
        """Enumerate all nodes with their paths."""
        if path is None:
            path = []
        
        nodes = [(path.copy(), ast)]
        
        if hasattr(ast, 'children') and ast.children:
            for i, child in enumerate(ast.children):
                if isinstance(child, LispNode):
                    child_path = path + [i]
                    nodes.extend(self._enumerate_nodes(child, child_path))
        
        return nodes
    
    def _nodes_equal(self, node1: LispNode, node2: LispNode) -> bool:
        """Check if two nodes are equal."""
        return (node1.node_type == node2.node_type and 
                node1.value == node2.value)
    
    def _node_summary(self, node: LispNode) -> str:
        """Get a summary string for a node."""
        if node.node_type == 'const':
            return f"const({node.value})"
        elif node.node_type == 'var':
            return f"var({node.value})"
        elif node.node_type == 'op':
            return f"op({node.value})"
        else:
            return f"{node.node_type}({node.value})"
    
    def _path_str(self, path: List[int]) -> str:
        """Convert path to string."""
        if not path:
            return "root"
        return "→".join(map(str, path))
    
    def format_diff(self, diffs: List[ASTDiff]) -> str:
        """Format differences for display."""
        if not diffs:
            return "No differences found."
        
        lines = ["AST Differences:", "=" * 40]
        
        for diff in diffs:
            symbol = {
                DiffType.ADDED: "+ ",
                DiffType.REMOVED: "- ",
                DiffType.CHANGED: "~ ",
                DiffType.MOVED: "↔ "
            }[diff.diff_type]
            
            lines.append(f"{symbol}{diff.description}")
        
        return "\n".join(lines)


class DeltaDebugger:
    """
    Delta-debugging shrinker for minimizing programs while preserving fitness.
    
    Uses binary search to find minimal program that maintains target fitness.
    """
    
    def __init__(self, interpreter: LispInterpreter, fitness_threshold: float = 0.01):
        """
        Initialize delta debugger.
        
        Args:
            interpreter: Interpreter for evaluation
            fitness_threshold: Minimum fitness difference to preserve
        """
        self.interpreter = interpreter
        self.fitness_threshold = fitness_threshold
    
    def shrink(self, program: LispNode, test_data: List[int], 
               target_fitness: float, max_iterations: int = 50) -> Tuple[LispNode, Dict[str, Any]]:
        """
        Shrink a program while preserving fitness.
        
        Args:
            program: Program to shrink
            test_data: Test data for evaluation
            target_fitness: Target fitness to maintain
            max_iterations: Maximum shrinking iterations
            
        Returns:
            Tuple of (shrunk_program, shrinking_stats)
        """
        current_program = copy.deepcopy(program)
        original_size = len(list(iter_nodes(program)))
        
        stats = {
            'original_size': original_size,
            'iterations': 0,
            'reductions': 0,
            'final_size': original_size,
            'size_reduction': 0.0
        }
        
        for iteration in range(max_iterations):
            stats['iterations'] = iteration + 1
            
            # Try to shrink the program
            shrunk = self._attempt_shrink(current_program, test_data, target_fitness)
            
            if shrunk is None:
                # No more shrinking possible
                break
            
            current_program = shrunk
            stats['reductions'] += 1
        
        final_size = len(list(iter_nodes(current_program)))
        stats['final_size'] = final_size
        stats['size_reduction'] = (original_size - final_size) / original_size if original_size > 0 else 0.0
        
        return current_program, stats
    
    def _attempt_shrink(self, program: LispNode, test_data: List[int], 
                       target_fitness: float) -> Optional[LispNode]:
        """Attempt to shrink the program by removing nodes."""
        # Get all removable nodes (not the root)
        removable_nodes = []
        
        for parent, child_idx, node in iter_nodes(program):
            if parent is not None:  # Not root
                removable_nodes.append((parent, child_idx, node))
        
        # Try removing each node
        for parent, child_idx, node in removable_nodes:
            # Create candidate by removing this node
            candidate = self._create_removal_candidate(program, parent, child_idx)
            
            if candidate is None:
                continue
            
            # Test fitness
            try:
                fitness, _ = evaluate_program_on_window(
                    self.interpreter, candidate, test_data, k=2
                )
                
                # If fitness is preserved, return the shrunk version
                if abs(fitness - target_fitness) <= self.fitness_threshold:
                    return candidate
                    
            except Exception:
                # Evaluation failed, skip this candidate
                continue
        
        # Try simplifying operations
        for parent, child_idx, node in iter_nodes(program):
            if node.node_type == 'op' and hasattr(node, 'children') and len(node.children) > 1:
                # Try replacing with first child
                candidate = self._create_simplification_candidate(program, parent, child_idx, 0)
                
                if candidate is not None:
                    try:
                        fitness, _ = evaluate_program_on_window(
                            self.interpreter, candidate, test_data, k=2
                        )
                        
                        if abs(fitness - target_fitness) <= self.fitness_threshold:
                            return candidate
                            
                    except Exception:
                        continue
        
        return None  # No shrinking possible
    
    def _create_removal_candidate(self, program: LispNode, parent: LispNode, 
                                child_idx: int) -> Optional[LispNode]:
        """Create a candidate by removing a node."""
        candidate = copy.deepcopy(program)
        
        # Find the corresponding parent in the copy
        parent_copy = self._find_node_in_copy(candidate, parent, program)
        
        if parent_copy is None or not hasattr(parent_copy, 'children'):
            return None
        
        if child_idx >= len(parent_copy.children):
            return None
        
        # Remove the child
        new_children = parent_copy.children[:child_idx] + parent_copy.children[child_idx+1:]
        
        # Handle special cases
        if parent_copy.node_type == 'op' and len(new_children) == 0:
            # Can't have operation with no children
            return None
        elif parent_copy.node_type == 'if' and len(new_children) < 3:
            # Can't have if with less than 3 children
            return None
        elif parent_copy.node_type == 'let' and len(new_children) < 3:
            # Can't have let with less than 3 children
            return None
        
        parent_copy.children = new_children
        return candidate
    
    def _create_simplification_candidate(self, program: LispNode, parent: LispNode,
                                       child_idx: int, replacement_child_idx: int) -> Optional[LispNode]:
        """Create a candidate by replacing a node with one of its children."""
        candidate = copy.deepcopy(program)
        
        # Find the corresponding nodes in the copy
        parent_copy = self._find_node_in_copy(candidate, parent, program)
        
        if parent_copy is None or not hasattr(parent_copy, 'children'):
            return None
        
        if child_idx >= len(parent_copy.children):
            return None
        
        node_copy = parent_copy.children[child_idx]
        
        if not hasattr(node_copy, 'children') or replacement_child_idx >= len(node_copy.children):
            return None
        
        # Replace the node with its child
        replacement = node_copy.children[replacement_child_idx]
        parent_copy.children[child_idx] = replacement
        
        return candidate
    
    def _find_node_in_copy(self, copy_root: LispNode, target_node: LispNode, 
                          original_root: LispNode) -> Optional[LispNode]:
        """Find corresponding node in a copied AST."""
        # Simple approach: use structural matching
        # This is a simplified implementation
        
        def find_recursive(copy_node: LispNode, orig_node: LispNode) -> Optional[LispNode]:
            if orig_node is target_node:
                return copy_node
            
            if (hasattr(orig_node, 'children') and hasattr(copy_node, 'children') and
                orig_node.children and copy_node.children):
                
                for i, (orig_child, copy_child) in enumerate(zip(orig_node.children, copy_node.children)):
                    if isinstance(orig_child, LispNode) and isinstance(copy_child, LispNode):
                        result = find_recursive(copy_child, orig_child)
                        if result is not None:
                            return result
            
            return None
        
        return find_recursive(copy_root, original_root)


def demo_diff_shrink():
    """Demonstrate diff and shrink functionality."""
    print("🔍 AST Diff & Shrink Demo")
    print("=" * 30)
    
    # Create two similar programs
    from .core import op, var, const, if_expr
    
    prog1 = if_expr(
        op('>', var('x'), const(0)),
        op('+', var('x'), const(1)),
        const(0)
    )
    
    prog2 = if_expr(
        op('>', var('x'), const(5)),  # Changed constant
        op('*', var('x'), const(2)),  # Changed operation
        const(-1)  # Changed constant
    )
    
    # Test diff
    differ = ASTDiffer()
    diffs = differ.diff(prog1, prog2)
    
    print("Diff between programs:")
    print(differ.format_diff(diffs))
    
    # Test shrinking
    print("\nShrinking demo:")
    
    # Create a more complex program
    complex_prog = if_expr(
        op('and',
           op('>', var('x'), const(0)),
           op('<', var('x'), const(10))),
        op('+',
           op('*', var('x'), const(2)),
           const(1)),
        op('-', const(0), const(1))
    )
    
    print(f"Original program: {complex_prog}")
    print(f"Original size: {len(list(iter_nodes(complex_prog)))} nodes")
    
    # Shrink it
    interpreter = LispInterpreter()
    debugger = DeltaDebugger(interpreter, fitness_threshold=0.1)
    
    test_data = [1, 0, 1, 1, 0] * 20
    
    try:
        # Get original fitness
        original_fitness, _ = evaluate_program_on_window(
            interpreter, complex_prog, test_data, k=2
        )
        
        print(f"Original fitness: {original_fitness:.3f}")
        
        # Shrink
        shrunk_prog, stats = debugger.shrink(
            complex_prog, test_data, original_fitness, max_iterations=20
        )
        
        print(f"Shrunk program: {shrunk_prog}")
        print(f"Shrinking stats: {stats}")
        print(f"Size reduction: {stats['size_reduction']:.1%}")
        
        # Verify fitness
        shrunk_fitness, _ = evaluate_program_on_window(
            interpreter, shrunk_prog, test_data, k=2
        )
        print(f"Shrunk fitness: {shrunk_fitness:.3f}")
        
    except Exception as e:
        print(f"Shrinking failed: {e}")
    
    return differ, debugger


if __name__ == "__main__":
    demo_diff_shrink()
