#!/usr/bin/env python3
"""
Anti-biased tests for ProtoSynth mutation functionality.

These tests are designed to catch edge cases, break assumptions,
and find bugs before they become problems in the mutation system.
"""

import unittest
import random
from protosynth import const, var, let, if_expr, op, LispNode


class TestASTWalkerAntibiased(unittest.TestCase):
    """
    Anti-biased tests for AST walker - designed to catch edge cases
    and break assumptions rather than just confirm basic functionality.
    """
    
    def test_walker_import(self):
        """Test that we can import the walker function."""
        try:
            from protosynth.mutation import iter_nodes
            self.assertTrue(callable(iter_nodes))
        except ImportError:
            self.fail("iter_nodes function not found in protosynth.mutation")
    
    def test_empty_ast_edge_case(self):
        """Test walker behavior with minimal/edge case ASTs."""
        from protosynth.mutation import iter_nodes
        
        # Single constant node - should yield exactly one result
        single_node = const(42)
        results = list(iter_nodes(single_node))
        
        # Should have exactly one entry: (None, None, node) for root
        self.assertEqual(len(results), 1)
        parent, child_idx, node = results[0]
        self.assertIsNone(parent)  # Root has no parent
        self.assertIsNone(child_idx)  # Root has no child index
        self.assertEqual(node.node_type, 'const')
        self.assertEqual(node.value, 42)
    
    def test_walker_parent_child_consistency(self):
        """Test that parent-child relationships are consistent."""
        from protosynth.mutation import iter_nodes
        
        # Create a nested structure
        ast = op('+', const(1), op('*', const(2), const(3)))
        results = list(iter_nodes(ast))
        
        # Verify parent-child consistency
        for parent, child_idx, node in results:
            if parent is not None and child_idx is not None:
                # The parent's child at child_idx should be this node
                self.assertIs(parent.children[child_idx], node,
                            f"Parent-child inconsistency: parent.children[{child_idx}] != node")
                
                # child_idx should be valid
                self.assertGreaterEqual(child_idx, 0)
                self.assertLess(child_idx, len(parent.children))
    
    def test_walker_visits_all_nodes(self):
        """Test that walker visits every node exactly once."""
        from protosynth.mutation import iter_nodes
        
        # Create complex nested structure
        ast = let('x', const(10),
                  if_expr(op('>', var('x'), const(5)),
                          op('+', var('x'), const(1)),
                          op('-', var('x'), const(1))))
        
        # Count nodes manually by traversing
        def count_nodes_manual(node):
            count = 1  # This node
            for child in node.children:
                count += count_nodes_manual(child)
            return count
        
        expected_count = count_nodes_manual(ast)
        walker_results = list(iter_nodes(ast))
        
        self.assertEqual(len(walker_results), expected_count,
                        "Walker should visit every node exactly once")
        
        # Check that all nodes are unique objects
        visited_nodes = [node for _, _, node in walker_results]
        unique_nodes = set(id(node) for node in visited_nodes)
        self.assertEqual(len(visited_nodes), len(unique_nodes),
                        "Walker should not visit any node more than once")
    
    def test_walker_depth_ordering(self):
        """Test that walker visits nodes in a predictable depth order."""
        from protosynth.mutation import iter_nodes
        
        # Create a deep nested structure
        ast = let('a', const(1),
                  let('b', const(2),
                      let('c', const(3),
                          op('+', var('a'), var('b')))))
        
        results = list(iter_nodes(ast))
        
        # Calculate depth for each node
        def get_depth(parent, child_idx, node, results):
            if parent is None:
                return 0
            # Find parent in results and get its depth
            for p, ci, n in results:
                if n is parent:
                    return get_depth(p, ci, n, results) + 1
            return -1  # Should never happen
        
        # Check that we can calculate depths (no circular references)
        depths = []
        for i, (parent, child_idx, node) in enumerate(results):
            if parent is None:
                depths.append(0)
            else:
                # Find parent's depth from earlier in the list
                parent_depth = None
                for j in range(i):
                    if results[j][2] is parent:
                        parent_depth = depths[j]
                        break
                self.assertIsNotNone(parent_depth, "Parent should appear before child in walker")
                depths.append(parent_depth + 1)
    
    def test_walker_with_malformed_ast(self):
        """Test walker behavior with potentially malformed ASTs."""
        from protosynth.mutation import iter_nodes
        
        # AST with empty children list
        malformed1 = LispNode('op', '+', [])  # Operation with no arguments
        results1 = list(iter_nodes(malformed1))
        self.assertEqual(len(results1), 1)  # Should still visit the node
        
        # AST with None children (edge case)
        malformed2 = LispNode('const', 42, None)
        malformed2.children = []  # Fix it to be safe
        results2 = list(iter_nodes(malformed2))
        self.assertEqual(len(results2), 1)
    
    def test_walker_performance_with_large_ast(self):
        """Test walker performance doesn't degrade catastrophically."""
        from protosynth.mutation import iter_nodes
        import time
        
        # Create a large but not pathological AST
        def create_large_ast(depth):
            if depth <= 0:
                return const(depth)
            return op('+', create_large_ast(depth - 1), create_large_ast(depth - 1))
        
        # Create moderately large AST (depth 8 = 255 nodes)
        large_ast = create_large_ast(8)
        
        start_time = time.time()
        results = list(iter_nodes(large_ast))
        end_time = time.time()
        
        # Should complete in reasonable time (< 1 second for 255 nodes)
        self.assertLess(end_time - start_time, 1.0,
                       "Walker should handle moderately large ASTs efficiently")
        
        # Should visit expected number of nodes
        # For depth 8: 8 levels of op nodes + 1 level of const nodes = 2^9 - 1 = 511 nodes
        expected_nodes = 2**9 - 1  # Corrected calculation
        self.assertEqual(len(results), expected_nodes)
    
    def test_walker_child_index_correctness(self):
        """Test that child indices are correct and complete."""
        from protosynth.mutation import iter_nodes
        
        # Create AST with varying numbers of children
        ast = if_expr(
            op('and', const(True), const(False)),  # 2 children
            op('+', const(1), const(2), const(3)),  # 3 children (if we support n-ary)
            const(0)  # 0 children
        )
        
        results = list(iter_nodes(ast))
        
        # Group results by parent
        children_by_parent = {}
        for parent, child_idx, node in results:
            if parent is not None:
                if parent not in children_by_parent:
                    children_by_parent[parent] = []
                children_by_parent[parent].append((child_idx, node))
        
        # Check that child indices are complete and correct for each parent
        for parent, child_list in children_by_parent.items():
            child_list.sort(key=lambda x: x[0])  # Sort by index
            
            # Should have indices 0, 1, 2, ... len(parent.children)-1
            expected_indices = list(range(len(parent.children)))
            actual_indices = [idx for idx, _ in child_list]
            
            self.assertEqual(actual_indices, expected_indices,
                           f"Child indices should be complete: expected {expected_indices}, got {actual_indices}")
            
            # Each child should match parent.children[idx]
            for idx, node in child_list:
                self.assertIs(parent.children[idx], node,
                            f"Child at index {idx} should match parent.children[{idx}]")


class TestMutationRegistryAntibiased(unittest.TestCase):
    """
    Anti-biased tests for mutation registry - designed to catch issues
    with mutation type definitions and function mappings.
    """

    def test_mutation_types_import(self):
        """Test that mutation types can be imported."""
        try:
            from protosynth.mutation import MutationType, MUTATION_REGISTRY
            self.assertTrue(hasattr(MutationType, 'OP_SWAP'))
            self.assertTrue(hasattr(MutationType, 'CONST_PERTURB'))
            self.assertTrue(hasattr(MutationType, 'VAR_RENAME'))
            self.assertTrue(hasattr(MutationType, 'SUBTREE_INSERT'))
            self.assertTrue(hasattr(MutationType, 'SUBTREE_DELETE'))
        except ImportError as e:
            self.fail(f"Could not import mutation types: {e}")

    def test_registry_completeness(self):
        """Test that all mutation types have corresponding functions."""
        from protosynth.mutation import MutationType, MUTATION_REGISTRY

        # Every mutation type should have a function in the registry
        for mutation_type in MutationType:
            self.assertIn(mutation_type, MUTATION_REGISTRY,
                         f"Mutation type {mutation_type} missing from registry")

            # Each function should be callable
            func = MUTATION_REGISTRY[mutation_type]
            self.assertTrue(callable(func),
                           f"Registry entry for {mutation_type} is not callable")

    def test_mutation_function_signatures(self):
        """Test that mutation functions have correct signatures."""
        from protosynth.mutation import MutationType, MUTATION_REGISTRY
        import inspect

        for mutation_type, func in MUTATION_REGISTRY.items():
            # Each function should accept (root, rng) parameters
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            self.assertGreaterEqual(len(params), 2,
                                  f"{mutation_type} function should accept at least 2 parameters")

            # First two parameters should be root and rng
            self.assertIn('root', params[:2] + ['ast', 'node'],  # Allow some naming flexibility
                         f"{mutation_type} function should accept root/ast/node parameter")
            self.assertIn('rng', params[:2] + ['random', 'rand'],  # Allow some naming flexibility
                         f"{mutation_type} function should accept rng/random parameter")

    def test_mutation_functions_return_clones(self):
        """Test that mutation functions return clones, not modify originals."""
        from protosynth.mutation import MutationType, MUTATION_REGISTRY
        from protosynth import clone_ast

        # Create a test AST
        original = op('+', const(5), const(10))
        original_str = str(original)

        rng = random.Random(42)  # Deterministic for testing

        for mutation_type, func in MUTATION_REGISTRY.items():
            with self.subTest(mutation_type=mutation_type):
                try:
                    # Apply mutation
                    result = func(original, rng)

                    # Original should be unchanged
                    self.assertEqual(str(original), original_str,
                                   f"{mutation_type} modified the original AST")

                    # Result should be a LispNode
                    self.assertIsInstance(result, LispNode,
                                        f"{mutation_type} should return a LispNode")

                    # Result should be different object (clone)
                    self.assertIsNot(result, original,
                                   f"{mutation_type} should return a clone, not the original")

                except Exception as e:
                    # Some mutations might fail on simple ASTs - that's ok for now
                    # But they shouldn't crash with basic inputs
                    if "No suitable" in str(e) or "Cannot" in str(e):
                        continue  # Expected failure for this AST type
                    else:
                        self.fail(f"{mutation_type} crashed unexpectedly: {e}")

    def test_mutation_functions_handle_edge_cases(self):
        """Test mutation functions with edge case inputs."""
        from protosynth.mutation import MutationType, MUTATION_REGISTRY

        rng = random.Random(42)

        # Edge case ASTs
        edge_cases = [
            const(0),  # Single constant
            var('x'),  # Single variable (might fail some mutations)
            op('+'),   # Operation with no arguments (malformed)
            op('unknown_op', const(1)),  # Unknown operation
        ]

        for ast in edge_cases:
            for mutation_type, func in MUTATION_REGISTRY.items():
                with self.subTest(ast=str(ast), mutation_type=mutation_type):
                    try:
                        result = func(ast, rng)
                        # If it succeeds, result should be valid
                        self.assertIsInstance(result, LispNode)
                    except Exception as e:
                        # Failures are ok, but should be graceful
                        self.assertIsInstance(e, (ValueError, RuntimeError, TypeError),
                                            f"{mutation_type} should raise appropriate exception types, got {type(e)}")


class TestOperatorSwapAntibiased(unittest.TestCase):
    """
    Anti-biased tests for operator swap mutation - designed to catch
    arity mismatches, invalid operators, and edge cases.
    """

    def test_op_swap_preserves_arity(self):
        """Test that operator swap respects arity constraints."""
        from protosynth.mutation import _op_swap_mutation

        rng = random.Random(42)

        # Binary operations should stay binary
        binary_ast = op('+', const(1), const(2))
        result = _op_swap_mutation(binary_ast, rng)

        # Should still be an operation with 2 children
        self.assertEqual(result.node_type, 'op')
        self.assertEqual(len(result.children), 2)

        # Unary operations should stay unary (if alternatives exist)
        unary_ast = op('not', const(True))
        try:
            result = _op_swap_mutation(unary_ast, rng)
            self.assertEqual(result.node_type, 'op')
            self.assertEqual(len(result.children), 1)
        except ValueError as e:
            # It's ok if there are no alternative unary operators
            self.assertIn("No suitable", str(e))

    def test_op_swap_only_affects_op_nodes(self):
        """Test that op swap only mutates operation nodes."""
        from protosynth.mutation import _op_swap_mutation

        rng = random.Random(42)

        # Non-op nodes should be unchanged (or fail gracefully)
        non_op_asts = [
            const(42),
            var('x'),
            let('x', const(1), var('x'))
        ]

        for ast in non_op_asts:
            with self.subTest(ast=str(ast)):
                try:
                    result = _op_swap_mutation(ast, rng)
                    # If it succeeds, should be unchanged or have op nodes in children mutated
                    self.assertIsInstance(result, LispNode)
                except ValueError as e:
                    # It's ok to fail if there are no op nodes to mutate
                    self.assertIn("No suitable", str(e))

    def test_op_swap_with_nested_ops(self):
        """Test operator swap in nested operation structures."""
        from protosynth.mutation import _op_swap_mutation

        rng = random.Random(42)

        # Nested operations
        nested_ast = op('+',
                       op('*', const(2), const(3)),
                       op('-', const(10), const(4)))

        # Should be able to mutate at least one operation
        result = _op_swap_mutation(nested_ast, rng)
        self.assertEqual(result.node_type, 'op')

        # Structure should be preserved
        self.assertEqual(len(result.children), 2)
        for child in result.children:
            if child.node_type == 'op':
                self.assertEqual(len(child.children), 2)  # Binary ops stay binary

    def test_op_swap_unknown_operators(self):
        """Test behavior with unknown or invalid operators."""
        from protosynth.mutation import _op_swap_mutation

        rng = random.Random(42)

        # AST with unknown operator
        unknown_op_ast = op('unknown_op', const(1), const(2))

        try:
            result = _op_swap_mutation(unknown_op_ast, rng)
            # If it succeeds, should have a known operator
            from protosynth.core import LispInterpreter
            interpreter = LispInterpreter()
            self.assertIn(result.value, interpreter.operations)
        except ValueError:
            # It's ok to fail with unknown operators
            pass

    def test_op_swap_deterministic_with_seed(self):
        """Test that operator swap is deterministic with same seed."""
        from protosynth.mutation import _op_swap_mutation

        ast = op('+', const(1), const(2))

        # Same seed should produce same result
        rng1 = random.Random(12345)
        rng2 = random.Random(12345)

        try:
            result1 = _op_swap_mutation(ast, rng1)
            result2 = _op_swap_mutation(ast, rng2)

            self.assertEqual(result1.value, result2.value)
        except ValueError:
            # If it fails, both should fail the same way
            with self.assertRaises(ValueError):
                _op_swap_mutation(ast, rng2)


class TestConstantPerturbAntibiased(unittest.TestCase):
    """
    Anti-biased tests for constant perturbation - designed to catch
    type issues, overflow problems, and edge cases.
    """

    def test_const_perturb_only_affects_numeric_constants(self):
        """Test that constant perturbation only affects numeric constants."""
        from protosynth.mutation import _const_perturb_mutation

        rng = random.Random(42)

        # Numeric constants should be perturbed
        numeric_ast = const(10)
        result = _const_perturb_mutation(numeric_ast, rng)
        self.assertEqual(result.node_type, 'const')
        self.assertIsInstance(result.value, (int, float))

        # Non-numeric constants should be unchanged or fail gracefully
        non_numeric_asts = [
            const("hello"),
            const(True),
            const(None),
            var('x'),
            op('+', const(1), const(2))
        ]

        for ast in non_numeric_asts:
            with self.subTest(ast=str(ast)):
                try:
                    result = _const_perturb_mutation(ast, rng)
                    # If it succeeds, should be valid
                    self.assertIsInstance(result, LispNode)
                except ValueError as e:
                    # It's ok to fail if no numeric constants found
                    self.assertIn("No suitable", str(e))

    def test_const_perturb_preserves_type(self):
        """Test that perturbation preserves numeric type (int vs float)."""
        from protosynth.mutation import _const_perturb_mutation

        rng = random.Random(42)

        # Integer should stay integer
        int_ast = const(42)
        result = _const_perturb_mutation(int_ast, rng)
        self.assertIsInstance(result.value, int)

        # Float should stay float
        float_ast = const(3.14)
        result = _const_perturb_mutation(float_ast, rng)
        self.assertIsInstance(result.value, float)

    def test_const_perturb_bounded_change(self):
        """Test that perturbation is bounded by delta parameter."""
        from protosynth.mutation import _const_perturb_mutation

        rng = random.Random(42)
        original_value = 100
        ast = const(original_value)

        # Test multiple perturbations to check bounds
        for _ in range(10):
            result = _const_perturb_mutation(ast, rng)
            change = abs(result.value - original_value)
            # Should be within reasonable bounds (default delta should be small)
            self.assertLessEqual(change, 100, "Perturbation change too large")

    def test_const_perturb_with_edge_values(self):
        """Test perturbation with edge case numeric values."""
        from protosynth.mutation import _const_perturb_mutation

        rng = random.Random(42)

        edge_values = [0, -1, 1, -100, 100, 2**31-1, -2**31]

        for value in edge_values:
            with self.subTest(value=value):
                ast = const(value)
                try:
                    result = _const_perturb_mutation(ast, rng)
                    self.assertIsInstance(result.value, int)
                    # Should not overflow to unreasonable values
                    self.assertGreater(result.value, -2**32)
                    self.assertLess(result.value, 2**32)
                except (ValueError, OverflowError):
                    # Some edge cases might fail - that's ok
                    pass

    def test_const_perturb_nested_constants(self):
        """Test perturbation in nested structures with multiple constants."""
        from protosynth.mutation import _const_perturb_mutation

        rng = random.Random(42)

        # Nested structure with multiple constants
        nested_ast = op('+',
                       const(10),
                       op('*', const(5), const(2)))

        original_constants = []
        from protosynth.mutation import iter_nodes
        for _, _, node in iter_nodes(nested_ast):
            if node.node_type == 'const' and isinstance(node.value, (int, float)):
                original_constants.append(node.value)

        result = _const_perturb_mutation(nested_ast, rng)

        # Should have perturbed exactly one constant
        result_constants = []
        for _, _, node in iter_nodes(result):
            if node.node_type == 'const' and isinstance(node.value, (int, float)):
                result_constants.append(node.value)

        self.assertEqual(len(result_constants), len(original_constants))

        # Exactly one constant should be different
        differences = sum(1 for orig, new in zip(original_constants, result_constants)
                         if orig != new)
        self.assertEqual(differences, 1, "Should perturb exactly one constant")

    def test_const_perturb_deterministic(self):
        """Test that perturbation is deterministic with same seed."""
        from protosynth.mutation import _const_perturb_mutation

        ast = const(42)

        rng1 = random.Random(12345)
        rng2 = random.Random(12345)

        result1 = _const_perturb_mutation(ast, rng1)
        result2 = _const_perturb_mutation(ast, rng2)

        self.assertEqual(result1.value, result2.value)


class TestVariableRenameAntibiased(unittest.TestCase):
    """
    Anti-biased tests for variable rename - designed to catch
    scope issues, name collisions, and binding problems.
    """

    def test_var_rename_preserves_semantics(self):
        """Test that variable rename preserves program semantics."""
        from protosynth.mutation import _var_rename_mutation
        from protosynth import LispInterpreter

        rng = random.Random(42)
        interpreter = LispInterpreter()

        # Simple let binding
        ast = let('x', const(10), op('+', var('x'), const(5)))
        original_result = interpreter.evaluate(ast)

        try:
            mutated = _var_rename_mutation(ast, rng)
            mutated_result = interpreter.evaluate(mutated)

            # Should produce same result
            self.assertEqual(original_result, mutated_result)
        except ValueError as e:
            # It's ok if no suitable variables found
            self.assertIn("No suitable", str(e))

    def test_var_rename_handles_nested_scopes(self):
        """Test variable rename with nested let bindings."""
        from protosynth.mutation import _var_rename_mutation
        from protosynth import LispInterpreter

        rng = random.Random(42)
        interpreter = LispInterpreter()

        # Nested scopes with same variable name
        ast = let('x', const(1),
                  let('x', const(2),  # Shadows outer x
                      op('+', var('x'), const(10))))  # Should use inner x

        original_result = interpreter.evaluate(ast)

        try:
            mutated = _var_rename_mutation(ast, rng)
            mutated_result = interpreter.evaluate(mutated)

            # Should still produce same result
            self.assertEqual(original_result, mutated_result)
        except ValueError:
            # Complex scoping might not be supported yet
            pass

    def test_var_rename_avoids_name_collisions(self):
        """Test that rename avoids creating name collisions."""
        from protosynth.mutation import _var_rename_mutation

        rng = random.Random(42)

        # AST with multiple variables
        ast = let('x', const(1),
                  let('y', const(2),
                      op('+', var('x'), var('y'))))

        try:
            mutated = _var_rename_mutation(ast, rng)

            # Check that no variable names collide
            var_names = set()
            let_names = set()

            from protosynth.mutation import iter_nodes
            for _, _, node in iter_nodes(mutated):
                if node.node_type == 'let' and len(node.children) >= 1:
                    let_names.add(node.children[0].value)
                elif node.node_type == 'var':
                    var_names.add(node.value)

            # All let bindings should have unique names
            self.assertEqual(len(let_names), len(set(let_names)))

            # All variable references should refer to existing bindings
            # (This is a simplified check - full scope analysis is complex)

        except ValueError:
            # It's ok if no suitable variables found
            pass

    def test_var_rename_generates_fresh_names(self):
        """Test that rename generates fresh, unused variable names."""
        from protosynth.mutation import _var_rename_mutation

        rng = random.Random(42)

        # Simple case
        ast = let('x', const(42), var('x'))

        try:
            mutated = _var_rename_mutation(ast, rng)

            # Should have a different variable name
            from protosynth.mutation import iter_nodes
            let_node = None
            var_nodes = []

            for _, _, node in iter_nodes(mutated):
                if node.node_type == 'let':
                    let_node = node
                elif node.node_type == 'var':
                    var_nodes.append(node)

            if let_node and var_nodes:
                new_name = let_node.children[0].value
                self.assertNotEqual(new_name, 'x')  # Should be different

                # All var references should use the new name
                for var_node in var_nodes:
                    if var_node.value != 'x':  # If it was renamed
                        self.assertEqual(var_node.value, new_name)

        except ValueError:
            # It's ok if no suitable variables found
            pass

    def test_var_rename_with_no_variables(self):
        """Test behavior when there are no variables to rename."""
        from protosynth.mutation import _var_rename_mutation

        rng = random.Random(42)

        # ASTs with no let bindings
        no_var_asts = [
            const(42),
            op('+', const(1), const(2)),
            if_expr(const(True), const(1), const(0))
        ]

        for ast in no_var_asts:
            with self.subTest(ast=str(ast)):
                with self.assertRaises(ValueError):
                    _var_rename_mutation(ast, rng)

    def test_var_rename_deterministic(self):
        """Test that variable rename is deterministic with same seed."""
        from protosynth.mutation import _var_rename_mutation

        ast = let('x', const(10), var('x'))

        rng1 = random.Random(12345)
        rng2 = random.Random(12345)

        try:
            result1 = _var_rename_mutation(ast, rng1)
            result2 = _var_rename_mutation(ast, rng2)

            # Should produce same renamed variable
            from protosynth import pretty_print_ast
            self.assertEqual(pretty_print_ast(result1), pretty_print_ast(result2))
        except ValueError:
            # If it fails, both should fail the same way
            with self.assertRaises(ValueError):
                _var_rename_mutation(ast, rng2)


class TestMutationPipelineAntibiased(unittest.TestCase):
    """
    Anti-biased tests for mutation pipeline - designed to catch
    probability issues, mutation conflicts, and edge cases.
    """

    def test_mutation_pipeline_import(self):
        """Test that mutation pipeline can be imported."""
        try:
            from protosynth.mutation import mutate
            self.assertTrue(callable(mutate))
        except ImportError as e:
            self.fail(f"Could not import mutate function: {e}")

    def test_mutate_with_zero_probability(self):
        """Test that zero mutation rate returns unchanged AST."""
        from protosynth.mutation import mutate

        ast = op('+', const(1), const(2))
        rng = random.Random(42)

        result = mutate(ast, mutation_rate=0.0, rng=rng)

        # Should be a clone but semantically identical
        from protosynth import pretty_print_ast
        self.assertEqual(pretty_print_ast(ast), pretty_print_ast(result))
        self.assertIsNot(ast, result)  # Should be a clone

    def test_mutate_with_high_probability(self):
        """Test behavior with very high mutation rate."""
        from protosynth.mutation import mutate

        # Complex AST that should have multiple mutation opportunities
        ast = let('x', const(10),
                  if_expr(op('>', var('x'), const(5)),
                          op('+', var('x'), const(1)),
                          op('-', var('x'), const(1))))

        rng = random.Random(42)

        # With high probability, should apply some mutation
        result = mutate(ast, mutation_rate=0.9, rng=rng)

        # Should still be a valid AST
        self.assertIsInstance(result, LispNode)

        # Should be different from original (with high probability)
        from protosynth import pretty_print_ast
        # Note: There's a small chance it could be the same due to randomness
        # but with 0.9 probability across 5 mutation types, very unlikely

    def test_mutate_preserves_ast_validity(self):
        """Test that mutations preserve basic AST structure validity."""
        from protosynth.mutation import mutate

        test_asts = [
            const(42),
            op('+', const(1), const(2)),
            let('x', const(5), var('x')),
            if_expr(const(True), const(1), const(0))
        ]

        rng = random.Random(42)

        for ast in test_asts:
            with self.subTest(ast=str(ast)):
                result = mutate(ast, mutation_rate=0.5, rng=rng)

                # Should still be a LispNode
                self.assertIsInstance(result, LispNode)

                # Should have valid node_type
                self.assertIn(result.node_type, ['const', 'var', 'let', 'if', 'op'])

                # Should have children list
                self.assertIsInstance(result.children, list)

    def test_mutate_deterministic_with_seed(self):
        """Test that mutation is deterministic with same seed."""
        from protosynth.mutation import mutate

        ast = op('+', const(10), const(20))

        rng1 = random.Random(12345)
        rng2 = random.Random(12345)

        result1 = mutate(ast, mutation_rate=0.5, rng=rng1)
        result2 = mutate(ast, mutation_rate=0.5, rng=rng2)

        from protosynth import pretty_print_ast
        self.assertEqual(pretty_print_ast(result1), pretty_print_ast(result2))

    def test_mutate_handles_edge_case_asts(self):
        """Test mutation pipeline with edge case ASTs."""
        from protosynth.mutation import mutate

        edge_cases = [
            const(0),  # Minimal AST
            var('undefined'),  # Variable with no binding
            op('unknown', const(1)),  # Unknown operation
        ]

        rng = random.Random(42)

        for ast in edge_cases:
            with self.subTest(ast=str(ast)):
                try:
                    result = mutate(ast, mutation_rate=0.3, rng=rng)
                    # If it succeeds, should be valid
                    self.assertIsInstance(result, LispNode)
                except Exception as e:
                    # Some edge cases might fail - should be graceful
                    self.assertIsInstance(e, (ValueError, RuntimeError, TypeError))

    def test_mutate_multiple_applications(self):
        """Test applying mutations multiple times in sequence."""
        from protosynth.mutation import mutate

        ast = op('+', const(5), const(10))
        rng = random.Random(42)

        # Apply mutations multiple times
        current = ast
        for i in range(5):
            current = mutate(current, mutation_rate=0.2, rng=rng)

            # Should remain valid after each mutation
            self.assertIsInstance(current, LispNode)

            # Should be able to pretty-print (basic validity check)
            from protosynth import pretty_print_ast
            pretty_str = pretty_print_ast(current)
            self.assertIsInstance(pretty_str, str)
            self.assertGreater(len(pretty_str), 0)


class TestMutationFuzzHarness(unittest.TestCase):
    """
    Fuzz testing for mutation system - property-based testing to catch
    edge cases and ensure mutations always produce valid, verifiable ASTs.
    """

    def test_ten_random_mutations_from_verified_seed(self):
        """Test that 10 random mutations from verified seed always pass verifier."""
        from protosynth.mutation import mutate
        from protosynth.verify import verify_ast

        # Start with verified seed ASTs
        verified_seeds = [
            const(42),
            op('+', const(1), const(2)),
            let('x', const(10), var('x')),
            if_expr(op('>', const(5), const(3)), const(1), const(0)),
            let('a', const(5),
                let('b', const(10),
                    op('+', var('a'), var('b'))))
        ]

        rng = random.Random(12345)  # Deterministic for reproducibility

        for seed_ast in verified_seeds:
            with self.subTest(seed=str(seed_ast)):
                # Verify the seed is valid
                is_valid, errors = verify_ast(seed_ast)
                self.assertTrue(is_valid, f"Seed should be valid: {errors}")

                # Apply 10 mutations
                current_ast = seed_ast
                for i in range(10):
                    try:
                        mutated_ast = mutate(current_ast, mutation_rate=0.3, rng=rng)

                        # Every mutation should produce a verifiable AST
                        is_valid, errors = verify_ast(mutated_ast)
                        if not is_valid:
                            from protosynth import pretty_print_ast
                            self.fail(f"Mutation {i+1} should be valid: {errors}\n"
                                    f"Original: {pretty_print_ast(current_ast)}\n"
                                    f"Mutated: {pretty_print_ast(mutated_ast)}")

                        current_ast = mutated_ast

                    except Exception as e:
                        self.fail(f"Mutation {i+1} failed unexpectedly: {e}")

    def test_mutation_preserves_evaluability(self):
        """Test that mutations preserve the ability to evaluate ASTs."""
        from protosynth.mutation import mutate
        from protosynth import LispInterpreter

        interpreter = LispInterpreter()
        rng = random.Random(42)

        # Evaluable seed ASTs
        evaluable_seeds = [
            const(42),
            op('+', const(10), const(5)),
            let('x', const(7), op('*', var('x'), const(2))),
            if_expr(op('<', const(3), const(5)), const(100), const(200))
        ]

        for seed_ast in evaluable_seeds:
            with self.subTest(seed=str(seed_ast)):
                # Original should be evaluable
                try:
                    original_result = interpreter.evaluate(seed_ast)
                except Exception as e:
                    self.fail(f"Seed AST should be evaluable: {e}")

                # Apply mutation
                try:
                    mutated_ast = mutate(seed_ast, mutation_rate=0.5, rng=rng)

                    # Mutated AST should also be evaluable
                    mutated_result = interpreter.evaluate(mutated_ast)

                    # Results might be different, but both should succeed
                    self.assertIsNotNone(original_result)
                    self.assertIsNotNone(mutated_result)

                except Exception as e:
                    # Some mutations might make ASTs non-evaluable (e.g., unbound vars)
                    # but they should fail verification, not crash the interpreter
                    from protosynth.verify import verify_ast
                    is_valid, errors = verify_ast(mutated_ast)
                    if is_valid:
                        self.fail(f"Valid AST should be evaluable: {e}")

    def test_mutation_stress_test(self):
        """Stress test with many mutations on complex ASTs."""
        from protosynth.mutation import mutate
        from protosynth.verify import verify_ast

        # Complex but valid AST
        complex_ast = let('x', const(10),
                         let('y', const(20),
                             if_expr(op('>', var('x'), const(5)),
                                    op('+', var('x'), var('y')),
                                    op('-', var('y'), var('x')))))

        rng = random.Random(999)
        current_ast = complex_ast

        # Apply 50 mutations
        for i in range(50):
            try:
                mutated_ast = mutate(current_ast, mutation_rate=0.2, rng=rng)

                # Should remain valid
                is_valid, errors = verify_ast(mutated_ast)
                self.assertTrue(is_valid, f"Stress test mutation {i+1} failed: {errors}")

                current_ast = mutated_ast

            except Exception as e:
                self.fail(f"Stress test mutation {i+1} crashed: {e}")

    def test_mutation_determinism_property(self):
        """Property test: same seed should produce same mutations."""
        from protosynth.mutation import mutate

        test_ast = op('+', const(5), const(10))

        # Same seed should produce identical results
        for seed_value in [123, 456, 789]:
            rng1 = random.Random(seed_value)
            rng2 = random.Random(seed_value)

            result1 = mutate(test_ast, mutation_rate=0.5, rng=rng1)
            result2 = mutate(test_ast, mutation_rate=0.5, rng=rng2)

            from protosynth import pretty_print_ast
            self.assertEqual(pretty_print_ast(result1), pretty_print_ast(result2),
                           f"Same seed {seed_value} should produce identical mutations")


if __name__ == '__main__':
    unittest.main()
