#!/usr/bin/env python3
"""
Anti-biased tests for ProtoSynth verification system.

These tests are designed to catch verification failures, edge cases,
and ensure that the verifier correctly identifies invalid ASTs.
"""

import unittest
from protosynth import const, var, let, if_expr, op, LispNode


class TestVerificationAntibiased(unittest.TestCase):
    """
    Anti-biased tests for AST verification - designed to catch
    verification failures and ensure invalid ASTs are rejected.
    """
    
    def test_verification_imports(self):
        """Test that verification functions can be imported."""
        try:
            from protosynth.verify import check_arity, check_free_vars, check_depth, verify_ast
            self.assertTrue(callable(check_arity))
            self.assertTrue(callable(check_free_vars))
            self.assertTrue(callable(check_depth))
            self.assertTrue(callable(verify_ast))
        except ImportError as e:
            self.fail(f"Could not import verification functions: {e}")
    
    def test_check_arity_catches_invalid_operations(self):
        """Test that arity checker catches operations with wrong number of arguments."""
        from protosynth.verify import check_arity
        
        # Valid operations should pass
        valid_ops = [
            op('+', const(1), const(2)),  # Binary
            op('not', const(True)),       # Unary
            op('*', const(2), const(3)),  # Binary
        ]
        
        for ast in valid_ops:
            with self.subTest(ast=str(ast)):
                is_valid, errors = check_arity(ast)
                self.assertTrue(is_valid, f"Valid operation should pass: {errors}")
        
        # Invalid operations should fail
        invalid_ops = [
            op('+', const(1)),           # Binary op with 1 arg
            op('+', const(1), const(2), const(3)),  # Binary op with 3 args
            op('not', const(True), const(False)),    # Unary op with 2 args
            op('not'),                   # Unary op with 0 args
        ]
        
        for ast in invalid_ops:
            with self.subTest(ast=str(ast)):
                is_valid, errors = check_arity(ast)
                self.assertFalse(is_valid, f"Invalid operation should fail: {ast}")
                self.assertGreater(len(errors), 0, "Should have error messages")
    
    def test_check_free_vars_catches_unbound_variables(self):
        """Test that free variable checker catches unbound variable references."""
        from protosynth.verify import check_free_vars
        
        # Valid variable usage should pass
        valid_vars = [
            let('x', const(1), var('x')),  # Bound variable
            const(42),                     # No variables
            op('+', const(1), const(2)),   # No variables
        ]
        
        for ast in valid_vars:
            with self.subTest(ast=str(ast)):
                is_valid, errors = check_free_vars(ast)
                self.assertTrue(is_valid, f"Valid variable usage should pass: {errors}")
        
        # Invalid variable usage should fail
        invalid_vars = [
            var('x'),                      # Unbound variable
            op('+', var('x'), const(1)),   # Unbound variable in operation
            let('x', const(1), var('y')),  # Wrong variable name
            if_expr(var('undefined'), const(1), const(0)),  # Unbound in condition
        ]
        
        for ast in invalid_vars:
            with self.subTest(ast=str(ast)):
                is_valid, errors = check_free_vars(ast)
                self.assertFalse(is_valid, f"Invalid variable usage should fail: {ast}")
                self.assertGreater(len(errors), 0, "Should have error messages")
    
    def test_check_depth_catches_excessive_nesting(self):
        """Test that depth checker catches excessively nested ASTs."""
        from protosynth.verify import check_depth
        
        # Shallow ASTs should pass
        shallow_ast = op('+', const(1), const(2))
        is_valid, errors = check_depth(shallow_ast, max_depth=5)
        self.assertTrue(is_valid, f"Shallow AST should pass: {errors}")
        
        # Deep ASTs should fail with low limit
        deep_ast = let('a', const(1),
                      let('b', const(2),
                          let('c', const(3),
                              let('d', const(4),
                                  var('d')))))  # Depth 4
        
        is_valid, errors = check_depth(deep_ast, max_depth=2)
        self.assertFalse(is_valid, f"Deep AST should fail with low limit: {deep_ast}")
        self.assertGreater(len(errors), 0, "Should have error messages")
        
        # Same AST should pass with higher limit
        is_valid, errors = check_depth(deep_ast, max_depth=10)
        self.assertTrue(is_valid, f"Deep AST should pass with high limit: {errors}")
    
    def test_verify_ast_comprehensive_checking(self):
        """Test that verify_ast performs comprehensive validation."""
        from protosynth.verify import verify_ast
        
        # Valid AST should pass all checks
        valid_ast = let('x', const(10),
                       if_expr(op('>', var('x'), const(5)),
                              op('+', var('x'), const(1)),
                              const(0)))
        
        is_valid, errors = verify_ast(valid_ast)
        self.assertTrue(is_valid, f"Valid AST should pass all checks: {errors}")
        
        # AST with multiple issues should fail
        invalid_ast = let('x', const(10),
                         op('+', var('y'), var('z')))  # Unbound variables
        
        is_valid, errors = verify_ast(invalid_ast)
        self.assertFalse(is_valid, f"Invalid AST should fail: {invalid_ast}")
        self.assertGreater(len(errors), 0, "Should have error messages")
    
    def test_verification_with_malformed_asts(self):
        """Test verification behavior with malformed AST structures."""
        from protosynth.verify import verify_ast
        
        # ASTs with structural issues
        malformed_asts = [
            LispNode('unknown_type', 'value'),  # Unknown node type
            LispNode('let', None, []),          # Let with no children
            LispNode('if', None, [const(True)]), # If with insufficient children
            LispNode('op', 'unknown_op', [const(1)]), # Unknown operation
        ]
        
        for ast in malformed_asts:
            with self.subTest(ast=str(ast)):
                is_valid, errors = verify_ast(ast)
                self.assertFalse(is_valid, f"Malformed AST should fail: {ast}")
                self.assertGreater(len(errors), 0, "Should have error messages")
    
    def test_verification_error_messages_informative(self):
        """Test that verification error messages are informative."""
        from protosynth.verify import verify_ast
        
        # Create AST with known issue
        ast_with_unbound_var = op('+', var('undefined'), const(1))
        
        is_valid, errors = verify_ast(ast_with_unbound_var)
        self.assertFalse(is_valid)
        
        # Error messages should be informative
        error_text = ' '.join(errors)
        self.assertIn('undefined', error_text.lower())
        self.assertIn('variable', error_text.lower())
    
    def test_verification_performance_with_large_ast(self):
        """Test that verification doesn't degrade catastrophically with large ASTs."""
        from protosynth.verify import verify_ast
        import time
        
        # Create a large but valid AST
        def create_large_valid_ast(depth):
            if depth <= 0:
                return const(depth)
            return let(f'x{depth}', const(depth),
                      op('+', var(f'x{depth}'), create_large_valid_ast(depth - 1)))
        
        large_ast = create_large_valid_ast(5)  # Depth 5 - within default limits

        start_time = time.time()
        is_valid, errors = verify_ast(large_ast, max_depth=15, max_nodes=200)  # Generous limits
        end_time = time.time()

        # Should complete in reasonable time
        self.assertLess(end_time - start_time, 1.0, "Verification should be fast")
        self.assertTrue(is_valid, f"Large valid AST should pass: {errors}")


if __name__ == '__main__':
    unittest.main()
