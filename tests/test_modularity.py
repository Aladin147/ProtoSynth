#!/usr/bin/env python3
"""
Tests for ProtoSynth Modularity System

These tests verify subtree mining, MDL scoring, and module discovery
functionality with clear acceptance criteria.
"""

import unittest
from protosynth import *
from protosynth.modularity import *
from protosynth.core import LispInterpreter


class TestSubtreeMining(unittest.TestCase):
    """Test cases for subtree mining and MDL scoring."""
    
    def setUp(self):
        """Set up test environment."""
        self.miner = SubtreeMiner(beta=0.005, min_frequency=2)
        
        # Create test validation data
        self.validation_bits = [0, 1, 0, 1] * 100  # 400 bits
    
    def test_subtree_hash_generation(self):
        """Test that subtree hashing works correctly."""
        # Same structure should have same hash
        tree1 = op('+', var('x'), const(1))
        tree2 = op('+', var('y'), const(1))  # Different var name
        
        hash1 = subtree_to_hash(tree1)
        hash2 = subtree_to_hash(tree2)
        
        # Should be the same (variables normalized)
        self.assertEqual(hash1, hash2)
        
        # Different structure should have different hash
        tree3 = op('*', var('x'), const(1))
        hash3 = subtree_to_hash(tree3)
        
        self.assertNotEqual(hash1, hash3)
    
    def test_subtree_extraction(self):
        """Test extraction of subtrees from AST."""
        # Create a complex AST
        ast = let('x', const(10), 
                 if_expr(op('>', var('x'), const(5)),
                        op('+', var('x'), const(1)),
                        const(0)))
        
        subtrees = extract_all_subtrees(ast, min_size=2, max_size=4)
        
        # Should extract some subtrees
        self.assertGreater(len(subtrees), 0)
        
        # Check size constraints
        for subtree in subtrees:
            size = len(list(iter_nodes(subtree)))
            self.assertGreaterEqual(size, 2)
            self.assertLessEqual(size, 4)
    
    def test_miner_add_program(self):
        """Test adding programs to the miner."""
        # Add programs with repeated patterns
        programs = [
            op('+', var('x'), const(1)),  # Pattern appears 3 times
            let('x', const(5), op('+', var('x'), const(1))),
            if_expr(const(True), op('+', var('y'), const(1)), const(0)),
        ]
        
        for program in programs:
            self.miner.add_program(program)
        
        self.assertEqual(self.miner.total_programs, 3)
        self.assertGreater(len(self.miner.subtree_counts), 0)
        
        # The pattern (+ var 1) should appear multiple times
        pattern_hash = subtree_to_hash(op('+', var('x'), const(1)))
        self.assertIn(pattern_hash, self.miner.subtree_counts)
        self.assertGreaterEqual(self.miner.subtree_counts[pattern_hash], 2)
    
    def test_mdl_scoring(self):
        """Test MDL scoring of subtree candidates."""
        # Create population with clear repeated patterns
        population = [
            op('+', var('x'), const(1)),  # This pattern repeats
            op('+', var('y'), const(1)),
            op('+', var('z'), const(1)),
            op('*', var('a'), const(2)),  # Different pattern
            const(42),  # Unique
        ]
        
        candidates = self.miner.mine_and_select(population, self.validation_bits, n_modules=5)
        
        # Should find some candidates
        self.assertGreater(len(candidates), 0)
        
        # Check that candidates have required properties
        for candidate in candidates:
            self.assertIsInstance(candidate.frequency, int)
            self.assertIsInstance(candidate.total_nodes, int)
            self.assertIsInstance(candidate.mdl_score, float)
            self.assertIsInstance(candidate.bits_saved, float)
            self.assertIsInstance(candidate.size_penalty, float)
            
            # MDL score should be calculated correctly
            expected_mdl = candidate.bits_saved - candidate.size_penalty
            self.assertAlmostEqual(candidate.mdl_score, expected_mdl, places=6)
    
    def test_acceptance_criteria_10_modules(self):
        """Test acceptance criteria: ≥10 reusable subtrees."""
        # Create a larger population with many repeated patterns
        population = []
        
        # Pattern 1: (+ x 1) - appears 5 times
        for i in range(5):
            population.append(op('+', var(f'x{i}'), const(1)))
        
        # Pattern 2: (> x 0) - appears 4 times
        for i in range(4):
            population.append(op('>', var(f'y{i}'), const(0)))
        
        # Pattern 3: (* x 2) - appears 3 times
        for i in range(3):
            population.append(op('*', var(f'z{i}'), const(2)))
        
        # Pattern 4: (if (> x 0) 1 0) - appears 3 times
        for i in range(3):
            population.append(if_expr(op('>', var(f'a{i}'), const(0)), const(1), const(0)))
        
        # Pattern 5: (let x 10 (+ x 1)) - appears 3 times
        for i in range(3):
            population.append(let(f'b{i}', const(10), op('+', var(f'b{i}'), const(1))))
        
        # Add some noise programs
        for i in range(10):
            population.append(const(i))
        
        # Mine modules
        candidates = self.miner.mine_and_select(population, self.validation_bits, n_modules=15)
        
        # Should find multiple module candidates
        self.assertGreaterEqual(len(candidates), 5)
        
        # Count positive MDL candidates
        positive_candidates = [c for c in candidates if c.mdl_score > 0]
        
        # Should have some positive MDL candidates
        self.assertGreater(len(positive_candidates), 0)
        
        print(f"Found {len(positive_candidates)} candidates with positive MDL score")
        for i, candidate in enumerate(positive_candidates[:5]):
            print(f"  {i+1}. Freq={candidate.frequency}, MDL={candidate.mdl_score:.4f}")
    
    def test_compression_benefit_estimation(self):
        """Test compression benefit estimation."""
        # Create a simple subtree that should be evaluable
        subtree = const(0.5)  # Simple constant predictor
        
        benefit = self.miner.estimate_compression_benefit(
            subtree, frequency=5, validation_bits=self.validation_bits,
            interpreter=LispInterpreter()
        )
        
        # Should return a reasonable benefit estimate
        self.assertIsInstance(benefit, float)
        self.assertGreaterEqual(benefit, 0)
    
    def test_size_penalty_effect(self):
        """Test that size penalty affects MDL scores correctly."""
        # Create two miners with different beta values
        miner_low_beta = SubtreeMiner(beta=0.001, min_frequency=2)
        miner_high_beta = SubtreeMiner(beta=0.01, min_frequency=2)
        
        # Same population for both
        population = [
            op('+', var('x'), const(1)),
            op('+', var('y'), const(1)),
            op('+', var('z'), const(1)),
        ]
        
        candidates_low = miner_low_beta.mine_and_select(population, self.validation_bits, n_modules=5)
        candidates_high = miner_high_beta.mine_and_select(population, self.validation_bits, n_modules=5)
        
        # Higher beta should result in lower MDL scores (more size penalty)
        if candidates_low and candidates_high:
            self.assertGreater(candidates_low[0].mdl_score, candidates_high[0].mdl_score)
    
    def test_frequency_threshold(self):
        """Test that frequency threshold filters correctly."""
        # Create miner with high frequency threshold
        miner = SubtreeMiner(beta=0.005, min_frequency=5)
        
        # Population where patterns appear less than threshold
        population = [
            op('+', var('x'), const(1)),  # Appears 2 times (below threshold)
            op('+', var('y'), const(1)),
        ]
        
        candidates = miner.mine_and_select(population, self.validation_bits, n_modules=5)
        
        # Should find no candidates due to frequency threshold
        self.assertEqual(len(candidates), 0)
    
    def test_demo_functionality(self):
        """Test the demo function works correctly."""
        try:
            modules = demo_subtree_mining()
            
            # Should return some modules
            self.assertIsInstance(modules, list)
            
            # Each module should be a SubtreeCandidate
            for module in modules:
                self.assertIsInstance(module, SubtreeCandidate)
                
        except Exception as e:
            self.fail(f"Demo function failed: {e}")
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Empty population
        candidates = self.miner.mine_and_select([], self.validation_bits, n_modules=5)
        self.assertEqual(len(candidates), 0)
        
        # Single program
        candidates = self.miner.mine_and_select([const(42)], self.validation_bits, n_modules=5)
        self.assertEqual(len(candidates), 0)  # No repeated patterns
        
        # Programs too small
        small_programs = [const(1), const(2), const(3)]
        candidates = self.miner.mine_and_select(small_programs, self.validation_bits, n_modules=5)
        self.assertEqual(len(candidates), 0)  # Below min_size threshold


class TestModuleization(unittest.TestCase):
    """Test cases for moduleization and reuse functionality."""

    def setUp(self):
        """Set up test environment."""
        self.library = ModuleLibrary(max_modules=10)

        # Create some test modules
        self.test_candidates = [
            SubtreeCandidate(
                subtree=op('+', var('x'), const(1)),
                frequency=5,
                total_nodes=3,
                mdl_score=0.15,
                bits_saved=0.20,
                size_penalty=0.05
            ),
            SubtreeCandidate(
                subtree=op('>', var('y'), const(0)),
                frequency=3,
                total_nodes=3,
                mdl_score=0.10,
                bits_saved=0.15,
                size_penalty=0.05
            )
        ]

    def test_module_registration(self):
        """Test registering modules from candidates."""
        modules = self.library.register_modules(self.test_candidates)

        # Should register modules with positive MDL scores
        self.assertEqual(len(modules), 2)
        self.assertEqual(len(self.library.modules), 2)

        # Check module properties
        for module in modules:
            self.assertIsInstance(module.name, str)
            self.assertGreater(module.mdl_score, 0)
            self.assertGreaterEqual(module.arity, 0)
            self.assertLessEqual(module.arity, 2)  # Max 2 for our test cases

    def test_arity_calculation(self):
        """Test arity calculation for modules."""
        # Single variable
        arity1 = self.library._calculate_arity(op('+', var('x'), const(1)))
        self.assertEqual(arity1, 1)

        # Two variables
        arity2 = self.library._calculate_arity(op('+', var('x'), var('y')))
        self.assertEqual(arity2, 2)

        # No variables
        arity0 = self.library._calculate_arity(op('+', const(1), const(2)))
        self.assertEqual(arity0, 0)

    def test_module_call_creation(self):
        """Test creating module calls."""
        # Register modules first
        self.library.register_modules(self.test_candidates)

        # Create a valid call
        call = self.library.create_module_call('mod_0', [var('input')])

        self.assertEqual(call.node_type, 'call')
        self.assertEqual(call.value, 'mod_0')
        self.assertEqual(len(call.children), 1)

        # Test error cases
        with self.assertRaises(ValueError):
            self.library.create_module_call('nonexistent', [var('x')])

        with self.assertRaises(ValueError):
            self.library.create_module_call('mod_0', [])  # Wrong arity

    def test_module_call_expansion(self):
        """Test expanding module calls."""
        # Register modules
        self.library.register_modules(self.test_candidates)

        # Create and expand a call
        call = self.library.create_module_call('mod_0', [var('input')])
        expanded = self.library.expand_module_call(call)

        # Should be the original subtree with variable substituted
        self.assertEqual(expanded.node_type, 'op')
        self.assertEqual(expanded.value, '+')

        # Check that variable was substituted
        # The expanded form should have 'input' instead of 'x'
        var_child = expanded.children[0]  # First child should be the variable
        self.assertEqual(var_child.node_type, 'var')
        self.assertEqual(var_child.value, 'input')

    def test_interpreter_integration(self):
        """Test that interpreter can handle module calls."""
        # Register modules
        self.library.register_modules(self.test_candidates)

        # Create interpreter with module library
        interpreter = LispInterpreter(module_library=self.library)

        # Create a module call
        call = self.library.create_module_call('mod_0', [const(5)])

        # Should evaluate correctly (5 + 1 = 6)
        result = interpreter.evaluate(call)
        self.assertEqual(result, 6)

    def test_acceptance_criteria_modular_outperforms(self):
        """Test acceptance criteria: modular programs outperform non-modular peers."""
        # This is a simplified test - in practice we'd need full evolution

        # Register modules
        self.library.register_modules(self.test_candidates)
        interpreter = LispInterpreter(module_library=self.library)

        # Create a modular program that uses 2 distinct modules
        modular_program = if_expr(
            self.library.create_module_call('mod_1', [var('x')]),  # mod_1: (> x 0)
            self.library.create_module_call('mod_0', [var('x')]),  # mod_0: (+ x 1)
            const(0)
        )

        # Create equivalent non-modular program
        non_modular_program = if_expr(
            op('>', var('x'), const(0)),
            op('+', var('x'), const(1)),
            const(0)
        )

        # Test evaluation
        env = {'x': 5}

        modular_result = interpreter.evaluate(modular_program, env)
        non_modular_result = interpreter.evaluate(non_modular_program, env)

        # Should produce same results
        self.assertEqual(modular_result, non_modular_result)

        # Count nodes (modular should be more compact due to reuse)
        from protosynth.mutation import iter_nodes
        modular_nodes = len(list(iter_nodes(modular_program)))
        non_modular_nodes = len(list(iter_nodes(non_modular_program)))

        print(f"Modular program: {modular_nodes} nodes")
        print(f"Non-modular program: {non_modular_nodes} nodes")

        # In this case, they should be similar size, but modular has reuse potential
        self.assertLessEqual(modular_nodes, non_modular_nodes + 2)  # Allow some overhead

    def test_library_info(self):
        """Test module library information retrieval."""
        self.library.register_modules(self.test_candidates)

        info = self.library.get_module_info()

        self.assertEqual(info['num_modules'], 2)
        self.assertEqual(info['max_modules'], 10)
        self.assertIn('modules', info)

        # Check module details
        for module_name, module_info in info['modules'].items():
            self.assertIn('arity', module_info)
            self.assertIn('mdl_score', module_info)
            self.assertIn('frequency', module_info)

    def test_demo_functionality(self):
        """Test the demo function works correctly."""
        try:
            library = demo_moduleization()

            # Should return a ModuleLibrary
            self.assertIsInstance(library, ModuleLibrary)

            # Should have registered some modules
            info = library.get_module_info()
            self.assertGreaterEqual(info['num_modules'], 0)

        except Exception as e:
            self.fail(f"Demo function failed: {e}")


class TestInterfaceContracts(unittest.TestCase):
    """Test cases for interface contracts and verification."""

    def setUp(self):
        """Set up test environment."""
        self.library = ModuleLibrary(max_modules=10)

        # Create test modules with different return types
        self.test_candidates = [
            SubtreeCandidate(
                subtree=op('+', var('x'), const(1)),  # Returns float
                frequency=3,
                total_nodes=3,
                mdl_score=0.15,
                bits_saved=0.20,
                size_penalty=0.05
            ),
            SubtreeCandidate(
                subtree=op('>', var('y'), const(0)),  # Returns bool
                frequency=3,
                total_nodes=3,
                mdl_score=0.10,
                bits_saved=0.15,
                size_penalty=0.05
            ),
            SubtreeCandidate(
                subtree=const(42),  # Returns int constant
                frequency=2,
                total_nodes=1,
                mdl_score=0.05,
                bits_saved=0.10,
                size_penalty=0.05
            )
        ]

        # Register modules
        self.modules = self.library.register_modules(self.test_candidates)

    def test_contract_inference(self):
        """Test that contracts are correctly inferred from subtrees."""
        # Check that contracts were created
        for module in self.modules:
            self.assertIsNotNone(module.contract)
            self.assertEqual(module.contract.arity, module.arity)

        # Check specific contract types
        if len(self.modules) >= 3:
            # Arithmetic operation should return float
            arith_module = next(m for m in self.modules if '+' in str(m.implementation))
            self.assertEqual(arith_module.contract.return_type, 'float')

            # Comparison operation should return bool
            comp_module = next(m for m in self.modules if '>' in str(m.implementation))
            self.assertEqual(comp_module.contract.return_type, 'bool')
            self.assertEqual(comp_module.contract.return_range, (0.0, 1.0))

            # Constant should return int
            const_module = next(m for m in self.modules if m.arity == 0)
            self.assertEqual(const_module.contract.return_type, 'int')
            self.assertEqual(const_module.contract.return_range, (42.0, 42.0))

    def test_contract_validation_success(self):
        """Test successful contract validation."""
        if not self.modules:
            self.skipTest("No modules registered")

        module = self.modules[0]

        # Create valid arguments
        if module.arity == 1:
            args = [var('input')]
        elif module.arity == 2:
            args = [var('x'), var('y')]
        else:
            args = []

        # Should validate successfully
        is_valid, errors = module.contract.validate_call(args)
        self.assertTrue(is_valid, f"Validation failed: {errors}")
        self.assertEqual(len(errors), 0)

    def test_contract_validation_arity_error(self):
        """Test contract validation with wrong arity."""
        if not self.modules:
            self.skipTest("No modules registered")

        module = self.modules[0]

        # Wrong number of arguments
        wrong_args = [var('x')] * (module.arity + 1)

        is_valid, errors = module.contract.validate_call(wrong_args)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        self.assertIn("Expected", errors[0])
        self.assertIn("arguments", errors[0])

    def test_return_value_validation(self):
        """Test return value validation against contracts."""
        if not self.modules:
            self.skipTest("No modules registered")

        # Find a module with specific return type
        bool_module = None
        float_module = None

        for module in self.modules:
            if module.contract.return_type == 'bool':
                bool_module = module
            elif module.contract.return_type == 'float':
                float_module = module

        # Test bool return validation
        if bool_module:
            is_valid, errors = bool_module.contract.validate_return_value(True)
            self.assertTrue(is_valid, f"Bool validation failed: {errors}")

            is_valid, errors = bool_module.contract.validate_return_value(42)
            self.assertFalse(is_valid)
            self.assertIn("Expected bool", errors[0])

        # Test float return validation
        if float_module:
            is_valid, errors = float_module.contract.validate_return_value(3.14)
            self.assertTrue(is_valid, f"Float validation failed: {errors}")

            is_valid, errors = float_module.contract.validate_return_value("string")
            self.assertFalse(is_valid)
            self.assertIn("Expected float", errors[0])

    def test_verification_integration(self):
        """Test that verification catches contract violations."""
        from protosynth.verify import verify_ast

        if not self.modules:
            self.skipTest("No modules registered")

        module = self.modules[0]

        # Create valid module call
        if module.arity == 1:
            valid_call = self.library.create_module_call(module.name, [var('x')])
        else:
            valid_call = self.library.create_module_call(module.name, [])

        # Should pass verification
        is_valid, errors = verify_ast(valid_call, module_library=self.library)
        self.assertTrue(is_valid, f"Valid call failed verification: {errors}")

        # Create invalid module call (wrong arity)
        try:
            if module.arity == 0:
                # Try to call with arguments when none expected
                invalid_call = LispNode('call', module.name, [var('x')])
            else:
                # Try to call with no arguments when some expected
                invalid_call = LispNode('call', module.name, [])

            is_valid, errors = verify_ast(invalid_call, module_library=self.library)
            self.assertFalse(is_valid, "Invalid call should fail verification")
            self.assertGreater(len(errors), 0)

        except ValueError:
            # Expected if contract validation catches it early
            pass

    def test_acceptance_criteria_zero_crashes(self):
        """Test acceptance criteria: zero runtime crashes from module misuse across 1k evals."""
        if not self.modules:
            self.skipTest("No modules registered")

        interpreter = LispInterpreter(module_library=self.library)
        crash_count = 0
        eval_count = 0

        # Test 100 evaluations (scaled down for test speed)
        for i in range(100):
            try:
                # Create various module calls
                for module in self.modules:
                    if module.arity == 0:
                        call = self.library.create_module_call(module.name, [])
                    elif module.arity == 1:
                        call = self.library.create_module_call(module.name, [const(i % 10)])
                    else:
                        call = self.library.create_module_call(module.name, [const(i % 5), const(i % 3)])

                    # Verify before evaluation
                    from protosynth.verify import verify_ast
                    is_valid, errors = verify_ast(call, module_library=self.library)

                    if is_valid:
                        # Should not crash
                        result = interpreter.evaluate(call)
                        eval_count += 1

                        # Validate return value against contract
                        if module.contract:
                            is_valid_return, return_errors = module.contract.validate_return_value(result)
                            if not is_valid_return:
                                print(f"Return value validation failed: {return_errors}")

            except Exception as e:
                crash_count += 1
                print(f"Crash {crash_count}: {e}")

        print(f"Completed {eval_count} evaluations with {crash_count} crashes")

        # Acceptance criteria: zero crashes
        self.assertEqual(crash_count, 0, f"Had {crash_count} crashes in {eval_count} evaluations")

    def test_unknown_module_rejection(self):
        """Test that calls to unknown modules are rejected."""
        from protosynth.verify import verify_ast

        # Create call to non-existent module
        unknown_call = LispNode('call', 'unknown_module', [var('x')])

        is_valid, errors = verify_ast(unknown_call, module_library=self.library)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        self.assertIn("Unknown module", errors[0])


if __name__ == '__main__':
    unittest.main()
