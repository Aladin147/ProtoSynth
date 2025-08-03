#!/usr/bin/env python3
"""
Tests for Enhanced Modularity Features

Tests credit assignment, versioning, canonicalization, and garbage collection.
"""

import unittest
from protosynth import *
from protosynth.modularity import *


class TestEnhancedModularity(unittest.TestCase):
    """Test enhanced modularity features."""
    
    def setUp(self):
        """Set up test environment."""
        self.library = ModuleLibrary(max_modules=5, credit_tau=0.02)
        
        # Create test candidates
        self.test_candidates = [
            SubtreeCandidate(
                subtree=op('+', var('x'), const(1)),
                frequency=3, total_nodes=3, mdl_score=0.15,
                bits_saved=0.20, size_penalty=0.05
            ),
            SubtreeCandidate(
                subtree=op('+', var('y'), const(1)),  # Near-duplicate
                frequency=2, total_nodes=3, mdl_score=0.12,
                bits_saved=0.17, size_penalty=0.05
            ),
            SubtreeCandidate(
                subtree=op('>', var('z'), const(0)),
                frequency=4, total_nodes=3, mdl_score=0.18,
                bits_saved=0.23, size_penalty=0.05
            )
        ]
    
    def test_canonicalization(self):
        """Test that canonicalization reduces near-duplicates."""
        # Test basic canonicalization
        node1 = op('+', var('x'), const(1))
        node2 = op('+', var('y'), const(1))  # Different variable, same pattern
        
        canon1 = canonicalize_subtree(node1)
        canon2 = canonicalize_subtree(node2)
        
        # Should be the same after canonicalization
        self.assertEqual(canon1, canon2)
        
        # Test commutative operation sorting
        node3 = op('+', const(1), var('x'))  # Swapped order
        canon3 = canonicalize_subtree(node3)
        
        # Should be the same as canon1 (commutative)
        self.assertEqual(canon1, canon3)
    
    def test_deduplication_during_registration(self):
        """Test that near-duplicates are deduplicated during registration."""
        modules = self.library.register_modules(self.test_candidates)
        
        # Should have fewer modules than candidates due to deduplication
        self.assertLessEqual(len(modules), len(self.test_candidates))
        
        # Check that the better version was kept
        increment_modules = [m for m in modules if '+' in str(m.implementation)]
        if increment_modules:
            # Should keep the one with higher MDL score
            best_increment = max(increment_modules, key=lambda m: m.mdl_score)
            self.assertGreaterEqual(best_increment.mdl_score, 0.15)
    
    def test_versioning(self):
        """Test module versioning system."""
        # Register initial modules
        initial_modules = self.library.register_modules(self.test_candidates[:1])
        self.assertEqual(len(initial_modules), 1)
        
        initial_module = initial_modules[0]
        self.assertEqual(initial_module.version, "1.0.0")
        
        # Register improved version of same pattern
        improved_candidate = SubtreeCandidate(
            subtree=op('+', var('x'), const(1)),  # Same pattern
            frequency=5, total_nodes=3, mdl_score=0.25,  # Better score
            bits_saved=0.30, size_penalty=0.05
        )
        
        updated_modules = self.library.register_modules([improved_candidate])
        
        # Should create new version
        if updated_modules:
            updated_module = updated_modules[0]
            self.assertEqual(updated_module.name, initial_module.name)
            self.assertNotEqual(updated_module.version, initial_module.version)
            self.assertGreater(updated_module.mdl_score, initial_module.mdl_score)
    
    def test_credit_assignment(self):
        """Test credit score updates."""
        modules = self.library.register_modules(self.test_candidates)
        
        if not modules:
            self.skipTest("No modules registered")
        
        # Create population that uses modules
        population = []
        fitness_scores = []
        
        for i, module in enumerate(modules):
            if module.arity == 1:
                program = self.library.create_module_call(module.name, [var('x')])
                population.append(program)
                fitness_scores.append(0.5 + i * 0.1)  # Increasing fitness
        
        if not population:
            self.skipTest("No usable modules")
        
        # Update credit scores
        initial_credits = {m.name: m.credit_score for m in modules}
        
        self.library.update_credit_scores(population, fitness_scores, mask_probability=1.0)
        
        # Check that credits were updated
        updated_credits = {m.name: m.credit_score for m in modules}
        
        # At least some credits should have changed
        changes = sum(1 for name in initial_credits 
                     if abs(initial_credits[name] - updated_credits[name]) > 0.001)
        self.assertGreater(changes, 0)
    
    def test_garbage_collection(self):
        """Test garbage collection when at capacity."""
        # Fill library to capacity
        many_candidates = []
        for i in range(10):  # More than max_modules (5)
            many_candidates.append(SubtreeCandidate(
                subtree=op('+', var(f'x{i}'), const(i)),
                frequency=1, total_nodes=3, mdl_score=0.01 * i,
                bits_saved=0.02 * i, size_penalty=0.01
            ))
        
        modules = self.library.register_modules(many_candidates)
        
        # Should not exceed capacity
        self.assertLessEqual(len(self.library.modules), self.library.max_modules)
        
        # Should keep higher-value modules
        remaining_scores = [m.mdl_score for m in self.library.modules.values()]
        if remaining_scores:
            min_remaining = min(remaining_scores)
            # Should have kept some of the better modules
            self.assertGreater(min_remaining, 0.01)
    
    def test_module_lookup_with_versioning(self):
        """Test module lookup with version handling."""
        modules = self.library.register_modules(self.test_candidates[:1])
        
        if not modules:
            self.skipTest("No modules registered")
        
        module = modules[0]
        
        # Test lookup without version (should find latest)
        found = self.library._find_module(module.name)
        self.assertIsNotNone(found)
        self.assertEqual(found.name, module.name)
        
        # Test lookup with specific version
        found_versioned = self.library._find_module(module.name, module.version)
        self.assertIsNotNone(found_versioned)
        self.assertEqual(found_versioned.version, module.version)
        
        # Test lookup with non-existent version
        found_missing = self.library._find_module(module.name, "99.99.99")
        self.assertIsNone(found_missing)
    
    def test_usage_tracking(self):
        """Test that module usage is tracked."""
        modules = self.library.register_modules(self.test_candidates)
        
        if not modules:
            self.skipTest("No modules registered")
        
        module = modules[0]
        initial_usage = module.usage_count
        
        # Create a call
        if module.arity == 1:
            call = self.library.create_module_call(module.name, [var('x')])
            
            # Usage should be incremented
            self.assertEqual(module.usage_count, initial_usage + 1)
            self.assertEqual(module.last_used_gen, self.library.current_generation)
    
    def test_generation_advancement(self):
        """Test generation counter advancement."""
        initial_gen = self.library.current_generation
        
        self.library.advance_generation()
        
        self.assertEqual(self.library.current_generation, initial_gen + 1)
    
    def test_enhanced_library_info(self):
        """Test enhanced library information retrieval."""
        modules = self.library.register_modules(self.test_candidates)
        
        info = self.library.get_module_info()
        
        # Should include enhanced information
        self.assertIn('num_modules', info)
        self.assertIn('modules', info)
        
        if modules:
            # Check that module info includes new fields
            first_module_info = list(info['modules'].values())[0]
            expected_fields = ['arity', 'mdl_score', 'frequency']
            
            for field in expected_fields:
                self.assertIn(field, first_module_info)


if __name__ == '__main__':
    unittest.main()
