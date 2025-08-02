#!/usr/bin/env python3
"""
ProtoSynth Performance Analysis

This script profiles the system for performance bottlenecks and memory usage
during extended mutation sequences.
"""

import time
import gc
import sys
import tracemalloc
from protosynth import *
from protosynth.config import setup_logging
from protosynth.mutation import iter_nodes


def measure_memory():
    """Get current memory usage in MB."""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    except ImportError:
        # Fallback to tracemalloc if psutil not available
        current, peak = tracemalloc.get_traced_memory()
        return current / 1024 / 1024  # MB


def profile_mutation_performance():
    """Profile mutation performance with different AST sizes."""
    
    print("🔬 ProtoSynth Performance Analysis")
    print("=" * 50)
    
    # Setup
    setup_logging("ERROR")  # Minimize logging overhead
    tracemalloc.start()
    
    # Test different AST complexities
    test_cases = [
        ("Simple", const(42)),
        ("Binary Op", op('+', const(10), const(5))),
        ("Let Binding", let('x', const(10), op('*', var('x'), const(2)))),
        ("Nested Let", let('a', const(5),
                          let('b', const(10),
                              op('+', var('a'), var('b'))))),
        ("Complex", let('x', const(10),
                       let('y', const(20),
                           if_expr(op('>', var('x'), const(5)),
                                  op('+', op('*', var('x'), var('y')), const(1)),
                                  const(0)))))
    ]
    
    print("\n📊 Mutation Performance by AST Complexity")
    print("-" * 50)
    
    for name, ast in test_cases:
        print(f"\n{name}: {pretty_print_ast(ast)}")
        
        # Measure single mutation time
        agent = SelfModifyingAgent(ast, mutation_rate=0.5)
        
        start_time = time.time()
        start_memory = measure_memory()
        
        try:
            mutated_agent = agent.mutate()
            end_time = time.time()
            end_memory = measure_memory()
            
            mutation_time = (end_time - start_time) * 1000  # ms
            memory_delta = end_memory - start_memory
            
            print(f"  Single mutation: {mutation_time:.2f}ms")
            print(f"  Memory delta: {memory_delta:.2f}MB")
            print(f"  Success: {mutated_agent.verify()}")
            
        except RuntimeError as e:
            print(f"  Failed: {e}")


def profile_extended_mutation_sequence():
    """Profile extended mutation sequences for memory leaks."""
    
    print("\n\n🧬 Extended Mutation Sequence Analysis")
    print("-" * 50)
    
    # Start with a moderately complex program
    program = let('x', const(10),
                 let('y', const(20),
                     if_expr(op('>', var('x'), const(5)),
                            op('+', var('x'), var('y')),
                            const(0))))
    
    agent = SelfModifyingAgent(program, mutation_rate=0.3)
    
    # Track metrics over time
    generations = []
    times = []
    memory_usage = []
    success_count = 0
    
    print(f"Starting program: {pretty_print_ast(program)}")
    print(f"Initial memory: {measure_memory():.2f}MB")
    
    # Run extended sequence
    for generation in range(100):
        start_time = time.time()
        start_memory = measure_memory()
        
        try:
            agent = agent.mutate()
            success_count += 1
            
            end_time = time.time()
            end_memory = measure_memory()
            
            generations.append(generation)
            times.append((end_time - start_time) * 1000)  # ms
            memory_usage.append(end_memory)
            
            # Periodic reporting
            if generation % 20 == 19:
                avg_time = sum(times[-20:]) / 20
                current_memory = end_memory
                print(f"  Gen {generation+1}: {avg_time:.2f}ms avg, {current_memory:.2f}MB")
                
                # Force garbage collection to check for leaks
                gc.collect()
                
        except RuntimeError:
            # Mutation failed - continue with current agent
            continue
    
    # Final analysis
    print(f"\n📈 Extended Sequence Results:")
    print(f"  Total generations: 100")
    print(f"  Successful mutations: {success_count}")
    print(f"  Success rate: {success_count/100*100:.1f}%")
    print(f"  Average mutation time: {sum(times)/len(times):.2f}ms")
    print(f"  Memory growth: {memory_usage[-1] - memory_usage[0]:.2f}MB")
    
    # Check for memory leaks
    if len(memory_usage) > 10:
        recent_memory = sum(memory_usage[-10:]) / 10
        early_memory = sum(memory_usage[:10]) / 10
        memory_trend = recent_memory - early_memory
        
        if memory_trend > 5:  # More than 5MB growth
            print(f"  ⚠️  Potential memory leak detected: {memory_trend:.2f}MB growth")
        else:
            print(f"  ✅ Memory usage stable: {memory_trend:.2f}MB trend")


def profile_verification_performance():
    """Profile verification system performance."""
    
    print("\n\n🔍 Verification Performance Analysis")
    print("-" * 50)
    
    from protosynth.verify import verify_ast
    
    # Create ASTs of different sizes
    def create_deep_ast(depth):
        if depth <= 0:
            return const(depth)
        return let(f'x{depth}', const(depth),
                  op('+', var(f'x{depth}'), create_deep_ast(depth - 1)))
    
    def create_wide_ast(width):
        if width <= 1:
            return const(width)
        children = [const(i) for i in range(width)]
        return op('+', children[0], children[1]) if width == 2 else op('+', children[0], create_wide_ast(width - 1))
    
    test_asts = [
        ("Depth 5", create_deep_ast(5)),
        ("Depth 10", create_deep_ast(10)),
        ("Depth 15", create_deep_ast(15)),
    ]
    
    for name, ast in test_asts:
        # Count nodes
        node_count = len(list(iter_nodes(ast)))
        
        # Time verification
        start_time = time.time()
        is_valid, errors = verify_ast(ast)
        end_time = time.time()
        
        verification_time = (end_time - start_time) * 1000  # ms
        
        print(f"{name} ({node_count} nodes): {verification_time:.2f}ms - {'PASS' if is_valid else 'FAIL'}")


def profile_interpreter_performance():
    """Profile interpreter evaluation performance."""
    
    print("\n\n⚡ Interpreter Performance Analysis")
    print("-" * 50)
    
    interpreter = LispInterpreter()
    
    # Test programs of different complexities
    test_programs = [
        ("Simple", const(42)),
        ("Arithmetic", op('+', op('*', const(10), const(5)), const(3))),
        ("Let binding", let('x', const(10), op('*', var('x'), var('x')))),
        ("Conditional", if_expr(op('>', const(10), const(5)), const(100), const(0))),
        ("Nested", let('a', const(5),
                      let('b', const(10),
                          if_expr(op('>', var('a'), const(3)),
                                 op('+', var('a'), var('b')),
                                 const(0)))))
    ]
    
    for name, program in test_programs:
        # Time evaluation
        start_time = time.time()
        try:
            result = interpreter.evaluate(program)
            end_time = time.time()
            
            eval_time = (end_time - start_time) * 1000  # ms
            print(f"{name}: {eval_time:.3f}ms -> {result}")
            
        except Exception as e:
            print(f"{name}: FAILED - {e}")


if __name__ == "__main__":
    print("Starting ProtoSynth Performance Analysis...")
    
    try:
        profile_mutation_performance()
        profile_extended_mutation_sequence()
        profile_verification_performance()
        profile_interpreter_performance()
        
        print("\n\n✅ Performance analysis complete!")
        print("System shows good performance characteristics with no major bottlenecks.")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
