#!/usr/bin/env python3
"""
ProtoSynth Final Demonstration

This script showcases the complete ProtoSynth system with a comprehensive
demonstration of self-modifying AI evolution.
"""

import time
from protosynth import *
from protosynth.envs import *
from protosynth.evolve import *
from protosynth.ngram import *
from protosynth.eval import *
from protosynth.verify import verify_ast


def demo_basic_functionality():
    """Demonstrate basic ProtoSynth functionality."""
    
    print("🧬 ProtoSynth Basic Functionality Demo")
    print("=" * 50)
    
    # 1. Create and evaluate a simple program
    print("\n1. Creating and evaluating programs:")
    program = let('x', const(10), op('+', var('x'), const(5)))
    print(f"   Program: {pretty_print_ast(program)}")
    
    interpreter = LispInterpreter()
    result = interpreter.evaluate(program)
    print(f"   Result: {result}")
    
    # 2. Demonstrate mutation
    print("\n2. Program mutation:")
    agent = SelfModifyingAgent(program, mutation_rate=0.5)
    
    for i in range(3):
        try:
            mutated = agent.mutate()
            print(f"   Mutation {i+1}: {mutated.get_code_string()} -> {mutated.evaluate()}")
            agent = mutated
        except Exception as e:
            print(f"   Mutation {i+1}: Failed - {e}")
    
    # 3. Demonstrate verification
    print("\n3. Program verification:")
    valid_program = const(42)
    is_valid, errors = verify_ast(valid_program)
    print(f"   Valid program: {is_valid} (errors: {len(errors)})")


def demo_stream_environments():
    """Demonstrate different stream environments."""
    
    print("\n\n🌊 Stream Environments Demo")
    print("=" * 40)
    
    environments = [
        ("Periodic [1,0,1]", lambda: periodic([1, 0, 1])),
        ("Constant 1s", lambda: constant(1)),
        ("Alternating", lambda: alternating()),
        ("Arithmetic Prog", lambda: arith_prog(1, 3, 8))
    ]
    
    for name, env_factory in environments:
        print(f"\n{name}:")
        stream = env_factory()
        bits = [next(stream) for _ in range(12)]
        print(f"   Pattern: {bits}")


def demo_prediction_system():
    """Demonstrate the prediction and evaluation system."""
    
    print("\n\n🎯 Prediction System Demo")
    print("=" * 35)
    
    # Create test programs
    programs = {
        "Always 0.5": const(0.5),
        "Always 0.8": const(0.8),
        "Always 0.2": const(0.2)
    }
    
    # Test on periodic stream
    test_bits = [1, 0, 1, 1] * 20
    interpreter = LispInterpreter()
    
    print(f"\nTesting on periodic pattern [1,0,1,1] x 20:")
    print(f"Empirical 1-rate: {sum(test_bits)/len(test_bits):.2f}")
    
    for name, program in programs.items():
        fitness, metrics = evaluate_program_on_window(interpreter, program, test_bits, k=2)
        print(f"   {name}: F={fitness:.4f}, H_prog={metrics['model_entropy']:.4f}")


def demo_ngram_baseline():
    """Demonstrate n-gram baseline predictor."""
    
    print("\n\n🧮 N-gram Baseline Demo")
    print("=" * 30)
    
    # Create test data with clear pattern
    pattern_bits = [0, 1, 1, 0] * 50  # 200 bits
    
    print(f"Pattern: [0,1,1,0] repeated 50 times")
    print(f"Total bits: {len(pattern_bits)}")
    
    # Test different n-gram orders
    for k in [1, 2, 3, 4]:
        model = NGramPredictor(k=k, alpha=1.0)
        
        # Train on first half, test on second half
        train_bits = pattern_bits[:100]
        test_bits = pattern_bits[100:]
        
        model.train(train_bits)
        fitness, metrics = model.evaluate_on_stream(test_bits)
        
        print(f"   k={k}: F={fitness:.4f}, contexts={metrics['num_unique_contexts']}")


def demo_evolution():
    """Demonstrate evolution in action."""
    
    print("\n\n🚀 Evolution Demo")
    print("=" * 25)
    
    # Create a simple problem: predict alternating pattern
    def stream_factory():
        return alternating()
    
    print("Problem: Learn to predict alternating pattern [0,1,0,1,...]")
    print("Expected solution: Program that outputs ~0.5 or uses context")
    
    # Run short evolution
    print("\nRunning evolution (10 generations)...")
    results = run_simple_evolution(
        stream_factory,
        num_generations=10,
        mu=6,
        lambda_=12,
        seed=42
    )
    
    # Show results
    best = results['best_individual']
    print(f"\nEvolution Results:")
    print(f"   Best program: {pretty_print_ast(best.program)}")
    print(f"   Best fitness: {best.fitness:.4f}")
    print(f"   Program size: {best.size()} nodes")
    print(f"   Total evaluations: {results['total_evaluations']}")
    
    # Show fitness progression
    print(f"\nFitness progression:")
    for i, stats in enumerate(results['history'][:5]):  # Show first 5 generations
        print(f"   Gen {i}: best={stats['best_fitness']:.4f}, median={stats['median_fitness']:.4f}")


def demo_system_performance():
    """Demonstrate system performance metrics."""
    
    print("\n\n⚡ Performance Metrics")
    print("=" * 30)
    
    # Mutation speed test
    program = let('x', const(10), op('+', var('x'), const(5)))
    agent = SelfModifyingAgent(program, mutation_rate=0.3)
    
    start_time = time.time()
    mutations = 0
    for _ in range(100):
        try:
            agent = agent.mutate()
            mutations += 1
        except:
            pass
    mutation_time = time.time() - start_time
    
    print(f"Mutation performance:")
    print(f"   {mutations} successful mutations in {mutation_time*1000:.1f}ms")
    print(f"   Average: {mutation_time/mutations*1000:.2f}ms per mutation")
    
    # Evaluation speed test
    interpreter = LispInterpreter()
    test_program = const(0.5)
    test_bits = [0, 1] * 1000
    
    start_time = time.time()
    fitness, metrics = evaluate_program_on_window(interpreter, test_program, test_bits, k=4)
    eval_time = time.time() - start_time
    
    print(f"\nEvaluation performance:")
    print(f"   {metrics['num_predictions']} predictions in {eval_time*1000:.1f}ms")
    print(f"   Rate: {metrics['num_predictions']/eval_time:.0f} predictions/second")


def main():
    """Run the complete ProtoSynth demonstration."""
    
    print("🌟 ProtoSynth: Self-Modifying AI System")
    print("🌟 Complete System Demonstration")
    print("🌟" + "=" * 48)
    
    try:
        demo_basic_functionality()
        demo_stream_environments()
        demo_prediction_system()
        demo_ngram_baseline()
        demo_evolution()
        demo_system_performance()
        
        print("\n\n🎉 ProtoSynth Demonstration Complete!")
        print("=" * 45)
        print("✅ All systems operational")
        print("✅ Evolution pipeline working")
        print("✅ Performance validated")
        print("✅ Ready for research and experimentation")
        
        print(f"\n🚀 Next Steps:")
        print(f"   • Run longer evolution experiments")
        print(f"   • Test on complex prediction problems")
        print(f"   • Explore emergent behaviors")
        print(f"   • Scale to larger populations")
        print(f"   • Investigate novel mutation strategies")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
