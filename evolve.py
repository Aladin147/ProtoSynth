#!/usr/bin/env python3
"""
ProtoSynth Evolution CLI

Command-line interface for running evolution experiments.

Usage:
    python evolve.py --env periodic --k 4 --gens 200 --mu 16 --lambda 48
    python evolve.py --env markov --gens 100 --seed 42
    python evolve.py --help
"""

import argparse
import json
import time
from pathlib import Path
from protosynth.envs import *
from protosynth.evolve import run_simple_evolution, run_evolution_with_overfitting_guard
from protosynth.ngram import NGramPredictor, compare_ngram_orders
from protosynth import pretty_print_ast


def create_stream_factory(env_type: str, **kwargs):
    """Create a stream factory function based on environment type."""
    
    if env_type == "periodic":
        pattern = kwargs.get("pattern", [1, 0, 1, 1])
        seed = kwargs.get("seed", 0)
        return lambda: periodic(pattern, seed)
    
    elif env_type == "markov":
        k = kwargs.get("k", 1)
        trans = kwargs.get("trans", {(0,): 0.8, (1,): 0.2})
        seed = kwargs.get("seed", 0)
        return lambda: k_order_markov(k, trans, seed)
    
    elif env_type == "arith":
        a = kwargs.get("a", 1)
        d = kwargs.get("d", 3)
        mod = kwargs.get("mod", 8)
        seed = kwargs.get("seed", 0)
        return lambda: arith_prog(a, d, mod, seed)
    
    elif env_type == "constant":
        value = kwargs.get("value", 1)
        seed = kwargs.get("seed", 0)
        return lambda: constant(value, seed)
    
    elif env_type == "alternating":
        seed = kwargs.get("seed", 0)
        return lambda: alternating(seed)
    
    else:
        raise ValueError(f"Unknown environment type: {env_type}")


def run_ngram_baseline(stream_factory, k_values=[2, 4, 6], alpha=1.0):
    """Run n-gram baseline comparison."""
    
    print(f"\n🧮 N-gram Baseline Comparison")
    print("=" * 50)
    
    # Collect test data
    test_bits = []
    stream = stream_factory()
    for i, bit in enumerate(stream):
        test_bits.append(bit)
        if i >= 5000:
            break
    
    print(f"Collected {len(test_bits)} bits for evaluation")
    
    # Compare different n-gram orders
    results = compare_ngram_orders(test_bits, max_k=max(k_values), alpha=alpha)
    
    print(f"\nN-gram Results (α={alpha}):")
    for k in sorted(results.keys()):
        fitness, metrics = results[k]
        print(f"  k={k}: F={fitness:.4f}, H_prog={metrics['model_entropy']:.4f}, "
              f"contexts={metrics['num_unique_contexts']}")
    
    return results


def save_results(results: dict, output_file: str):
    """Save evolution results to file."""
    
    # Convert results to JSON-serializable format
    json_results = {
        'best_fitness': results['best_individual'].fitness if results['best_individual'] else None,
        'best_program': pretty_print_ast(results['best_individual'].program) if results['best_individual'] else None,
        'best_size': results['best_individual'].size() if results['best_individual'] else None,
        'final_generation': results['history'][-1]['generation'] if results['history'] else 0,
        'total_evaluations': results['total_evaluations'],
        'total_mutations': results['total_mutations'],
        'history': results['history'],
        'summary': results['summary']
    }
    
    # Add validation history if present
    if 'validation_history' in results:
        json_results['validation_history'] = results['validation_history']
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {output_file}")


def print_evolution_summary(results: dict):
    """Print a summary of evolution results."""
    
    print(f"\n🎯 Evolution Summary")
    print("=" * 40)
    
    best = results['best_individual']
    if best:
        print(f"Best Individual:")
        print(f"  Program: {pretty_print_ast(best.program)}")
        print(f"  Fitness: {best.fitness:.4f}")
        print(f"  Size: {best.size()} nodes")
        print(f"  Generation: {best.generation}")
    
    print(f"\nStatistics:")
    print(f"  Total evaluations: {results['total_evaluations']}")
    print(f"  Total mutations: {results['total_mutations']}")
    print(f"  Generations: {len(results['history'])}")
    
    if results['history']:
        final_stats = results['history'][-1]
        print(f"  Final best fitness: {final_stats['best_fitness']:.4f}")
        print(f"  Final median fitness: {final_stats['median_fitness']:.4f}")
        print(f"  Final best size: {final_stats['best_size']}")


def main():
    parser = argparse.ArgumentParser(description="ProtoSynth Evolution CLI")
    
    # Environment parameters
    parser.add_argument("--env", default="periodic", 
                       choices=["periodic", "markov", "arith", "constant", "alternating"],
                       help="Environment type")
    parser.add_argument("--pattern", default="1011", 
                       help="Pattern for periodic environment (e.g., '1011')")
    parser.add_argument("--k", type=int, default=6,
                       help="Context length for prediction")
    
    # Evolution parameters
    parser.add_argument("--gens", type=int, default=50,
                       help="Number of generations")
    parser.add_argument("--mu", type=int, default=16,
                       help="Population size (number of elites)")
    parser.add_argument("--lambda", type=int, default=64, dest="lambda_",
                       help="Number of offspring per generation")
    parser.add_argument("--mutation-rate", type=float, default=0.10,
                       help="Mutation rate per node")
    parser.add_argument("--N", type=int, default=4096,
                       help="Number of symbols to evaluate on")
    
    # Experiment parameters
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--output", default=None,
                       help="Output file for results (JSON)")
    parser.add_argument("--overfitting-guard", action="store_true",
                       help="Use overfitting guard with train/val split")
    parser.add_argument("--baseline", action="store_true",
                       help="Run n-gram baseline comparison")
    
    args = parser.parse_args()
    
    print(f"🧬 ProtoSynth Evolution Experiment")
    print(f"Environment: {args.env}")
    print(f"Context length: {args.k}")
    print(f"Generations: {args.gens}")
    print(f"Population: μ={args.mu}, λ={args.lambda_}")
    print(f"Seed: {args.seed}")
    print("=" * 50)
    
    # Create stream factory
    if args.env == "periodic":
        pattern = [int(b) for b in args.pattern]
        stream_factory = create_stream_factory("periodic", pattern=pattern, seed=args.seed or 0)
    else:
        stream_factory = create_stream_factory(args.env, seed=args.seed or 0)
    
    # Run n-gram baseline if requested
    if args.baseline:
        ngram_results = run_ngram_baseline(stream_factory)
    
    # Run evolution
    start_time = time.time()
    
    if args.overfitting_guard:
        print(f"\n🛡️  Running evolution with overfitting guard...")
        results = run_evolution_with_overfitting_guard(
            stream_factory,
            num_generations=args.gens,
            mu=args.mu,
            lambda_=args.lambda_,
            seed=args.seed
        )
    else:
        print(f"\n🚀 Running standard evolution...")
        results = run_simple_evolution(
            stream_factory,
            num_generations=args.gens,
            mu=args.mu,
            lambda_=args.lambda_,
            seed=args.seed
        )
    
    evolution_time = time.time() - start_time
    print(f"\nEvolution completed in {evolution_time:.1f} seconds")
    
    # Print summary
    print_evolution_summary(results)
    
    # Save results if requested
    if args.output:
        save_results(results, args.output)
    
    print(f"\n✅ Experiment complete!")


if __name__ == "__main__":
    main()
