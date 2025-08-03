#!/usr/bin/env python3
"""Command-line interface for ProtoSynth."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

from .envs import get_stream_factory
from .evolve import EvolutionEngine, create_initial_population
from .utils import collect_run_metadata, format_metadata_summary


def evolve_main():
    """Main entry point for protosynth-evolve command."""
    parser = argparse.ArgumentParser(description="Evolve ProtoSynth programs")
    parser.add_argument(
        "--env", required=True, help="Environment name (periodic_k4, markov_k2)"
    )
    parser.add_argument("--gens", type=int, default=100, help="Number of generations")
    parser.add_argument("--mu", type=int, default=16, help="Population size")
    parser.add_argument(
        "--lambda", dest="lambda_", type=int, default=48, help="Offspring size"
    )
    parser.add_argument("--k", type=int, default=4, help="Context length")
    parser.add_argument("--N", type=int, default=2048, help="Evaluation length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-bundle", help="Save results bundle to path")
    parser.add_argument(
        "--determinism-check",
        action="store_true",
        help="Run determinism check (same seed twice)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Run determinism check if requested
    if args.determinism_check:
        print("Running determinism check...")
        _run_determinism_check(args)
        return

    # Create evolution engine
    engine = EvolutionEngine(
        mu=args.mu,
        lambda_=args.lambda_,
        seed=args.seed,
        k=args.k,
        N=args.N,
        env_name=args.env,
    )

    # Initialize population
    initial_programs = create_initial_population(args.mu, args.seed)
    engine.initialize_population(initial_programs)

    # Get stream factory
    stream_factory = get_stream_factory(args.env)

    print(f"ProtoSynth Evolution")
    print(f"Environment: {args.env}")
    print(f"Generations: {args.gens}")
    print(f"Population: μ={args.mu}, λ={args.lambda_}")
    print(f"Context: k={args.k}, N={args.N}")
    print(f"Seed: {args.seed}")
    print("-" * 40)

    start_time = time.time()
    best_fitness = -float("inf")

    try:
        for gen in range(args.gens):
            stream = stream_factory()
            metrics = engine.evolve_generation(stream)

            current_best = metrics.get("best_fitness", -float("inf"))
            if current_best > best_fitness:
                best_fitness = current_best

            if args.verbose or gen % 10 == 0:
                print(
                    f"Gen {gen:3d}: F_best={current_best:.6f}, "
                    f"F_med={metrics.get('median_fitness', 0):.6f}, "
                    f"ctx={metrics.get('context_fraction', 0):.2f}"
                )

            # Early stopping for successful learning
            target_fitness = 0.25 if "periodic" in args.env else 0.1
            if current_best >= target_fitness:
                print(
                    f"\nSUCCESS: Target F >= {target_fitness} reached in {gen+1} generations!"
                )
                break

    except KeyboardInterrupt:
        print("\nEvolution interrupted by user")

    elapsed = time.time() - start_time
    print(f"\nFinal best fitness: {best_fitness:.6f}")
    print(f"Time elapsed: {elapsed:.1f}s")

    # Get best program
    if engine.population:
        best_individual = max(engine.population, key=lambda x: x.fitness)
        print(f"Best program: {best_individual.program}")

        # Save bundle if requested
        if args.save_bundle:
            # Collect run metadata for reproducibility
            metadata = collect_run_metadata()
            metadata["timestamp"] = time.strftime(
                "%Y-%m-%d %H:%M:%S UTC", time.gmtime()
            )
            # Convert metrics to JSON-serializable format
            serializable_metrics = {}
            for key, value in best_individual.metrics.items():
                if isinstance(key, tuple):
                    key = str(key)  # Convert tuple keys to strings
                if isinstance(value, (dict, list, tuple)):
                    try:
                        json.dumps(value)  # Test if serializable
                        serializable_metrics[key] = value
                    except (TypeError, ValueError):
                        serializable_metrics[key] = str(
                            value
                        )  # Convert to string if not
                else:
                    serializable_metrics[key] = value

            bundle = {
                "metadata": metadata,
                "version": "0.1.0",
                "env": args.env,
                "config": {
                    "mu": args.mu,
                    "lambda_": args.lambda_,
                    "k": args.k,
                    "N": args.N,
                    "seed": args.seed,
                    "generations": args.gens,
                },
                "results": {
                    "best_fitness": best_fitness,
                    "final_generation": gen if "gen" in locals() else args.gens,
                    "elapsed_time": elapsed,
                },
                "best_program": str(best_individual.program),
                "best_metrics": serializable_metrics,
            }

            Path(args.save_bundle).parent.mkdir(parents=True, exist_ok=True)
            with open(args.save_bundle, "w") as f:
                json.dump(bundle, f, indent=2)
            print(f"Bundle saved to: {args.save_bundle}")


def replay_main():
    """Main entry point for protosynth-replay command."""
    parser = argparse.ArgumentParser(description="Replay ProtoSynth bundle")
    parser.add_argument("--bundle", required=True, help="Bundle JSON file to replay")

    args = parser.parse_args()

    try:
        with open(args.bundle, "r") as f:
            bundle = json.load(f)

        print(f"ProtoSynth Replay")
        print(f"Bundle: {args.bundle}")
        print(f"Version: {bundle.get('version', 'unknown')}")
        print(f"Environment: {bundle.get('env', 'unknown')}")
        print("-" * 40)

        config = bundle.get("config", {})
        results = bundle.get("results", {})

        print(f"Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        print(f"\nResults:")
        for key, value in results.items():
            print(f"  {key}: {value}")

        print(f"\nBest program:")
        print(f"  {bundle.get('best_program', 'N/A')}")

        best_metrics = bundle.get("best_metrics", {})
        if best_metrics:
            print(f"\nBest metrics:")
            for key, value in best_metrics.items():
                if isinstance(value, (int, float)):
                    print(
                        f"  {key}: {value:.6f}"
                        if isinstance(value, float)
                        else f"  {key}: {value}"
                    )
                else:
                    print(f"  {key}: {value}")

    except FileNotFoundError:
        print(f"Error: Bundle file not found: {args.bundle}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in bundle file: {args.bundle}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def info_main():
    """Main entry point for protosynth-info command."""
    parser = argparse.ArgumentParser(description="Show ProtoSynth system information")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    metadata = collect_run_metadata()

    if args.json:
        print(json.dumps(metadata, indent=2))
    else:
        print(format_metadata_summary(metadata))


def _run_determinism_check(args):
    """Run determinism check: same seed should produce identical results."""
    import hashlib

    print(f"Testing determinism with seed {args.seed}...")

    results = []
    for run in range(2):
        print(f"  Run {run + 1}/2...")

        # Create fresh engine for each run
        engine = EvolutionEngine(
            mu=args.mu,
            lambda_=args.lambda_,
            seed=args.seed,
            k=args.k,
            N=args.N,
            env_name=args.env,
        )

        # Initialize with same seed
        initial_pop = create_initial_population(args.mu, seed=args.seed)
        engine.initialize_population(initial_pop)

        # Run short evolution (5 generations for speed)
        stream_factory = get_stream_factory(args.env)
        for gen in range(5):
            stream = stream_factory()
            engine.evolve_generation(stream)

        # Get results
        best = max(engine.population, key=lambda x: x.fitness)
        result = {
            "fitness": best.fitness,
            "program_hash": hashlib.md5(str(best.program).encode()).hexdigest(),
            "program_str": str(best.program),
        }
        results.append(result)
        print(f"    Fitness: {result['fitness']:.6f}")
        print(f"    Program: {result['program_str']}")

    # Check determinism
    fitness_diff = abs(results[0]["fitness"] - results[1]["fitness"])
    programs_match = results[0]["program_hash"] == results[1]["program_hash"]

    print(f"\nDeterminism Check Results:")
    print(f"  Fitness difference: {fitness_diff:.2e}")
    print(f"  Programs identical: {programs_match}")

    if fitness_diff < 1e-9 and programs_match:
        print("  ✅ PASS: Results are deterministic")
    else:
        print("  ❌ FAIL: Results are not deterministic")
        print(
            f"  Run 1: F={results[0]['fitness']:.6f}, Hash={results[0]['program_hash'][:8]}"
        )
        print(
            f"  Run 2: F={results[1]['fitness']:.6f}, Hash={results[1]['program_hash'][:8]}"
        )
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "replay":
            # Remove 'replay' from args and call replay_main
            sys.argv.pop(1)
            replay_main()
        elif sys.argv[1] == "info":
            # Remove 'info' from args and call info_main
            sys.argv.pop(1)
            info_main()
        else:
            evolve_main()
    else:
        evolve_main()
