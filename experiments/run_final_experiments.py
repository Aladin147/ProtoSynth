#!/usr/bin/env python3
"""
ProtoSynth Final Experiments Runner

Runs all final validation experiments:
A) Scaling laws sweep
B) Baseline comparisons  
C) Transfer learning
D) Adversarial environment

Total estimated runtime: 2-3 days
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
from typing import Dict, Any

# Import experiment modules
from scaling_laws import ScalingLawsExperiment
from baseline_comparison import BaselineComparison
from transfer_learning import TransferLearningExperiment
from adversarial_env import AdversarialExperiment


class FinalExperimentsRunner:
    """Master runner for all final experiments."""
    
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self.results = {}
    
    def run_experiment_a_scaling_laws(self) -> Dict[str, Any]:
        """Run Experiment A: Scaling Laws."""
        print("🔬 EXPERIMENT A: SCALING LAWS")
        print("=" * 50)
        print("Vary μ, λ, N, mutation_rate, verifier_strictness")
        print("Goal: Log slopes of best_F vs compute; identify smooth regime")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            experiment = ScalingLawsExperiment(base_seed=self.base_seed)
            results = experiment.run_experiment()
            
            # Success criteria
            analysis = results['analysis']
            success = (
                analysis['scaling_detected'] and
                analysis['correlation'] > 0.4 and
                results['success_rate'] > 0.5
            )
            
            results['experiment_success'] = success
            runtime = time.time() - start_time
            
            print(f"\n✅ Experiment A Complete: {'SUCCESS' if success else 'PARTIAL'}")
            print(f"Runtime: {runtime:.1f}s")
            
            return results
            
        except Exception as e:
            print(f"❌ Experiment A Failed: {e}")
            return {
                'experiment': 'scaling_laws',
                'experiment_success': False,
                'error': str(e),
                'runtime_seconds': time.time() - start_time
            }
    
    def run_experiment_b_baselines(self) -> Dict[str, Any]:
        """Run Experiment B: Baseline Comparisons."""
        print("\n🔬 EXPERIMENT B: BASELINE COMPARISONS")
        print("=" * 50)
        print("Add LZMA + tiny HMM on periodic(k), Markov(k), noisy(0.1)")
        print("Goal: match/exceed HMM; approach LZMA on periodic/low-k Markov")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            experiment = BaselineComparison(base_seed=self.base_seed)
            results = experiment.run_experiment()
            
            runtime = time.time() - start_time
            
            print(f"\n✅ Experiment B Complete: {'SUCCESS' if results['overall_success'] else 'PARTIAL'}")
            print(f"Runtime: {runtime:.1f}s")
            
            return results
            
        except Exception as e:
            print(f"❌ Experiment B Failed: {e}")
            return {
                'experiment': 'baseline_comparison',
                'overall_success': False,
                'error': str(e),
                'runtime_seconds': time.time() - start_time
            }
    
    def run_experiment_c_transfer(self) -> Dict[str, Any]:
        """Run Experiment C: Transfer Learning."""
        print("\n🔬 EXPERIMENT C: TRANSFER LEARNING")
        print("=" * 50)
        print("Freeze top-8 modules from Periodic/Markov")
        print("Goal: gens-to-F=0.3 speedup on Noisy/Markov↑ vs scratch")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            experiment = TransferLearningExperiment(base_seed=self.base_seed, target_fitness=0.3)
            results = experiment.run_experiment()
            
            runtime = time.time() - start_time
            
            print(f"\n✅ Experiment C Complete: {'SUCCESS' if results['overall_success'] else 'PARTIAL'}")
            print(f"Runtime: {runtime:.1f}s")
            
            return results
            
        except Exception as e:
            print(f"❌ Experiment C Failed: {e}")
            return {
                'experiment': 'transfer_learning',
                'overall_success': False,
                'error': str(e),
                'runtime_seconds': time.time() - start_time
            }
    
    def run_experiment_d_adversarial(self) -> Dict[str, Any]:
        """Run Experiment D: Adversarial Environment."""
        print("\n🔬 EXPERIMENT D: ADVERSARIAL ENVIRONMENT")
        print("=" * 50)
        print("Teacher mutates transition table under small budget")
        print("Goal: agent recovers ≥60% of baseline F after 50 gens")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            experiment = AdversarialExperiment(base_seed=self.base_seed)
            results = experiment.run_experiment()
            
            runtime = time.time() - start_time
            
            print(f"\n✅ Experiment D Complete: {'SUCCESS' if results['overall_success'] else 'PARTIAL'}")
            print(f"Runtime: {runtime:.1f}s")
            
            return results
            
        except Exception as e:
            print(f"❌ Experiment D Failed: {e}")
            return {
                'experiment': 'adversarial_environment',
                'overall_success': False,
                'error': str(e),
                'runtime_seconds': time.time() - start_time
            }
    
    def run_all_experiments(self, skip_experiments: list = None) -> Dict[str, Any]:
        """Run all final experiments."""
        print("🌟 PROTOSYNTH FINAL EXPERIMENTS")
        print("=" * 60)
        print("Comprehensive scientific validation suite")
        print("Estimated runtime: 2-3 days")
        print("=" * 60)
        
        overall_start_time = time.time()
        skip_experiments = skip_experiments or []
        
        # Run experiments
        experiments = [
            ('A', 'scaling_laws', self.run_experiment_a_scaling_laws),
            ('B', 'baselines', self.run_experiment_b_baselines),
            ('C', 'transfer', self.run_experiment_c_transfer),
            ('D', 'adversarial', self.run_experiment_d_adversarial),
        ]
        
        for exp_id, exp_name, exp_func in experiments:
            if exp_name in skip_experiments:
                print(f"\n⏭️  Skipping Experiment {exp_id}: {exp_name}")
                continue
            
            self.results[exp_name] = exp_func()
        
        # Overall analysis
        total_runtime = time.time() - overall_start_time
        
        completed_experiments = len(self.results)
        successful_experiments = sum(
            1 for result in self.results.values() 
            if result.get('experiment_success', result.get('overall_success', False))
        )
        
        overall_success_rate = successful_experiments / completed_experiments if completed_experiments > 0 else 0.0
        
        # Final summary
        print(f"\n🎯 FINAL EXPERIMENTS SUMMARY")
        print("=" * 40)
        print(f"Completed experiments: {completed_experiments}")
        print(f"Successful experiments: {successful_experiments}")
        print(f"Overall success rate: {overall_success_rate:.1%}")
        print(f"Total runtime: {total_runtime:.1f}s ({total_runtime/3600:.1f}h)")
        
        # Detailed results
        for exp_name, result in self.results.items():
            success = result.get('experiment_success', result.get('overall_success', False))
            runtime = result.get('runtime_seconds', 0)
            status = "✅ SUCCESS" if success else "❌ PARTIAL/FAILED"
            print(f"  {exp_name}: {status} ({runtime:.1f}s)")
        
        # Overall assessment
        if overall_success_rate >= 0.75:
            print(f"\n🌟 EXCELLENT: ProtoSynth passes rigorous scientific validation!")
        elif overall_success_rate >= 0.5:
            print(f"\n✅ GOOD: ProtoSynth demonstrates strong scientific validity")
        else:
            print(f"\n⚠️  PARTIAL: ProtoSynth shows promise but needs refinement")
        
        # Save comprehensive results
        final_results = {
            'experiment_suite': 'protosynth_final_validation',
            'timestamp': time.time(),
            'completed_experiments': completed_experiments,
            'successful_experiments': successful_experiments,
            'overall_success_rate': overall_success_rate,
            'total_runtime_seconds': total_runtime,
            'individual_results': self.results
        }
        
        with open('final_experiments_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n📁 Results saved to: final_experiments_results.json")
        
        return final_results
    
    def run_quick_validation(self) -> Dict[str, Any]:
        """Run a quick validation subset for testing."""
        print("🚀 QUICK VALIDATION MODE")
        print("=" * 30)
        print("Running subset of experiments for rapid validation")
        
        # Run only baseline comparison (fastest)
        self.results['baselines'] = self.run_experiment_b_baselines()
        
        success = self.results['baselines'].get('overall_success', False)
        
        print(f"\n🎯 Quick Validation: {'SUCCESS' if success else 'PARTIAL'}")
        
        return {
            'mode': 'quick_validation',
            'success': success,
            'results': self.results
        }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ProtoSynth final experiments')
    parser.add_argument('--quick', action='store_true', help='Run quick validation only')
    parser.add_argument('--skip', nargs='*', default=[], 
                       choices=['scaling_laws', 'baselines', 'transfer', 'adversarial'],
                       help='Skip specific experiments')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    
    args = parser.parse_args()
    
    runner = FinalExperimentsRunner(base_seed=args.seed)
    
    if args.quick:
        results = runner.run_quick_validation()
    else:
        results = runner.run_all_experiments(skip_experiments=args.skip)
    
    return results


if __name__ == "__main__":
    main()
