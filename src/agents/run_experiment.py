"""
Main Experiment Script for Problem 2.

This script runs the complete LLM agent experiment to validate
the Hammer-Spammer model algorithm.

Usage:
    python -m src.agents.run_experiment --num-questions 50 --dataset gsm8k
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.dataset import DatasetLoader
from src.agents.agent_config import AGENT_CONFIGS
from src.agents.agent_system import AgentSystem
from src.algorithm.hammer_spammer import HammerSpammerRanking
from src.evaluation.validator import Validator
from src.evaluation.baselines import Baselines


class ExperimentRunner:
    """Main experiment runner for Problem 2."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize experiment runner.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.results = []
        self.start_time = None
        self.end_time = None

        # Initialize components
        print("Initializing experiment components...")

        # Load API key
        load_dotenv()
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")

        # Initialize dataset
        print(f"\nLoading dataset: {config['dataset']}")
        self.dataset = DatasetLoader(
            dataset_name=config['dataset'],
            split=config['split'],
            max_samples=config['num_questions']
        )

        # Initialize agent system
        print(f"\nInitializing {config['num_agents']} agents...")
        agent_configs = AGENT_CONFIGS[:config['num_agents']]
        self.agent_system = AgentSystem(
            agent_configs=agent_configs,
            api_key=api_key,
            parallel_generation=config['parallel_generation'],
            parallel_comparison=config['parallel_comparison']
        )

        # Initialize algorithm
        print(f"\nInitializing Hammer-Spammer algorithm (beta={config['beta']})...")
        self.algorithm = HammerSpammerRanking(
            beta=config['beta'],
            epsilon=config['epsilon']
        )

        # Initialize validator
        print(f"\nInitializing validator (oracle={config['oracle_model']})...")
        self.validator = Validator(
            oracle_model=config['oracle_model'],
            api_key=api_key
        )

        print("\n" + "="*60)
        print("Initialization complete!")
        print("="*60)

    def run_single_question(self, question_idx: int) -> Dict[str, Any]:
        """
        Run experiment for a single question.

        Args:
            question_idx: Index of question in dataset

        Returns:
            Results dictionary for this question
        """
        # Get question and ground truth
        question, ground_truth = self.dataset.get_question(question_idx)

        print(f"\nQuestion {question_idx + 1}/{len(self.dataset)}")
        print(f"Q: {question[:100]}...")

        # Run agent system to get answers and comparison matrix
        experiment_data = self.agent_system.run_experiment(
            question=question,
            ground_truth=ground_truth,
            verbose=self.config['verbose']
        )

        R = experiment_data['comparison_matrix']
        answers = experiment_data['answers']
        answer_strings = [a['answer'] for a in answers]

        # Apply Hammer-Spammer algorithm
        if self.config['verbose']:
            print("\nApplying Hammer-Spammer algorithm...")

        algo_idx, quality_scores = self.algorithm.select_best(R, return_scores=True)
        algo_answer = answer_strings[algo_idx]

        # Apply baselines
        if self.config['verbose']:
            print("Applying baseline methods...")

        baseline_selections = Baselines.apply_all_baselines(
            answers=answer_strings,
            comparison_matrix=R
        )

        # Validate algorithm selection with ground truth
        if self.config['verbose']:
            print("Validating with ground truth...")

        algo_validation = self.validator.validate_with_ground_truth(
            selected_answer=algo_answer,
            ground_truth=ground_truth
        )

        # Validate baselines
        baseline_validations = {}
        for baseline_name, baseline_idx in baseline_selections.items():
            baseline_answer = answer_strings[baseline_idx]
            baseline_validations[baseline_name] = self.validator.validate_with_ground_truth(
                selected_answer=baseline_answer,
                ground_truth=ground_truth
            )

        # Oracle validation (if enabled)
        oracle_validation = None
        if self.config['use_oracle']:
            if self.config['verbose']:
                print("Validating with oracle model...")

            try:
                oracle_validation = self.validator.validate_with_oracle(
                    question=question,
                    selected_answer=algo_answer,
                    all_answers=answer_strings
                )
            except Exception as e:
                print(f"Oracle validation failed: {e}")
                oracle_validation = None

        # Compile results
        result = {
            'question_idx': question_idx,
            'question': question,
            'ground_truth': ground_truth,
            'answers': answers,
            'comparison_matrix': R.tolist(),
            'quality_scores': quality_scores.tolist(),

            # Algorithm results
            'algorithm_selected_idx': algo_idx,
            'algorithm_answer': algo_answer,
            'algorithm_correct': algo_validation['correct'],
            'algorithm_validation': algo_validation,

            # Baseline results
            'baseline_selections': baseline_selections,
            'baseline_validations': baseline_validations,

            # Oracle results
            'oracle_validation': oracle_validation,

            # Metadata
            'timestamp': experiment_data['timestamp'],
            'elapsed_time': experiment_data['elapsed_time']
        }

        return result

    def run_all_questions(self):
        """Run experiment on all questions."""
        self.start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Running experiment on {len(self.dataset)} questions")
        print(f"{'='*60}\n")

        num_questions = len(self.dataset)

        for i in range(num_questions):
            try:
                result = self.run_single_question(i)
                self.results.append(result)

                # Print summary
                print(f"\nResult:")
                print(f"  Algorithm: {'✓' if result['algorithm_correct'] else '✗'} (Answer: {result['algorithm_answer']})")
                print(f"  Ground truth: {result['ground_truth']}")

                for baseline, validation in result['baseline_validations'].items():
                    symbol = '✓' if validation['correct'] else '✗'
                    print(f"  {baseline}: {symbol}")

                # Save intermediate results
                if (i + 1) % self.config['save_frequency'] == 0:
                    self.save_results(intermediate=True)

            except Exception as e:
                print(f"\nError processing question {i}: {e}")
                import traceback
                traceback.print_exc()
                continue

        self.end_time = time.time()

        print(f"\n{'='*60}")
        print(f"Experiment completed!")
        print(f"Total time: {self.end_time - self.start_time:.1f}s")
        print(f"{'='*60}\n")

    def compute_summary_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics from results."""
        if not self.results:
            return {}

        num_questions = len(self.results)

        # Algorithm performance
        algo_correct = sum(1 for r in self.results if r['algorithm_correct'])
        algo_accuracy = algo_correct / num_questions * 100

        # Baseline performance
        baseline_accuracies = {}
        for baseline_name in self.results[0]['baseline_validations'].keys():
            correct = sum(1 for r in self.results
                         if r['baseline_validations'][baseline_name]['correct'])
            baseline_accuracies[baseline_name] = correct / num_questions * 100

        # Oracle performance (if available)
        oracle_stats = None
        if self.config['use_oracle']:
            oracle_results = [r['oracle_validation'] for r in self.results
                            if r['oracle_validation'] is not None]
            if oracle_results:
                oracle_stats = {
                    'avg_score': np.mean([o['oracle_score'] for o in oracle_results]),
                    'matches_oracle': sum(1 for o in oracle_results if o['matches_oracle']),
                    'oracle_agreement_rate': sum(1 for o in oracle_results if o['matches_oracle']) / len(oracle_results) * 100
                }

        # Agent statistics
        agent_stats = self.agent_system.get_all_stats()

        return {
            'num_questions': num_questions,
            'algorithm_accuracy': algo_accuracy,
            'algorithm_correct': algo_correct,
            'baseline_accuracies': baseline_accuracies,
            'oracle_stats': oracle_stats,
            'agent_stats': agent_stats,
            'total_time': self.end_time - self.start_time if self.end_time else None
        }

    def print_summary(self):
        """Print experiment summary."""
        stats = self.compute_summary_statistics()

        print(f"\n{'='*60}")
        print(f"EXPERIMENT SUMMARY")
        print(f"{'='*60}\n")

        print(f"Dataset: {self.config['dataset']}")
        print(f"Number of questions: {stats['num_questions']}")
        print(f"Number of agents: {self.config['num_agents']}")
        print(f"Total time: {stats['total_time']:.1f}s\n")

        print(f"{'='*60}")
        print(f"ACCURACY RESULTS")
        print(f"{'='*60}\n")

        print(f"Hammer-Spammer Algorithm: {stats['algorithm_accuracy']:.2f}% ({stats['algorithm_correct']}/{stats['num_questions']})")
        print()

        # Sort baselines by accuracy
        sorted_baselines = sorted(stats['baseline_accuracies'].items(),
                                 key=lambda x: x[1], reverse=True)

        print("Baseline Methods:")
        for baseline, accuracy in sorted_baselines:
            print(f"  {baseline:<25}: {accuracy:.2f}%")

        # Oracle stats
        if stats['oracle_stats']:
            print(f"\n{'='*60}")
            print(f"ORACLE VALIDATION")
            print(f"{'='*60}\n")
            print(f"Average oracle score: {stats['oracle_stats']['avg_score']:.2f}/10")
            print(f"Oracle agreement rate: {stats['oracle_stats']['oracle_agreement_rate']:.2f}%")

        # Agent usage stats
        print(f"\n{'='*60}")
        print(f"AGENT USAGE STATISTICS")
        print(f"{'='*60}\n")

        total_tokens = sum(a['total_tokens_used'] for a in stats['agent_stats'])
        total_calls = sum(a['total_calls'] for a in stats['agent_stats'])

        print(f"Total API calls: {total_calls:,}")
        print(f"Total tokens: {total_tokens:,}")

    def save_results(self, intermediate: bool = False):
        """
        Save results to file.

        Args:
            intermediate: Whether this is an intermediate save
        """
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "intermediate_" if intermediate else "final_"
        filename = f"{prefix}results_{timestamp}.json"
        filepath = output_dir / filename

        # Prepare data
        data = {
            'config': self.config,
            'results': self.results,
            'summary': self.compute_summary_statistics()
        }

        # Save
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to: {filepath}")

        return filepath


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Hammer-Spammer validation experiment")

    # Dataset options
    parser.add_argument('--dataset', type=str, default='gsm8k',
                       choices=['gsm8k', 'math', 'math500'],
                       help='Dataset to use')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'test'],
                       help='Dataset split')
    parser.add_argument('--num-questions', type=int, default=50,
                       help='Number of questions to run')

    # Agent options
    parser.add_argument('--num-agents', type=int, default=15,
                       help='Number of agents to use')
    parser.add_argument('--no-parallel-generation', action='store_true',
                       help='Disable parallel answer generation')
    parser.add_argument('--no-parallel-comparison', action='store_true',
                       help='Disable parallel comparisons')

    # Algorithm options
    parser.add_argument('--beta', type=float, default=5.0,
                       help='Quality gap parameter')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Minimum quality threshold')

    # Validation options
    parser.add_argument('--oracle-model', type=str, default='openai/gpt-4o',
                       help='Oracle model for validation')
    parser.add_argument('--no-oracle', action='store_true',
                       help='Disable oracle validation')

    # Output options
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--save-frequency', type=int, default=10,
                       help='Save intermediate results every N questions')

    # Other options
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode (use 3 questions)')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Override for debug mode
    if args.debug:
        args.num_questions = 3
        args.verbose = True
        print("DEBUG MODE: Running with 3 questions only")

    # Build config
    config = {
        'dataset': args.dataset,
        'split': args.split,
        'num_questions': args.num_questions,
        'num_agents': args.num_agents,
        'parallel_generation': not args.no_parallel_generation,
        'parallel_comparison': not args.no_parallel_comparison,
        'beta': args.beta,
        'epsilon': args.epsilon,
        'oracle_model': args.oracle_model,
        'use_oracle': not args.no_oracle,
        'output_dir': args.output_dir,
        'save_frequency': args.save_frequency,
        'verbose': args.verbose
    }

    print(f"\n{'='*60}")
    print(f"Hammer-Spammer Model Validation Experiment")
    print(f"Problem 2 - LLM Agent System (N={config['num_agents']})")
    print(f"{'='*60}\n")

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Run experiment
    runner = ExperimentRunner(config)
    runner.run_all_questions()

    # Print summary
    runner.print_summary()

    # Save final results
    output_path = runner.save_results(intermediate=False)

    print(f"\n{'='*60}")
    print(f"Experiment complete!")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
