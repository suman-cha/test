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
from src.algorithm.ccrr import CCRRanking
from src.algorithm.swra import SWRARanking
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
        difficulty_str = f" (difficulty: {config.get('difficulty', 'all')})" if config.get('difficulty') else ""
        print(f"Difficulty filter: {config.get('difficulty', 'all')}{difficulty_str}")
        self.dataset = DatasetLoader(
            dataset_name=config['dataset'],
            split=config['split'],
            max_samples=config['num_questions'],
            difficulty_filter=config.get('difficulty')
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

        # Initialize algorithms
        print(f"\nInitializing algorithms (beta={config['beta']})...")
        self.algorithm_svd = HammerSpammerRanking(
            beta=config['beta'],
            epsilon=config['epsilon']
        )
        self.algorithm_ccrr = CCRRanking(
            beta=config['beta'],
            epsilon=config['epsilon'],
            T=config.get('ccrr_iterations', 5)
        )
        self.algorithm_swra = SWRARanking(
            k=config.get('swra_k', 2),
            max_iter=config.get('swra_iterations', 15),
            version=config.get('swra_version', 'iterative')
        )
        print("  - SVD-based Hammer-Spammer")
        print("  - CCRR (Cross-Consistency Robust Ranking)")
        print("  - SWRA v2 (Spectral Weighted Rank Aggregation)")

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

        # Apply algorithms
        if self.config['verbose']:
            print("\nApplying algorithms...")

        # SVD-based Hammer-Spammer
        svd_idx, svd_scores = self.algorithm_svd.select_best(R, return_scores=True)
        svd_answer = answer_strings[svd_idx]

        # CCRR
        ccrr_idx, ccrr_scores, ccrr_weights = self.algorithm_ccrr.select_best(R, return_details=True)
        ccrr_answer = answer_strings[ccrr_idx]

        # SWRA v2
        swra_idx, swra_scores, swra_weights = self.algorithm_swra.select_best(R, return_details=True)
        swra_answer = answer_strings[swra_idx]

        # Apply baselines
        if self.config['verbose']:
            print("Applying baseline methods...")

        baseline_selections = Baselines.apply_all_baselines(
            answers=answer_strings,
            comparison_matrix=R
        )

        # Validate algorithms with ground truth
        if self.config['verbose']:
            print("Validating with ground truth...")

        svd_validation = self.validator.validate_with_ground_truth(
            selected_answer=svd_answer,
            ground_truth=ground_truth
        )

        ccrr_validation = self.validator.validate_with_ground_truth(
            selected_answer=ccrr_answer,
            ground_truth=ground_truth
        )

        swra_validation = self.validator.validate_with_ground_truth(
            selected_answer=swra_answer,
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

        # Oracle validation (if enabled) - validate CCRR since it's the main algorithm
        oracle_validation = None
        if self.config['use_oracle']:
            if self.config['verbose']:
                print("Validating with oracle model...")

            try:
                oracle_validation = self.validator.validate_with_oracle(
                    question=question,
                    selected_answer=ccrr_answer,
                    all_answers=answer_strings
                )
            except Exception as e:
                print(f"Oracle validation failed: {e}")
                oracle_validation = None

        # Evaluate ALL agent answers for latent score estimation
        agent_evaluation = None
        if self.config['use_oracle'] or ground_truth:
            if self.config['verbose']:
                print("Evaluating all agent answers for latent score estimation...")

            try:
                agent_evaluation = self.validator.evaluate_all_answers(
                    question=question,
                    all_answers=answer_strings,
                    ground_truth=ground_truth
                )
            except Exception as e:
                print(f"Agent evaluation failed: {e}")
                agent_evaluation = None

        # Compile results
        result = {
            'question_idx': question_idx,
            'question': question,
            'ground_truth': ground_truth,
            'answers': answers,
            'comparison_matrix': R.tolist(),

            # SVD Algorithm results
            'svd_selected_idx': svd_idx,
            'svd_answer': svd_answer,
            'svd_correct': svd_validation['correct'],
            'svd_validation': svd_validation,
            'svd_scores': svd_scores.tolist(),

            # CCRR Algorithm results
            'ccrr_selected_idx': ccrr_idx,
            'ccrr_answer': ccrr_answer,
            'ccrr_correct': ccrr_validation['correct'],
            'ccrr_validation': ccrr_validation,
            'ccrr_scores': ccrr_scores.tolist(),
            'ccrr_weights': ccrr_weights.tolist(),

            # SWRA v2 Algorithm results
            'swra_selected_idx': swra_idx,
            'swra_answer': swra_answer,
            'swra_correct': swra_validation['correct'],
            'swra_validation': swra_validation,
            'swra_scores': swra_scores.tolist(),
            'swra_weights': swra_weights.tolist(),

            # Baseline results
            'baseline_selections': baseline_selections,
            'baseline_validations': baseline_validations,

            # Oracle results
            'oracle_validation': oracle_validation,

            # Agent latent score evaluation
            'agent_evaluation': agent_evaluation,

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
                print(f"  CCRR: {'✓' if result['ccrr_correct'] else '✗'}")
                print(f"    Answer: {result['ccrr_answer']}")
                print(f"  SVD: {'✓' if result['svd_correct'] else '✗'}")
                print(f"    Answer: {result['svd_answer']}")
                print(f"  SWRA v2: {'✓' if result['swra_correct'] else '✗'}")
                print(f"    Answer: {result['swra_answer']}")
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

        # CCRR performance
        ccrr_correct = sum(1 for r in self.results if r['ccrr_correct'])
        ccrr_accuracy = ccrr_correct / num_questions * 100

        # SVD performance
        svd_correct = sum(1 for r in self.results if r['svd_correct'])
        svd_accuracy = svd_correct / num_questions * 100

        # SWRA performance
        swra_correct = sum(1 for r in self.results if r['swra_correct'])
        swra_accuracy = swra_correct / num_questions * 100

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

        # Latent score correlation analysis
        latent_score_analysis = self._compute_latent_score_correlation()

        return {
            'num_questions': num_questions,
            'ccrr_accuracy': ccrr_accuracy,
            'ccrr_correct': ccrr_correct,
            'svd_accuracy': svd_accuracy,
            'svd_correct': svd_correct,
            'swra_accuracy': swra_accuracy,
            'swra_correct': swra_correct,
            'baseline_accuracies': baseline_accuracies,
            'oracle_stats': oracle_stats,
            'agent_stats': agent_stats,
            'latent_score_analysis': latent_score_analysis,
            'total_time': self.end_time - self.start_time if self.end_time else None
        }

    def _compute_latent_score_correlation(self) -> Dict[str, Any]:
        """
        Compute correlation between algorithm's estimated scores and oracle/ground truth scores.

        This validates the algorithm's ability to identify high-quality agents (hammers)
        vs low-quality agents (spammers).

        Returns:
            Dictionary with correlation metrics
        """
        if not self.results:
            return {}

        num_agents = self.config['num_agents']

        # Aggregate agent scores across all questions
        ccrr_scores_per_question = []
        svd_scores_per_question = []
        swra_scores_per_question = []
        oracle_scores_per_question = []
        gt_scores_per_question = []

        for result in self.results:
            # Algorithm estimated scores
            ccrr_scores_per_question.append(result['ccrr_scores'])
            svd_scores_per_question.append(result['svd_scores'])
            swra_scores_per_question.append(result['swra_scores'])

            # Oracle/GT scores
            if result.get('agent_evaluation'):
                eval_data = result['agent_evaluation']
                if eval_data.get('agent_scores_oracle'):
                    oracle_scores_per_question.append(eval_data['agent_scores_oracle'])
                if eval_data.get('agent_scores_ground_truth'):
                    gt_scores_per_question.append(eval_data['agent_scores_ground_truth'])

        # Compute average scores per agent across all questions
        ccrr_avg = np.mean(ccrr_scores_per_question, axis=0) if ccrr_scores_per_question else None
        svd_avg = np.mean(svd_scores_per_question, axis=0) if svd_scores_per_question else None
        swra_avg = np.mean(swra_scores_per_question, axis=0) if swra_scores_per_question else None
        oracle_avg = np.mean(oracle_scores_per_question, axis=0) if oracle_scores_per_question else None
        gt_avg = np.mean(gt_scores_per_question, axis=0) if gt_scores_per_question else None

        analysis = {
            'num_agents': num_agents,
            'num_questions_evaluated': len(self.results),
            'ccrr_scores': ccrr_avg.tolist() if ccrr_avg is not None else None,
            'svd_scores': svd_avg.tolist() if svd_avg is not None else None,
            'swra_scores': swra_avg.tolist() if swra_avg is not None else None,
            'oracle_scores': oracle_avg.tolist() if oracle_avg is not None else None,
            'ground_truth_scores': gt_avg.tolist() if gt_avg is not None else None,
        }

        # Compute correlations if oracle scores available
        if oracle_avg is not None:
            from scipy.stats import pearsonr, spearmanr

            if ccrr_avg is not None:
                ccrr_pearson, ccrr_p = pearsonr(ccrr_avg, oracle_avg)
                ccrr_spearman, ccrr_sp = spearmanr(ccrr_avg, oracle_avg)
                analysis['ccrr_vs_oracle'] = {
                    'pearson_correlation': float(ccrr_pearson),
                    'pearson_p_value': float(ccrr_p),
                    'spearman_correlation': float(ccrr_spearman),
                    'spearman_p_value': float(ccrr_sp)
                }

            if svd_avg is not None:
                svd_pearson, svd_p = pearsonr(svd_avg, oracle_avg)
                svd_spearman, svd_sp = spearmanr(svd_avg, oracle_avg)
                analysis['svd_vs_oracle'] = {
                    'pearson_correlation': float(svd_pearson),
                    'pearson_p_value': float(svd_p),
                    'spearman_correlation': float(svd_spearman),
                    'spearman_p_value': float(svd_sp)
                }

            if swra_avg is not None:
                swra_pearson, swra_p = pearsonr(swra_avg, oracle_avg)
                swra_spearman, swra_sp = spearmanr(swra_avg, oracle_avg)
                analysis['swra_vs_oracle'] = {
                    'pearson_correlation': float(swra_pearson),
                    'pearson_p_value': float(swra_p),
                    'spearman_correlation': float(swra_spearman),
                    'spearman_p_value': float(swra_sp)
                }

        # Compute correlations with ground truth if available
        if gt_avg is not None:
            from scipy.stats import pearsonr, spearmanr

            if ccrr_avg is not None:
                ccrr_pearson_gt, ccrr_p_gt = pearsonr(ccrr_avg, gt_avg)
                ccrr_spearman_gt, ccrr_sp_gt = spearmanr(ccrr_avg, gt_avg)
                analysis['ccrr_vs_ground_truth'] = {
                    'pearson_correlation': float(ccrr_pearson_gt),
                    'pearson_p_value': float(ccrr_p_gt),
                    'spearman_correlation': float(ccrr_spearman_gt),
                    'spearman_p_value': float(ccrr_sp_gt)
                }

            if svd_avg is not None:
                svd_pearson_gt, svd_p_gt = pearsonr(svd_avg, gt_avg)
                svd_spearman_gt, svd_sp_gt = spearmanr(svd_avg, gt_avg)
                analysis['svd_vs_ground_truth'] = {
                    'pearson_correlation': float(svd_pearson_gt),
                    'pearson_p_value': float(svd_p_gt),
                    'spearman_correlation': float(svd_spearman_gt),
                    'spearman_p_value': float(svd_sp_gt)
                }

            if swra_avg is not None:
                swra_pearson_gt, swra_p_gt = pearsonr(swra_avg, gt_avg)
                swra_spearman_gt, swra_sp_gt = spearmanr(swra_avg, gt_avg)
                analysis['swra_vs_ground_truth'] = {
                    'pearson_correlation': float(swra_pearson_gt),
                    'pearson_p_value': float(swra_p_gt),
                    'spearman_correlation': float(swra_spearman_gt),
                    'spearman_p_value': float(swra_sp_gt)
                }

        return analysis

    def print_summary(self):
        """Print experiment summary."""
        stats = self.compute_summary_statistics()

        print(f"\n{'='*60}")
        print(f"EXPERIMENT SUMMARY")
        print(f"{'='*60}\n")

        print(f"Dataset: {self.config['dataset']}")

        # Handle case when no questions were processed
        if not stats:
            print(f"Number of questions: 0")
            print(f"Number of agents: {self.config['num_agents']}")
            print(f"\nNo questions were processed. This may be due to:")
            print(f"  - Difficulty filter returned 0 questions")
            print(f"  - Dataset loading error")
            print(f"  - max_samples set to 0")
            return

        print(f"Number of questions: {stats['num_questions']}")
        print(f"Number of agents: {self.config['num_agents']}")
        print(f"Total time: {stats['total_time']:.1f}s\n")

        print(f"{'='*60}")
        print(f"ACCURACY RESULTS")
        print(f"{'='*60}\n")

        print("Algorithms:")
        print(f"  CCRR (Cross-Consistency): {stats['ccrr_accuracy']:.2f}% ({stats['ccrr_correct']}/{stats['num_questions']})")
        print(f"  SVD (Hammer-Spammer):     {stats['svd_accuracy']:.2f}% ({stats['svd_correct']}/{stats['num_questions']})")
        print(f"  SWRA v2 (Spectral Weighted): {stats['swra_accuracy']:.2f}% ({stats['swra_correct']}/{stats['num_questions']})")
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

        # Latent score correlation analysis
        if stats.get('latent_score_analysis') and stats['latent_score_analysis']:
            lsa = stats['latent_score_analysis']
            print(f"\n{'='*60}")
            print(f"LATENT SCORE CORRELATION ANALYSIS")
            print(f"{'='*60}\n")
            print("This measures how well the algorithms identify agent quality (Hammers vs Spammers)\n")

            # CCRR vs Oracle
            if lsa.get('ccrr_vs_oracle'):
                corr = lsa['ccrr_vs_oracle']
                print(f"CCRR Algorithm vs Oracle:")
                print(f"  Pearson correlation:  {corr['pearson_correlation']:.4f} (p={corr['pearson_p_value']:.4f})")
                print(f"  Spearman correlation: {corr['spearman_correlation']:.4f} (p={corr['spearman_p_value']:.4f})")

            # SVD vs Oracle
            if lsa.get('svd_vs_oracle'):
                corr = lsa['svd_vs_oracle']
                print(f"\nSVD Algorithm vs Oracle:")
                print(f"  Pearson correlation:  {corr['pearson_correlation']:.4f} (p={corr['pearson_p_value']:.4f})")
                print(f"  Spearman correlation: {corr['spearman_correlation']:.4f} (p={corr['spearman_p_value']:.4f})")

            # SWRA vs Oracle
            if lsa.get('swra_vs_oracle'):
                corr = lsa['swra_vs_oracle']
                print(f"\nSWRA v2 Algorithm vs Oracle:")
                print(f"  Pearson correlation:  {corr['pearson_correlation']:.4f} (p={corr['pearson_p_value']:.4f})")
                print(f"  Spearman correlation: {corr['spearman_correlation']:.4f} (p={corr['spearman_p_value']:.4f})")

            # CCRR vs Ground Truth
            if lsa.get('ccrr_vs_ground_truth'):
                corr = lsa['ccrr_vs_ground_truth']
                print(f"\nCCRR Algorithm vs Ground Truth:")
                print(f"  Pearson correlation:  {corr['pearson_correlation']:.4f} (p={corr['pearson_p_value']:.4f})")
                print(f"  Spearman correlation: {corr['spearman_correlation']:.4f} (p={corr['spearman_p_value']:.4f})")

            # SVD vs Ground Truth
            if lsa.get('svd_vs_ground_truth'):
                corr = lsa['svd_vs_ground_truth']
                print(f"\nSVD Algorithm vs Ground Truth:")
                print(f"  Pearson correlation:  {corr['pearson_correlation']:.4f} (p={corr['pearson_p_value']:.4f})")
                print(f"  Spearman correlation: {corr['spearman_correlation']:.4f} (p={corr['spearman_p_value']:.4f})")

            # SWRA vs Ground Truth
            if lsa.get('swra_vs_ground_truth'):
                corr = lsa['swra_vs_ground_truth']
                print(f"\nSWRA v2 Algorithm vs Ground Truth:")
                print(f"  Pearson correlation:  {corr['pearson_correlation']:.4f} (p={corr['pearson_p_value']:.4f})")
                print(f"  Spearman correlation: {corr['spearman_correlation']:.4f} (p={corr['spearman_p_value']:.4f})")

            print(f"\nInterpretation:")
            print(f"  - Correlation close to +1: Algorithm correctly identifies agent quality")
            print(f"  - Higher Spearman: Algorithm correctly ranks agents (Hammers > Spammers)")
            print(f"  - p-value < 0.05: Statistically significant correlation")

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
    parser.add_argument('--difficulty', type=str, default=None,
                       choices=['easy', 'medium', 'hard'],
                       help='Difficulty filter (MATH only): easy (L1-2), medium (L3), hard (L4-5)')

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
        'difficulty': args.difficulty,
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
