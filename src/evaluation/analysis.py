"""
Results Analysis and Visualization.

This module analyzes experiment results and generates visualizations
and comprehensive reports.

Usage:
    python -m src.evaluation.analysis --results results/final_results_*.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Algorithm keys as produced by run_experiment.py
ALGORITHM_KEYS = {
    'CCRR': {'correct': 'ccrr_correct', 'idx': 'ccrr_selected_idx', 'scores': 'ccrr_scores'},
    'SVD': {'correct': 'svd_correct', 'idx': 'svd_selected_idx', 'scores': 'svd_scores'},
    'SWRA v2': {'correct': 'swra_correct', 'idx': 'swra_selected_idx', 'scores': 'swra_scores'},
}


class ResultsAnalyzer:
    """Analyze and visualize experiment results."""

    def __init__(self, results_path: str):
        """
        Initialize analyzer.

        Args:
            results_path: Path to results JSON file
        """
        self.results_path = Path(results_path)
        self.data = None
        self.results = None
        self.config = None
        self.summary = None

        self._load_results()

    def _load_results(self):
        """Load results from JSON file."""
        print(f"Loading results from: {self.results_path}")

        with open(self.results_path, 'r') as f:
            self.data = json.load(f)

        self.results = self.data.get('results', [])
        self.config = self.data.get('config', {})
        self.summary = self.data.get('summary', {})

        print(f"Loaded {len(self.results)} question results")

    def _get_method_correctness(self, result: Dict, method_name: str) -> bool:
        """
        Get correctness for a method from a single result.

        Args:
            result: Single question result dict
            method_name: Method name (algorithm name or baseline name)

        Returns:
            True if the method was correct for this question
        """
        # Check if it's an algorithm
        if method_name in ALGORITHM_KEYS:
            return result[ALGORITHM_KEYS[method_name]['correct']]
        # Otherwise it's a baseline
        return result['baseline_validations'][method_name]['correct']

    def analyze_accuracy(self) -> pd.DataFrame:
        """
        Analyze accuracy of all algorithms and baselines.

        Returns:
            DataFrame with accuracy comparison
        """
        num_questions = len(self.results)

        accuracies = {
            'Method': [],
            'Correct': [],
            'Total': [],
            'Accuracy': []
        }

        # All three algorithms
        for algo_name, keys in ALGORITHM_KEYS.items():
            correct = sum(1 for r in self.results if r[keys['correct']])
            accuracies['Method'].append(algo_name)
            accuracies['Correct'].append(correct)
            accuracies['Total'].append(num_questions)
            accuracies['Accuracy'].append(correct / num_questions * 100)

        # Baselines
        baseline_names = list(self.results[0]['baseline_validations'].keys())
        for baseline in baseline_names:
            correct = sum(1 for r in self.results
                         if r['baseline_validations'][baseline]['correct'])
            accuracies['Method'].append(baseline)
            accuracies['Correct'].append(correct)
            accuracies['Total'].append(num_questions)
            accuracies['Accuracy'].append(correct / num_questions * 100)

        df = pd.DataFrame(accuracies)
        df = df.sort_values('Accuracy', ascending=False)

        return df

    def statistical_comparison(self, method1: str = 'CCRR',
                              method2: str = 'majority_voting') -> Dict[str, Any]:
        """
        Perform statistical comparison between two methods.

        Args:
            method1: First method name (algorithm or baseline)
            method2: Second method name (algorithm or baseline)

        Returns:
            Dictionary with statistical test results
        """
        results1 = []
        results2 = []

        for r in self.results:
            results1.append(1 if self._get_method_correctness(r, method1) else 0)
            results2.append(1 if self._get_method_correctness(r, method2) else 0)

        results1 = np.array(results1)
        results2 = np.array(results2)

        # McNemar's test (for paired binary data)
        both_correct = np.sum((results1 == 1) & (results2 == 1))
        method1_only = np.sum((results1 == 1) & (results2 == 0))
        method2_only = np.sum((results1 == 0) & (results2 == 1))
        both_wrong = np.sum((results1 == 0) & (results2 == 0))

        if method1_only + method2_only > 0:
            mcnemar_stat = (abs(method1_only - method2_only) - 1)**2 / (method1_only + method2_only)
            p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
        else:
            mcnemar_stat = 0
            p_value = 1.0

        acc1 = np.mean(results1) * 100
        acc2 = np.mean(results2) * 100
        diff = acc1 - acc2

        return {
            'method1': method1,
            'method2': method2,
            'method1_accuracy': acc1,
            'method2_accuracy': acc2,
            'difference': diff,
            'both_correct': int(both_correct),
            'method1_only': int(method1_only),
            'method2_only': int(method2_only),
            'both_wrong': int(both_wrong),
            'mcnemar_statistic': float(mcnemar_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }

    def analyze_agent_quality(self, algorithm: str = 'CCRR') -> pd.DataFrame:
        """
        Analyze agent quality scores across all questions.

        Args:
            algorithm: Which algorithm's scores to use ('CCRR', 'SVD', or 'SWRA v2')

        Returns:
            DataFrame with agent quality statistics
        """
        N = self.config['num_agents']
        scores_key = ALGORITHM_KEYS[algorithm]['scores']
        idx_key = ALGORITHM_KEYS[algorithm]['idx']

        # Collect quality scores for each agent
        all_scores = [[] for _ in range(N)]

        for r in self.results:
            quality_scores = r[scores_key]
            for i, score in enumerate(quality_scores):
                all_scores[i].append(score)

        # Compute statistics
        agent_stats = {
            'Agent': [],
            'Mean_Quality': [],
            'Std_Quality': [],
            'Min_Quality': [],
            'Max_Quality': [],
            'Times_Selected': []
        }

        for i in range(N):
            scores = all_scores[i]
            times_selected = sum(1 for r in self.results
                                if r[idx_key] == i)

            agent_stats['Agent'].append(f"Agent {i}")
            agent_stats['Mean_Quality'].append(np.mean(scores))
            agent_stats['Std_Quality'].append(np.std(scores))
            agent_stats['Min_Quality'].append(np.min(scores))
            agent_stats['Max_Quality'].append(np.max(scores))
            agent_stats['Times_Selected'].append(times_selected)

        df = pd.DataFrame(agent_stats)
        df = df.sort_values('Mean_Quality', ascending=False)

        return df

    def plot_accuracy_comparison(self, output_path: Optional[str] = None):
        """
        Plot accuracy comparison between methods.

        Args:
            output_path: Path to save plot (optional)
        """
        df = self.analyze_accuracy()

        plt.figure(figsize=(12, 6))

        # Color algorithms differently from baselines
        algo_names = set(ALGORITHM_KEYS.keys())
        colors = ['#2ecc71' if m in algo_names else '#3498db'
                 for m in df['Method']]

        plt.barh(df['Method'], df['Accuracy'], color=colors)
        plt.xlabel('Accuracy (%)', fontsize=12)
        plt.ylabel('Method', fontsize=12)
        plt.title('Accuracy Comparison: Algorithms vs Baselines', fontsize=14, fontweight='bold')
        plt.xlim(0, 100)

        # Add value labels
        for i, (method, acc) in enumerate(zip(df['Method'], df['Accuracy'])):
            plt.text(acc + 1, i, f'{acc:.1f}%', va='center', fontsize=10)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {output_path}")

        plt.show()

    def plot_agent_quality(self, output_path: Optional[str] = None, algorithm: str = 'CCRR'):
        """
        Plot agent quality distribution.

        Args:
            output_path: Path to save plot (optional)
            algorithm: Which algorithm's scores to use
        """
        df = self.analyze_agent_quality(algorithm=algorithm)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Mean quality scores
        colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
        ax1.barh(df['Agent'], df['Mean_Quality'], color=colors, xerr=df['Std_Quality'])
        ax1.set_xlabel('Quality Score', fontsize=12)
        ax1.set_ylabel('Agent', fontsize=12)
        ax1.set_title(f'Agent Quality Scores - {algorithm} (Mean +/- Std)', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)

        # Plot 2: Times selected
        ax2.barh(df['Agent'], df['Times_Selected'], color=colors)
        ax2.set_xlabel('Number of Times Selected', fontsize=12)
        ax2.set_ylabel('Agent', fontsize=12)
        ax2.set_title(f'Selection Frequency by Agent - {algorithm}', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {output_path}")

        plt.show()

    def plot_quality_vs_selection(self, output_path: Optional[str] = None, algorithm: str = 'CCRR'):
        """
        Plot quality score vs selection rate.

        Args:
            output_path: Path to save plot (optional)
            algorithm: Which algorithm's scores to use
        """
        df = self.analyze_agent_quality(algorithm=algorithm)

        plt.figure(figsize=(10, 6))

        plt.scatter(df['Mean_Quality'], df['Times_Selected'],
                   s=100, alpha=0.6, c=range(len(df)), cmap='viridis')

        # Add labels for each agent
        for i, row in df.iterrows():
            plt.annotate(row['Agent'], (row['Mean_Quality'], row['Times_Selected']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

        plt.xlabel('Mean Quality Score', fontsize=12)
        plt.ylabel('Number of Times Selected', fontsize=12)
        plt.title(f'Agent Quality vs Selection Frequency - {algorithm}', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)

        # Add trend line
        if len(df) > 1:
            z = np.polyfit(df['Mean_Quality'], df['Times_Selected'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df['Mean_Quality'].min(), df['Mean_Quality'].max(), 100)
            plt.plot(x_line, p(x_line), "r--", alpha=0.5, label='Trend')
            plt.legend()

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {output_path}")

        plt.show()

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive text report.

        Args:
            output_path: Path to save report (optional)

        Returns:
            Report text
        """
        lines = []
        lines.append("="*80)
        lines.append("HAMMER-SPAMMER MODEL VALIDATION REPORT")
        lines.append("Problem 2 - LLM Agent System Experiment")
        lines.append("="*80)
        lines.append("")

        # Configuration
        lines.append("EXPERIMENT CONFIGURATION")
        lines.append("-"*80)
        lines.append(f"Dataset: {self.config['dataset']}")
        lines.append(f"Number of questions: {len(self.results)}")
        lines.append(f"Number of agents: {self.config['num_agents']}")
        lines.append(f"Algorithm parameters: beta={self.config['beta']}, epsilon={self.config['epsilon']}")
        lines.append("")

        # Accuracy results
        lines.append("ACCURACY RESULTS")
        lines.append("-"*80)
        df_acc = self.analyze_accuracy()
        lines.append(df_acc.to_string(index=False))
        lines.append("")

        # Statistical comparisons for each algorithm vs majority voting
        lines.append("STATISTICAL COMPARISONS (vs Majority Voting)")
        lines.append("-"*80)

        for algo_name in ALGORITHM_KEYS:
            stat_result = self.statistical_comparison(method1=algo_name, method2='majority_voting')
            lines.append(f"\n{algo_name} vs majority_voting:")
            lines.append(f"  {algo_name} accuracy: {stat_result['method1_accuracy']:.2f}%")
            lines.append(f"  majority_voting accuracy: {stat_result['method2_accuracy']:.2f}%")
            lines.append(f"  Difference: {stat_result['difference']:+.2f}%")
            lines.append(f"  McNemar's test: chi2 = {stat_result['mcnemar_statistic']:.4f}, p = {stat_result['p_value']:.4f}")
            lines.append(f"  Statistically significant: {'YES' if stat_result['significant'] else 'NO'} (alpha=0.05)")
            lines.append(f"  Contingency: both_correct={stat_result['both_correct']}, "
                        f"{algo_name}_only={stat_result['method1_only']}, "
                        f"mv_only={stat_result['method2_only']}, "
                        f"both_wrong={stat_result['both_wrong']}")

        lines.append("")

        # Agent quality analysis (using CCRR as primary)
        lines.append("AGENT QUALITY ANALYSIS (CCRR scores)")
        lines.append("-"*80)
        df_quality = self.analyze_agent_quality(algorithm='CCRR')
        lines.append(df_quality.to_string(index=False))
        lines.append("")

        # Oracle statistics (if available)
        if self.summary.get('oracle_stats'):
            lines.append("ORACLE VALIDATION")
            lines.append("-"*80)
            oracle = self.summary['oracle_stats']
            lines.append(f"Average oracle score: {oracle['avg_score']:.2f}/10")
            lines.append(f"Oracle agreement rate: {oracle['oracle_agreement_rate']:.2f}%")
            lines.append("")

        # Conclusions
        lines.append("CONCLUSIONS")
        lines.append("-"*80)

        # Compare CCRR (primary algorithm) vs majority voting
        stat_ccrr = self.statistical_comparison(method1='CCRR', method2='majority_voting')
        algo_acc = stat_ccrr['method1_accuracy']
        baseline_acc = stat_ccrr['method2_accuracy']

        if algo_acc > baseline_acc and stat_ccrr['significant']:
            lines.append(f"The CCRR algorithm OUTPERFORMS majority voting")
            lines.append(f"  with a statistically significant improvement of {stat_ccrr['difference']:.2f}%.")
        elif algo_acc > baseline_acc:
            lines.append(f"The CCRR algorithm performs slightly better than majority voting")
            lines.append(f"  ({stat_ccrr['difference']:.2f}%), but the difference is not statistically significant.")
        else:
            lines.append(f"The CCRR algorithm does not outperform majority voting.")

        lines.append("")

        # Identify hammers vs spammers
        df_quality_sorted = df_quality.sort_values('Mean_Quality', ascending=False)
        top_agents = df_quality_sorted.head(5)['Agent'].tolist()
        bottom_agents = df_quality_sorted.tail(5)['Agent'].tolist()

        lines.append(f"Top 5 agents (Hammers): {', '.join(top_agents)}")
        lines.append(f"Bottom 5 agents (Spammers): {', '.join(bottom_agents)}")
        lines.append("")

        lines.append("="*80)

        report_text = "\n".join(lines)

        # Save to file
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {output_path}")

        return report_text

    def generate_all_outputs(self, output_dir: str):
        """
        Generate all analysis outputs.

        Args:
            output_dir: Directory to save outputs
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating analysis outputs to: {output_path}")

        # Generate plots
        print("\n1. Generating accuracy comparison plot...")
        self.plot_accuracy_comparison(output_path / "accuracy_comparison.png")

        print("\n2. Generating agent quality plot...")
        self.plot_agent_quality(output_path / "agent_quality.png")

        print("\n3. Generating quality vs selection plot...")
        self.plot_quality_vs_selection(output_path / "quality_vs_selection.png")

        # Generate report
        print("\n4. Generating comprehensive report...")
        report = self.generate_report(output_path / "analysis_report.txt")

        # Also print to console
        print("\n" + "="*80)
        print(report)

        print(f"\nAll outputs saved to: {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze experiment results")

    parser.add_argument('--results', type=str, required=True,
                       help='Path to results JSON file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for analysis (default: same as results)')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Use same directory as results file
        output_dir = Path(args.results).parent / "analysis"

    # Run analysis
    analyzer = ResultsAnalyzer(args.results)
    analyzer.generate_all_outputs(str(output_dir))


if __name__ == "__main__":
    main()
