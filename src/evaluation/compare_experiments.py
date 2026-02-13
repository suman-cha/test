"""
Cross-experiment comparison for spammer ratio analysis.

Compares results across different epsilon values and datasets.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

sns.set_style("whitegrid")


def load_experiment_results(results_dir: str) -> Dict[str, Any]:
    """
    Load all experiment results from a directory.

    Args:
        results_dir: Directory containing experiment results

    Returns:
        Dictionary mapping experiment name to results
    """
    results_path = Path(results_dir)
    experiments = {}

    for exp_dir in results_path.iterdir():
        if not exp_dir.is_dir():
            continue

        # Find final results JSON
        json_files = list(exp_dir.glob("final_results_*.json"))
        if not json_files:
            continue

        # Load most recent
        latest_json = max(json_files, key=lambda p: p.stat().st_mtime)

        with open(latest_json, 'r') as f:
            data = json.load(f)

        # Extract experiment parameters from directory name
        exp_name = exp_dir.name
        experiments[exp_name] = {
            'path': str(latest_json),
            'data': data,
            'name': exp_name
        }

    return experiments


def extract_epsilon_from_name(exp_name: str) -> float:
    """Extract epsilon value from experiment name."""
    if 'eps0_1' in exp_name or 'eps01' in exp_name:
        return 0.1
    elif 'eps0_3' in exp_name or 'eps03' in exp_name:
        return 0.3
    elif 'eps0_5' in exp_name or 'eps05' in exp_name:
        return 0.5
    return None


def extract_dataset_from_name(exp_name: str) -> str:
    """Extract dataset name from experiment name."""
    if 'gsm8k_easy' in exp_name:
        return 'GSM8K Easy'
    elif 'gsm8k_hard' in exp_name:
        return 'GSM8K Hard'
    elif 'math_medium' in exp_name:
        return 'MATH Medium'
    elif 'math_hard' in exp_name:
        return 'MATH Hard'
    elif 'math' in exp_name:
        return 'MATH'
    elif 'gsm8k' in exp_name:
        return 'GSM8K'
    return 'Unknown'


def compare_by_epsilon(experiments: Dict[str, Any], output_dir: str = None):
    """
    Compare algorithm performance across different epsilon values.

    Args:
        experiments: Dictionary of experiment results
        output_dir: Optional output directory for plots
    """
    # Organize by epsilon and dataset
    results_by_eps = defaultdict(lambda: defaultdict(dict))

    for exp_name, exp_data in experiments.items():
        epsilon = extract_epsilon_from_name(exp_name)
        dataset = extract_dataset_from_name(exp_name)

        if epsilon is None:
            continue

        summary = exp_data['data']['summary']

        results_by_eps[dataset][epsilon] = {
            'ccrr': summary['ccrr_accuracy'],
            'svd': summary['svd_accuracy'],
            'swra': summary['swra_accuracy'],
            'majority_voting': summary['baseline_accuracies'].get('majority_voting', 0),
            'random': summary['baseline_accuracies'].get('random', 0),
            'num_questions': summary['num_questions']
        }

    # Print comparison table
    print("\n" + "="*80)
    print("SPAMMER RATIO COMPARISON")
    print("="*80 + "\n")

    for dataset in sorted(results_by_eps.keys()):
        print(f"\n{dataset}")
        print("-" * 80)
        print(f"{'Epsilon':<10} {'CCRR':<10} {'SVD':<10} {'SWRA':<10} {'MajVote':<10} {'Random':<10}")
        print("-" * 80)

        for epsilon in sorted(results_by_eps[dataset].keys()):
            data = results_by_eps[dataset][epsilon]
            print(f"{epsilon:<10.1f} "
                  f"{data['ccrr']:<10.2f} "
                  f"{data['svd']:<10.2f} "
                  f"{data['swra']:<10.2f} "
                  f"{data['majority_voting']:<10.2f} "
                  f"{data['random']:<10.2f}")

    # Plot comparison
    if output_dir:
        plot_epsilon_comparison(results_by_eps, output_dir)


def plot_epsilon_comparison(results_by_eps: Dict, output_dir: str):
    """
    Create plots comparing performance across epsilon values.

    Args:
        results_by_eps: Results organized by dataset and epsilon
        output_dir: Output directory for plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Plot 1: Algorithm performance vs epsilon (all datasets)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    datasets = sorted(results_by_eps.keys())
    for idx, dataset in enumerate(datasets[:4]):  # Max 4 subplots
        ax = axes[idx]
        data = results_by_eps[dataset]

        epsilons = sorted(data.keys())
        ccrr_accs = [data[e]['ccrr'] for e in epsilons]
        svd_accs = [data[e]['svd'] for e in epsilons]
        swra_accs = [data[e]['swra'] for e in epsilons]
        maj_accs = [data[e]['majority_voting'] for e in epsilons]
        rand_accs = [data[e]['random'] for e in epsilons]

        ax.plot(epsilons, ccrr_accs, 'o-', label='CCRR', linewidth=2, markersize=8)
        ax.plot(epsilons, svd_accs, 's-', label='SVD', linewidth=2, markersize=8)
        ax.plot(epsilons, swra_accs, '^-', label='SWRA', linewidth=2, markersize=8)
        ax.plot(epsilons, maj_accs, 'x--', label='Majority Vote', linewidth=1.5, markersize=8, alpha=0.7)
        ax.plot(epsilons, rand_accs, '+-', label='Random', linewidth=1, markersize=8, alpha=0.5)

        ax.set_xlabel('Spammer Ratio (ε)', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title(dataset, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xticks(epsilons)

    plt.suptitle('Algorithm Performance vs Spammer Ratio', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'epsilon_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'epsilon_comparison.png'}")
    plt.close()

    # Plot 2: Relative improvement over baseline
    fig, ax = plt.subplots(figsize=(10, 6))

    for dataset in datasets[:4]:
        data = results_by_eps[dataset]
        epsilons = sorted(data.keys())
        improvements = []

        for e in epsilons:
            best_baseline = max(data[e]['majority_voting'], data[e]['random'])
            ccrr_improvement = data[e]['ccrr'] - best_baseline
            improvements.append(ccrr_improvement)

        ax.plot(epsilons, improvements, 'o-', label=dataset, linewidth=2, markersize=8)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Spammer Ratio (ε)', fontsize=12, fontweight='bold')
    ax.set_ylabel('CCRR Improvement over Best Baseline (%)', fontsize=12, fontweight='bold')
    ax.set_title('Algorithm Robustness to Noise', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks(epsilons)

    plt.tight_layout()
    plt.savefig(output_path / 'robustness_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'robustness_analysis.png'}")
    plt.close()


def generate_summary_report(experiments: Dict[str, Any], output_file: str):
    """
    Generate text summary report.

    Args:
        experiments: Dictionary of experiment results
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SPAMMER RATIO EXPERIMENT SUMMARY\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total experiments: {len(experiments)}\n\n")

        # Organize by epsilon
        by_epsilon = defaultdict(list)
        for exp_name, exp_data in experiments.items():
            epsilon = extract_epsilon_from_name(exp_name)
            if epsilon:
                by_epsilon[epsilon].append((exp_name, exp_data))

        for epsilon in sorted(by_epsilon.keys()):
            f.write(f"\nEPSILON = {epsilon}\n")
            f.write("-" * 80 + "\n")

            for exp_name, exp_data in by_epsilon[epsilon]:
                dataset = extract_dataset_from_name(exp_name)
                summary = exp_data['data']['summary']

                f.write(f"\n{dataset}:\n")
                f.write(f"  CCRR:           {summary['ccrr_accuracy']:.2f}%\n")
                f.write(f"  SVD:            {summary['svd_accuracy']:.2f}%\n")
                f.write(f"  SWRA:           {summary['swra_accuracy']:.2f}%\n")
                f.write(f"  Majority Vote:  {summary['baseline_accuracies'].get('majority_voting', 0):.2f}%\n")
                f.write(f"  Random:         {summary['baseline_accuracies'].get('random', 0):.2f}%\n")
                f.write(f"  Questions:      {summary['num_questions']}\n")

    print(f"Summary report saved: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare spammer ratio experiments")
    parser.add_argument('--results-dir', type=str, default='results/spammer_experiments',
                       help='Directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default='cross_analysis',
                       help='Output directory for analysis')

    args = parser.parse_args()

    print("Loading experiment results...")
    experiments = load_experiment_results(args.results_dir)

    if not experiments:
        print(f"No experiments found in {args.results_dir}")
        return

    print(f"Found {len(experiments)} experiments")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compare by epsilon
    compare_by_epsilon(experiments, str(output_dir))

    # Generate summary report
    generate_summary_report(experiments, str(output_dir / 'summary_report.txt'))

    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
