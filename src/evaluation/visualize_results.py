"""
Visualization tools for comparing algorithms vs baselines.

Creates plots to demonstrate algorithm superiority.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def plot_accuracy_comparison(stats: Dict[str, Any], output_path: str = None):
    """
    Create bar plot comparing algorithm vs baseline accuracies.

    Args:
        stats: Summary statistics dictionary
        output_path: Optional path to save figure
    """
    methods = ['CCRR', 'SVD', 'SWRA v2', 'Majority\nVoting', 'Random']
    accuracies = [
        stats['ccrr_accuracy'],
        stats['svd_accuracy'],
        stats['swra_accuracy'],
        stats['baseline_accuracies'].get('majority_voting', 0),
        stats['baseline_accuracies'].get('random', 0)
    ]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#95a5a6', '#e74c3c']
    method_types = ['Algorithm', 'Algorithm', 'Algorithm', 'Baseline', 'Baseline']

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add legend for method types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.8, edgecolor='black', label='Algorithms'),
        Patch(facecolor='#e74c3c', alpha=0.8, edgecolor='black', label='Baselines')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Method', fontsize=13, fontweight='bold')
    ax.set_title('Algorithm vs Baseline Accuracy Comparison', fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim(0, min(105, max(accuracies) + 15))
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_relative_improvement(stats: Dict[str, Any], output_path: str = None):
    """
    Plot relative improvement of algorithms over best baseline.

    Args:
        stats: Summary statistics
        output_path: Optional path to save figure
    """
    best_baseline = max(stats['baseline_accuracies'].values())

    algorithms = ['CCRR', 'SVD', 'SWRA v2']
    algorithm_accs = [
        stats['ccrr_accuracy'],
        stats['svd_accuracy'],
        stats['swra_accuracy']
    ]

    improvements = [acc - best_baseline for acc in algorithm_accs]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
    bars = ax.barh(algorithms, improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, imp in zip(bars, improvements):
        width = bar.get_width()
        ax.text(width + (0.5 if width > 0 else -0.5), bar.get_y() + bar.get_height()/2.,
                f'{imp:+.1f}%',
                ha='left' if width > 0 else 'right', va='center', fontsize=12, fontweight='bold')

    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Improvement over Best Baseline (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Relative Improvement (Best Baseline: {best_baseline:.1f}%)',
                fontsize=15, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_win_loss_heatmap(results: List[Dict[str, Any]], output_path: str = None):
    """
    Create heatmap showing per-question correctness.

    Args:
        results: List of experiment results
        output_path: Optional path to save figure
    """
    methods = ['CCRR', 'SVD', 'SWRA v2', 'Majority Voting', 'Random']
    method_keys = ['ccrr', 'svd', 'swra', 'majority_voting', 'random']

    # Build correctness matrix: rows=questions, cols=methods
    correctness_matrix = []
    for result in results:
        row = []
        for key in method_keys:
            if key in ['ccrr', 'svd', 'swra']:
                row.append(1 if result[f'{key}_correct'] else 0)
            else:
                row.append(1 if result['baseline_validations'][key]['correct'] else 0)
        correctness_matrix.append(row)

    correctness_matrix = np.array(correctness_matrix)

    fig, ax = plt.subplots(figsize=(8, max(6, len(results) * 0.3)))

    # Create heatmap
    im = ax.imshow(correctness_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(results)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_yticklabels([f"Q{i+1}" for i in range(len(results))])

    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Question', fontsize=12, fontweight='bold')
    ax.set_title('Per-Question Correctness (Green=Correct, Red=Wrong)',
                fontsize=13, fontweight='bold', pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Wrong', 'Correct'])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_latent_score_correlation(stats: Dict[str, Any], output_path: str = None):
    """
    Plot correlation between estimated and oracle latent scores.

    Args:
        stats: Summary statistics with latent_score_analysis
        output_path: Optional path to save figure
    """
    lsa = stats.get('latent_score_analysis')
    if not lsa or not lsa.get('ccrr_vs_oracle'):
        print("No latent score correlation data available")
        return

    methods = []
    correlations = []

    for method in ['ccrr', 'svd', 'swra']:
        key = f'{method}_vs_oracle'
        if key in lsa:
            methods.append(method.upper())
            correlations.append(lsa[key]['spearman_correlation'])

    if not methods:
        print("No correlation data available")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['#2ecc71', '#3498db', '#9b59b6']
    bars = ax.bar(methods, correlations, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{corr:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Spearman Correlation', fontsize=13, fontweight='bold')
    ax.set_xlabel('Algorithm', fontsize=13, fontweight='bold')
    ax.set_title('Latent Score Correlation with Oracle', fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()

    plt.close()


def generate_all_plots(results_json_path: str, output_dir: str = None):
    """
    Generate all visualization plots from experiment results.

    Args:
        results_json_path: Path to experiment results JSON file
        output_dir: Directory to save plots (if None, displays plots)
    """
    # Load results
    with open(results_json_path, 'r') as f:
        data = json.load(f)

    stats = data['summary']
    results = data['results']

    # Create output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("Generating visualizations...")

    plot_accuracy_comparison(
        stats,
        output_path=str(output_dir / 'accuracy_comparison.png') if output_dir else None
    )

    plot_relative_improvement(
        stats,
        output_path=str(output_dir / 'relative_improvement.png') if output_dir else None
    )

    plot_win_loss_heatmap(
        results,
        output_path=str(output_dir / 'per_question_heatmap.png') if output_dir else None
    )

    plot_latent_score_correlation(
        stats,
        output_path=str(output_dir / 'latent_score_correlation.png') if output_dir else None
    )

    print(f"\n{'='*60}")
    print("All visualizations generated successfully!")
    if output_dir:
        print(f"Saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.evaluation.visualize_results <results_json_path> [output_dir]")
        print("\nExample:")
        print("  python -m src.evaluation.visualize_results results/final_results_20240213_120000.json plots/")
        sys.exit(1)

    results_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    generate_all_plots(results_path, output_dir)
