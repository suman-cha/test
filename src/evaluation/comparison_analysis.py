"""
Enhanced comparison analysis for algorithms vs baselines.

This module provides metrics and visualizations to demonstrate
algorithm superiority over simple baselines.
"""

import numpy as np
from typing import Dict, List, Any
from scipy import stats


def compute_relative_improvements(results: Dict[str, float], baselines: List[str]) -> Dict[str, Any]:
    """
    Compute relative improvement of algorithms over baselines.

    Args:
        results: Dictionary with method_name -> accuracy
        baselines: List of baseline method names (e.g., ['random', 'majority_voting'])

    Returns:
        Dictionary with improvement metrics
    """
    # Extract algorithm results
    algorithms = {k: v for k, v in results.items() if k not in baselines}
    baseline_results = {k: v for k, v in results.items() if k in baselines}

    # Best baseline accuracy
    best_baseline = max(baseline_results.values()) if baseline_results else 0
    best_baseline_name = max(baseline_results.items(), key=lambda x: x[1])[0] if baseline_results else None

    improvements = {}
    for alg_name, alg_acc in algorithms.items():
        improvements[alg_name] = {
            'absolute_improvement': alg_acc - best_baseline,
            'relative_improvement': ((alg_acc - best_baseline) / best_baseline * 100) if best_baseline > 0 else 0,
            'best_baseline': best_baseline_name,
            'best_baseline_acc': best_baseline
        }

    return {
        'improvements': improvements,
        'best_baseline': best_baseline_name,
        'best_baseline_accuracy': best_baseline,
        'algorithms': algorithms,
        'baselines': baseline_results
    }


def compute_statistical_significance(results_per_question: List[Dict[str, bool]],
                                     method1: str, method2: str) -> Dict[str, Any]:
    """
    Compute statistical significance between two methods using McNemar's test.

    Args:
        results_per_question: List of dicts with method_name -> correct (bool) for each question
        method1: First method name
        method2: Second method name

    Returns:
        Dictionary with statistical test results
    """
    # Count outcomes
    both_correct = sum(1 for r in results_per_question if r[method1] and r[method2])
    both_wrong = sum(1 for r in results_per_question if not r[method1] and not r[method2])
    m1_only = sum(1 for r in results_per_question if r[method1] and not r[method2])
    m2_only = sum(1 for r in results_per_question if not r[method1] and r[method2])

    # McNemar's test (for paired binary outcomes)
    # Tests if two methods have significantly different error rates
    if m1_only + m2_only > 0:
        mcnemar_stat = (abs(m1_only - m2_only) - 1) ** 2 / (m1_only + m2_only)
        p_value = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
    else:
        mcnemar_stat = 0
        p_value = 1.0

    return {
        'method1': method1,
        'method2': method2,
        'both_correct': both_correct,
        'both_wrong': both_wrong,
        'method1_only_correct': m1_only,
        'method2_only_correct': m2_only,
        'mcnemar_statistic': mcnemar_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'winner': method1 if m1_only > m2_only else method2 if m2_only > m1_only else 'tie'
    }


def compute_win_tie_loss(results_per_question: List[Dict[str, bool]],
                         algorithm: str, baseline: str) -> Dict[str, Any]:
    """
    Compute win/tie/loss statistics for algorithm vs baseline.

    Args:
        results_per_question: List of dicts with method_name -> correct (bool)
        algorithm: Algorithm name
        baseline: Baseline name

    Returns:
        Win/tie/loss counts
    """
    wins = sum(1 for r in results_per_question if r[algorithm] and not r[baseline])
    ties = sum(1 for r in results_per_question if r[algorithm] == r[baseline])
    losses = sum(1 for r in results_per_question if not r[algorithm] and r[baseline])

    return {
        'wins': wins,
        'ties': ties,
        'losses': losses,
        'win_rate': wins / len(results_per_question) * 100 if results_per_question else 0,
        'advantage': wins - losses
    }


def compute_oracle_alignment(results: List[Dict[str, Any]], methods: List[str]) -> Dict[str, float]:
    """
    Compute how well each method's selections align with oracle rankings.

    Args:
        results: List of experiment results with oracle scores
        methods: List of method names to evaluate

    Returns:
        Dictionary mapping method -> oracle alignment score
    """
    alignment = {method: [] for method in methods}

    for result in results:
        if 'agent_evaluation' not in result or result['agent_evaluation'] is None:
            continue

        agent_eval = result['agent_evaluation']
        if 'agent_scores_oracle' not in agent_eval:
            continue

        oracle_scores = agent_eval['agent_scores_oracle']
        best_oracle_idx = int(np.argmax(oracle_scores))

        for method in methods:
            selected_key = f'{method}_selected_idx' if method in ['ccrr', 'svd', 'swra'] else None

            if selected_key and selected_key in result:
                selected_idx = result[selected_key]
                # Score based on how close selected answer is to oracle best
                alignment[method].append(oracle_scores[selected_idx])
            elif method in result.get('baseline_selections', {}):
                selected_idx = result['baseline_selections'][method]
                alignment[method].append(oracle_scores[selected_idx])

    # Average oracle score of selections
    return {method: np.mean(scores) if scores else 0.0 for method, scores in alignment.items()}


def print_comparison_report(summary_stats: Dict[str, Any], results_per_question: List[Dict[str, bool]]):
    """
    Print comprehensive comparison report.

    Args:
        summary_stats: Summary statistics from experiment
        results_per_question: Per-question correctness for each method
    """
    print(f"\n{'='*70}")
    print(f"ALGORITHM vs BASELINE COMPARISON")
    print(f"{'='*70}\n")

    # Extract accuracies
    accuracies = {
        'CCRR': summary_stats['ccrr_accuracy'],
        'SVD': summary_stats['svd_accuracy'],
        'SWRA': summary_stats['swra_accuracy'],
    }
    accuracies.update(summary_stats['baseline_accuracies'])

    # Compute relative improvements
    improvements = compute_relative_improvements(
        accuracies,
        baselines=['random', 'majority_voting']
    )

    # Print accuracy comparison
    print("1. ACCURACY COMPARISON")
    print("-" * 70)
    print(f"{'Method':<25} {'Accuracy':<15} {'Type':<15}")
    print("-" * 70)

    # Sort by accuracy
    sorted_methods = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
    for method, acc in sorted_methods:
        method_type = "Baseline" if method in ['random', 'majority_voting'] else "Algorithm"
        print(f"{method:<25} {acc:>6.2f}% {'':<8} {method_type:<15}")

    # Print relative improvements
    print(f"\n2. RELATIVE IMPROVEMENT OVER BEST BASELINE")
    print("-" * 70)
    best_baseline = improvements['best_baseline']
    best_baseline_acc = improvements['best_baseline_accuracy']
    print(f"Best Baseline: {best_baseline} ({best_baseline_acc:.2f}%)\n")

    print(f"{'Algorithm':<25} {'Absolute':<15} {'Relative':<15}")
    print("-" * 70)
    for alg, metrics in improvements['improvements'].items():
        abs_imp = metrics['absolute_improvement']
        rel_imp = metrics['relative_improvement']
        print(f"{alg:<25} {abs_imp:>+6.2f}% {'':<8} {rel_imp:>+6.2f}%")

    # Win/Tie/Loss analysis
    print(f"\n3. WIN/TIE/LOSS vs BEST BASELINE ({best_baseline})")
    print("-" * 70)
    print(f"{'Algorithm':<25} {'Wins':<10} {'Ties':<10} {'Losses':<10} {'Advantage':<10}")
    print("-" * 70)

    for alg in ['CCRR', 'SVD', 'SWRA']:
        alg_key = alg.lower().replace(' ', '_')
        wtl = compute_win_tie_loss(results_per_question, alg_key, best_baseline)
        print(f"{alg:<25} {wtl['wins']:<10} {wtl['ties']:<10} {wtl['losses']:<10} {wtl['advantage']:>+3}")

    # Statistical significance
    print(f"\n4. STATISTICAL SIGNIFICANCE (vs {best_baseline})")
    print("-" * 70)
    print(f"{'Algorithm':<25} {'p-value':<15} {'Significant':<15} {'Winner':<15}")
    print("-" * 70)

    for alg in ['CCRR', 'SVD', 'SWRA']:
        alg_key = alg.lower().replace(' ', '_')
        sig_test = compute_statistical_significance(results_per_question, alg_key, best_baseline)
        sig_str = "✓ YES" if sig_test['significant'] else "✗ NO"
        winner = "Algorithm" if sig_test['winner'] == alg_key else "Baseline" if sig_test['winner'] == best_baseline else "Tie"
        print(f"{alg:<25} {sig_test['p_value']:<15.4f} {sig_str:<15} {winner:<15}")

    print(f"\nNote: p < 0.05 indicates statistically significant difference (McNemar's test)")

    # Oracle alignment (if available)
    if summary_stats.get('oracle_stats'):
        print(f"\n5. ORACLE ALIGNMENT")
        print("-" * 70)
        print(f"Oracle Agreement Rate (CCRR): {summary_stats['oracle_stats']['oracle_agreement_rate']:.2f}%")
        print(f"Average Oracle Score (CCRR): {summary_stats['oracle_stats']['avg_score']:.2f}/10")

    print(f"\n{'='*70}\n")


def generate_latex_table(summary_stats: Dict[str, Any]) -> str:
    """
    Generate LaTeX table for paper/report.

    Args:
        summary_stats: Summary statistics

    Returns:
        LaTeX table string
    """
    accuracies = {
        'Random': summary_stats['baseline_accuracies'].get('random', 0),
        'Majority Voting': summary_stats['baseline_accuracies'].get('majority_voting', 0),
        'SVD': summary_stats['svd_accuracy'],
        'CCRR': summary_stats['ccrr_accuracy'],
        'SWRA v2': summary_stats['swra_accuracy'],
    }

    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\begin{tabular}{lcc}\n"
    latex += "\\hline\n"
    latex += "Method & Accuracy (\\%) & Type \\\\\n"
    latex += "\\hline\n"

    for method, acc in accuracies.items():
        method_type = "Baseline" if method in ['Random', 'Majority Voting'] else "Algorithm"
        latex += f"{method} & {acc:.2f} & {method_type} \\\\\n"

    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\caption{Comparison of algorithms vs baselines}\n"
    latex += "\\label{tab:results}\n"
    latex += "\\end{table}\n"

    return latex


if __name__ == "__main__":
    # Example usage
    print("Comparison Analysis Module")
    print("=" * 60)
    print("This module provides enhanced comparison metrics for")
    print("demonstrating algorithm superiority over baselines:")
    print()
    print("1. Relative improvement (absolute & percentage)")
    print("2. Win/Tie/Loss statistics")
    print("3. Statistical significance (McNemar's test)")
    print("4. Oracle alignment scores")
    print("5. LaTeX table generation")
    print()
    print("Use via run_experiment.py with --verbose flag")
