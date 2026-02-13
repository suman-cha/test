"""
Critical Analysis: Row Sum vs Answer Correctness

This validates the fundamental Bradley-Terry assumption:
  High quality agent → High score → More wins → High row sum

If row sum does NOT correlate with correctness, the algorithm
fundamentally cannot work.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt


def analyze_rowsum_correctness_correlation(results_file: str):
    """
    Analyze correlation between row sum and correctness across all data.

    Args:
        results_file: Path to experiment results JSON
    """
    # Load results
    with open(results_file) as f:
        data = json.load(f)

    results = data['results']
    config = data.get('config', {})

    print("="*70)
    print("ROW SUM vs CORRECTNESS CORRELATION ANALYSIS")
    print("="*70)
    print(f"\nDataset: {config.get('dataset', 'unknown')}")
    print(f"Number of questions: {len(results)}")
    print(f"Number of agents: {config.get('num_agents', 'unknown')}")
    print()

    # Collect data: (row_sum, is_correct) pairs
    all_data = []  # List of (question_idx, agent_idx, row_sum, is_correct)
    per_agent_data = {}  # agent_idx -> [(row_sum, is_correct), ...]
    per_question_data = {}  # question_idx -> [(agent_idx, row_sum, is_correct), ...]

    N_agents = len(results[0]['answers']) if results else 0

    # Initialize per-agent storage
    for i in range(N_agents):
        per_agent_data[i] = []

    # Process each question
    for result in results:
        q_idx = result['question_idx']
        R = np.array(result['comparison_matrix'])
        N = R.shape[0]

        answers = [a['answer'] for a in result['answers']]
        ground_truth = result['ground_truth']

        per_question_data[q_idx] = []

        # For each agent
        for i in range(N):
            # Compute row sum (excluding diagonal)
            row_sum = (R[i, :].sum() - R[i, i]) / (N - 1)

            # Check correctness
            from src.utils.dataset import compare_answers
            is_correct = compare_answers(answers[i], ground_truth)

            # Store data
            all_data.append((q_idx, i, row_sum, is_correct))
            per_agent_data[i].append((row_sum, is_correct))
            per_question_data[q_idx].append((i, row_sum, is_correct))

    # Convert to arrays for analysis
    all_row_sums = np.array([d[2] for d in all_data])
    all_correct = np.array([d[3] for d in all_data], dtype=float)

    # Overall correlation
    print("="*70)
    print("OVERALL CORRELATION")
    print("="*70)

    pearson_r, pearson_p = pearsonr(all_row_sums, all_correct)
    spearman_r, spearman_p = spearmanr(all_row_sums, all_correct)

    print(f"\nAcross {len(all_data)} (question, agent) pairs:")
    print(f"  Pearson correlation:  r = {pearson_r:+.4f}, p = {pearson_p:.4e}")
    print(f"  Spearman correlation: ρ = {spearman_r:+.4f}, p = {spearman_p:.4e}")
    print()

    # Interpretation
    print("Interpretation:")
    if pearson_r > 0.3:
        print(f"  ✓ POSITIVE correlation (r = {pearson_r:.3f})")
        print(f"    → High row sum correlates with correctness")
        print(f"    → Bradley-Terry assumption holds!")
        print(f"    → Algorithm should work")
    elif pearson_r > 0.1:
        print(f"  ⚠️  WEAK positive correlation (r = {pearson_r:.3f})")
        print(f"    → Some signal, but noisy")
        print(f"    → Algorithm will struggle")
    elif pearson_r > -0.1:
        print(f"  ❌ NO correlation (r ≈ {pearson_r:.3f})")
        print(f"    → Row sum is random noise")
        print(f"    → Algorithm cannot work!")
    else:
        print(f"  ❌❌ NEGATIVE correlation (r = {pearson_r:.3f})!!!")
        print(f"    → High row sum → MORE LIKELY WRONG")
        print(f"    → Algorithm will select WRONG answers")
        print(f"    → Complete failure of Bradley-Terry model")

    # Compare correct vs incorrect agents
    correct_row_sums = all_row_sums[all_correct == 1]
    incorrect_row_sums = all_row_sums[all_correct == 0]

    print()
    print("-"*70)
    print("Row sum distribution:")
    print(f"  Correct answers:   mean = {correct_row_sums.mean():.3f}, std = {correct_row_sums.std():.3f}")
    print(f"  Incorrect answers: mean = {incorrect_row_sums.mean():.3f}, std = {incorrect_row_sums.std():.3f}")
    print(f"  Difference:        {correct_row_sums.mean() - incorrect_row_sums.mean():+.3f}")
    print()

    if correct_row_sums.mean() < incorrect_row_sums.mean():
        print("  ❌ CRITICAL: Incorrect answers have HIGHER row sums!")
        print("     → Algorithm will systematically prefer wrong answers")

    # Per-agent analysis
    print()
    print("="*70)
    print("PER-AGENT ANALYSIS")
    print("="*70)
    print()

    agent_correlations = []

    print(f"{'Agent':<8} {'N_correct':<12} {'Correlation':<14} {'Interpretation'}")
    print("-"*70)

    for agent_idx in range(N_agents):
        data = per_agent_data[agent_idx]
        if len(data) < 3:
            continue

        row_sums = np.array([d[0] for d in data])
        correct = np.array([d[1] for d in data], dtype=float)

        n_correct = int(correct.sum())
        n_total = len(correct)

        # Correlation
        if np.std(row_sums) > 1e-6 and np.std(correct) > 1e-6:
            r, p = pearsonr(row_sums, correct)
            agent_correlations.append((agent_idx, r, p, n_correct, n_total))

            # Interpretation
            if r > 0.2:
                interp = "✓ Reliable"
            elif r > 0:
                interp = "~ Weak"
            elif r > -0.2:
                interp = "⚠️  Random"
            else:
                interp = "✗ Inverted"

            print(f"{agent_idx:<8} {n_correct}/{n_total:<9} r={r:+.3f} (p={p:.3f})  {interp}")
        else:
            print(f"{agent_idx:<8} {n_correct}/{n_total:<9} {'(constant)':<14} -")

    # Summary
    print()
    print("-"*70)
    if agent_correlations:
        avg_r = np.mean([r for _, r, _, _, _ in agent_correlations])
        positive_agents = sum(1 for _, r, _, _, _ in agent_correlations if r > 0.1)
        negative_agents = sum(1 for _, r, _, _, _ in agent_correlations if r < -0.1)

        print(f"Summary:")
        print(f"  Average correlation: {avg_r:+.3f}")
        print(f"  Reliable agents (r > 0.1): {positive_agents}/{len(agent_correlations)}")
        print(f"  Inverted agents (r < -0.1): {negative_agents}/{len(agent_correlations)}")

        if negative_agents > positive_agents:
            print()
            print("  ❌ MORE agents have INVERTED correlation!")
            print("     → For these agents, confidence anti-correlates with correctness")

    # Per-question analysis (find problematic questions)
    print()
    print("="*70)
    print("PROBLEMATIC QUESTIONS")
    print("="*70)

    problematic_questions = []

    for q_idx, q_data in per_question_data.items():
        if len(q_data) < 3:
            continue

        row_sums = np.array([d[1] for d in q_data])
        correct = np.array([d[2] for d in q_data], dtype=float)

        if np.std(row_sums) > 1e-6 and np.std(correct) > 1e-6:
            r, p = pearsonr(row_sums, correct)

            if r < -0.2:  # Significant negative correlation
                problematic_questions.append((q_idx, r, p))

    if problematic_questions:
        print(f"\nFound {len(problematic_questions)} questions with NEGATIVE correlation:")
        print(f"(These questions cause algorithm to prefer wrong answers)")
        print()

        problematic_questions.sort(key=lambda x: x[1])  # Sort by correlation

        for q_idx, r, p in problematic_questions[:10]:  # Show worst 10
            result = results[q_idx]
            question = result['question'][:60] + "..." if len(result['question']) > 60 else result['question']
            print(f"  Q{q_idx:3d}: r={r:+.3f} (p={p:.3f})")
            print(f"       {question}")
            print()
    else:
        print("\n✓ No questions with strong negative correlation")

    # Create visualization if matplotlib available
    print()
    print("="*70)
    print("VISUALIZATION")
    print("="*70)

    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Overall scatter
        ax = axes[0, 0]
        colors = ['red' if c == 0 else 'green' for c in all_correct]
        ax.scatter(all_row_sums, all_correct + np.random.normal(0, 0.02, len(all_correct)),
                  c=colors, alpha=0.3, s=10)
        ax.set_xlabel('Row Sum (Normalized)')
        ax.set_ylabel('Correctness (0=Wrong, 1=Correct)')
        ax.set_title(f'Overall: r={pearson_r:.3f}, p={pearson_p:.3e}')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

        # Plot 2: Distribution comparison
        ax = axes[0, 1]
        ax.hist(correct_row_sums, bins=20, alpha=0.5, label='Correct', color='green')
        ax.hist(incorrect_row_sums, bins=20, alpha=0.5, label='Incorrect', color='red')
        ax.set_xlabel('Row Sum')
        ax.set_ylabel('Frequency')
        ax.set_title('Row Sum Distribution by Correctness')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Per-agent correlation
        ax = axes[1, 0]
        if agent_correlations:
            agent_indices = [a[0] for a in agent_correlations]
            agent_rs = [a[1] for a in agent_correlations]
            colors = ['green' if r > 0.1 else 'red' if r < -0.1 else 'gray' for r in agent_rs]

            ax.bar(agent_indices, agent_rs, color=colors, alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Reliable threshold')
            ax.axhline(y=-0.1, color='red', linestyle='--', alpha=0.5, label='Inverted threshold')
            ax.set_xlabel('Agent Index')
            ax.set_ylabel('Correlation (r)')
            ax.set_title('Per-Agent: Row Sum vs Correctness')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 4: Cumulative correctness by row sum quantile
        ax = axes[1, 1]
        # Bin by row sum quantiles
        quantiles = np.percentile(all_row_sums, [0, 25, 50, 75, 100])
        quantile_labels = ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
        quantile_accuracy = []

        for i in range(4):
            mask = (all_row_sums >= quantiles[i]) & (all_row_sums < quantiles[i+1])
            if i == 3:  # Include upper bound for last quantile
                mask = all_row_sums >= quantiles[i]

            if mask.sum() > 0:
                accuracy = all_correct[mask].mean()
                quantile_accuracy.append(accuracy)
            else:
                quantile_accuracy.append(0)

        colors = ['green' if a > 0.5 else 'red' for a in quantile_accuracy]
        ax.bar(quantile_labels, quantile_accuracy, color=colors, alpha=0.7)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy by Row Sum Quantile')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        for i, acc in enumerate(quantile_accuracy):
            ax.text(i, acc + 0.02, f'{acc:.2%}', ha='center', va='bottom')

        plt.tight_layout()

        # Save plot
        output_dir = Path(results_file).parent
        plot_file = output_dir / "rowsum_correctness_analysis.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {plot_file}")

        # Also show if possible
        # plt.show()  # Commented out for non-interactive environments

    except Exception as e:
        print(f"\nVisualization failed: {e}")
        print("(This is OK - the numerical analysis is complete)")

    # Final conclusion
    print()
    print("="*70)
    print("FINAL CONCLUSION")
    print("="*70)
    print()

    if pearson_r > 0.2:
        print("✓ Row sum correlates with correctness (r > 0.2)")
        print("  → Bradley-Terry assumption approximately holds")
        print("  → Algorithm should provide some benefit over majority voting")
        print()
        print("Recommendation: Proceed with current algorithms")

    elif pearson_r > 0:
        print("⚠️  Weak correlation (0 < r < 0.2)")
        print("  → Bradley-Terry assumption only weakly holds")
        print("  → Algorithm may provide marginal benefit")
        print()
        print("Recommendation: Report limitations, consider hybrid approaches")

    else:
        print("❌ NO or NEGATIVE correlation (r ≤ 0)")
        print("  → Bradley-Terry assumption VIOLATED")
        print("  → Algorithm fundamentally cannot work")
        print()
        print("Key insight: Comparison agreements do not reflect answer quality.")
        print("This happens when agents judge based on:")
        print("  - Confidence/verbosity rather than correctness")
        print("  - Wrong answer clusters have high internal agreement")
        print("  - Correct minority appears inconsistent")
        print()
        print("Recommendation: Report as negative result showing model limitations")

    print("="*70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_rowsum_correctness.py <results_file>")
        print()
        print("Example:")
        print("  python analyze_rowsum_correctness.py results/final_results_20240213.json")
        sys.exit(1)

    results_file = sys.argv[1]

    if not Path(results_file).exists():
        print(f"Error: File not found: {results_file}")
        sys.exit(1)

    analyze_rowsum_correctness_correlation(results_file)
