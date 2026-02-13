"""
Diagnostic script to understand why majority voting beats algorithms.
"""

import json
import numpy as np
from pathlib import Path

def diagnose_results(results_file):
    """Analyze why algorithms underperform vs majority voting."""

    with open(results_file) as f:
        data = json.load(f)

    results = data['results']

    print("="*70)
    print("DIAGNOSTIC ANALYSIS: Why Majority Voting Wins")
    print("="*70)

    # 1. Answer diversity
    print("\n1. ANSWER DIVERSITY")
    print("-"*70)
    diversities = []
    for r in results[:10]:
        answers = [a['answer'] for a in r['answers']]
        unique = len(set(answers))
        diversities.append(unique / len(answers))
        print(f"Q{r['question_idx']:2d}: {unique:2d}/{len(answers):2d} unique answers ({unique/len(answers):.1%})")

    avg_diversity = np.mean(diversities)
    print(f"\nAverage diversity: {avg_diversity:.1%}")
    if avg_diversity < 0.4:
        print("⚠️  WARNING: Low diversity! Problems may be too easy.")

    # 2. Self-preference bias
    print("\n2. SELF-PREFERENCE BIAS")
    print("-"*70)
    self_prefs = []
    for r in results[:10]:
        R = np.array(r['comparison_matrix'])
        # Each agent compares with N-1 others, should win ~50% if unbiased
        row_sums = (R.sum(axis=1) - 1) / (len(R) - 1)  # Exclude diagonal
        self_pref = row_sums.mean()
        self_prefs.append(self_pref)
        print(f"Q{r['question_idx']:2d}: Average row sum = {self_pref:.2f} (0=always loses, 1=always wins)")

    avg_self_pref = np.mean(self_prefs)
    print(f"\nAverage self-preference: {avg_self_pref:.2f}")
    if avg_self_pref > 0.6:
        print("⚠️  WARNING: Strong self-preference bias! Agents favor own answers.")

    # 3. Algorithm weight accuracy
    print("\n3. ALGORITHM WEIGHT ACCURACY")
    print("-"*70)

    weight_correlations = {'ccrr': [], 'swra': []}

    for r in results:
        answers = [a['answer'] for a in r['answers']]
        ground_truth = r['ground_truth']

        # True quality: 1 if correct, 0 if wrong
        from src.utils.dataset import compare_answers
        true_quality = [1.0 if compare_answers(ans, ground_truth) else 0.0
                       for ans in answers]

        # Algorithm weights
        ccrr_weights = r['ccrr_weights']
        swra_weights = r['swra_weights']

        # Correlation
        if np.std(true_quality) > 0:  # At least some variation
            ccrr_corr = np.corrcoef(ccrr_weights, true_quality)[0, 1]
            swra_corr = np.corrcoef(swra_weights, true_quality)[0, 1]
            weight_correlations['ccrr'].append(ccrr_corr)
            weight_correlations['swra'].append(swra_corr)

    print(f"CCRR weight vs true quality correlation: {np.mean(weight_correlations['ccrr']):.3f}")
    print(f"SWRA weight vs true quality correlation: {np.mean(weight_correlations['swra']):.3f}")

    if np.mean(weight_correlations['ccrr']) < 0.2:
        print("⚠️  WARNING: Weights barely correlate with quality! Algorithm failing.")

    # 4. When algorithms fail
    print("\n4. WHEN ALGORITHMS FAIL VS MAJORITY VOTING")
    print("-"*70)

    mv_right_algo_wrong = []
    algo_right_mv_wrong = []

    for r in results:
        mv_correct = r['baseline_validations']['majority_voting']['correct']
        ccrr_correct = r['ccrr_correct']

        if mv_correct and not ccrr_correct:
            mv_right_algo_wrong.append(r['question_idx'])
        elif not mv_correct and ccrr_correct:
            algo_right_mv_wrong.append(r['question_idx'])

    print(f"MV right, Algorithm wrong: {len(mv_right_algo_wrong)} cases")
    print(f"Algorithm right, MV wrong: {len(algo_right_mv_wrong)} cases")
    print(f"Net advantage: {'MV' if len(mv_right_algo_wrong) > len(algo_right_mv_wrong) else 'Algorithm'}")

    if len(mv_right_algo_wrong) > 0:
        print(f"\nFailed question indices: {mv_right_algo_wrong[:5]}")

    # 5. Summary diagnosis
    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)

    issues = []
    if avg_diversity < 0.4:
        issues.append("Low answer diversity (problems too easy)")
    if avg_self_pref > 0.6:
        issues.append("Strong self-preference bias (model artifacts)")
    if np.mean(weight_correlations['ccrr']) < 0.2:
        issues.append("Algorithm weights don't match quality (wrong model)")

    if not issues:
        print("✓ No obvious issues detected. Check parameter tuning (β, ε).")
    else:
        print("Issues detected:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

    print("\n" + "="*70)

if __name__ == "__main__":
    import sys

    # Find latest results file
    results_dir = Path("results")
    if not results_dir.exists():
        print("No results directory found. Run experiments first.")
        sys.exit(1)

    # Get latest results file
    results_files = list(results_dir.glob("**/final_results_*.json"))
    if not results_files:
        print("No results files found. Run experiments first.")
        sys.exit(1)

    latest = max(results_files, key=lambda p: p.stat().st_mtime)
    print(f"Analyzing: {latest}\n")

    diagnose_results(latest)
