"""
Debug Tool: Analyze why algorithm failed on specific question.

This tool extracts detailed information about a failed question to
understand the "wrong answer cluster" phenomenon.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys


def analyze_failure_case(results_file: str, question_idx: int = None):
    """
    Deep analysis of a failure case.

    Args:
        results_file: Path to results JSON
        question_idx: Specific question to analyze (None = find failures)
    """
    # Load results
    with open(results_file) as f:
        data = json.load(f)

    results = data['results']

    # Find failure cases if not specified
    if question_idx is None:
        failure_cases = []
        for r in results:
            # Check if algorithm wrong but MV right
            algo_correct = r['ccrr_correct']
            mv_correct = r['baseline_validations']['majority_voting']['correct']

            if not algo_correct and mv_correct:
                failure_cases.append(r['question_idx'])

        if not failure_cases:
            print("No failure cases found!")
            return

        print(f"Found {len(failure_cases)} failure cases:")
        print(f"Question indices: {failure_cases}")
        question_idx = failure_cases[0]
        print(f"\nAnalyzing first failure: Question {question_idx}")

    # Get the specific result
    result = results[question_idx]

    print("="*70)
    print(f"FAILURE CASE ANALYSIS: Question {question_idx}")
    print("="*70)

    # Basic info
    question = result['question']
    ground_truth = result['ground_truth']
    answers = result['answers']

    print(f"\nQuestion: {question[:100]}...")
    print(f"Ground Truth: {ground_truth}")
    print()

    # Answers
    print("="*70)
    print("ANSWERS FROM AGENTS")
    print("="*70)

    answer_strings = [a['answer'] for a in answers]
    N = len(answer_strings)

    for i, ans in enumerate(answer_strings):
        is_correct = _check_answer_match(ans, ground_truth)
        marker = "✓" if is_correct else "✗"
        print(f"Agent {i:2d} {marker}: {ans}")

    print()

    # Selections
    ccrr_idx = result['ccrr_selected_idx']
    swra_idx = result['swra_selected_idx']
    mv_selections = result['baseline_selections']
    mv_idx = mv_selections['majority_voting']

    print("="*70)
    print("SELECTIONS")
    print("="*70)
    print(f"CCRR selected:  Agent {ccrr_idx} → '{answer_strings[ccrr_idx]}' "
          f"{'✓ CORRECT' if result['ccrr_correct'] else '✗ WRONG'}")
    print(f"SWRA selected:  Agent {swra_idx} → '{answer_strings[swra_idx]}' "
          f"{'✓ CORRECT' if result['swra_correct'] else '✗ WRONG'}")
    print(f"MV selected:    Agent {mv_idx} → '{answer_strings[mv_idx]}' "
          f"{'✓ CORRECT' if result['baseline_validations']['majority_voting']['correct'] else '✗ WRONG'}")
    print()

    # Comparison matrix
    R = np.array(result['comparison_matrix'])

    # Weights and scores
    ccrr_weights = np.array(result['ccrr_weights'])
    ccrr_scores = np.array(result['ccrr_scores'])
    swra_weights = np.array(result['swra_weights'])

    # Identify correct and wrong answer agents
    correct_agents = []
    wrong_agents = []
    answer_groups = {}  # answer -> [agent_indices]

    for i, ans in enumerate(answer_strings):
        is_correct = _check_answer_match(ans, ground_truth)
        if is_correct:
            correct_agents.append(i)
        else:
            wrong_agents.append(i)

        ans_key = ans.strip().lower()
        if ans_key not in answer_groups:
            answer_groups[ans_key] = []
        answer_groups[ans_key].append(i)

    print("="*70)
    print("ANSWER GROUPS")
    print("="*70)
    for ans, agents in sorted(answer_groups.items(), key=lambda x: len(x[1]), reverse=True):
        is_correct = _check_answer_match(ans, ground_truth)
        marker = "✓" if is_correct else "✗"
        print(f"{marker} '{ans}': {len(agents)} agents {agents}")
    print()

    # CRITICAL ANALYSIS: Weights for correct vs wrong
    print("="*70)
    print("CRITICAL: WEIGHTS FOR CORRECT VS WRONG AGENTS")
    print("="*70)

    if correct_agents:
        print(f"\nCorrect agents (answered '{ground_truth}'):")
        for i in correct_agents:
            row_sum = (R[i, :].sum() - R[i, i]) / (N - 1)  # Exclude diagonal
            print(f"  Agent {i:2d}: weight={ccrr_weights[i]:.3f}, "
                  f"score={ccrr_scores[i]:+.3f}, row_sum={row_sum:.3f}")

        avg_correct_weight = np.mean(ccrr_weights[correct_agents])
        avg_correct_score = np.mean(ccrr_scores[correct_agents])
        print(f"  AVERAGE: weight={avg_correct_weight:.3f}, score={avg_correct_score:+.3f}")
    else:
        print("\nNo agents answered correctly!")

    # Wrong agents (focus on selected wrong answer)
    selected_wrong_answer = answer_strings[ccrr_idx]
    wrong_cluster_agents = answer_groups.get(selected_wrong_answer.strip().lower(), [])

    if wrong_cluster_agents and ccrr_idx in wrong_cluster_agents:
        print(f"\nWrong answer cluster (answered '{selected_wrong_answer}'):")
        for i in wrong_cluster_agents:
            row_sum = (R[i, :].sum() - R[i, i]) / (N - 1)
            print(f"  Agent {i:2d}: weight={ccrr_weights[i]:.3f}, "
                  f"score={ccrr_scores[i]:+.3f}, row_sum={row_sum:.3f}")

        avg_wrong_weight = np.mean(ccrr_weights[wrong_cluster_agents])
        avg_wrong_score = np.mean(ccrr_scores[wrong_cluster_agents])
        print(f"  AVERAGE: weight={avg_wrong_weight:.3f}, score={avg_wrong_score:+.3f}")

    # Comparison
    print()
    print("-"*70)
    if correct_agents and wrong_cluster_agents:
        print("COMPARISON:")
        print(f"  Correct agents avg weight: {avg_correct_weight:.3f}")
        print(f"  Wrong cluster avg weight:  {avg_wrong_weight:.3f}")
        print(f"  Difference: {avg_wrong_weight - avg_correct_weight:+.3f}")
        print()

        if avg_wrong_weight > avg_correct_weight:
            print("  ❌ PROBLEM: Wrong agents have HIGHER weights than correct!")
            print("     → Algorithm thinks wrong cluster are 'hammers'")
        else:
            print("  ⚠️  PARADOX: Correct agents have higher weights, but wrong score won")
            print("     → Check score computation")

    # Row-norm analysis (Phase 1)
    print()
    print("="*70)
    print("PHASE 1: ROW-NORM ANALYSIS")
    print("="*70)

    # Compute row-norms (ρ_i)
    M = R.astype(float).copy()
    np.fill_diagonal(M, 0.0)
    G = M @ M.T / (N - 2) if N > 2 else M @ M.T
    np.fill_diagonal(G, 0.0)
    rho = np.sum(G ** 2, axis=1)

    if correct_agents:
        print(f"\nCorrect agents row-norms (ρ_i):")
        for i in correct_agents:
            print(f"  Agent {i:2d}: ρ={rho[i]:.3f}")
        avg_correct_rho = np.mean(rho[correct_agents])
        print(f"  AVERAGE: ρ={avg_correct_rho:.3f}")

    if wrong_cluster_agents:
        print(f"\nWrong cluster row-norms:")
        for i in wrong_cluster_agents:
            print(f"  Agent {i:2d}: ρ={rho[i]:.3f}")
        avg_wrong_rho = np.mean(rho[wrong_cluster_agents])
        print(f"  AVERAGE: ρ={avg_wrong_rho:.3f}")

    tau = np.median(rho) / 2.0
    print(f"\nThreshold τ = {tau:.3f}")
    print(f"Agents with ρ > τ classified as hammers")

    if correct_agents:
        correct_above_threshold = sum(1 for i in correct_agents if rho[i] > tau)
        print(f"\nCorrect agents above threshold: {correct_above_threshold}/{len(correct_agents)}")

    if wrong_cluster_agents:
        wrong_above_threshold = sum(1 for i in wrong_cluster_agents if rho[i] > tau)
        print(f"Wrong cluster above threshold: {wrong_above_threshold}/{len(wrong_cluster_agents)}")

    # Agreement analysis
    print()
    print("="*70)
    print("AGREEMENT ANALYSIS")
    print("="*70)

    if len(wrong_cluster_agents) >= 2:
        # Compute pairwise agreement within wrong cluster
        wrong_agreements = []
        for i in wrong_cluster_agents:
            for j in wrong_cluster_agents:
                if i < j:
                    agreement = R[i, :] @ R[j, :]  # Dot product
                    wrong_agreements.append(agreement / N)

        avg_wrong_agreement = np.mean(wrong_agreements) if wrong_agreements else 0
        print(f"Average agreement within wrong cluster: {avg_wrong_agreement:.3f}")

    if len(correct_agents) >= 2:
        correct_agreements = []
        for i in correct_agents:
            for j in correct_agents:
                if i < j:
                    agreement = R[i, :] @ R[j, :]
                    correct_agreements.append(agreement / N)

        avg_correct_agreement = np.mean(correct_agreements) if correct_agreements else 0
        print(f"Average agreement within correct group: {avg_correct_agreement:.3f}")

    # Cross-agreement
    if correct_agents and wrong_cluster_agents:
        cross_agreements = []
        for i in correct_agents:
            for j in wrong_cluster_agents:
                agreement = R[i, :] @ R[j, :]
                cross_agreements.append(agreement / N)

        avg_cross_agreement = np.mean(cross_agreements)
        print(f"Average agreement between correct and wrong: {avg_cross_agreement:.3f}")

    # DIAGNOSIS
    print()
    print("="*70)
    print("DIAGNOSIS")
    print("="*70)

    diagnosis = []

    if correct_agents and wrong_cluster_agents:
        if avg_wrong_weight > avg_correct_weight:
            diagnosis.append("✗ Wrong cluster has higher weights than correct")

        if 'avg_wrong_rho' in locals() and 'avg_correct_rho' in locals():
            if avg_wrong_rho > avg_correct_rho:
                diagnosis.append("✗ Wrong cluster has higher row-norms (misclassified as hammers)")

        if 'avg_wrong_agreement' in locals() and 'avg_correct_agreement' in locals():
            if avg_wrong_agreement > avg_correct_agreement:
                diagnosis.append("✗ Wrong cluster has higher internal agreement")

    if diagnosis:
        print("\nProblems detected:")
        for i, issue in enumerate(diagnosis, 1):
            print(f"  {i}. {issue}")

        print("\n" + "="*70)
        print("CONCLUSION: Wrong Answer Cluster Hypothesis CONFIRMED")
        print("="*70)
        print("""
The algorithm is systematically preferring a cluster of wrong answers because:

1. Agents giving the same wrong answer agree with each other
2. This high agreement → high row-norms → classified as "hammers"
3. Correct agents are minority → low agreement → classified as "spammers"
4. Algorithm selects from the wrong "hammer" cluster

This is a FUNDAMENTAL FLAW in the hammer-spammer model when:
- Wrong answers are more common than correct ones
- Agents judge based on confidence/length rather than correctness
- Agreement does not imply correctness
        """)
    else:
        print("\nNo clear pattern detected. Further investigation needed.")

    print("="*70)


def _check_answer_match(answer: str, ground_truth: str) -> bool:
    """Check if answer matches ground truth."""
    from src.utils.dataset import compare_answers
    return compare_answers(answer, ground_truth)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_failure_case.py <results_file> [question_idx]")
        print("\nExample:")
        print("  python debug_failure_case.py results/final_results_20240213.json")
        print("  python debug_failure_case.py results/final_results_20240213.json 5")
        sys.exit(1)

    results_file = sys.argv[1]
    question_idx = int(sys.argv[2]) if len(sys.argv) > 2 else None

    analyze_failure_case(results_file, question_idx)
