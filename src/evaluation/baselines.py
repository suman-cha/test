"""
Baseline methods for comparison.

This module implements various baseline answer selection methods
to compare against the Hammer-Spammer algorithm.
"""

import numpy as np
from typing import List, Dict, Any
from collections import Counter


class Baselines:
    """Collection of baseline methods for answer selection."""

    @staticmethod
    def majority_voting(answers: List[str]) -> int:
        """
        Select most common answer (Majority Voting).

        Args:
            answers: List of N answer strings

        Returns:
            Index of most popular answer (first occurrence if tie)
        """
        # Import normalize_answer from run_experiment
        # (we need proper numeric normalization: "8" = "8.00")
        import sys
        import re

        def normalize_answer_local(s: str) -> str:
            """Normalize answer (same logic as run_experiment)."""
            s = str(s).strip().lower()
            s = s.replace(',', '').replace('$', '').replace('%', '')
            s = s.replace('\\', '').replace('{', '').replace('}', '')
            s = s.strip('.')

            # Try extracting and normalizing a number
            match = re.search(r'-?\d+\.?\d*', s)
            if match:
                num_str = match.group()
                try:
                    num = float(num_str)
                    if num == int(num):
                        return str(int(num))
                    else:
                        return str(num)
                except ValueError:
                    return num_str
            return s

        # Normalize answers for comparison
        normalized = [normalize_answer_local(ans) for ans in answers]

        # Count occurrences
        counter = Counter(normalized)

        # Find most common
        if not counter:
            return 0

        most_common_answer = counter.most_common(1)[0][0]

        # Return index of first occurrence of most common answer
        for i, norm_ans in enumerate(normalized):
            if norm_ans == most_common_answer:
                return i

        return 0

    @staticmethod
    def weighted_majority_voting(answers: List[str], weights: List[float]) -> int:
        """
        Weighted majority voting based on agent quality.

        Args:
            answers: List of N answer strings
            weights: List of N weights (e.g., agent quality scores)

        Returns:
            Index of answer with highest weighted vote
        """
        if len(answers) != len(weights):
            raise ValueError("Number of answers must match number of weights")

        # Normalize answers
        normalized = [ans.strip().lower() for ans in answers]

        # Accumulate weighted votes
        vote_weights = {}
        for i, norm_ans in enumerate(normalized):
            if norm_ans not in vote_weights:
                vote_weights[norm_ans] = 0.0
            vote_weights[norm_ans] += weights[i]

        # Find answer with highest weight
        best_answer = max(vote_weights.items(), key=lambda x: x[1])[0]

        # Return index of first occurrence
        for i, norm_ans in enumerate(normalized):
            if norm_ans == best_answer:
                return i

        return 0

    @staticmethod
    def random_selection(N: int, seed: int = None) -> int:
        """
        Random baseline - select random answer.

        Args:
            N: Number of answers
            seed: Random seed for reproducibility

        Returns:
            Random index in [0, N-1]
        """
        if seed is not None:
            np.random.seed(seed)
        return np.random.randint(0, N)

    @staticmethod
    def best_single_model() -> int:
        """
        Best single model baseline - always select the strongest model's answer.

        Assumes the first agent (index 0) is the strongest model (e.g., GPT-4o).
        This baseline tests whether using multiple models provides any benefit
        over simply using the best available model.

        Returns:
            Always returns 0 (strongest model's answer)
        """
        return 0

    @staticmethod
    def oracle_voting(answers: List[str], oracle_scores: List[float]) -> int:
        """
        Select answer with highest oracle score.

        This is a "cheating" baseline that uses oracle knowledge.
        Used for upper bound comparison.

        Args:
            answers: List of N answer strings
            oracle_scores: Oracle quality scores for each answer

        Returns:
            Index of answer with highest oracle score
        """
        if len(answers) != len(oracle_scores):
            raise ValueError("Number of answers must match number of oracle scores")

        return int(np.argmax(oracle_scores))


    @staticmethod
    def row_sum(comparison_matrix: np.ndarray) -> int:
        """
        Row-sum baseline: select agent with highest row sum in R.

        This is the most natural rank-aggregation baseline for pairwise
        comparison matrices. The row sum counts how many agents each
        agent judges itself to be better than.

        Args:
            comparison_matrix: N×N comparison matrix R with values in {-1, 1}

        Returns:
            Index of agent with highest row sum (excluding diagonal)
        """
        R = comparison_matrix.astype(float).copy()
        np.fill_diagonal(R, 0.0)  # Exclude self-comparisons
        row_sums = R.sum(axis=1)
        return int(np.argmax(row_sums))

    @staticmethod
    def column_sum(comparison_matrix: np.ndarray) -> int:
        """
        Column-sum baseline: select agent with lowest column sum in R.

        R[j,i] = -1 means agent j thinks agent i is better.
        A low column sum means many agents think this agent is better.

        Args:
            comparison_matrix: N×N comparison matrix R with values in {-1, 1}

        Returns:
            Index of agent with lowest column sum (best peer assessment)
        """
        R = comparison_matrix.astype(float).copy()
        np.fill_diagonal(R, 0.0)  # Exclude self-comparisons
        col_sums = R.sum(axis=0)
        return int(np.argmin(col_sums))

    @staticmethod
    def borda_count(comparison_matrix: np.ndarray) -> int:
        """
        Borda count baseline: combine row and column rankings.

        Averages the row-sum rank and column-sum rank for each agent.

        Args:
            comparison_matrix: N×N comparison matrix R with values in {-1, 1}

        Returns:
            Index of agent with best combined Borda score
        """
        R = comparison_matrix.astype(float).copy()
        np.fill_diagonal(R, 0.0)

        row_sums = R.sum(axis=1)
        col_sums = -R.sum(axis=0)  # Negate so higher = better

        # Convert to ranks (higher value = better rank)
        row_ranks = np.argsort(np.argsort(row_sums)).astype(float)
        col_ranks = np.argsort(np.argsort(col_sums)).astype(float)

        combined = row_ranks + col_ranks
        return int(np.argmax(combined))

    @staticmethod
    def apply_all_baselines(answers: List[str],
                           comparison_matrix: np.ndarray = None,
                           weights: List[float] = None) -> Dict[str, int]:
        """
        Apply all applicable baselines.

        Args:
            answers: List of N answer strings
            comparison_matrix: Optional N×N comparison matrix (unused)
            weights: Optional agent weights (unused)

        Returns:
            Dictionary mapping baseline name to selected index
        """
        N = len(answers)
        results = {}

        # Core baselines: Random and Majority Voting only
        results['random'] = Baselines.random_selection(N, seed=42)
        results['majority_voting'] = Baselines.majority_voting(answers)

        return results


def compare_baseline_performance(baseline_results: List[Dict[str, Any]],
                                ground_truths: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Compare performance of different baselines.

    Args:
        baseline_results: List of result dicts, each containing:
            - answers: List[str]
            - baseline_selections: Dict[str, int] (baseline name -> selected index)
        ground_truths: List of ground truth answers

    Returns:
        Dictionary mapping baseline name to performance metrics
    """
    from ..utils.dataset import compare_answers

    if len(baseline_results) != len(ground_truths):
        raise ValueError("Number of results must match number of ground truths")

    # Collect all baseline names
    all_baselines = set()
    for result in baseline_results:
        all_baselines.update(result['baseline_selections'].keys())

    # Initialize counters
    performance = {
        baseline: {'correct': 0, 'total': 0, 'accuracy': 0.0}
        for baseline in all_baselines
    }

    # Evaluate each question
    for result, ground_truth in zip(baseline_results, ground_truths):
        answers = result['answers']
        selections = result['baseline_selections']

        for baseline, selected_idx in selections.items():
            selected_answer = answers[selected_idx]
            is_correct = compare_answers(selected_answer, ground_truth)

            performance[baseline]['total'] += 1
            if is_correct:
                performance[baseline]['correct'] += 1

    # Calculate accuracies
    for baseline in performance:
        total = performance[baseline]['total']
        if total > 0:
            performance[baseline]['accuracy'] = performance[baseline]['correct'] / total * 100

    return performance


def print_baseline_comparison(performance: Dict[str, Dict[str, Any]]):
    """Print baseline performance comparison."""
    print(f"\n{'='*60}")
    print(f"Baseline Performance Comparison")
    print(f"{'='*60}\n")

    # Sort by accuracy
    sorted_baselines = sorted(performance.items(), key=lambda x: x[1]['accuracy'], reverse=True)

    print(f"{'Baseline':<25} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 60)

    for baseline, stats in sorted_baselines:
        print(f"{baseline:<25} {stats['correct']:<10} {stats['total']:<10} {stats['accuracy']:.2f}%")


if __name__ == "__main__":
    # Test baselines
    print("Testing Baseline Methods...")

    # Test data
    test_answers = ["42", "42", "43", "42", "40"]
    test_weights = [1.0, 0.8, 0.6, 0.9, 0.5]
    N = len(test_answers)

    # Test comparison matrix (example)
    R = np.array([
        [ 1,  1,  1,  1,  1],
        [ 1,  1,  1,  1, -1],
        [-1, -1,  1, -1, -1],
        [ 1,  1,  1,  1,  1],
        [-1, -1,  1, -1,  1]
    ])

    print(f"\nTest answers: {test_answers}")
    print(f"Test weights: {test_weights}")
    print(f"\nComparison matrix R:")
    print(R)

    # Apply all baselines
    print("\n\n=== Baseline Results ===")
    results = Baselines.apply_all_baselines(test_answers, R, test_weights)

    for baseline, idx in results.items():
        print(f"{baseline:<25} -> Index {idx} (Answer: {test_answers[idx]})")

    # Test majority voting specifically
    print("\n\n=== Testing Majority Voting ===")
    test_cases = [
        (["A", "A", "B", "A", "C"], "A (index 0)"),
        (["1", "2", "1", "1", "3"], "1 (index 0)"),
        (["X", "Y", "X", "Z", "X"], "X (index 0)")
    ]

    for answers, expected in test_cases:
        idx = Baselines.majority_voting(answers)
        print(f"Answers: {answers}")
        print(f"  Selected: {answers[idx]} at index {idx}")
        print(f"  Expected: {expected}")
        print()
