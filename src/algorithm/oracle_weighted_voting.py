"""
Oracle-Weighted Voting: Use observed quality to weight votes.

Instead of inferring agent quality from the comparison matrix R
(which doesn't work for LLMs), directly observe agent quality from
their actual answer correctness over multiple questions.
"""

import numpy as np
from typing import Tuple, List, Dict
from collections import defaultdict


class OracleWeightedVoting:
    """
    Weighted voting using empirically observed agent quality.

    This addresses the fundamental problem: the comparison matrix R
    doesn't reliably indicate agent quality for real LLMs.
    """

    def __init__(self, smoothing: float = 0.1):
        """
        Initialize oracle-weighted voting.

        Args:
            smoothing: Laplace smoothing parameter (default 0.1)
        """
        self.smoothing = smoothing
        self.agent_quality = {}  # agent_idx -> success rate
        self.agent_history = defaultdict(lambda: {'correct': 0, 'total': 0})

    def update_agent_quality(self, agent_idx: int, was_correct: bool):
        """
        Update observed quality for an agent.

        Args:
            agent_idx: Agent index
            was_correct: Whether this agent's answer was correct
        """
        self.agent_history[agent_idx]['total'] += 1
        if was_correct:
            self.agent_history[agent_idx]['correct'] += 1

    def get_agent_weight(self, agent_idx: int) -> float:
        """
        Get empirical quality weight for an agent.

        Uses Laplace smoothing to handle limited data.

        Args:
            agent_idx: Agent index

        Returns:
            Weight in [0, 1] representing estimated quality
        """
        if agent_idx not in self.agent_history:
            return 0.5  # Neutral prior

        correct = self.agent_history[agent_idx]['correct']
        total = self.agent_history[agent_idx]['total']

        # Laplace smoothing
        weight = (correct + self.smoothing) / (total + 2 * self.smoothing)
        return weight

    def select_best_with_learned_weights(self, answers: List[str]) -> Tuple[int, np.ndarray]:
        """
        Select best answer using learned agent quality weights.

        Args:
            answers: List of N answer strings

        Returns:
            Tuple of (best_idx, weights)
        """
        N = len(answers)
        weights = np.array([self.get_agent_weight(i) for i in range(N)])

        # Weighted majority voting
        # Group by answer, sum weights
        answer_weights = defaultdict(float)
        answer_to_idx = {}

        for i, ans in enumerate(answers):
            ans_normalized = ans.strip().lower()
            answer_weights[ans_normalized] += weights[i]
            if ans_normalized not in answer_to_idx:
                answer_to_idx[ans_normalized] = i

        # Select answer with highest total weight
        best_answer = max(answer_weights.items(), key=lambda x: x[1])[0]
        best_idx = answer_to_idx[best_answer]

        return best_idx, weights

    def select_best_with_comparison_weights(self, R: np.ndarray, answers: List[str]) -> Tuple[int, np.ndarray]:
        """
        Select best answer using both learned weights AND comparison matrix.

        Hybrid approach: Use learned weights to filter reliable judges,
        then use their comparison votes.

        Args:
            R: Comparison matrix (N, N)
            answers: List of N answer strings

        Returns:
            Tuple of (best_idx, weights)
        """
        N = len(answers)
        learned_weights = np.array([self.get_agent_weight(i) for i in range(N)])

        # Use learned weights to gate the row sums
        # Only trust comparisons from high-quality agents
        R_tilde = R.astype(float).copy()
        np.fill_diagonal(R_tilde, 0.0)

        # Weighted row sums: each agent's vote weighted by their quality
        weighted_row_sums = (R_tilde.T * learned_weights).T.sum(axis=1)

        # Weighted column sums: how much do good agents endorse each answer?
        weighted_col_sums = (R_tilde.T @ learned_weights) * -1  # Negative because R[i,j]=1 means i beats j

        # Combine
        scores = np.zeros(N)

        # Normalize and combine
        if weighted_row_sums.std() > 1e-6:
            scores += (weighted_row_sums - weighted_row_sums.mean()) / weighted_row_sums.std()
        if weighted_col_sums.std() > 1e-6:
            scores += (weighted_col_sums - weighted_col_sums.mean()) / weighted_col_sums.std()

        best_idx = int(np.argmax(scores))

        return best_idx, learned_weights


class AdaptiveAlgorithmSelector:
    """
    Meta-algorithm that adaptively selects which method to use.

    Tracks performance of multiple algorithms and selects the best one
    for each question based on past performance.
    """

    def __init__(self, algorithms: Dict[str, callable]):
        """
        Initialize adaptive selector.

        Args:
            algorithms: Dictionary of {name: selection_function}
        """
        self.algorithms = algorithms
        self.performance = {name: {'correct': 0, 'total': 0}
                           for name in algorithms}

    def update_performance(self, algorithm_name: str, was_correct: bool):
        """Update performance tracking for an algorithm."""
        self.performance[algorithm_name]['total'] += 1
        if was_correct:
            self.performance[algorithm_name]['correct'] += 1

    def get_algorithm_quality(self, algorithm_name: str) -> float:
        """Get success rate for an algorithm."""
        perf = self.performance[algorithm_name]
        if perf['total'] == 0:
            return 0.5  # Neutral prior
        return perf['correct'] / perf['total']

    def select_best_algorithm(self) -> str:
        """Select the algorithm with best historical performance."""
        if all(p['total'] == 0 for p in self.performance.values()):
            return list(self.algorithms.keys())[0]  # Default to first

        return max(self.algorithms.keys(),
                  key=lambda name: self.get_algorithm_quality(name))

    def select_with_ensemble(self, R: np.ndarray, answers: List[str]) -> int:
        """
        Use ensemble voting: each algorithm votes, weighted by its quality.

        Args:
            R: Comparison matrix
            answers: List of answers

        Returns:
            Selected answer index
        """
        N = len(answers)
        vote_weights = np.zeros(N)

        for name, algo_func in self.algorithms.items():
            algo_quality = self.get_algorithm_quality(name)
            selected_idx = algo_func(R, answers)
            vote_weights[selected_idx] += algo_quality

        return int(np.argmax(vote_weights))


def create_hybrid_system():
    """
    Create a hybrid system combining multiple approaches.

    Returns:
        Configured OracleWeightedVoting instance
    """
    return OracleWeightedVoting(smoothing=0.1)


if __name__ == "__main__":
    # Test oracle-weighted voting
    print("Testing Oracle-Weighted Voting...")

    owv = OracleWeightedVoting()

    # Simulate learning phase
    print("\n1. Learning agent quality from 10 questions:")
    np.random.seed(42)
    N = 5
    true_quality = [0.9, 0.7, 0.5, 0.3, 0.1]  # Agent quality

    for q in range(10):
        for agent_idx, quality in enumerate(true_quality):
            was_correct = np.random.random() < quality
            owv.update_agent_quality(agent_idx, was_correct)

    print("Learned weights:")
    for i in range(N):
        learned = owv.get_agent_weight(i)
        true = true_quality[i]
        print(f"  Agent {i}: learned={learned:.2f}, true={true:.2f}")

    # Test selection
    print("\n2. Using learned weights for selection:")
    test_answers = ["42", "42", "43", "42", "41"]
    selected_idx, weights = owv.select_best_with_learned_weights(test_answers)

    print(f"Answers: {test_answers}")
    print(f"Weights: {weights}")
    print(f"Selected: Answer from agent {selected_idx} = {test_answers[selected_idx]}")
    print(f"(High-quality agents 0,1 say '42', so should select '42')")
