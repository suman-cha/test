"""
Practical Weighted Voting: Empirically-grounded approach for LLM agents.

Key insight: For real LLMs, we should directly observe and track agent
quality rather than inferring it from comparison matrices.
"""

import numpy as np
from typing import Tuple, Dict, List
from collections import Counter, defaultdict


class PracticalWeightedVoting:
    """
    Weighted voting that learns from actual outcomes.

    Combines:
    1. Empirical agent quality (from past correctness)
    2. Answer popularity (majority voting signal)
    3. Optional: Comparison matrix as tiebreaker
    """

    def __init__(self, learning_rate: float = 0.1):
        """
        Initialize practical weighted voting.

        Args:
            learning_rate: How quickly to update agent weights (0-1)
        """
        self.learning_rate = learning_rate
        self.agent_weights = {}  # agent_idx -> weight
        self.agent_updates = defaultdict(int)  # Track update count

    def select_best(self, R: np.ndarray = None, answers: List[str] = None,
                   return_details: bool = False) -> int:
        """
        Select best answer using learned weights + majority voting.

        Args:
            R: Comparison matrix (optional, for tiebreaking)
            answers: List of answer strings (required)
            return_details: If True, return (idx, scores, weights)

        Returns:
            Best answer index (or tuple if return_details=True)
        """
        if answers is None:
            raise ValueError("answers must be provided")

        N = len(answers)

        # Get current weights for all agents
        weights = np.array([self.agent_weights.get(i, 0.5) for i in range(N)])

        # Compute weighted votes for each unique answer
        answer_scores = defaultdict(float)
        answer_to_idx = {}

        for i, ans in enumerate(answers):
            ans_key = ans.strip().lower()
            answer_scores[ans_key] += weights[i]
            if ans_key not in answer_to_idx:
                answer_to_idx[ans_key] = i

        # Select answer with highest weighted vote
        best_answer_key = max(answer_scores.items(), key=lambda x: x[1])[0]
        best_idx = answer_to_idx[best_answer_key]

        # Compute individual scores (for diagnostics)
        scores = np.array([answer_scores[answers[i].strip().lower()]
                          for i in range(N)])

        if return_details:
            return best_idx, scores, weights
        else:
            return best_idx

    def update_agent_quality(self, agent_idx: int, was_correct: bool):
        """
        Update agent weight based on observed correctness.

        Uses exponential moving average for stability.

        Args:
            agent_idx: Index of agent
            was_correct: Whether agent's answer was correct
        """
        current_weight = self.agent_weights.get(agent_idx, 0.5)
        target = 1.0 if was_correct else 0.0

        # Exponential moving average
        new_weight = (1 - self.learning_rate) * current_weight + \
                     self.learning_rate * target

        self.agent_weights[agent_idx] = new_weight
        self.agent_updates[agent_idx] += 1

    def get_diagnostics(self) -> Dict:
        """Get current state of learned weights."""
        return {
            'agent_weights': dict(self.agent_weights),
            'agent_updates': dict(self.agent_updates),
            'mean_weight': np.mean(list(self.agent_weights.values())) if self.agent_weights else 0.5,
            'weight_std': np.std(list(self.agent_weights.values())) if len(self.agent_weights) > 1 else 0.0
        }


class HybridSelector:
    """
    Hybrid: Start with algorithms, gracefully fall back to weighted majority voting.
    """

    def __init__(self, ccrr_algo=None, swra_algo=None):
        """
        Initialize hybrid selector.

        Args:
            ccrr_algo: CCRR algorithm instance (optional)
            swra_algo: SWRA algorithm instance (optional)
        """
        self.ccrr = ccrr_algo
        self.swra = swra_algo
        self.pwv = PracticalWeightedVoting(learning_rate=0.15)

        self.method_performance = {
            'ccrr': {'correct': 0, 'total': 0},
            'swra': {'correct': 0, 'total': 0},
            'pwv': {'correct': 0, 'total': 0},
            'mv': {'correct': 0, 'total': 0}
        }

    def select_best(self, R: np.ndarray, answers: List[str],
                   strategy: str = 'adaptive') -> Tuple[int, str]:
        """
        Select best answer using hybrid strategy.

        Args:
            R: Comparison matrix
            answers: List of answers
            strategy: 'adaptive', 'ensemble', or 'cascade'

        Returns:
            Tuple of (selected_idx, method_used)
        """
        if strategy == 'adaptive':
            return self._adaptive_select(R, answers)
        elif strategy == 'ensemble':
            return self._ensemble_select(R, answers)
        elif strategy == 'cascade':
            return self._cascade_select(R, answers)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _adaptive_select(self, R: np.ndarray, answers: List[str]) -> Tuple[int, str]:
        """Select using best-performing method so far."""
        # Get performance scores
        scores = {}
        for method, perf in self.method_performance.items():
            if perf['total'] > 0:
                scores[method] = perf['correct'] / perf['total']
            else:
                scores[method] = 0.5  # Neutral prior

        # Use best method
        best_method = max(scores.items(), key=lambda x: x[1])[0]

        if best_method == 'ccrr' and self.ccrr:
            idx = self.ccrr.select_best(R)
            return idx, 'ccrr'
        elif best_method == 'swra' and self.swra:
            idx = self.swra.select_best(R)
            return idx, 'swra'
        elif best_method == 'pwv':
            idx = self.pwv.select_best(answers=answers)
            return idx, 'pwv'
        else:  # mv (majority voting)
            idx = self._majority_voting(answers)
            return idx, 'mv'

    def _ensemble_select(self, R: np.ndarray, answers: List[str]) -> Tuple[int, str]:
        """Ensemble: each method votes, weighted by past performance."""
        N = len(answers)
        vote_weights = np.zeros(N)

        # Get votes from each method
        methods_and_weights = []

        if self.ccrr:
            idx = self.ccrr.select_best(R)
            perf = self.method_performance['ccrr']
            weight = perf['correct'] / perf['total'] if perf['total'] > 0 else 0.5
            vote_weights[idx] += weight
            methods_and_weights.append(('ccrr', weight))

        if self.swra:
            idx = self.swra.select_best(R)
            perf = self.method_performance['swra']
            weight = perf['correct'] / perf['total'] if perf['total'] > 0 else 0.5
            vote_weights[idx] += weight
            methods_and_weights.append(('swra', weight))

        # PWV
        idx = self.pwv.select_best(answers=answers)
        perf = self.method_performance['pwv']
        weight = perf['correct'] / perf['total'] if perf['total'] > 0 else 0.5
        vote_weights[idx] += weight
        methods_and_weights.append(('pwv', weight))

        # MV
        idx = self._majority_voting(answers)
        perf = self.method_performance['mv']
        weight = perf['correct'] / perf['total'] if perf['total'] > 0 else 0.5
        vote_weights[idx] += weight
        methods_and_weights.append(('mv', weight))

        # Select highest-voted answer
        best_idx = int(np.argmax(vote_weights))
        return best_idx, 'ensemble'

    def _cascade_select(self, R: np.ndarray, answers: List[str]) -> Tuple[int, str]:
        """Cascade: Try algorithms, fall back to MV if low confidence."""
        # Try CCRR first
        if self.ccrr:
            ccrr_idx, ccrr_scores, ccrr_weights = self.ccrr.select_best(R, return_details=True)

            # Check confidence: is the selected agent high-weight?
            if ccrr_weights[ccrr_idx] > 0.7:
                return ccrr_idx, 'ccrr'

        # Try SWRA
        if self.swra:
            swra_idx, swra_scores, swra_weights = self.swra.select_best(R, return_details=True)

            # Check confidence
            if swra_weights[swra_idx] > 0.7:
                return swra_idx, 'swra'

        # Fall back to practical weighted voting
        pwv_idx = self.pwv.select_best(answers=answers)
        return pwv_idx, 'pwv'

    def _majority_voting(self, answers: List[str]) -> int:
        """Simple majority voting."""
        normalized = [ans.strip().lower() for ans in answers]
        counter = Counter(normalized)
        most_common = counter.most_common(1)[0][0]

        # Return first index with this answer
        for i, ans in enumerate(normalized):
            if ans == most_common:
                return i
        return 0

    def update_performance(self, method: str, was_correct: bool):
        """Update performance tracking for a method."""
        if method in self.method_performance:
            self.method_performance[method]['total'] += 1
            if was_correct:
                self.method_performance[method]['correct'] += 1

    def update_agent_quality(self, agent_idx: int, was_correct: bool):
        """Forward to PWV for agent quality updates."""
        self.pwv.update_agent_quality(agent_idx, was_correct)


if __name__ == "__main__":
    print("Testing Practical Weighted Voting...")

    pwv = PracticalWeightedVoting(learning_rate=0.2)

    # Simulate learning
    np.random.seed(42)
    N = 5
    true_quality = [0.9, 0.7, 0.5, 0.3, 0.1]

    print("\n1. Learning phase (20 questions):")
    for q in range(20):
        for i, quality in enumerate(true_quality):
            was_correct = np.random.random() < quality
            pwv.update_agent_quality(i, was_correct)

    print(pwv.get_diagnostics())

    # Test selection
    print("\n2. Selection test:")
    test_answers = ["42", "42", "43", "42", "41"]
    selected = pwv.select_best(answers=test_answers)

    print(f"Answers: {test_answers}")
    print(f"Selected: agent {selected} = '{test_answers[selected]}'")
    print(f"Expected: '42' (from high-quality agents 0,1,3)")
