"""
Validation system for evaluating selected answers.

This module provides two validation methods:
1. Ground Truth Validation - compare against dataset answers
2. Oracle Validation - use GPT-4o to evaluate quality
"""

import re
import requests
from typing import Dict, List, Any, Optional
from ..utils.dataset import normalize_answer, compare_answers


class Validator:
    """Validate selected answers against Ground Truth and Oracle Model."""

    def __init__(self, oracle_model: str = 'openai/gpt-4o', api_key: Optional[str] = None):
        """
        Initialize validator.

        Args:
            oracle_model: Strong model for validation (default GPT-4o)
            api_key: OpenRouter API key for oracle validation
        """
        self.oracle_model = oracle_model
        self.api_key = api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

        # Statistics
        self.total_validations = 0
        self.ground_truth_correct = 0
        self.oracle_calls = 0

    def validate_with_ground_truth(self, selected_answer: str, ground_truth: str) -> Dict[str, Any]:
        """
        Compare selected answer with ground truth.

        Args:
            selected_answer: The answer selected by the algorithm
            ground_truth: The ground truth answer from dataset

        Returns:
            Dictionary with:
                - correct: bool (whether answers match)
                - selected_answer_parsed: str (normalized selected answer)
                - ground_truth_parsed: str (normalized ground truth)
                - match_type: str (exact, numerical, or none)
        """
        self.total_validations += 1

        # Normalize both answers
        selected_norm = normalize_answer(selected_answer)
        ground_truth_norm = normalize_answer(ground_truth)

        # Check if they match
        is_correct = compare_answers(selected_answer, ground_truth)

        if is_correct:
            self.ground_truth_correct += 1

        # Determine match type
        if selected_norm == ground_truth_norm:
            match_type = 'exact'
        elif is_correct:
            match_type = 'numerical'
        else:
            match_type = 'none'

        return {
            'correct': is_correct,
            'selected_answer_parsed': selected_norm,
            'ground_truth_parsed': ground_truth_norm,
            'match_type': match_type
        }

    def validate_with_oracle(self, question: str, selected_answer: str,
                           all_answers: List[str],
                           all_reasonings: Optional[List[str]] = None,
                           selected_idx: Optional[int] = None,
                           max_retries: int = 3) -> Dict[str, Any]:
        """
        Use Oracle Model (GPT-4o) to evaluate quality.

        The oracle:
        1. Grades the selected answer (0-10 scale)
        2. Identifies the best answer among all answers
        3. Compares selected vs oracle's choice

        Args:
            question: The original question
            selected_answer: The answer selected by the algorithm
            all_answers: All N final answer strings
            all_reasonings: Full reasoning/solution from each agent (for quality assessment)
            selected_idx: Index of the selected answer (avoids ambiguity with duplicates)
            max_retries: Maximum retry attempts

        Returns:
            Dictionary with:
                - oracle_score: float (0-10 quality score)
                - oracle_best_idx: int (index of best answer according to oracle)
                - matches_oracle: bool (whether selection matches oracle)
                - oracle_reasoning: str (oracle's explanation)
        """
        if not self.api_key:
            raise ValueError("API key required for oracle validation")

        self.oracle_calls += 1

        # Resolve selected index safely (handle duplicate answers)
        if selected_idx is None:
            try:
                selected_idx = all_answers.index(selected_answer)
            except ValueError:
                selected_idx = 0

        # Format answers with full reasoning if available
        if all_reasonings:
            answers_text = "\n\n".join([
                f"Answer {i+1}:\nFinal Answer: {ans}\nReasoning: {reasoning}"
                for i, (ans, reasoning) in enumerate(zip(all_answers, all_reasonings))
            ])
        else:
            answers_text = "\n\n".join([
                f"Answer {i+1}: {ans}"
                for i, ans in enumerate(all_answers)
            ])

        prompt = f"""You are an expert mathematics evaluator. Evaluate the answers to the following question.

Question: {question}

Here are {len(all_answers)} different answers:

{answers_text}

Tasks:
1. Grade Answer {selected_idx + 1} on a scale of 0-10 for correctness and quality of reasoning.
2. Identify which answer is the BEST (most correct and well-reasoned).
3. Provide brief reasoning for your evaluation.

Respond in the following format:
SCORE: [0-10 score for Answer {selected_idx + 1}]
BEST: [number of the best answer]
REASONING: [your explanation]
"""

        messages = [
            {"role": "user", "content": prompt}
        ]

        # Make API call to oracle
        for attempt in range(max_retries):
            try:
                response = self._make_oracle_call(messages)

                if 'choices' in response and len(response['choices']) > 0:
                    content = response['choices'][0]['message']['content']

                    # Parse response
                    score = self._extract_score(content)
                    best_idx = self._extract_best_answer(content, len(all_answers))
                    reasoning = self._extract_reasoning(content)

                    # Check if selection matches oracle
                    matches_oracle = (selected_idx == best_idx)

                    return {
                        'oracle_score': score,
                        'oracle_best_idx': best_idx,
                        'matches_oracle': matches_oracle,
                        'oracle_reasoning': reasoning,
                        'selected_idx': selected_idx
                    }

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Oracle validation error (attempt {attempt+1}): {e}")
                    continue
                else:
                    # Return default values on failure
                    return {
                        'oracle_score': 5.0,
                        'oracle_best_idx': 0,
                        'matches_oracle': False,
                        'oracle_reasoning': f"Error: {e}",
                        'selected_idx': selected_idx
                    }

    def _make_oracle_call(self, messages: list) -> Dict[str, Any]:
        """Make API call to oracle model."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/hammer-spammer-validation",
            "X-Title": "Hammer-Spammer Oracle Validation"
        }

        payload = {
            "model": self.oracle_model,
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.0  # Deterministic for reproducibility
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Oracle API error {response.status_code}: {response.text}")

    def _extract_score(self, text: str) -> float:
        """Extract score from oracle response."""
        match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            return max(0.0, min(10.0, score))  # Clamp to [0, 10]

        # Fallback: look for any score/rating
        match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*10', text)
        if match:
            return float(match.group(1))

        return 5.0  # Default middle score

    def _extract_best_answer(self, text: str, num_answers: int) -> int:
        """Extract best answer index from oracle response."""
        match = re.search(r'BEST:\s*(?:Answer\s*)?(\d+)', text, re.IGNORECASE)
        if match:
            idx = int(match.group(1)) - 1  # Convert to 0-indexed
            if 0 <= idx < num_answers:
                return idx

        # Fallback: look for "Answer X is best"
        match = re.search(r'Answer\s*(\d+)\s+is\s+(?:the\s+)?best', text, re.IGNORECASE)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < num_answers:
                return idx

        return 0  # Default to first answer

    def _extract_reasoning(self, text: str) -> str:
        """Extract reasoning from oracle response."""
        match = re.search(r'REASONING:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

        # Return full text if no reasoning section found
        return text.strip()

    def evaluate_all_answers(self, question: str, all_answers: List[str],
                            all_reasonings: Optional[List[str]] = None,
                            ground_truth: Optional[str] = None,
                            max_retries: int = 3) -> Dict[str, Any]:
        """
        Evaluate ALL agent answers to estimate latent quality scores.

        This is for validating the algorithm's ability to identify high-quality
        agents (hammers) vs low-quality agents (spammers).

        Args:
            question: The original question
            all_answers: All N answers from agents
            all_reasonings: Full reasoning/solution from each agent (for quality assessment)
            ground_truth: Optional ground truth answer
            max_retries: Maximum retry attempts

        Returns:
            Dictionary with:
                - agent_scores_oracle: List[float] - Oracle's quality scores for each agent (0-10)
                - agent_scores_ground_truth: List[float] - Ground truth correctness (0 or 1)
                - oracle_best_idx: int - Oracle's choice for best answer
                - ground_truth_correct_indices: List[int] - Indices of correct answers
        """
        result = {
            'agent_scores_oracle': [],
            'agent_scores_ground_truth': [],
            'oracle_best_idx': None,
            'ground_truth_correct_indices': []
        }

        # Evaluate with ground truth if available
        if ground_truth:
            for i, answer in enumerate(all_answers):
                is_correct = compare_answers(answer, ground_truth)
                result['agent_scores_ground_truth'].append(1.0 if is_correct else 0.0)
                if is_correct:
                    result['ground_truth_correct_indices'].append(i)

        # Evaluate with oracle if API key available
        if not self.api_key:
            return result

        self.oracle_calls += 1

        # Format all answers with full reasoning if available
        if all_reasonings:
            answers_text = "\n\n".join([
                f"Answer {i+1}:\nFinal Answer: {ans}\nReasoning: {reasoning}"
                for i, (ans, reasoning) in enumerate(zip(all_answers, all_reasonings))
            ])
        else:
            answers_text = "\n\n".join([
                f"Answer {i+1}: {ans}"
                for i, ans in enumerate(all_answers)
            ])

        prompt = f"""You are an expert mathematics evaluator. Grade each answer to the following question.

Question: {question}

Here are {len(all_answers)} different answers:

{answers_text}

Tasks:
1. Grade EACH answer on a scale of 0-10 for correctness and quality of reasoning.
2. Identify which answer is the BEST overall.

Respond in the following format:
SCORES: [Answer 1: X.X, Answer 2: X.X, ..., Answer {len(all_answers)}: X.X]
BEST: [number of the best answer]
REASONING: [brief explanation]
"""

        messages = [{"role": "user", "content": prompt}]

        # Make API call
        for attempt in range(max_retries):
            try:
                response = self._make_oracle_call(messages)

                if 'choices' in response and len(response['choices']) > 0:
                    content = response['choices'][0]['message']['content']

                    # Parse scores for each answer
                    scores = self._extract_all_scores(content, len(all_answers))
                    result['agent_scores_oracle'] = scores

                    # Parse best answer
                    best_idx = self._extract_best_answer(content, len(all_answers))
                    result['oracle_best_idx'] = best_idx

                    return result

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Oracle evaluation error (attempt {attempt+1}): {e}")
                    continue
                else:
                    # Return default scores on failure
                    result['agent_scores_oracle'] = [5.0] * len(all_answers)
                    result['oracle_best_idx'] = 0
                    return result

        return result

    def _extract_all_scores(self, text: str, num_answers: int) -> List[float]:
        """Extract individual scores for all answers from oracle response."""
        scores = []

        # Look for "SCORES:" section
        scores_match = re.search(r'SCORES:\s*\[(.*?)\]', text, re.IGNORECASE | re.DOTALL)
        if scores_match:
            scores_text = scores_match.group(1)
            # Extract all numbers (scores)
            numbers = re.findall(r'(\d+(?:\.\d+)?)', scores_text)
            for num_str in numbers[:num_answers]:
                try:
                    score = float(num_str)
                    scores.append(max(0.0, min(10.0, score)))
                except ValueError:
                    scores.append(5.0)

        # Fallback: look for individual "Answer X: Y.Y" patterns
        if len(scores) < num_answers:
            scores = []
            for i in range(1, num_answers + 1):
                pattern = rf'Answer {i}:?\s*(\d+(?:\.\d+)?)'
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    scores.append(float(match.group(1)))
                else:
                    scores.append(5.0)

        # Ensure we have exactly num_answers scores
        while len(scores) < num_answers:
            scores.append(5.0)

        return scores[:num_answers]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get validation statistics.

        Returns:
            Dictionary with validation statistics
        """
        accuracy = (self.ground_truth_correct / self.total_validations * 100
                   if self.total_validations > 0 else 0.0)

        return {
            'total_validations': self.total_validations,
            'ground_truth_correct': self.ground_truth_correct,
            'accuracy': accuracy,
            'oracle_calls': self.oracle_calls
        }

    def print_statistics(self):
        """Print validation statistics."""
        stats = self.get_statistics()
        print(f"\n{'='*60}")
        print(f"Validation Statistics")
        print(f"{'='*60}")
        print(f"Total validations: {stats['total_validations']}")
        print(f"Ground truth correct: {stats['ground_truth_correct']}")
        print(f"Accuracy: {stats['accuracy']:.2f}%")
        print(f"Oracle calls: {stats['oracle_calls']}")


if __name__ == "__main__":
    # Test the validator
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv('OPENROUTER_API_KEY')

    print("Testing Validator...")

    validator = Validator(oracle_model='openai/gpt-4o', api_key=api_key)

    # Test ground truth validation
    print("\n=== Testing Ground Truth Validation ===")
    test_cases = [
        ("42", "42"),
        ("1000", "1,000"),
        ("$50", "50"),
        ("100", "99")
    ]

    for selected, ground_truth in test_cases:
        result = validator.validate_with_ground_truth(selected, ground_truth)
        print(f"Selected: '{selected}' vs Ground truth: '{ground_truth}'")
        print(f"  Correct: {result['correct']}, Match type: {result['match_type']}")

    # Test oracle validation
    if api_key:
        print("\n\n=== Testing Oracle Validation ===")
        question = "What is 15 × 23?"
        selected = "345"
        all_answers = ["345", "340", "350", "355"]

        result = validator.validate_with_oracle(question, selected, all_answers)
        print(f"Question: {question}")
        print(f"Selected answer: {selected}")
        print(f"Oracle score: {result['oracle_score']}/10")
        print(f"Oracle best answer: Answer {result['oracle_best_idx']+1}")
        print(f"Matches oracle: {result['matches_oracle']}")
        print(f"Reasoning: {result['oracle_reasoning'][:200]}...")

    # Print statistics
    validator.print_statistics()
