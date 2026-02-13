"""
Dataset loader for GSM8K and MATH500 datasets.

This module provides utilities to load mathematical reasoning datasets
for validating the Hammer-Spammer model.
"""

import os
import json
import re
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset


class DatasetLoader:
    """Load and manage GSM8K/MATH500 datasets for reasoning questions."""

    def __init__(self, dataset_name: str = 'gsm8k', split: str = 'test', max_samples: Optional[int] = 100):
        """
        Initialize dataset loader.

        Args:
            dataset_name: 'gsm8k' or 'math500'
            split: 'train' or 'test'
            max_samples: Maximum number of questions to load (None for all)
        """
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.max_samples = max_samples
        self.dataset = None
        self._load_dataset()

    def _load_dataset(self):
        """Load dataset from Hugging Face."""
        print(f"Loading {self.dataset_name} dataset ({self.split} split)...")

        try:
            if self.dataset_name == 'gsm8k':
                # GSM8K dataset
                self.dataset = load_dataset('gsm8k', 'main', split=self.split)
            elif self.dataset_name == 'math' or self.dataset_name == 'math500':
                # MATH dataset (lighteval/MATH)
                self.dataset = load_dataset('lighteval/MATH', split=self.split)
            else:
                raise ValueError(f"Unknown dataset: {self.dataset_name}")

            # Limit samples if specified
            if self.max_samples is not None and self.max_samples < len(self.dataset):
                self.dataset = self.dataset.select(range(self.max_samples))

            print(f"Loaded {len(self.dataset)} questions from {self.dataset_name}")

        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def get_question(self, idx: int) -> Tuple[str, str]:
        """
        Get a single question and its ground truth answer.

        Args:
            idx: Question index

        Returns:
            Tuple of (question_text, ground_truth_answer)
        """
        if idx >= len(self.dataset):
            raise IndexError(f"Index {idx} out of range (dataset has {len(self.dataset)} questions)")

        item = self.dataset[idx]

        if self.dataset_name == 'gsm8k':
            # GSM8K format: {'question': str, 'answer': str}
            question = item['question']
            # Extract numerical answer from the answer string
            # GSM8K answers are in format "#### NUMBER"
            answer = self._extract_gsm8k_answer(item['answer'])

        elif self.dataset_name in ['math', 'math500']:
            # MATH format: {'problem': str, 'solution': str}
            question = item['problem']
            # MATH solutions contain the final answer, extract it
            answer = self._extract_math_answer(item['solution'])

        else:
            raise ValueError(f"Unknown dataset format: {self.dataset_name}")

        return question, answer

    def _extract_gsm8k_answer(self, answer_text: str) -> str:
        """
        Extract the numerical answer from GSM8K answer format.

        GSM8K answers end with "#### NUMBER" where NUMBER is the final answer.

        Args:
            answer_text: Full answer text from GSM8K

        Returns:
            Extracted numerical answer as string
        """
        # Look for "#### " pattern
        match = re.search(r'####\s*(.+)', answer_text)
        if match:
            answer = match.group(1).strip()
            # Remove commas from numbers (e.g., "1,000" -> "1000")
            answer = answer.replace(',', '')
            return answer
        else:
            # Fallback: return the full answer text
            return answer_text.strip()

    def _extract_math_answer(self, solution_text: str) -> str:
        """
        Extract the final answer from MATH solution format.

        MATH solutions often use \\boxed{answer} to denote final answer.

        Args:
            solution_text: Full solution text from MATH

        Returns:
            Extracted answer as string
        """
        # Look for \boxed{...} pattern
        match = re.search(r'\\boxed\{([^}]+)\}', solution_text)
        if match:
            return match.group(1).strip()

        # Fallback: look for last line or return full solution
        lines = solution_text.strip().split('\n')
        if lines:
            return lines[-1].strip()

        return solution_text.strip()

    def get_batch(self, start_idx: int, batch_size: int) -> List[Tuple[str, str]]:
        """
        Get multiple questions as a batch.

        Args:
            start_idx: Starting index
            batch_size: Number of questions to retrieve

        Returns:
            List of (question, ground_truth) tuples
        """
        end_idx = min(start_idx + batch_size, len(self.dataset))
        return [self.get_question(i) for i in range(start_idx, end_idx)]

    def __len__(self) -> int:
        """Return number of questions in the dataset."""
        return len(self.dataset) if self.dataset is not None else 0

    def get_all_questions(self) -> List[Tuple[str, str]]:
        """
        Get all questions in the dataset.

        Returns:
            List of (question, ground_truth) tuples
        """
        return [self.get_question(i) for i in range(len(self))]


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer for comparison.

    This handles:
    - Removing whitespace
    - Converting to lowercase
    - Removing common formatting (commas, dollar signs, etc.)

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer string
    """
    # Convert to lowercase
    answer = answer.lower().strip()

    # Remove common formatting characters
    answer = answer.replace(',', '')  # Remove commas from numbers
    answer = answer.replace('$', '')  # Remove dollar signs
    answer = answer.replace('%', '')  # Remove percent signs

    # Remove extra whitespace
    answer = ' '.join(answer.split())

    return answer


def extract_number(text: str) -> Optional[float]:
    """
    Extract the first number from a text string.

    Args:
        text: Text containing a number

    Returns:
        Extracted number as float, or None if no number found
    """
    # Try to find numbers in the text (including decimals and negatives)
    matches = re.findall(r'-?\d+\.?\d*', text)
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            pass
    return None


def compare_answers(answer1: str, answer2: str) -> bool:
    """
    Compare two answers for equivalence.

    This handles:
    - Direct string comparison
    - Numerical comparison (extracts numbers from text)
    - Case insensitive matching

    Args:
        answer1: First answer
        answer2: Second answer

    Returns:
        True if answers are equivalent, False otherwise
    """
    norm1 = normalize_answer(answer1)
    norm2 = normalize_answer(answer2)

    # Direct string comparison
    if norm1 == norm2:
        return True

    # Try numerical comparison
    # First, try direct conversion
    try:
        num1 = float(norm1)
        num2 = float(norm2)
        return abs(num1 - num2) < 1e-6
    except ValueError:
        pass

    # If direct conversion fails, try extracting numbers from text
    # e.g., "3 bolts" -> 3, "$50 dollars" -> 50
    num1 = extract_number(norm1)
    num2 = extract_number(norm2)

    if num1 is not None and num2 is not None:
        return abs(num1 - num2) < 1e-6

    return False


if __name__ == "__main__":
    # Test the dataset loader
    print("Testing DatasetLoader...")

    # Test GSM8K
    print("\n=== Testing GSM8K ===")
    loader = DatasetLoader('gsm8k', split='test', max_samples=5)
    print(f"Dataset size: {len(loader)}")

    for i in range(min(3, len(loader))):
        question, answer = loader.get_question(i)
        print(f"\nQuestion {i}:")
        print(f"Q: {question}")
        print(f"A: {answer}")

    # Test batch retrieval
    print("\n=== Testing Batch Retrieval ===")
    batch = loader.get_batch(0, 2)
    print(f"Retrieved batch of {len(batch)} questions")

    # Test answer normalization and comparison
    print("\n=== Testing Answer Comparison ===")
    test_answers = [
        ("1,000", "1000", True),
        ("$50", "50", True),
        ("  42  ", "42", True),
        ("3.14159", "3.14159", True),
        ("3 bolts", "3", True),  # Number extraction
        ("$70,000", "70000", True),  # Complex formatting
        ("The answer is 42", "42", True),  # Number in text
        ("yes", "no", False),  # Different strings
    ]
    for a1, a2, expected in test_answers:
        result = compare_answers(a1, a2)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{a1}' == '{a2}': {result} (expected {expected})")
