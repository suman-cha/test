"""
LLM Agent wrapper for OpenRouter API.

This module provides a unified interface for calling different LLM models
via the OpenRouter API for answer generation and pairwise comparisons.
"""

import os
import time
import json
import re
import requests
from typing import Dict, Optional, Any
from datetime import datetime


class LLMAgent:
    """Unified interface for calling LLM models via OpenRouter API."""

    def __init__(self, model_id: str, name: str, api_key: str,
                 temperature: float = 0.0, max_tokens: int = 500):
        """
        Initialize LLM agent.

        Args:
            model_id: OpenRouter model identifier (e.g., "openai/gpt-4o")
            name: Human-readable agent name
            api_key: OpenRouter API key
            temperature: Sampling temperature (default 0.0 for reproducibility)
            max_tokens: Maximum tokens in response (default 500)
        """
        self.model_id = model_id
        self.name = name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

        # Track statistics
        self.total_tokens_used = 0
        self.total_calls = 0
        self.total_cost = 0.0

    def _make_api_call(self, messages: list, max_retries: int = 3) -> Dict[str, Any]:
        """
        Make API call to OpenRouter with retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_retries: Maximum number of retry attempts

        Returns:
            API response dictionary

        Raises:
            Exception if all retries fail
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/hammer-spammer-validation",
            "X-Title": "Hammer-Spammer Model Validation"
        }

        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()

                    # Update statistics
                    if 'usage' in result:
                        tokens = result['usage'].get('total_tokens', 0)
                        self.total_tokens_used += tokens
                        self.total_calls += 1

                    return result

                elif response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Rate limit hit for {self.name}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                else:
                    error_msg = f"API error {response.status_code}: {response.text}"
                    if attempt < max_retries - 1:
                        print(f"{error_msg} - Retrying...")
                        time.sleep(1)
                        continue
                    else:
                        raise Exception(error_msg)

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"Timeout for {self.name}, retrying...")
                    time.sleep(1)
                    continue
                else:
                    raise Exception(f"Request timeout after {max_retries} attempts")

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error for {self.name}: {e} - Retrying...")
                    time.sleep(1)
                    continue
                else:
                    raise

        raise Exception(f"All {max_retries} API call attempts failed")

    def generate_answer(self, question: str) -> Dict[str, Any]:
        """
        Generate answer to a question.

        Args:
            question: The question to answer

        Returns:
            Dictionary with:
                - answer: str (the final answer)
                - reasoning: str (the full response/reasoning)
                - timestamp: float
                - tokens_used: int
                - agent_name: str
                - model_id: str
        """
        prompt = f"""Solve the following math problem. Show your reasoning and provide a clear final answer.

Question: {question}

Please solve this step-by-step and clearly state your final answer at the end using the format "Final Answer: [your answer]"."""

        messages = [
            {"role": "user", "content": prompt}
        ]

        start_time = time.time()
        response = self._make_api_call(messages)
        elapsed_time = time.time() - start_time

        # Extract response content
        if 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
        else:
            content = "ERROR: No response generated"

        # Extract final answer
        final_answer = self._extract_final_answer(content)

        tokens_used = response.get('usage', {}).get('total_tokens', 0)

        return {
            'answer': final_answer,
            'reasoning': content,
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': elapsed_time,
            'tokens_used': tokens_used,
            'agent_name': self.name,
            'model_id': self.model_id
        }

    def _extract_final_answer(self, response_text: str) -> str:
        """
        Extract the final answer from the response.

        Looks for patterns like:
        - "Final Answer: X"
        - "Answer: X"
        - "Therefore, X"
        - Last number in the text

        Args:
            response_text: Full response from LLM

        Returns:
            Extracted final answer
        """
        # Try pattern: "Final Answer: ..."
        match = re.search(r'Final Answer:\s*(.+?)(?:\n|$)', response_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Try pattern: "Answer: ..."
        match = re.search(r'(?:^|\n)Answer:\s*(.+?)(?:\n|$)', response_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Try pattern: "Therefore, ..."
        match = re.search(r'Therefore,?\s+(.+?)(?:\n|$)', response_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Try to find the last number in the text
        numbers = re.findall(r'-?\d+\.?\d*', response_text)
        if numbers:
            return numbers[-1]

        # Fallback: return last line
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        if lines:
            return lines[-1]

        return response_text.strip()

    def compare_answers(self, question: str, my_answer: str, other_answer: str) -> int:
        """
        Compare own answer against another answer.

        Args:
            question: The original question
            my_answer: This agent's answer
            other_answer: Another agent's answer

        Returns:
            1 if my_answer is better or equal, -1 if other_answer is better
        """
        prompt = f"""You are evaluating two answers to a math problem. Determine which answer is better.

Question: {question}

Answer A (your answer): {my_answer}

Answer B (other answer): {other_answer}

Which answer is better? Consider correctness, reasoning, and clarity.
Respond with ONLY ONE WORD: either "A" if Answer A is better or equal, or "B" if Answer B is clearly better.

Your response (A or B):"""

        messages = [
            {"role": "user", "content": prompt}
        ]

        try:
            response = self._make_api_call(messages)

            if 'choices' in response and len(response['choices']) > 0:
                content = response['choices'][0]['message']['content'].strip().upper()

                # Parse response - look for A or B
                if 'A' in content[:10]:  # Check first few characters
                    return 1  # My answer is better/equal
                elif 'B' in content[:10]:
                    return -1  # Other answer is better
                else:
                    # Default: prefer own answer if unclear
                    return 1
            else:
                # Default: prefer own answer if no response
                return 1

        except Exception as e:
            print(f"Error in comparison for {self.name}: {e}")
            # Default: prefer own answer on error
            return 1

    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics.

        Returns:
            Dictionary with usage statistics
        """
        return {
            'agent_name': self.name,
            'model_id': self.model_id,
            'total_calls': self.total_calls,
            'total_tokens_used': self.total_tokens_used,
            'total_cost': self.total_cost
        }

    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"LLMAgent(name={self.name}, model={self.model_id})"


if __name__ == "__main__":
    # Test the LLM agent
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv('OPENROUTER_API_KEY')

    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found in environment")
        exit(1)

    print("Testing LLMAgent...")

    # Create a test agent
    agent = LLMAgent(
        model_id="openai/gpt-3.5-turbo",
        name="test-agent",
        api_key=api_key
    )

    # Test answer generation
    print("\n=== Testing Answer Generation ===")
    question = "What is 15 × 23?"
    print(f"Question: {question}")

    result = agent.generate_answer(question)
    print(f"\nAnswer: {result['answer']}")
    print(f"Reasoning: {result['reasoning'][:200]}...")
    print(f"Tokens used: {result['tokens_used']}")
    print(f"Time: {result['elapsed_time']:.2f}s")

    # Test comparison
    print("\n\n=== Testing Answer Comparison ===")
    my_answer = "345"
    other_answer = "340"
    print(f"My answer: {my_answer}")
    print(f"Other answer: {other_answer}")

    comparison = agent.compare_answers(question, my_answer, other_answer)
    print(f"Comparison result: {comparison} ({'prefer my answer' if comparison == 1 else 'prefer other'})")

    # Show stats
    print("\n\n=== Agent Statistics ===")
    stats = agent.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
