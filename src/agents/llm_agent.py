"""
LLM Agent wrapper for OpenRouter API.

This module provides a unified interface for calling different LLM models
via the OpenRouter API for answer generation and pairwise comparisons.
"""

import os
import time
import json
import re
import random
import requests
from typing import Dict, Optional, Any
from datetime import datetime


class LLMAgent:
    """Unified interface for calling LLM models via OpenRouter API."""

    def __init__(self, model_id: str, name: str, api_key: str,
                 persona: str = '', temperature: float = 0.0, max_tokens: int = 500):
        """
        Initialize LLM agent.

        Args:
            model_id: OpenRouter model identifier (e.g., "openai/gpt-4o")
            name: Human-readable agent name
            api_key: OpenRouter API key
            persona: System prompt that shapes the agent's behavior.
                     Empty string = use model's default behavior.
                     Example: "You are a careful mathematician who shows all work step-by-step."
                     Same model + different persona = different agent (방법 2).
            temperature: Sampling temperature (default 0.0 for reproducibility)
            max_tokens: Maximum tokens in response (default 500)
        """
        self.model_id = model_id
        self.name = name
        self.api_key = api_key
        self.persona = persona
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

                    # Small delay to be gentle with API rate limits
                    # Reduced to 0.2s for faster comparisons
                    time.sleep(0.2)

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
        prompt = f"""Solve the following math problem step-by-step.

Question: {question}

IMPORTANT: After showing your work, you MUST end your response with a line in EXACTLY this format:
Final Answer: [your answer here]

The final answer should be concise (a number, expression, or short phrase), not a full explanation."""

        messages = []
        if self.persona:
            messages.append({"role": "system", "content": self.persona})
        messages.append({"role": "user", "content": prompt})

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
        - \boxed{X}
        - "Therefore, X"
        - Last substantial line

        Args:
            response_text: Full response from LLM

        Returns:
            Extracted final answer
        """
        # Try pattern: "Final Answer: ..." (most explicit)
        # Look for this pattern, preferring the LAST occurrence
        matches = list(re.finditer(r'Final Answer:\s*(.+?)(?:\n|$)', response_text, re.IGNORECASE))
        if matches:
            answer = matches[-1].group(1).strip()
            # Remove trailing periods
            answer = answer.rstrip('.')
            # If it starts with LaTeX delimiters, clean them
            answer = re.sub(r'^\$+|\$+$', '', answer).strip()
            if answer:
                return answer

        # Try LaTeX \boxed{...} (common in math problems)
        matches = list(re.finditer(r'\\boxed\{([^}]+)\}', response_text))
        if matches:
            return matches[-1].group(1).strip()

        # Try pattern: "Answer: ..." (look for last occurrence)
        matches = list(re.finditer(r'(?:^|\n)Answer:\s*(.+?)(?:\n|$)', response_text, re.IGNORECASE))
        if matches:
            answer = matches[-1].group(1).strip().rstrip('.')
            answer = re.sub(r'^\$+|\$+$', '', answer).strip()
            if answer:
                return answer

        # Try pattern: "Therefore, ..."
        matches = list(re.finditer(r'Therefore,?\s+(?:the answer is\s+)?(.+?)(?:\.|$)', response_text, re.IGNORECASE))
        if matches:
            answer = matches[-1].group(1).strip()
            answer = re.sub(r'^\$+|\$+$', '', answer).strip()
            if answer:
                return answer

        # Try pattern: "the answer is X" or "answer is X"
        matches = list(re.finditer(r'(?:the\s+)?answer\s+is\s+(.+?)(?:\.|$)', response_text, re.IGNORECASE))
        if matches:
            answer = matches[-1].group(1).strip()
            answer = re.sub(r'^\$+|\$+$', '', answer).strip()
            # If it's short enough, use it
            if len(answer) < 100:
                return answer

        # For math problems: try to find standalone numbers or short expressions
        # Look for lines that are ONLY numbers/math (not part of explanation)
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]

        # Filter for short lines that look like answers
        candidate_lines = []
        for line in reversed(lines):  # Start from end
            # Skip long lines (likely explanations)
            if len(line) > 150:
                continue
            # Skip lines starting with explanation phrases
            if re.match(r'^(To\s+find|To\s+solve|We\s+need|We\s+first|Step|First|Let|The\s+equation|The\s+problem)', line, re.IGNORECASE):
                continue
            # Skip markdown headers
            if re.match(r'^#+\s', line):
                continue
            # Prefer lines with numbers or short expressions
            if re.search(r'\d', line) or len(line) < 50:
                candidate_lines.append(line)
                if len(candidate_lines) >= 5:
                    break

        # From candidates, pick the shortest one (likely the answer)
        if candidate_lines:
            # Sort by length and pick shortest
            candidate_lines.sort(key=len)
            for candidate in candidate_lines:
                # Clean LaTeX
                answer = re.sub(r'^\$+|\$+$', '', candidate).strip()
                answer = re.sub(r'^\\\[|\\\]$', '', answer).strip()
                answer = re.sub(r'^\\text\{|\}$', '', answer).strip()
                # If it looks reasonable, use it
                if answer and len(answer) < 100:
                    return answer

        # Last resort: find standalone number at end
        # Look for a number that appears alone on a line or at the end
        number_lines = [line for line in reversed(lines) if re.match(r'^-?\d+\.?\d*$', line.strip())]
        if number_lines:
            return number_lines[0]

        # Absolute last resort: return full text (truncated)
        result = response_text.strip()
        if len(result) > 300:
            result = result[:300] + "..."
        return result

    def compare_answers(self, question: str, my_answer: str, other_answer: str,
                        my_reasoning: str = '', other_reasoning: str = '') -> int:
        """
        Compare own answer against another answer.

        Args:
            question: The original question
            my_answer: This agent's final answer
            other_answer: Another agent's final answer
            my_reasoning: This agent's full reasoning/solution
            other_reasoning: Other agent's full reasoning/solution

        Returns:
            1 if Answer A is better, -1 if Answer B is better
        """
        # Format answers with reasoning if available
        if my_reasoning and other_reasoning:
            answer_a_text = f"Final Answer: {my_answer}\nReasoning:\n{my_reasoning}"
            answer_b_text = f"Final Answer: {other_answer}\nReasoning:\n{other_reasoning}"
        else:
            answer_a_text = my_answer
            answer_b_text = other_answer

        prompt = f"""You are evaluating two answers to a math problem. Determine which answer is better.

Question: {question}

Answer A:
{answer_a_text}

Answer B:
{answer_b_text}

Which answer is better in terms of correctness and quality of reasoning?
Respond with ONLY ONE WORD: "A" or "B"."""

        messages = []
        if self.persona:
            messages.append({"role": "system", "content": self.persona})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._make_api_call(messages)

            if 'choices' in response and len(response['choices']) > 0:
                content = response['choices'][0]['message']['content'].strip().upper()

                # Parse response - look for A or B
                if 'A' in content[:10]:
                    return 1
                elif 'B' in content[:10]:
                    return -1
                else:
                    # Ambiguous response — random to avoid systematic bias
                    return random.choice([1, -1])
            else:
                return random.choice([1, -1])

        except Exception as e:
            print(f"Error in comparison for {self.name}: {e}")
            return random.choice([1, -1])

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
        persona_str = f", persona='{self.persona[:30]}...'" if self.persona else ""
        return f"LLMAgent(name={self.name}, model={self.model_id}{persona_str})"


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
