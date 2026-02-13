"""
Agent System for orchestrating multiple LLM agents.

This module manages N=15 agents to answer questions and perform
pairwise comparisons to construct the comparison matrix R.
"""

import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from .llm_agent import LLMAgent


class AgentSystem:
    """Orchestrate N=15 agents to answer questions and perform comparisons."""

    def __init__(self, agent_configs: List[Dict], api_key: str,
                 epsilon: float = 0.1,
                 parallel_generation: bool = True,
                 parallel_comparison: bool = True):
        """
        Initialize agent system with Hammer-Spammer model.

        Args:
            agent_configs: List of agent configuration dictionaries
            api_key: OpenRouter API key
            epsilon: Spammer probability P(z_i = S) = epsilon
            parallel_generation: Use parallel execution for answer generation
            parallel_comparison: Use parallel execution for comparisons
        """
        self.agent_configs = agent_configs
        self.api_key = api_key
        self.N = len(agent_configs)
        self.epsilon = epsilon
        self.parallel_generation = parallel_generation
        self.parallel_comparison = parallel_comparison

        # Draw agent types: z_i ∈ {H, S} with P(z_i = S) = epsilon
        # 1 = hammer, 0 = spammer (fixed for all questions)
        np.random.seed(42)  # Seed for reproducibility
        self.agent_types = (np.random.random(self.N) >= self.epsilon).astype(int)
        num_hammers = int(np.sum(self.agent_types))
        num_spammers = self.N - num_hammers

        # Initialize agents
        print(f"Initializing {self.N} agents...")
        self.agents = []
        for config in agent_configs:
            agent = LLMAgent(
                model_id=config['model'],
                name=config['name'],
                api_key=api_key
            )
            self.agents.append(agent)

        print(f"Agent system ready with {self.N} agents")
        print(f"  Hammer-Spammer assignment (ε={self.epsilon}): {num_hammers} hammers, {num_spammers} spammers")
        for i in range(self.N):
            type_str = "H" if self.agent_types[i] == 1 else "S"
            print(f"    Agent {i} ({self.agents[i].name}): {type_str}")

    def generate_all_answers(self, question: str, verbose: bool = True) -> List[Dict[str, Any]]:
        """
        All N agents generate answers to the question.

        Args:
            question: The question to answer
            verbose: Show progress bars

        Returns:
            List of N answer dictionaries
        """
        if verbose:
            print(f"\nGenerating answers from {self.N} agents...")

        answers = [None] * self.N

        if self.parallel_generation:
            # Parallel execution for faster generation
            # Reduced to 2 workers to avoid rate limiting
            with ThreadPoolExecutor(max_workers=min(2, self.N)) as executor:
                # Submit all tasks
                future_to_idx = {
                    executor.submit(agent.generate_answer, question): i
                    for i, agent in enumerate(self.agents)
                }

                # Collect results with progress bar
                if verbose:
                    pbar = tqdm(total=self.N, desc="Generating answers")

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        answers[idx] = future.result()
                        if verbose:
                            pbar.update(1)
                    except Exception as e:
                        print(f"Error generating answer for agent {idx}: {e}")
                        # Create error placeholder
                        answers[idx] = {
                            'answer': 'ERROR',
                            'reasoning': str(e),
                            'timestamp': time.time(),
                            'tokens_used': 0,
                            'agent_name': self.agents[idx].name,
                            'model_id': self.agents[idx].model_id
                        }
                        if verbose:
                            pbar.update(1)

                if verbose:
                    pbar.close()

        else:
            # Sequential execution
            iterator = tqdm(enumerate(self.agents), total=self.N, desc="Generating answers") if verbose else enumerate(self.agents)
            for i, agent in iterator:
                try:
                    answers[i] = agent.generate_answer(question)
                except Exception as e:
                    print(f"Error generating answer for agent {i}: {e}")
                    answers[i] = {
                        'answer': 'ERROR',
                        'reasoning': str(e),
                        'timestamp': time.time(),
                        'tokens_used': 0,
                        'agent_name': agent.name,
                        'model_id': agent.model_id
                    }

        return answers

    def construct_comparison_matrix(self, question: str, answers: List[Dict[str, Any]],
                                   verbose: bool = True) -> np.ndarray:
        """
        Construct R ∈ {±1}^(N×N) via pairwise comparisons.

        Implements the Hammer-Spammer model:
        - Hammer agents (z_i = H): use real LLM comparison with full reasoning
        - Spammer agents (z_i = S): R_ij ~ uniform({-1, 1}), no API call
        - R_ii = 1 for all agents

        Args:
            question: The original question
            answers: List of N answer dictionaries
            verbose: Show progress bars

        Returns:
            R: numpy array of shape (N, N) with values in {-1, 1}
        """
        if verbose:
            num_hammers = int(np.sum(self.agent_types))
            num_spammers = self.N - num_hammers
            print(f"\nConstructing comparison matrix ({self.N}×{self.N})...")
            print(f"  Agent types: {num_hammers} hammers, {num_spammers} spammers (ε={self.epsilon})")

        # Initialize comparison matrix with diagonal = 1
        R = np.ones((self.N, self.N), dtype=int)

        # === Spammer rows: R_ij ~ uniform({-1, 1}), no API calls ===
        for i in range(self.N):
            if self.agent_types[i] == 0:  # Spammer
                for j in range(self.N):
                    if i != j:
                        R[i, j] = 1 if np.random.random() < 0.5 else -1

        # === Hammer rows: real LLM comparison with full reasoning ===
        hammer_tasks = []
        for i in range(self.N):
            if self.agent_types[i] == 1:  # Hammer
                for j in range(self.N):
                    if i != j:
                        hammer_tasks.append((i, j))

        if verbose and hammer_tasks:
            print(f"  Running {len(hammer_tasks)} LLM comparisons (hammers only)...")

        if hammer_tasks and self.parallel_comparison:
            # Parallel execution for hammer comparisons
            with ThreadPoolExecutor(max_workers=min(8, len(hammer_tasks))) as executor:
                future_to_task = {}
                for i, j in hammer_tasks:
                    future = executor.submit(
                        self.agents[i].compare_answers,
                        question,
                        answers[i]['answer'],
                        answers[j]['answer'],
                        answers[i].get('reasoning', ''),
                        answers[j].get('reasoning', '')
                    )
                    future_to_task[future] = (i, j)

                if verbose:
                    pbar = tqdm(total=len(future_to_task), desc="Pairwise comparisons")

                for future in as_completed(future_to_task):
                    i, j = future_to_task[future]
                    try:
                        R[i, j] = future.result()
                    except Exception as e:
                        print(f"Error in comparison ({i}, {j}): {e}")
                        R[i, j] = 1 if np.random.random() < 0.5 else -1
                    if verbose:
                        pbar.update(1)

                if verbose:
                    pbar.close()

        elif hammer_tasks:
            # Sequential execution for hammer comparisons
            if verbose:
                pbar = tqdm(total=len(hammer_tasks), desc="Pairwise comparisons")

            for i, j in hammer_tasks:
                try:
                    R[i, j] = self.agents[i].compare_answers(
                        question,
                        answers[i]['answer'],
                        answers[j]['answer'],
                        answers[i].get('reasoning', ''),
                        answers[j].get('reasoning', '')
                    )
                except Exception as e:
                    print(f"Error in comparison ({i}, {j}): {e}")
                    R[i, j] = 1 if np.random.random() < 0.5 else -1

                if verbose:
                    pbar.update(1)

            if verbose:
                pbar.close()

        # Verify matrix properties
        assert R.shape == (self.N, self.N), f"Wrong shape: {R.shape}"
        assert np.all(np.diag(R) == 1), "Diagonal should be all 1s"
        assert np.all(np.isin(R, [-1, 1])), "All values should be ±1"

        if verbose:
            print(f"Comparison matrix constructed: shape={R.shape}")
            print(f"  Positive comparisons: {np.sum(R == 1)} ({100*np.sum(R == 1)/R.size:.1f}%)")
            print(f"  Negative comparisons: {np.sum(R == -1)} ({100*np.sum(R == -1)/R.size:.1f}%)")

        return R

    def run_experiment(self, question: str, ground_truth: Optional[str] = None,
                      verbose: bool = True) -> Dict[str, Any]:
        """
        Complete experiment pipeline for one question.

        Args:
            question: The question to answer
            ground_truth: Optional ground truth answer
            verbose: Show progress information

        Returns:
            Dictionary with:
                - question: str
                - answers: List[Dict]
                - comparison_matrix: ndarray (N, N)
                - ground_truth: str (optional)
                - timestamp: str
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running experiment for question:")
            print(f"{question[:100]}...")
            print(f"{'='*60}")

        start_time = time.time()

        # Step 1: Generate all answers
        answers = self.generate_all_answers(question, verbose=verbose)

        # Step 2: Construct comparison matrix
        R = self.construct_comparison_matrix(question, answers, verbose=verbose)

        elapsed_time = time.time() - start_time

        if verbose:
            print(f"\nExperiment completed in {elapsed_time:.1f}s")
            print(f"Total tokens used: {sum(a['tokens_used'] for a in answers)}")

        return {
            'question': question,
            'answers': answers,
            'comparison_matrix': R,
            'agent_types': self.agent_types.tolist(),
            'agent_tiers': [config['tier'] for config in self.agent_configs],
            'agent_names': [config['name'] for config in self.agent_configs],
            'ground_truth': ground_truth,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'elapsed_time': elapsed_time
        }

    def get_all_stats(self) -> List[Dict[str, Any]]:
        """
        Get statistics for all agents.

        Returns:
            List of agent statistics
        """
        return [agent.get_stats() for agent in self.agents]

    def print_summary(self):
        """Print summary of all agents."""
        print(f"\n{'='*60}")
        print(f"Agent System Summary (N={self.N})")
        print(f"{'='*60}\n")

        total_calls = sum(agent.total_calls for agent in self.agents)
        total_tokens = sum(agent.total_tokens_used for agent in self.agents)

        print(f"Total API calls: {total_calls}")
        print(f"Total tokens used: {total_tokens:,}")
        print(f"\nPer-agent breakdown:")
        print(f"{'Agent':<20} {'Model':<35} {'Calls':<8} {'Tokens':<10}")
        print("-" * 75)

        for agent in self.agents:
            print(f"{agent.name:<20} {agent.model_id:<35} {agent.total_calls:<8} {agent.total_tokens_used:<10,}")

    def inject_artificial_spammers(self, R: np.ndarray, k: int, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject k artificial spammers into comparison matrix R for Track A analysis.

        This replaces k rows of R with random noise, simulating k spammer agents.
        NO API CALLS - just matrix manipulation on already-collected data.

        Args:
            R: N×N comparison matrix from Track B
            k: Number of spammers to inject (0 to N)
            seed: Random seed for reproducibility

        Returns:
            (R_modified, spammer_mask):
                - R_modified: N×N matrix with k rows replaced by random {-1, +1}
                - spammer_mask: N-length boolean array (True = spammer)
        """
        if seed is not None:
            np.random.seed(seed)

        N = R.shape[0]
        R_new = R.copy()

        # Choose k agents randomly to be spammers
        spammer_indices = np.random.choice(N, size=min(k, N), replace=False)
        spammer_mask = np.zeros(N, dtype=bool)
        spammer_mask[spammer_indices] = True

        # Replace their rows with random comparisons
        for i in spammer_indices:
            # Random {-1, +1} for all comparisons
            R_new[i, :] = np.random.choice([-1, 1], size=N)
            # Diagonal stays 1 (self-comparison)
            R_new[i, i] = 1

        return R_new, spammer_mask


if __name__ == "__main__":
    # Test the agent system
    import os
    from dotenv import load_dotenv
    from .agent_config import AGENT_CONFIGS

    load_dotenv()
    api_key = os.getenv('OPENROUTER_API_KEY')

    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found")
        exit(1)

    print("Testing AgentSystem with 3 agents...")

    # Use only 3 agents for testing
    test_configs = AGENT_CONFIGS[:3]

    system = AgentSystem(test_configs, api_key, parallel_generation=True, parallel_comparison=True)

    # Test question
    question = "What is 7 × 8?"
    ground_truth = "56"

    # Run experiment
    result = system.run_experiment(question, ground_truth, verbose=True)

    print("\n\n=== Results ===")
    print(f"Question: {result['question']}")
    print(f"Ground truth: {result['ground_truth']}")
    print(f"\nAnswers:")
    for i, ans in enumerate(result['answers']):
        print(f"  Agent {i} ({ans['agent_name']}): {ans['answer']}")

    print(f"\nComparison matrix R:")
    print(result['comparison_matrix'])

    # Print stats
    system.print_summary()
