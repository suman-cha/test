"""
Main Experiment Script for Problem 2.

Runs the complete LLM agent experiment to validate the Hammer-Spammer
model algorithm against baselines.

Two tracks:
  Track B ("natural"):    All agents use real LLM — tests practical utility.
  Track A ("artificial"): Post-hoc spammer injection — validates theory.

Usage:
    # Track B: natural experiment (50 questions, 15 agents)
    python run_experiment.py --num-questions 50

    # Track A: inject spammers and compare (uses saved Track B data)
    python run_experiment.py --num-questions 50 --track-a --spammer-counts 0,2,4,6

    # Debug mode: 3 questions, verbose
    python run_experiment.py --debug
"""

import os
import sys
import json
import argparse
import time
import re
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

# ── Local imports ──
from .agent_config import AGENT_CONFIGS, print_agent_summary
from .agent_system import AgentSystem
from ..algorithm.ccrr import CCRRanking
from ..algorithm.spectral_ranking import SpectralRanking
from ..evaluation.baselines import Baselines
from ..utils.dataset import compare_answers


# ═════════════════════════════════════════════════════════════════
# Dataset loading (self-contained, no external dependency)
# ═════════════════════════════════════════════════════════════════

def load_gsm8k(num_questions: int = 50, split: str = "test",
               start_index: int = 0) -> List[Dict[str, str]]:
    """
    Load GSM8K dataset from HuggingFace datasets.

    Each item has 'question' and 'answer' (final numerical answer).

    Returns:
        List of {'question': str, 'answer': str} dicts
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split=split)
    except Exception as e:
        print(f"Failed to load GSM8K from HuggingFace: {e}")
        print("Falling back to manual test questions...")
        return _fallback_questions(num_questions)

    data = []
    end_index = min(start_index + num_questions, len(ds))

    for i in range(start_index, end_index):
        item = ds[i]
        question = item['question']

        # Extract numerical answer from GSM8K format:
        # "... #### <number>"
        raw_answer = item['answer']
        match = re.search(r'####\s*(.+)', raw_answer)
        if match:
            answer = match.group(1).strip().replace(',', '')
        else:
            answer = raw_answer.strip()

        data.append({'question': question, 'answer': answer})

    print(f"Loaded {len(data)} questions from GSM8K ({split}[{start_index}:{end_index}])")
    return data


def _fallback_questions(n: int) -> List[Dict[str, str]]:
    """Fallback math questions if dataset loading fails."""
    questions = [
        {"question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?", "answer": "18"},
        {"question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?", "answer": "3"},
        {"question": "Josh decides to try flipping a house. He buys a house for $80,000 and puts $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?", "answer": "70000"},
        {"question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?", "answer": "624"},
        {"question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. If the carry-over from the morning to the afternoon is 5 cups, how many cups of feed does she need to give her chickens in the final meal of the day?", "answer": "20"},
    ]
    return questions[:n]


# ═════════════════════════════════════════════════════════════════
# Answer matching utilities
# ═════════════════════════════════════════════════════════════════

def normalize_answer(s: str) -> str:
    """Normalize an answer string for comparison."""
    s = str(s).strip().lower()
    s = s.replace(',', '').replace('$', '').replace('%', '')
    s = s.replace('\\', '').replace('{', '').replace('}', '')
    s = s.strip('.')

    # Try extracting and normalizing a number
    match = re.search(r'-?\d+\.?\d*', s)
    if match:
        num_str = match.group()
        try:
            # Convert to float, then check if it's a whole number
            num = float(num_str)
            if num == int(num):
                return str(int(num))  # Return as integer string (8.00 → "8")
            else:
                return str(num)  # Return as float string
        except ValueError:
            return num_str
    return s


def check_correct_with_llm(answer: str, ground_truth: str, api_key: str) -> bool:
    """
    Check if an answer is correct using GPT-4o as judge.

    This handles mathematical equivalence better than string matching.
    For example: "x \\in [-2, 7]" vs "[-2, 7]" vs "-2 ≤ x ≤ 7"
    """
    import requests

    prompt = f"""You are a mathematical answer checker. Compare the student's answer with the ground truth answer.

Ground Truth: {ground_truth}
Student Answer: {answer}

Determine if the student's answer is mathematically equivalent to the ground truth, even if the format is different.

Consider these equivalent:
- "x \\in [-2, 7]" = "[-2, 7]" = "-2 ≤ x ≤ 7"
- "42" = "42.0" = "42.00"
- "1/2" = "0.5"
- Different LaTeX formatting of the same expression

Respond with ONLY "CORRECT" or "INCORRECT" - nothing else."""

    try:
        response = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
            },
            json={
                'model': 'openai/gpt-4o',
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.0,
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            judgment = result['choices'][0]['message']['content'].strip().upper()
            return 'CORRECT' in judgment
        else:
            # Fallback to string matching if API fails
            return normalize_answer(answer) == normalize_answer(ground_truth)

    except Exception as e:
        # Fallback to string matching if error occurs
        print(f"  Warning: LLM judge failed ({e}), using string match")
        return normalize_answer(answer) == normalize_answer(ground_truth)


def check_correct(answer: str, ground_truth: str, api_key: str = None, use_llm: bool = False) -> bool:
    """
    Check if an answer matches the ground truth.

    Args:
        answer: The answer to check
        ground_truth: The correct answer
        api_key: OpenRouter API key (required if use_llm=True)
        use_llm: If True, use GPT-4o for semantic checking; if False, use string matching (default: False)

    Returns:
        True if correct, False otherwise
    """
    if use_llm and api_key:
        return check_correct_with_llm(answer, ground_truth, api_key)
    else:
        return normalize_answer(answer) == normalize_answer(ground_truth)


# ═════════════════════════════════════════════════════════════════
# Experiment Runner
# ═════════════════════════════════════════════════════════════════

class ExperimentRunner:
    """Main experiment runner for Problem 2."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = []

        # Load API key
        load_dotenv()
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found. Set it in .env or environment.")

        # Load dataset
        from ..utils.dataset import DatasetLoader
        dataset_name = config.get('dataset', 'gsm8k')
        difficulty = config.get('difficulty', None)
        print(f"\nLoading dataset: {dataset_name.upper()}")
        if difficulty:
            print(f"  Difficulty filter: {difficulty}")

        loader = DatasetLoader(
            dataset_name=dataset_name,
            split=config.get('split', 'test'),
            max_samples=config['num_questions'],
            difficulty_filter=difficulty,
            start_index=config.get('start_index', 0),
        )

        # Convert to expected format
        self.dataset = [
            {'question': q, 'answer': a}
            for q, a in loader.get_all_questions()
        ]

        # Initialize agent system (Track B: natural mode)
        print_agent_summary()
        agent_configs = AGENT_CONFIGS[:config['num_agents']]
        self.agent_system = AgentSystem(
            agent_configs=agent_configs,
            api_key=api_key,
            parallel_generation=config.get('parallel_generation', True),
            parallel_comparison=config.get('parallel_comparison', True),
        )

        self.api_key = api_key
        print(f"\nExperiment ready: {len(self.dataset)} questions, {config['num_agents']} agents")

    # ─────────────────────────────────────────────────────────────
    # Run one question
    # ─────────────────────────────────────────────────────────────

    def run_single_question(self, idx: int) -> Dict[str, Any]:
        """Run full pipeline for one question."""
        item = self.dataset[idx]
        question = item['question']
        ground_truth = item['answer']

        print(f"\n{'─'*60}")
        print(f"Question {idx+1}/{len(self.dataset)}")
        print(f"Q: {question[:120]}...")
        print(f"GT: {ground_truth}")

        # ── Step 1-2: Generate answers & build R (via AgentSystem) ──
        exp_data = self.agent_system.run_experiment(
            question=question,
            ground_truth=ground_truth,
            verbose=self.config.get('verbose', False),
        )

        R = exp_data['comparison_matrix']
        answers = exp_data['answers']
        answer_strings = [a['answer'] for a in answers]

        # ── Step 3: Apply algorithms ──
        # CCRR Algorithm
        ccrr = CCRRanking(beta=5.0, epsilon=0.1)
        ccrr_idx, ccrr_scores, ccrr_weights = ccrr.select_best(R, return_details=True)

        # Spectral Ranking Algorithm
        spectral = SpectralRanking(beta=5.0)
        spectral_idx, spectral_scores, spectral_weights, spectral_details = spectral.select_best(R, return_details=True)

        # Majority Voting (baseline)
        mv_idx = Baselines.majority_voting(answer_strings)

        # Random Selection (baseline)
        rand_idx = Baselines.random_selection(len(answer_strings), seed=idx)

        # ── Step 4: Validate against ground truth ──
        # Use GPT-4o as oracle judge for semantic equivalence
        ccrr_correct = check_correct(answer_strings[ccrr_idx], ground_truth, self.api_key)
        spectral_correct = check_correct(answer_strings[spectral_idx], ground_truth, self.api_key)
        mv_correct = check_correct(answer_strings[mv_idx], ground_truth, self.api_key)
        rand_correct = check_correct(answer_strings[rand_idx], ground_truth, self.api_key)

        # Check how many agents got the right answer (for context)
        agent_correctness = [check_correct(a, ground_truth, self.api_key) for a in answer_strings]
        num_correct_agents = sum(agent_correctness)

        # Oracle: best single agent (upper bound)
        oracle_idx = 0  # Best single model (first agent)
        for i, ans in enumerate(answer_strings):
            if check_correct(ans, ground_truth, self.api_key):
                oracle_idx = i
                break
        oracle_correct = check_correct(answer_strings[oracle_idx], ground_truth, self.api_key)

        # ── Print result ──
        print(f"\n  Results:")
        print(f"    CCRR Algorithm:     {'✓' if ccrr_correct else '✗'}  → {answer_strings[ccrr_idx]}")
        print(f"    Spectral Ranking:   {'✓' if spectral_correct else '✗'}  → {answer_strings[spectral_idx]}")
        print(f"    Majority Vote:      {'✓' if mv_correct else '✗'}  → {answer_strings[mv_idx]}")
        print(f"    Random:             {'✓' if rand_correct else '✗'}  → {answer_strings[rand_idx]}")
        print(f"    Oracle (upper):     {'✓' if oracle_correct else '✗'}")
        print(f"    Agents correct:     {num_correct_agents}/{len(answer_strings)}")

        # ── Compile result ──
        return {
            'question_idx': idx,
            'question': question,
            'ground_truth': ground_truth,
            'answers': [{'agent': a['agent_name'], 'answer': a['answer'],
                         'model': a['model_id']} for a in answers],
            'comparison_matrix': R.tolist(),

            # Algorithm results
            'ccrr_idx': ccrr_idx,
            'ccrr_answer': answer_strings[ccrr_idx],
            'ccrr_correct': ccrr_correct,
            'ccrr_scores': ccrr_scores.tolist(),
            'ccrr_weights': ccrr_weights.tolist(),

            'spectral_idx': spectral_idx,
            'spectral_answer': answer_strings[spectral_idx],
            'spectral_correct': spectral_correct,
            'spectral_scores': spectral_scores.tolist(),
            'spectral_weights': spectral_weights.tolist(),
            'spectral_gap': spectral_details['spectral_gap'],

            'mv_idx': mv_idx,
            'mv_answer': answer_strings[mv_idx],
            'mv_correct': mv_correct,

            'rand_idx': rand_idx,
            'rand_correct': rand_correct,

            'oracle_correct': oracle_correct,
            'num_correct_agents': num_correct_agents,
            'agent_correctness': agent_correctness,
            'agent_tiers': exp_data['agent_tiers'],
            'agent_names': exp_data['agent_names'],

            'elapsed_time': exp_data['elapsed_time'],
        }

    # ─────────────────────────────────────────────────────────────
    # Run all questions (Track B)
    # ─────────────────────────────────────────────────────────────

    def run_all(self):
        """Run Track B experiment on all questions."""
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Track B: Natural LLM Experiment ({len(self.dataset)} questions)")
        print(f"{'='*60}")

        for i in range(len(self.dataset)):
            try:
                result = self.run_single_question(i)
                self.results.append(result)

                # Intermediate save
                if (i + 1) % self.config.get('save_frequency', 10) == 0:
                    self._save("intermediate")

            except Exception as e:
                print(f"\n  ERROR on question {i}: {e}")
                import traceback
                traceback.print_exc()

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Track B complete! {len(self.results)} questions in {total_time:.0f}s")
        print(f"{'='*60}")

    # ─────────────────────────────────────────────────────────────
    # Track A: Post-hoc spammer injection
    # ─────────────────────────────────────────────────────────────

    def run_track_a(self, spammer_counts: List[int], num_trials: int = 5):
        """
        Run Track A analysis on saved Track B results.

        For each spammer count k, inject k artificial spammers into
        the existing R matrices and re-run algorithms. This avoids
        re-calling the LLM API.

        Args:
            spammer_counts: List of k values to test (e.g., [0, 2, 4, 6, 8])
            num_trials: Number of random trials per k (for variance estimation)
        """
        if not self.results:
            print("ERROR: No Track B results to inject spammers into. Run Track B first.")
            return {}

        print(f"\n{'='*60}")
        print(f"Track A: Artificial Spammer Injection")
        print(f"  Spammer counts to test: {spammer_counts}")
        print(f"  Trials per count: {num_trials}")
        print(f"{'='*60}")

        track_a_results = {}
        track_a_detailed = {}  # Per-question detailed results

        for k in spammer_counts:
            ccrr_accs = []
            spectral_accs = []
            mv_accs = []
            track_a_detailed[k] = {}  # Store per-question results for this k

            for trial in range(num_trials):
                ccrr_correct_count = 0
                spectral_correct_count = 0
                mv_correct_count = 0

                for q_idx, result in enumerate(self.results):
                    R_orig = np.array(result['comparison_matrix'])
                    gt = result['ground_truth']
                    answer_strings_orig = [a['answer'] for a in result['answers']]

                    # Inject k spammers into comparison matrix R
                    R_new, spammer_mask = self.agent_system.inject_artificial_spammers(
                        R_orig, k=k, seed=trial * 1000 + result['question_idx']
                    )

                    # Also corrupt the spammer agents' answers
                    # This ensures MV is also affected by spammer injection
                    answer_strings_new = answer_strings_orig.copy()
                    spammer_indices = np.where(spammer_mask)[0]

                    # For each spammer, replace their answer with a WRONG answer
                    np.random.seed(trial * 1000 + result['question_idx'])
                    for spammer_idx in spammer_indices:
                        # Strategy 1: Try to use actual wrong answers from the pool
                        wrong_answers = [
                            ans for i, ans in enumerate(answer_strings_orig)
                            if i != spammer_idx and not check_correct(ans, gt, self.api_key)
                        ]

                        if wrong_answers:
                            # Use an actual wrong answer from the pool (most realistic)
                            answer_strings_new[spammer_idx] = np.random.choice(wrong_answers)
                        else:
                            # Strategy 2: Generate a wrong answer based on GT
                            # For numeric answers, add random offset from GT
                            import re
                            try:
                                # Try to extract numeric value from GT
                                gt_match = re.search(r'-?\d+\.?\d*', str(gt))
                                if gt_match:
                                    gt_num = float(gt_match.group())
                                    # Add large random offset to ensure it's wrong
                                    offset = np.random.choice([-100, -50, -20, -10, 10, 20, 50, 100])
                                    wrong_num = gt_num + offset
                                    # Format as integer if GT was integer
                                    if gt_num == int(gt_num):
                                        answer_strings_new[spammer_idx] = str(int(wrong_num))
                                    else:
                                        answer_strings_new[spammer_idx] = str(wrong_num)
                                else:
                                    # Non-numeric GT: use fixed wrong marker
                                    answer_strings_new[spammer_idx] = "WRONG_ANSWER"
                            except:
                                # Parsing failed: use fixed wrong marker
                                answer_strings_new[spammer_idx] = "WRONG_ANSWER"

                    # Re-run algorithms on modified data
                    ccrr_new = CCRRanking(beta=5.0, epsilon=0.1)
                    ccrr_idx_new = ccrr_new.select_best(R_new)

                    spectral_new = SpectralRanking(beta=5.0)
                    spectral_idx_new = spectral_new.select_best(R_new)

                    mv_idx_new = Baselines.majority_voting(answer_strings_new)

                    # Check correctness for each method
                    ccrr_correct = check_correct(answer_strings_new[ccrr_idx_new], gt, self.api_key)
                    spectral_correct = check_correct(answer_strings_new[spectral_idx_new], gt, self.api_key)
                    mv_correct = check_correct(answer_strings_new[mv_idx_new], gt, self.api_key)

                    # Store per-question results (first trial only for clarity)
                    if trial == 0:
                        if q_idx not in track_a_detailed[k]:
                            track_a_detailed[k][q_idx] = {
                                'question': result['question'][:80] + '...' if len(result['question']) > 80 else result['question'],
                                'gt': gt,
                                'ccrr': [],
                                'spectral': [],
                                'mv': []
                            }
                        track_a_detailed[k][q_idx]['ccrr'].append(ccrr_correct)
                        track_a_detailed[k][q_idx]['spectral'].append(spectral_correct)
                        track_a_detailed[k][q_idx]['mv'].append(mv_correct)

                    if ccrr_correct:
                        ccrr_correct_count += 1
                    if spectral_correct:
                        spectral_correct_count += 1
                    if mv_correct:
                        mv_correct_count += 1

                ccrr_acc = ccrr_correct_count / len(self.results) * 100
                spectral_acc = spectral_correct_count / len(self.results) * 100
                mv_acc = mv_correct_count / len(self.results) * 100
                ccrr_accs.append(ccrr_acc)
                spectral_accs.append(spectral_acc)
                mv_accs.append(mv_acc)

            track_a_results[k] = {
                'ccrr_mean': float(np.mean(ccrr_accs)),
                'ccrr_std': float(np.std(ccrr_accs)),
                'spectral_mean': float(np.mean(spectral_accs)),
                'spectral_std': float(np.std(spectral_accs)),
                'mv_mean': float(np.mean(mv_accs)),
                'mv_std': float(np.std(mv_accs)),
            }

            print(f"\n  k={k} spammers (out of {self.config['num_agents']} agents):")
            print(f"    CCRR:            {np.mean(ccrr_accs):.1f}% ± {np.std(ccrr_accs):.1f}%")
            print(f"    Spectral:        {np.mean(spectral_accs):.1f}% ± {np.std(spectral_accs):.1f}%")
            print(f"    Majority Vote:   {np.mean(mv_accs):.1f}% ± {np.std(mv_accs):.1f}%")

        # Print detailed per-question breakdown
        if verbose:
            print(f"\n{'='*60}")
            print(f"Track A: Per-Question Detailed Results")
            print(f"{'='*60}\n")

            for q_idx in range(len(self.results)):
                print(f"Question {q_idx + 1}: {track_a_detailed[spammer_counts[0]][q_idx]['question']}")
                print(f"  GT: {track_a_detailed[spammer_counts[0]][q_idx]['gt']}")
                print(f"\n  {'k':<5} {'CCRR':<10} {'Spectral':<10} {'MV':<10}")
                print(f"  {'-'*40}")

                for k in spammer_counts:
                    if q_idx in track_a_detailed[k]:
                        ccrr_result = '✓' if track_a_detailed[k][q_idx]['ccrr'][0] else '✗'
                        spectral_result = '✓' if track_a_detailed[k][q_idx]['spectral'][0] else '✗'
                        mv_result = '✓' if track_a_detailed[k][q_idx]['mv'][0] else '✗'
                        print(f"  {k:<5} {ccrr_result:<10} {spectral_result:<10} {mv_result:<10}")
                print()

        return track_a_results, track_a_detailed

    # ─────────────────────────────────────────────────────────────
    # Summary & Reporting
    # ─────────────────────────────────────────────────────────────

    def print_summary(self, track_a_results: Dict = None):
        """Print final experiment summary."""
        if not self.results:
            print("No results to summarize.")
            return

        n = len(self.results)

        ccrr_acc = sum(r['ccrr_correct'] for r in self.results) / n * 100
        spectral_acc = sum(r['spectral_correct'] for r in self.results) / n * 100
        mv_acc = sum(r['mv_correct'] for r in self.results) / n * 100
        rand_acc = sum(r['rand_correct'] for r in self.results) / n * 100
        oracle_acc = sum(r['oracle_correct'] for r in self.results) / n * 100
        avg_correct = np.mean([r['num_correct_agents'] for r in self.results])

        print(f"\n{'═'*60}")
        print(f"  EXPERIMENT SUMMARY")
        print(f"{'═'*60}")
        print(f"  Questions: {n},  Agents: {self.config['num_agents']}")
        print(f"  Avg agents with correct answer: {avg_correct:.1f}/{self.config['num_agents']}")

        print(f"\n  {'Method':<30} {'Accuracy':>10}")
        print(f"  {'─'*41}")
        print(f"  {'CCRR Algorithm':<30} {ccrr_acc:>9.1f}%")
        print(f"  {'Spectral Ranking':<30} {spectral_acc:>9.1f}%")
        print(f"  {'Majority Voting':<30} {mv_acc:>9.1f}%")
        print(f"  {'Random Selection':<25} {rand_acc:>9.1f}%")
        print(f"  {'Oracle (upper bound)':<25} {oracle_acc:>9.1f}%")

        # CCRR weight analysis: do weights correlate with agent quality?
        print(f"\n{'═'*60}")
        print(f"  CCRR WEIGHT ANALYSIS (Agent Reliability Detection)")
        print(f"{'═'*60}")

        # Average weights across all questions, grouped by tier
        all_weights = np.array([r['ccrr_weights'] for r in self.results])
        avg_weights = all_weights.mean(axis=0)
        tiers = self.results[0]['agent_tiers']
        names = self.results[0]['agent_names']

        # Average weight per tier
        for tier in ['strong', 'mid', 'weak']:
            mask = [t == tier for t in tiers]
            tier_avg = avg_weights[mask].mean() if any(mask) else 0
            print(f"  {tier.upper():<8} avg weight: {tier_avg:.4f}")

        # Correlation between weights and individual agent accuracy
        agent_accs = np.zeros(len(names))
        for r in self.results:
            for i, correct in enumerate(r['agent_correctness']):
                agent_accs[i] += correct
        agent_accs /= n

        try:
            from scipy.stats import spearmanr
            corr, pval = spearmanr(avg_weights, agent_accs)
            print(f"\n  Spearman(CCRR weight, agent accuracy) = {corr:.3f} (p={pval:.3f})")
            if corr > 0.3 and pval < 0.05:
                print(f"  → CCRR successfully identifies more reliable agents!")
            elif pval >= 0.05:
                print(f"  → Not statistically significant (need more questions)")
        except ImportError:
            print(f"\n  (Install scipy for correlation analysis)")

        # Print per-agent breakdown
        print(f"\n  {'Agent':<22} {'Tier':<8} {'Accuracy':>8} {'CCRR Wt':>8}")
        print(f"  {'─'*48}")
        order = np.argsort(-avg_weights)
        for idx in order:
            print(f"  {names[idx]:<22} {tiers[idx]:<8} {agent_accs[idx]*100:>7.1f}% {avg_weights[idx]:>.4f}")

        # Track A results
        if track_a_results:
            print(f"\n{'═'*70}")
            print(f"  TRACK A: Robustness to Spammer Injection")
            print(f"{'═'*70}")
            print(f"  {'Spammers':<10} {'CCRR':>10} {'Spectral':>10} {'MV':>10} {'Best Δ':>10}")
            print(f"  {'─'*70}")
            for k in sorted(track_a_results.keys()):
                r = track_a_results[k]
                ccrr_delta = r['ccrr_mean'] - r['mv_mean']
                spectral_delta = r['spectral_mean'] - r['mv_mean']
                best_delta = max(ccrr_delta, spectral_delta)
                print(f"  k={k:<8} {r['ccrr_mean']:>9.1f}% {r['spectral_mean']:>9.1f}% {r['mv_mean']:>9.1f}% {best_delta:>+9.1f}%")
            print(f"\n  Note: 'Best Δ' shows max(CCRR-MV, Spectral-MV)")
            print(f"  Positive values = SVD-based methods outperform Majority Voting")

        print()

    # ─────────────────────────────────────────────────────────────
    # Save results
    # ─────────────────────────────────────────────────────────────

    def _save(self, prefix: str = "final"):
        """Save results to JSON."""
        output_dir = Path(self.config.get('output_dir', 'results'))
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"{prefix}_results_{timestamp}.json"

        data = {
            'config': self.config,
            'results': self.results,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"  Saved to {filepath}")
        return filepath

    def save_results(self, track_a_results: Dict = None, track_a_detailed: Dict = None):
        """Save final results including Track A if available."""
        output_dir = Path(self.config.get('output_dir', 'results'))
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"final_results_{timestamp}.json"

        data = {
            'config': self.config,
            'results': self.results,
            'track_a': track_a_results,
            'track_a_detailed': track_a_detailed,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"\nFinal results saved to: {filepath}")
        return filepath


# ═════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Hammer-Spammer LLM Experiment")

    parser.add_argument('--num-questions', type=int, default=50,
                        help='Number of questions to run')
    parser.add_argument('--num-agents', type=int, default=10,
                        help='Number of agents (max 10)')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Start index in dataset')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'test'])
    parser.add_argument('--dataset', type=str, default='gsm8k',
                        choices=['gsm8k', 'math'],
                        help='Dataset to use (gsm8k or math)')
    parser.add_argument('--difficulty', type=str, default=None,
                        choices=['easy', 'medium', 'hard'],
                        help='Difficulty filter for MATH dataset')

    parser.add_argument('--track-a', action='store_true',
                        help='Also run Track A (spammer injection) analysis')
    parser.add_argument('--spammer-counts', type=str, default='0,3,5,7,8',
                        help='Comma-separated spammer counts for Track A (default: 0,3,5,7,8)')
    parser.add_argument('--track-a-trials', type=int, default=5,
                        help='Random trials per spammer count in Track A')

    parser.add_argument('--no-parallel-gen', action='store_true')
    parser.add_argument('--no-parallel-cmp', action='store_true')
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--save-frequency', type=int, default=10)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode: 3 questions, verbose')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.debug:
        args.num_questions = 3
        args.verbose = True
        print("DEBUG MODE: 3 questions\n")

    config = {
        'num_questions': args.num_questions,
        'num_agents': min(args.num_agents, len(AGENT_CONFIGS)),
        'start_index': args.start_index,
        'split': args.split,
        'dataset': args.dataset,
        'difficulty': args.difficulty,
        'parallel_generation': not args.no_parallel_gen,
        'parallel_comparison': not args.no_parallel_cmp,
        'output_dir': args.output_dir,
        'save_frequency': args.save_frequency,
        'verbose': args.verbose,
    }

    print(f"{'='*60}")
    print(f"  Hammer-Spammer Model: LLM Agent Experiment")
    print(f"  N={config['num_agents']} agents, {config['num_questions']} questions")
    print(f"{'='*60}")

    # ── Track B: Natural experiment ──
    runner = ExperimentRunner(config)
    runner.run_all()

    # ── Track A: Spammer injection (optional) ──
    track_a_results = None
    if args.track_a:
        spammer_counts = [int(x) for x in args.spammer_counts.split(',')]
        track_a_results, track_a_detailed = runner.run_track_a(
            spammer_counts=spammer_counts,
            num_trials=args.track_a_trials,
        )
    else:
        track_a_detailed = None

    # ── Print summary & save ──
    runner.print_summary(track_a_results)
    filepath = runner.save_results(track_a_results, track_a_detailed)

    print(f"\nDone! Results at: {filepath}")


if __name__ == "__main__":
    main()