"""
Quick Pilot + Row Sum Analysis

This runs a minimal pilot experiment and immediately analyzes
the row sum vs correctness correlation.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from scipy.stats import pearsonr

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.dataset import DatasetLoader, compare_answers
from src.agents.agent_system import AgentSystem


# Working mid-tier models (verified for OpenRouter)
WORKING_MID_TIER_CONFIGS = [
    {
        "model": "openai/gpt-3.5-turbo",
        "name": "gpt35-turbo",
        "description": "OpenAI GPT-3.5 Turbo"
    },
    {
        "model": "openai/gpt-4o-mini",
        "name": "gpt4o-mini",
        "description": "OpenAI GPT-4o Mini"
    },
    {
        "model": "meta-llama/llama-3.1-70b-instruct",
        "name": "llama-70b",
        "description": "Meta Llama 3.1 70B"
    },
    {
        "model": "qwen/qwen-2.5-72b-instruct",
        "name": "qwen-72b",
        "description": "Qwen 2.5 72B"
    },
    {
        "model": "google/gemini-2.0-flash-exp",
        "name": "gemini-flash",
        "description": "Google Gemini 2.0 Flash"
    },
]


def run_minimal_pilot(num_questions: int = 5):
    """Run minimal pilot to get data for analysis."""

    print("="*70)
    print("QUICK PILOT: Row Sum vs Correctness Analysis")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Models: {len(WORKING_MID_TIER_CONFIGS)} mid-tier")
    print(f"  Questions: {num_questions}")
    print(f"  Estimated cost: $0.80-1.50")
    print()

    # Load environment
    load_dotenv()
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env")

    # Load dataset
    print("Loading dataset...")
    loader = DatasetLoader(
        dataset_name='math',
        split='test',
        max_samples=num_questions,
        difficulty_filter='hard'
    )
    print(f"Loaded {len(loader)} questions\n")

    # Initialize agent system
    print("Initializing agents...")
    agent_system = AgentSystem(
        agent_configs=WORKING_MID_TIER_CONFIGS,
        api_key=api_key,
        epsilon=0.1,
        parallel_generation=False,  # Sequential to avoid rate limits
        parallel_comparison=False
    )
    print(f"Initialized {len(WORKING_MID_TIER_CONFIGS)} agents\n")

    # Run experiments
    results = []

    print("="*70)
    print("RUNNING EXPERIMENTS")
    print("="*70)
    print()

    for q_idx in range(len(loader)):
        question, ground_truth = loader.get_question(q_idx)

        print(f"Question {q_idx + 1}/{len(loader)}")
        print(f"Q: {question[:80]}...")

        try:
            # Get answers and comparisons
            exp_data = agent_system.run_experiment(
                question=question,
                ground_truth=ground_truth,
                verbose=False
            )

            R = exp_data['comparison_matrix']
            answers = exp_data['answers']

            # Store result
            result = {
                'question_idx': q_idx,
                'question': question,
                'ground_truth': ground_truth,
                'answers': answers,
                'comparison_matrix': R.tolist(),
            }
            results.append(result)

            print(f"  ✓ Completed")
            print()

        except Exception as e:
            print(f"  ✗ Error: {e}")
            print()
            continue

    # Save results
    output_dir = Path("pilot_results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"pilot_rowsum_analysis_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'num_questions': num_questions,
                'dataset': 'math',
                'difficulty': 'hard',
                'agents': WORKING_MID_TIER_CONFIGS
            },
            'results': results
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return results_file, results


def analyze_results_inline(results):
    """Analyze row sum vs correctness correlation inline."""

    print("\n" + "="*70)
    print("ROW SUM vs CORRECTNESS ANALYSIS")
    print("="*70)
    print()

    # Collect data
    all_row_sums = []
    all_correct = []
    per_agent_data = {}

    N_agents = len(results[0]['answers']) if results else 0

    for i in range(N_agents):
        per_agent_data[i] = []

    # Process each question
    for result in results:
        R = np.array(result['comparison_matrix'])
        N = R.shape[0]

        answers = [a['answer'] for a in result['answers']]
        ground_truth = result['ground_truth']

        # For each agent
        for i in range(N):
            # Compute row sum (excluding diagonal)
            row_sum = (R[i, :].sum() - R[i, i]) / (N - 1)

            # Check correctness
            is_correct = compare_answers(answers[i], ground_truth)

            all_row_sums.append(row_sum)
            all_correct.append(is_correct)
            per_agent_data[i].append((row_sum, is_correct))

    # Convert to arrays
    all_row_sums = np.array(all_row_sums)
    all_correct = np.array(all_correct, dtype=float)

    # Overall correlation
    print("OVERALL CORRELATION")
    print("-"*70)

    if len(all_row_sums) > 2:
        pearson_r, pearson_p = pearsonr(all_row_sums, all_correct)

        print(f"\nAcross {len(all_row_sums)} (question, agent) pairs:")
        print(f"  Pearson correlation:  r = {pearson_r:+.4f}, p = {pearson_p:.4e}")
        print()

        # Interpretation
        print("INTERPRETATION:")
        print("-"*70)

        if pearson_r > 0.3:
            print(f"✓ POSITIVE correlation (r = {pearson_r:.3f})")
            print(f"  → High row sum correlates with correctness")
            print(f"  → Bradley-Terry assumption holds!")
            print(f"  → Algorithm should work ✓")

        elif pearson_r > 0.1:
            print(f"⚠️  WEAK positive correlation (r = {pearson_r:.3f})")
            print(f"  → Some signal, but noisy")
            print(f"  → Algorithm will struggle")

        elif pearson_r > -0.1:
            print(f"❌ NO correlation (r ≈ {pearson_r:.3f})")
            print(f"  → Row sum is random noise")
            print(f"  → Algorithm cannot work!")

        else:
            print(f"❌❌ NEGATIVE correlation (r = {pearson_r:.3f})!!!")
            print(f"  → High row sum → MORE LIKELY WRONG")
            print(f"  → Algorithm will select WRONG answers")
            print(f"  → Wrong Answer Cluster hypothesis CONFIRMED!")

        # Distribution comparison
        print()
        print("ROW SUM DISTRIBUTION:")
        print("-"*70)

        correct_row_sums = all_row_sums[all_correct == 1]
        incorrect_row_sums = all_row_sums[all_correct == 0]

        if len(correct_row_sums) > 0:
            print(f"  Correct answers:   mean = {correct_row_sums.mean():.3f}, "
                  f"std = {correct_row_sums.std():.3f}, n = {len(correct_row_sums)}")
        else:
            print(f"  Correct answers:   (none)")

        if len(incorrect_row_sums) > 0:
            print(f"  Incorrect answers: mean = {incorrect_row_sums.mean():.3f}, "
                  f"std = {incorrect_row_sums.std():.3f}, n = {len(incorrect_row_sums)}")
        else:
            print(f"  Incorrect answers: (none)")

        if len(correct_row_sums) > 0 and len(incorrect_row_sums) > 0:
            diff = correct_row_sums.mean() - incorrect_row_sums.mean()
            print(f"  Difference:        {diff:+.3f}")
            print()

            if diff < -0.05:
                print("  ❌ CRITICAL: Incorrect answers have HIGHER row sums!")
                print("     → Algorithm systematically prefers wrong answers")

        # Per-agent analysis
        print()
        print("PER-AGENT BREAKDOWN:")
        print("-"*70)
        print(f"{'Agent':<8} {'N_correct':<12} {'Correlation':<14} {'Status'}")
        print("-"*70)

        for agent_idx in range(N_agents):
            data = per_agent_data[agent_idx]
            if len(data) < 2:
                continue

            row_sums = np.array([d[0] for d in data])
            correct = np.array([d[1] for d in data], dtype=float)

            n_correct = int(correct.sum())
            n_total = len(correct)

            if np.std(row_sums) > 1e-6 and np.std(correct) > 1e-6:
                r, p = pearsonr(row_sums, correct)

                if r > 0.2:
                    status = "✓ Reliable"
                elif r > 0:
                    status = "~ Weak"
                elif r > -0.2:
                    status = "⚠️  Random"
                else:
                    status = "✗ Inverted"

                print(f"{agent_idx:<8} {n_correct}/{n_total:<9} r={r:+.3f}          {status}")
            else:
                print(f"{agent_idx:<8} {n_correct}/{n_total:<9} {'(constant)'}     -")

    else:
        print("Not enough data for correlation analysis")

    print()
    print("="*70)


def main():
    """Main execution."""
    import sys

    num_questions = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    print("\n" + "="*70)
    print("QUICK PILOT + ROW SUM ANALYSIS")
    print("="*70)
    print()

    # Run pilot
    results_file, results = run_minimal_pilot(num_questions)

    # Analyze
    if results:
        analyze_results_inline(results)

        print()
        print("="*70)
        print("NEXT STEPS")
        print("="*70)
        print()
        print("Results saved. For detailed analysis:")
        print(f"  python analyze_rowsum_correctness.py {results_file}")
        print()
    else:
        print("\n❌ No results to analyze")


if __name__ == "__main__":
    main()
