"""
Pilot Study: Estimate α (self-preference bias) and β (discrimination).

This small-scale experiment checks if mid-tier models have better
bias characteristics than low-tier models.

Cost: ~$0.50-1.00 for 3 questions × 5 agents
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import os
from dotenv import load_dotenv

from src.utils.dataset import DatasetLoader
from src.agents.agent_system import AgentSystem
from src.agents.llm_agent import LLMAgent


# Mid-tier models for pilot
PILOT_MID_TIER_CONFIGS = [
    {
        "model": "openai/gpt-3.5-turbo",
        "name": "gpt35-turbo",
        "description": "OpenAI GPT-3.5 Turbo"
    },
    {
        "model": "anthropic/claude-3-haiku-20240307",
        "name": "claude-haiku",
        "description": "Anthropic Claude 3 Haiku"
    },
    {
        "model": "google/gemini-flash-1.5",
        "name": "gemini-flash",
        "description": "Google Gemini 1.5 Flash"
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
]


def estimate_bias_parameters(R: np.ndarray) -> Dict:
    """
    Estimate α (self-preference bias) and β (discrimination) from R.

    Args:
        R: Comparison matrix (N, N) with values in {-1, 1}

    Returns:
        Dictionary with estimated parameters
    """
    N = R.shape[0]

    # Convert {-1, 1} to {0, 1}
    R_01 = (R + 1) / 2

    # 1. Estimate α from average row sum (excluding diagonal)
    row_sums = []
    for i in range(N):
        row_sum = (R_01[i, :].sum() - R_01[i, i]) / (N - 1)
        row_sums.append(row_sum)

    avg_row_sum = np.mean(row_sums)

    # α = logit(p) where p = P(R_ij = 1)
    p_win = avg_row_sum
    if p_win <= 0.01:
        p_win = 0.01
    elif p_win >= 0.99:
        p_win = 0.99

    alpha_est = np.log(p_win / (1 - p_win))

    # 2. Check if α is constant across agents
    row_sum_variance = np.var(row_sums)
    row_sum_std = np.std(row_sums)

    # 3. Estimate β from pairwise differences
    # If we knew true scores s_i, we'd have:
    # R_ij ≈ sign(β(s_i - s_j) + α)
    # Without knowing s, use a proxy: assume row sum ~ score

    proxy_scores = np.array(row_sums)

    # For pairs where proxy score difference is large,
    # check if R_ij follows the sign
    beta_estimates = []
    for i in range(N):
        for j in range(i+1, N):
            if i == j:
                continue
            score_diff = proxy_scores[i] - proxy_scores[j]
            if abs(score_diff) < 0.1:
                continue  # Skip similar scores

            # Expected: R_ij = 1 if score_diff > 0
            expected_sign = 1 if score_diff > 0 else -1
            actual = R[i, j]

            # If they match, β is working
            if actual == expected_sign:
                # Rough estimate: β ≈ logit(p) / score_diff
                # But this is very rough
                beta_estimates.append(1.0)  # Placeholder

    # For now, just report if comparisons are consistent
    consistency = len(beta_estimates) / (N * (N-1) / 2) if N > 1 else 0

    return {
        'alpha_estimate': float(alpha_est),
        'avg_row_sum': float(avg_row_sum),
        'row_sum_std': float(row_sum_std),
        'row_sum_variance': float(row_sum_variance),
        'row_sums_per_agent': [float(x) for x in row_sums],
        'comparison_consistency': float(consistency),
        'interpretation': {
            'bias_magnitude': 'low' if abs(alpha_est) < 0.3 else 'medium' if abs(alpha_est) < 0.7 else 'high',
            'bias_direction': 'self-preference' if alpha_est > 0 else 'anti-self-preference' if alpha_est < 0 else 'neutral',
            'bias_uniformity': 'uniform' if row_sum_std < 0.15 else 'varied',
        }
    }


def run_pilot_study(num_questions: int = 3, dataset: str = 'math',
                   difficulty: str = 'hard') -> Dict:
    """
    Run pilot study to estimate bias parameters.

    Args:
        num_questions: Number of questions (default 3)
        dataset: Dataset name
        difficulty: Difficulty level

    Returns:
        Results dictionary
    """
    print("="*70)
    print("PILOT STUDY: Bias Parameter Estimation")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  - Agents: {len(PILOT_MID_TIER_CONFIGS)} mid-tier models")
    print(f"  - Questions: {num_questions} ({dataset} {difficulty})")
    print(f"  - Estimated cost: $0.50-1.00")
    print()

    # Load environment
    load_dotenv()
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found")

    # Load dataset
    print("Loading dataset...")
    loader = DatasetLoader(
        dataset_name=dataset,
        split='test',
        max_samples=num_questions,
        difficulty_filter=difficulty
    )

    # Initialize agent system
    print("Initializing agents...")
    agent_system = AgentSystem(
        agent_configs=PILOT_MID_TIER_CONFIGS,
        api_key=api_key,
        epsilon=0.1,
        parallel_generation=True,
        parallel_comparison=True
    )

    # Run experiments
    print(f"\nProcessing {len(loader)} questions...\n")

    results = []
    all_bias_estimates = []

    for q_idx in range(len(loader)):
        question, ground_truth = loader.get_question(q_idx)

        print(f"Question {q_idx + 1}/{len(loader)}")
        print(f"Q: {question[:80]}...")

        # Get answers and comparisons
        exp_data = agent_system.run_experiment(
            question=question,
            ground_truth=ground_truth,
            verbose=False
        )

        R = exp_data['comparison_matrix']

        # Estimate bias parameters
        bias_params = estimate_bias_parameters(R)
        all_bias_estimates.append(bias_params)

        print(f"  α estimate: {bias_params['alpha_estimate']:.3f}")
        print(f"  Avg row sum: {bias_params['avg_row_sum']:.3f}")
        print(f"  Row sum std: {bias_params['row_sum_std']:.3f}")
        print(f"  Bias: {bias_params['interpretation']['bias_direction']}")
        print()

        results.append({
            'question_idx': q_idx,
            'question': question,
            'ground_truth': ground_truth,
            'comparison_matrix': R.tolist(),
            'bias_parameters': bias_params,
            'answers': exp_data['answers']
        })

    # Aggregate results
    print("="*70)
    print("AGGREGATED RESULTS")
    print("="*70)

    alpha_values = [b['alpha_estimate'] for b in all_bias_estimates]
    row_sum_values = [b['avg_row_sum'] for b in all_bias_estimates]
    row_sum_stds = [b['row_sum_std'] for b in all_bias_estimates]

    avg_alpha = np.mean(alpha_values)
    avg_row_sum = np.mean(row_sum_values)
    avg_row_sum_std = np.mean(row_sum_stds)

    print(f"\nBias Parameters (averaged over {num_questions} questions):")
    print(f"  α (self-preference bias): {avg_alpha:.3f}")
    print(f"  Average row sum: {avg_row_sum:.3f}")
    print(f"  Row sum std (uniformity): {avg_row_sum_std:.3f}")

    print(f"\nInterpretation:")
    if abs(avg_alpha) < 0.3:
        print(f"  ✓ LOW BIAS: α ≈ {avg_alpha:.2f} is close to 0")
        print(f"    → Agents have minimal self-preference bias")
        print(f"    → Bradley-Terry model assumption holds!")
    elif abs(avg_alpha) < 0.7:
        print(f"  ⚠️  MODERATE BIAS: α ≈ {avg_alpha:.2f}")
        print(f"    → Some bias present, but may be manageable")
    else:
        print(f"  ✗ HIGH BIAS: α ≈ {avg_alpha:.2f}")
        print(f"    → Strong systematic bias")
        print(f"    → Bradley-Terry model violated")

    if avg_row_sum_std < 0.15:
        print(f"\n  ✓ UNIFORM BIAS: std = {avg_row_sum_std:.3f}")
        print(f"    → Bias is consistent across agents")
        print(f"    → Can potentially be corrected")
    else:
        print(f"\n  ✗ VARIED BIAS: std = {avg_row_sum_std:.3f}")
        print(f"    → Different agents have different biases")
        print(f"    → Harder to correct")

    print(f"\nRECOMMENDATION:")
    if abs(avg_alpha) < 0.5 and avg_row_sum > 0.4:
        print(f"  ✅ PROCEED with mid-tier models for full experiment")
        print(f"     Algorithms should work reasonably well.")
    elif abs(avg_alpha) < 0.7:
        print(f"  ⚠️  BORDERLINE: Consider algorithm modifications")
        print(f"     or use even higher-tier models.")
    else:
        print(f"  ❌ STOP: Bias too high, algorithms unlikely to work")
        print(f"     Consider: bias correction, different models,")
        print(f"     or honest reporting of limitations.")

    print("="*70)

    # Save results
    output_dir = Path("pilot_results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"pilot_study_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'num_questions': num_questions,
                'dataset': dataset,
                'difficulty': difficulty,
                'agents': PILOT_MID_TIER_CONFIGS
            },
            'results': results,
            'summary': {
                'avg_alpha': float(avg_alpha),
                'avg_row_sum': float(avg_row_sum),
                'avg_row_sum_std': float(avg_row_sum_std),
                'alpha_per_question': [float(x) for x in alpha_values]
            }
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return {
        'avg_alpha': avg_alpha,
        'avg_row_sum': avg_row_sum,
        'results': results
    }


if __name__ == "__main__":
    from datetime import datetime
    import sys

    # Parse args
    num_q = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    dataset = sys.argv[2] if len(sys.argv) > 2 else 'math'
    difficulty = sys.argv[3] if len(sys.argv) > 3 else 'hard'

    # Run pilot
    results = run_pilot_study(num_q, dataset, difficulty)
