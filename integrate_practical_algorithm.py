"""
Quick integration of practical weighted voting into experiment runner.

This modifies the experiment to track agent quality across questions
and use it for selection.
"""

from src.algorithm.practical_weighted_voting import HybridSelector, PracticalWeightedVoting
from src.algorithm.ccrr import CCRRanking
from src.algorithm.swra import SWRARanking


def run_with_practical_algorithm(dataset='math', difficulty='hard', num_questions=30):
    """
    Run experiment with practical weighted voting that learns from outcomes.
    """
    print("="*70)
    print("EXPERIMENT: Practical Weighted Voting (Learning from Outcomes)")
    print("="*70)

    # Initialize
    from src.utils.dataset import DatasetLoader, compare_answers
    from src.agents.agent_config import AGENT_CONFIGS
    from src.agents.agent_system import AgentSystem
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv('OPENROUTER_API_KEY')

    # Load dataset
    dataset_loader = DatasetLoader(
        dataset_name=dataset,
        split='test',
        max_samples=num_questions,
        difficulty_filter=difficulty
    )

    # Initialize agents
    agent_configs = AGENT_CONFIGS[:15]
    agent_system = AgentSystem(
        agent_configs=agent_configs,
        api_key=api_key,
        epsilon=0.1,
        parallel_generation=True,
        parallel_comparison=True
    )

    # Initialize algorithms
    ccrr = CCRRanking(beta=5.0, epsilon=0.1, T=5)
    swra = SWRARanking(k=2, max_iter=15, version='iterative')
    pwv = PracticalWeightedVoting(learning_rate=0.15)
    hybrid = HybridSelector(ccrr_algo=ccrr, swra_algo=swra)

    # Track results
    results = {
        'ccrr': {'correct': 0, 'total': 0},
        'swra': {'correct': 0, 'total': 0},
        'pwv': {'correct': 0, 'total': 0},
        'hybrid_adaptive': {'correct': 0, 'total': 0},
        'hybrid_ensemble': {'correct': 0, 'total': 0},
        'mv': {'correct': 0, 'total': 0}
    }

    print(f"\nProcessing {len(dataset_loader)} questions...\n")

    for q_idx in range(len(dataset_loader)):
        question, ground_truth = dataset_loader.get_question(q_idx)

        print(f"Question {q_idx + 1}/{len(dataset_loader)}")

        # Get answers and comparisons
        exp_data = agent_system.run_experiment(
            question=question,
            ground_truth=ground_truth,
            verbose=False
        )

        R = exp_data['comparison_matrix']
        answers = [a['answer'] for a in exp_data['answers']]

        # Apply all methods
        ccrr_idx = ccrr.select_best(R)
        swra_idx = swra.select_best(R)
        pwv_idx = pwv.select_best(answers=answers)
        hybrid_adap_idx, _ = hybrid.select_best(R, answers, strategy='adaptive')
        hybrid_ens_idx, _ = hybrid.select_best(R, answers, strategy='ensemble')

        # Majority voting
        from collections import Counter
        normalized = [a.strip().lower() for a in answers]
        mv_answer = Counter(normalized).most_common(1)[0][0]
        mv_idx = normalized.index(mv_answer)

        # Check correctness
        methods_and_indices = {
            'ccrr': ccrr_idx,
            'swra': swra_idx,
            'pwv': pwv_idx,
            'hybrid_adaptive': hybrid_adap_idx,
            'hybrid_ensemble': hybrid_ens_idx,
            'mv': mv_idx
        }

        for method, idx in methods_and_indices.items():
            is_correct = compare_answers(answers[idx], ground_truth)
            results[method]['total'] += 1
            if is_correct:
                results[method]['correct'] += 1

        # UPDATE LEARNING (key step!)
        # Update PWV agent quality
        for agent_idx, answer in enumerate(answers):
            was_correct = compare_answers(answer, ground_truth)
            pwv.update_agent_quality(agent_idx, was_correct)
            hybrid.update_agent_quality(agent_idx, was_correct)

        # Update hybrid method performance
        for method, idx in methods_and_indices.items():
            if method.startswith('hybrid'):
                continue  # Don't update hybrid with itself
            is_correct = compare_answers(answers[idx], ground_truth)
            hybrid.update_performance(method, is_correct)

        # Print progress
        print(f"  CCRR: {'✓' if compare_answers(answers[ccrr_idx], ground_truth) else '✗'}")
        print(f"  PWV:  {'✓' if compare_answers(answers[pwv_idx], ground_truth) else '✗'}")
        print(f"  MV:   {'✓' if compare_answers(answers[mv_idx], ground_truth) else '✗'}")
        print()

    # Print final results
    print("="*70)
    print("FINAL RESULTS")
    print("="*70)
    print()

    print(f"{'Method':<25} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-"*70)

    for method, stats in sorted(results.items(), key=lambda x: x[1]['correct'] / x[1]['total'], reverse=True):
        accuracy = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"{method:<25} {stats['correct']:<10} {stats['total']:<10} {accuracy:<10.2f}%")

    print()
    print("="*70)
    print("KEY INSIGHT:")
    print("="*70)
    print()
    print("Practical Weighted Voting (PWV) learns from actual outcomes,")
    print("avoiding the model mismatch problem of Bradley-Terry assumptions.")
    print()
    print("Hybrid approaches can adaptively select the best method for each")
    print("question based on historical performance.")
    print()


if __name__ == "__main__":
    import sys

    # Default to MATH hard, 30 questions
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'math'
    difficulty = sys.argv[2] if len(sys.argv) > 2 else 'hard'
    num_q = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    run_with_practical_algorithm(dataset, difficulty, num_q)
