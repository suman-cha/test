#!/usr/bin/env python3
"""
Quick Beta Parameter Test - 30 questions only for fast analysis
"""
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.algorithm.ccrr import CCRRanking

def normalize_answer(s: str) -> str:
    """Normalize an answer string for comparison."""
    s = str(s).strip().lower()
    s = s.replace(',', '').replace('$', '').replace('%', '')
    s = s.replace('\\', '').replace('{', '').replace('}', '')
    s = s.strip('.')

    match = re.search(r'-?\d+\.?\d*', s)
    if match:
        num_str = match.group()
        try:
            if '.' in num_str:
                return str(float(num_str))
            else:
                return str(int(num_str))
        except ValueError:
            pass
    return s

# Load results - use only first 30 questions for speed
results_file = 'results/gsm8k_100q/intermediate_results_20260214_024252.json'
with open(results_file) as f:
    data = json.load(f)

results = data['results'][:30]  # Only 30 questions
n_questions = len(results)
n_agents = 10

print("="*60)
print("Quick Beta Sensitivity Test")
print("="*60)
print(f"Questions: {n_questions} (subset for speed)")
print(f"Agents: {n_agents}")
print()

# Test fewer beta values but cover the range
beta_values = [1.0, 2.0, 5.0, 10.0, 20.0]
spammer_counts = [2, 4, 6, 8]  # 20%, 40%, 60%, 80%
num_trials = 3  # Reduced trials for speed

print(f"Beta values: {beta_values}")
print(f"Spammer counts: {spammer_counts}")
print(f"Trials: {num_trials}")
print()

beta_results = {beta: {k: [] for k in spammer_counts} for beta in beta_values}

print("Running experiments...")
print()

for beta in beta_values:
    print(f"Testing beta = {beta}")

    for k in spammer_counts:
        accuracies = []

        for trial in range(num_trials):
            np.random.seed(trial * 1000 + k * 100 + int(beta * 10))
            correct_count = 0

            for q_idx, result in enumerate(results):
                answer_strings_orig = result['answers']
                comparison_matrix_orig = np.array(result['comparison_matrix'])
                gt = result['ground_truth']

                # Select spammers
                spammer_indices = np.random.choice(n_agents, size=k, replace=False)

                # Inject wrong answers
                answer_strings_new = answer_strings_orig.copy()
                for spammer_idx in spammer_indices:
                    wrong_answers = []
                    for i, ans in enumerate(answer_strings_orig):
                        if i != spammer_idx:
                            ans_str = ans['answer'] if isinstance(ans, dict) else ans
                            if normalize_answer(ans_str) != normalize_answer(str(gt)):
                                wrong_answers.append(ans)

                    if wrong_answers:
                        answer_strings_new[spammer_idx] = np.random.choice(wrong_answers)

                # Inject random comparisons
                comparison_matrix_new = comparison_matrix_orig.copy()
                for spammer_idx in spammer_indices:
                    for j in range(n_agents):
                        if j != spammer_idx:
                            comparison_matrix_new[spammer_idx, j] = np.random.choice([-1, 1])

                # Run CCRR
                ccrr = CCRRanking(beta=beta, epsilon=0.1, T=500)
                try:
                    ccrr_idx, _, _ = ccrr.select_best(comparison_matrix_new, return_details=True)
                    ccrr_answer = answer_strings_new[ccrr_idx]['answer'] if isinstance(answer_strings_new[ccrr_idx], dict) else answer_strings_new[ccrr_idx]

                    if normalize_answer(ccrr_answer) == normalize_answer(str(gt)):
                        correct_count += 1
                except:
                    pass

            accuracy = 100 * correct_count / n_questions
            accuracies.append(accuracy)

        beta_results[beta][k] = accuracies
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        print(f"  k={k}: {mean_acc:.1f}% ± {std_acc:.1f}%")

    print()

# Save and visualize
output_dir = Path('results/beta_analysis')
output_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1: Lines for each beta
ax1 = axes[0]
spammer_ratios = [k / n_agents * 100 for k in spammer_counts]
colors = plt.cm.viridis(np.linspace(0, 1, len(beta_values)))

for beta, color in zip(beta_values, colors):
    means = [np.mean(beta_results[beta][k]) for k in spammer_counts]
    stds = [np.std(beta_results[beta][k]) for k in spammer_counts]
    ax1.plot(spammer_ratios, means, 'o-', linewidth=2, markersize=7,
             label=f'β={beta}', color=color, alpha=0.8)
    ax1.fill_between(spammer_ratios, np.array(means) - np.array(stds),
                      np.array(means) + np.array(stds),
                      alpha=0.1, color=color)

ax1.set_xlabel('Spammer Ratio (%)', fontsize=11)
ax1.set_ylabel('Accuracy (%)', fontsize=11)
ax1.set_title('Beta Sensitivity Analysis', fontsize=12)
ax1.legend(fontsize=9, loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 100)

# Plot 2: Accuracy at high spammer ratio
ax2 = axes[1]
k_high = 8
means_high = [np.mean(beta_results[beta][k_high]) for beta in beta_values]
stds_high = [np.std(beta_results[beta][k_high]) for beta in beta_values]

ax2.errorbar(beta_values, means_high, yerr=stds_high,
             fmt='o-', linewidth=2, markersize=8, capsize=5,
             color='#2E7D32', alpha=0.8)

optimal_idx = np.argmax(means_high)
optimal_beta = beta_values[optimal_idx]
ax2.axvline(x=optimal_beta, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax2.text(optimal_beta, max(means_high) + 2, f'β={optimal_beta}',
         ha='center', fontsize=10, color='red')

ax2.set_xlabel('Beta Parameter', fontsize=11)
ax2.set_ylabel('Accuracy (%)', fontsize=11)
ax2.set_title(f'Accuracy at 80% Spammers', fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'beta_sensitivity_quick.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'beta_sensitivity_quick.pdf', bbox_inches='tight')

print(f"✅ Results saved:")
print(f"   - {output_dir / 'beta_sensitivity_quick.png'}")
print(f"   - {output_dir / 'beta_sensitivity_quick.pdf'}")
print()

# Summary
print("="*60)
print("Summary at 80% Spammers:")
print("="*60)
for beta in beta_values:
    acc = np.mean(beta_results[beta][8])
    std = np.std(beta_results[beta][8])
    marker = " ★" if beta == optimal_beta else ""
    print(f"  β={beta:5.1f}: {acc:5.1f}% ± {std:.1f}%{marker}")

print()
print(f"Optimal beta: {optimal_beta}")
print(f"Note: Based on {n_questions} questions × {num_trials} trials")
