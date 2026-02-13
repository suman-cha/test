#!/usr/bin/env python3
"""
Visualize Hammer-Spammer Experiment Results - Clean Version
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

# Load results
results_file = 'results/gsm8k_100q/intermediate_results_20260214_024252.json'
with open(results_file) as f:
    data = json.load(f)

results = data['results']
n_questions = len(results)

# ============================================================
# Track A Data (from user's reported results)
# ============================================================
track_a_data = {
    'spammer_ratios': [0.0, 0.1, 0.2, 0.4, 0.6, 0.8],
    'ccrr_mean': [91.0, 89.2, 88.8, 82.6, 71.2, 51.0],
    'ccrr_std': [0.0, 1.6, 1.2, 1.7, 3.9, 6.2],
    'spectral_mean': [89.0, 84.6, 76.2, 61.8, 41.2, 20.8],
    'spectral_std': [0.0, 2.4, 2.5, 3.3, 3.2, 3.7],
    'mv_mean': [95.0, 95.0, 93.8, 78.2, 41.2, 6.0],
    'mv_std': [0.0, 0.0, 1.2, 2.9, 1.7, 1.9],
}

# ============================================================
# Track B Data
# ============================================================
ccrr_correct = sum(r['ccrr_correct'] for r in results)
spectral_correct = sum(r['spectral_correct'] for r in results)
mv_correct = sum(r['mv_correct'] for r in results)
rand_correct = sum(r['rand_correct'] for r in results)
oracle_correct = sum(r['oracle_correct'] for r in results)

track_b_algorithms = {
    'Ours': 100 * ccrr_correct / n_questions,
    'ColSum': 100 * spectral_correct / n_questions,
    'MV': 100 * mv_correct / n_questions,
    'Random': 100 * rand_correct / n_questions,
    'Oracle': 100 * oracle_correct / n_questions,
}

# Individual agents
agent_names = results[0]['agent_names']
agent_tiers = results[0]['agent_tiers']
n_agents = len(agent_names)
agent_correct_counts = [0] * n_agents

for r in results:
    for i, correct in enumerate(r['agent_correctness']):
        if correct:
            agent_correct_counts[i] += 1

agent_accuracies = [100 * count / n_questions for count in agent_correct_counts]

output_dir = Path('results/gsm8k_100q')
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# Figure 1: Track A Robustness Curve
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

x = np.array(track_a_data['spammer_ratios']) * 100

ax.plot(x, track_a_data['ccrr_mean'], 'o-', linewidth=2.5, markersize=8,
        label='Ours', color='#2E7D32')
ax.fill_between(x,
                 np.array(track_a_data['ccrr_mean']) - np.array(track_a_data['ccrr_std']),
                 np.array(track_a_data['ccrr_mean']) + np.array(track_a_data['ccrr_std']),
                 alpha=0.15, color='#2E7D32')

ax.plot(x, track_a_data['spectral_mean'], 's-', linewidth=2.5, markersize=8,
        label='ColSum', color='#1976D2')
ax.fill_between(x,
                 np.array(track_a_data['spectral_mean']) - np.array(track_a_data['spectral_std']),
                 np.array(track_a_data['spectral_mean']) + np.array(track_a_data['spectral_std']),
                 alpha=0.15, color='#1976D2')

ax.plot(x, track_a_data['mv_mean'], '^-', linewidth=2.5, markersize=8,
        label='Majority Vote', color='#C62828')
ax.fill_between(x,
                 np.array(track_a_data['mv_mean']) - np.array(track_a_data['mv_std']),
                 np.array(track_a_data['mv_mean']) + np.array(track_a_data['mv_std']),
                 alpha=0.15, color='#C62828')

ax.axvline(x=40, color='gray', linestyle='--', alpha=0.4, linewidth=1.5)

ax.set_xlabel('Spammer Ratio (%)', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Track A: Robustness to Artificial Spammer Injection', fontsize=13)
ax.legend(fontsize=11, loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-2, 82)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(output_dir / 'fig1_track_a_robustness.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'fig1_track_a_robustness.pdf', bbox_inches='tight')
plt.close()

print(f"✓ Figure 1 saved: track_a_robustness")

# ============================================================
# Figure 2: Track B Algorithm Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

algorithms = list(track_b_algorithms.keys())
accuracies = list(track_b_algorithms.values())
colors = ['#2E7D32', '#1976D2', '#C62828', '#757575', '#FFD700']

bars = ax.bar(algorithms, accuracies, color=colors, alpha=0.7,
              edgecolor='black', linewidth=1)

for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Track B: Algorithm Comparison (Natural Experiment)', fontsize=13)
ax.set_ylim(0, 105)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'fig2_track_b_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'fig2_track_b_comparison.pdf', bbox_inches='tight')
plt.close()

print(f"✓ Figure 2 saved: track_b_comparison")

# ============================================================
# Figure 3: Individual Agent Performance
# ============================================================
fig, ax = plt.subplots(figsize=(8, 7))

tier_colors = {'high': '#2E7D32', 'mid': '#1976D2', 'low': '#F57C00'}
bar_colors = [tier_colors[tier] for tier in agent_tiers]

display_names = [name.replace('llama-', 'llm-').replace('claude-', 'cld-')
                 .replace('gemini-', 'gmn-').replace('mistral', 'mstrl')
                 for name in agent_names]

bars = ax.barh(display_names, agent_accuracies, color=bar_colors, alpha=0.7,
               edgecolor='black', linewidth=1)

for bar, acc in zip(bars, agent_accuracies):
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
            f'{acc:.0f}%', ha='left', va='center', fontsize=9)

ax.set_xlabel('Accuracy (%)', fontsize=12)
ax.set_title('Individual Agent Performance', fontsize=13)
ax.set_xlim(0, 105)
ax.grid(axis='x', alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2E7D32', alpha=0.7, label='High-tier'),
    Patch(facecolor='#1976D2', alpha=0.7, label='Mid-tier'),
    Patch(facecolor='#F57C00', alpha=0.7, label='Low-tier')
]
ax.legend(handles=legend_elements, fontsize=10, loc='best')

plt.tight_layout()
plt.savefig(output_dir / 'fig3_agent_performance.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'fig3_agent_performance.pdf', bbox_inches='tight')
plt.close()

print(f"✓ Figure 3 saved: agent_performance")

# ============================================================
# Figure 4: Performance Degradation
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

ccrr_baseline = track_a_data['ccrr_mean'][0]
spectral_baseline = track_a_data['spectral_mean'][0]
mv_baseline = track_a_data['mv_mean'][0]

ccrr_degradation = [(ccrr_baseline - acc) for acc in track_a_data['ccrr_mean']]
spectral_degradation = [(spectral_baseline - acc) for acc in track_a_data['spectral_mean']]
mv_degradation = [(mv_baseline - acc) for acc in track_a_data['mv_mean']]

x_pos = np.arange(len(x))
width = 0.25

bars1 = ax.bar(x_pos - width, ccrr_degradation, width, label='Ours',
               color='#2E7D32', alpha=0.7, edgecolor='black', linewidth=1)
bars2 = ax.bar(x_pos, spectral_degradation, width, label='ColSum',
               color='#1976D2', alpha=0.7, edgecolor='black', linewidth=1)
bars3 = ax.bar(x_pos + width, mv_degradation, width, label='Majority Vote',
               color='#C62828', alpha=0.7, edgecolor='black', linewidth=1)

ax.set_xlabel('Spammer Ratio (%)', fontsize=12)
ax.set_ylabel('Accuracy Degradation (percentage points)', fontsize=12)
ax.set_title('Performance Degradation from Baseline', fontsize=13)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{int(r)}%' for r in x])
ax.legend(fontsize=11, loc='best')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(mv_degradation) * 1.1)

plt.tight_layout()
plt.savefig(output_dir / 'fig4_degradation.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'fig4_degradation.pdf', bbox_inches='tight')
plt.close()

print(f"✓ Figure 4 saved: degradation")

print(f"\n✅ All figures saved to: {output_dir}/")
print(f"   - fig1_track_a_robustness.png/pdf")
print(f"   - fig2_track_b_comparison.png/pdf")
print(f"   - fig3_agent_performance.png/pdf")
print(f"   - fig4_degradation.png/pdf")
