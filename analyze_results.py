#!/usr/bin/env python3
"""
Analyze intermediate experiment results.

Usage:
    python analyze_results.py <path_to_intermediate_results.json>
    python analyze_results.py results/main_test/gsm8k_*/intermediate_*.json

Analyzes both Track B (natural experiment) and Track A (spammer injection) results.
"""

import json
import sys
import glob
from pathlib import Path


def analyze_results(file_path: str):
    """Analyze a single intermediate results file."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {file_path}")
    print(f"{'='*70}")

    with open(file_path, 'r') as f:
        data = json.load(f)

    config = data['config']
    results = data['results']
    track_a = data.get('track_a_results', {})

    # Basic info
    print(f"\nDataset: {config['dataset']}")
    print(f"Questions completed: {len(results)}")
    print(f"Start index: {config.get('start_index', 0)}")
    print(f"Agents: {config['num_agents']}")
    print(f"Difficulty: {config.get('difficulty', 'all')}")

    # Track B: Natural Experiment
    n = len(results)
    if n == 0:
        print("\n⚠️  No results to analyze!")
        return

    ccrr_correct = sum(r['ccrr_correct'] for r in results)
    spectral_correct = sum(r['spectral_correct'] for r in results)
    mv_correct = sum(r['mv_correct'] for r in results)
    rand_correct = sum(r.get('rand_correct', 0) for r in results)

    ccrr_acc = ccrr_correct / n * 100
    spectral_acc = spectral_correct / n * 100
    mv_acc = mv_correct / n * 100
    rand_acc = rand_correct / n * 100

    print(f"\n{'='*70}")
    print(f"Track B: Natural Experiment (Real Agents)")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 70)
    print(f"{'CCRR':<20} {ccrr_correct:<10} {n:<10} {ccrr_acc:>6.1f}%")
    print(f"{'Spectral Ranking':<20} {spectral_correct:<10} {n:<10} {spectral_acc:>6.1f}%")
    print(f"{'Majority Voting':<20} {mv_correct:<10} {n:<10} {mv_acc:>6.1f}%")
    if rand_correct > 0:
        print(f"{'Random (baseline)':<20} {rand_correct:<10} {n:<10} {rand_acc:>6.1f}%")

    # Comparison
    print(f"\n{'Comparison':<30} {'vs MV':<15}")
    print("-" * 45)
    ccrr_diff = ccrr_acc - mv_acc
    spectral_diff = spectral_acc - mv_acc
    print(f"{'CCRR':<30} {ccrr_diff:>+6.1f}%")
    print(f"{'Spectral Ranking':<30} {spectral_diff:>+6.1f}%")

    # Track A: Spammer Injection
    if track_a:
        print(f"\n{'='*70}")
        print(f"Track A: Artificial Spammer Injection (Robustness Test)")
        print(f"{'='*70}")
        print(f"\nTests how each algorithm handles increasing numbers of spammer agents.")
        print(f"Spammer agents have their answers replaced with random wrong answers.\n")

        # Header
        print(f"{'k (spammers)':<15} {'CCRR':<15} {'Spectral':<15} {'MV':<15}")
        print("-" * 70)

        # Sort by k value
        for k in sorted(track_a.keys(), key=int):
            stats = track_a[k]
            ccrr_str = f"{stats['ccrr_mean']:.1f}% ± {stats['ccrr_std']:.1f}%"
            spectral_str = f"{stats['spectral_mean']:.1f}% ± {stats['spectral_std']:.1f}%"
            mv_str = f"{stats['mv_mean']:.1f}% ± {stats['mv_std']:.1f}%"

            print(f"{k + '/' + str(config['num_agents']):<15} {ccrr_str:<15} {spectral_str:<15} {mv_str:<15}")

        # Analysis
        print(f"\n{'Analysis':<30}")
        print("-" * 70)
        if len(track_a) >= 2:
            k_vals = sorted([int(k) for k in track_a.keys()])
            k0 = str(k_vals[0])
            k_max = str(k_vals[-1])

            # Degradation from k=0 to k=max
            ccrr_deg = track_a[k0]['ccrr_mean'] - track_a[k_max]['ccrr_mean']
            spectral_deg = track_a[k0]['spectral_mean'] - track_a[k_max]['spectral_mean']
            mv_deg = track_a[k0]['mv_mean'] - track_a[k_max]['mv_mean']

            print(f"Accuracy drop from k={k0} to k={k_max}:")
            print(f"  CCRR:     {ccrr_deg:>6.1f}% {'(most robust)' if ccrr_deg == min(ccrr_deg, spectral_deg, mv_deg) else ''}")
            print(f"  Spectral: {spectral_deg:>6.1f}% {'(most robust)' if spectral_deg == min(ccrr_deg, spectral_deg, mv_deg) else ''}")
            print(f"  MV:       {mv_deg:>6.1f}% {'(least robust)' if mv_deg == max(ccrr_deg, spectral_deg, mv_deg) else ''}")

            # Best at high spammer rate
            print(f"\nAt k={k_max} spammers (worst case):")
            best_at_kmax = max(
                ('CCRR', track_a[k_max]['ccrr_mean']),
                ('Spectral', track_a[k_max]['spectral_mean']),
                ('MV', track_a[k_max]['mv_mean']),
                key=lambda x: x[1]
            )
            print(f"  Best: {best_at_kmax[0]} ({best_at_kmax[1]:.1f}%)")
    else:
        print(f"\n⚠️  No Track A results (older version without spammer injection)")

    print(f"\n{'='*70}\n")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <path_to_results.json>")
        print("\nExamples:")
        print("  python analyze_results.py results/main_test/gsm8k_20260213_134034/intermediate_results_20260213_141033.json")
        print("  python analyze_results.py 'results/main_test/*/intermediate_*.json'  # analyze all")
        sys.exit(1)

    # Handle glob patterns
    pattern = sys.argv[1]
    files = glob.glob(pattern)

    if not files:
        # Try as literal path
        if Path(pattern).exists():
            files = [pattern]
        else:
            print(f"Error: No files found matching: {pattern}")
            sys.exit(1)

    # Sort by modification time (most recent first)
    files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)

    print(f"\nFound {len(files)} result file(s)")

    # Analyze each file
    for file_path in files:
        try:
            analyze_results(file_path)
        except Exception as e:
            print(f"\n❌ Error analyzing {file_path}: {e}\n")
            continue


if __name__ == "__main__":
    main()
