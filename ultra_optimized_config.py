"""
Ultra-Optimized Agent Configuration: Minimum cost, maximum value.

Strategy: Use 2-3 cheap, reliable low-tier models with persona diversity.
- 8x Llama 3.1 8B (primary workhorse)
- 4x Mistral 7B (architecture diversity)
- 3x Gemma 2 9B (Google's approach)

Total cost for 50 questions: ~$0.30-0.50 (vs $2.50-3.50 original)
Total cost for 100 questions: ~$0.60-1.00 (fits minimal budget!)
"""

ULTRA_OPTIMIZED_CONFIGS = [
    # ===== LLAMA 3.1 8B (8 agents) =====
    # Most reliable, cheapest, best value
    {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "name": "llama-careful",
        "tier": "low",
        "persona": "You are a careful mathematician who shows all work step-by-step and double-checks calculations.",
        "description": "Llama 3.1 8B - Careful"
    },
    {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "name": "llama-quick",
        "tier": "low",
        "persona": "You are a quick problem solver who finds shortcuts and patterns.",
        "description": "Llama 3.1 8B - Quick"
    },
    {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "name": "llama-creative",
        "tier": "low",
        "persona": "You think creatively and explore multiple approaches before solving.",
        "description": "Llama 3.1 8B - Creative"
    },
    {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "name": "llama-formal",
        "tier": "low",
        "persona": "You approach problems methodically using formal mathematical notation.",
        "description": "Llama 3.1 8B - Formal"
    },
    {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "name": "llama-intuitive",
        "tier": "low",
        "persona": "You rely on intuition and pattern recognition to solve problems.",
        "description": "Llama 3.1 8B - Intuitive"
    },
    {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "name": "llama-detailed",
        "tier": "low",
        "persona": "You are detail-oriented and break down complex problems into small steps.",
        "description": "Llama 3.1 8B - Detailed"
    },
    {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "name": "llama-efficient",
        "tier": "low",
        "persona": "You prioritize speed while maintaining accuracy.",
        "description": "Llama 3.1 8B - Efficient"
    },
    {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "name": "llama-practical",
        "tier": "low",
        "persona": "You test answers with concrete examples before finalizing.",
        "description": "Llama 3.1 8B - Practical"
    },

    # ===== MISTRAL 7B (4 agents) =====
    # Different architecture for diversity
    {
        "model": "mistralai/mistral-7b-instruct",
        "name": "mistral-logical",
        "tier": "low",
        "persona": "You are a logical reasoner who builds proofs step-by-step.",
        "description": "Mistral 7B - Logical"
    },
    {
        "model": "mistralai/mistral-7b-instruct",
        "name": "mistral-visual",
        "tier": "low",
        "persona": "You are a visual thinker who uses geometric reasoning.",
        "description": "Mistral 7B - Visual"
    },
    {
        "model": "mistralai/mistral-7b-instruct",
        "name": "mistral-systematic",
        "tier": "low",
        "persona": "You follow systematic algorithmic approaches to problem-solving.",
        "description": "Mistral 7B - Systematic"
    },
    {
        "model": "mistralai/mistral-7b-instruct",
        "name": "mistral-heuristic",
        "tier": "low",
        "persona": "You use heuristics and estimation before detailed calculation.",
        "description": "Mistral 7B - Heuristic"
    },

    # ===== GEMMA 2 9B (3 agents) =====
    # Google architecture for additional diversity
    {
        "model": "google/gemma-2-9b-it",
        "name": "gemma-educator",
        "tier": "low",
        "persona": "You are an experienced educator who explains reasoning clearly.",
        "description": "Gemma 2 9B - Educator"
    },
    {
        "model": "google/gemma-2-9b-it",
        "name": "gemma-competitive",
        "tier": "low",
        "persona": "You are a competitive problem solver seeking elegant solutions.",
        "description": "Gemma 2 9B - Competitive"
    },
    {
        "model": "google/gemma-2-9b-it",
        "name": "gemma-conservative",
        "tier": "low",
        "persona": "You are conservative and verify every step multiple times.",
        "description": "Gemma 2 9B - Conservative"
    },
]

assert len(ULTRA_OPTIMIZED_CONFIGS) == 15


def print_ultra_cost_analysis():
    """Print detailed cost analysis."""
    print("="*70)
    print("ULTRA-OPTIMIZED CONFIGURATION ANALYSIS")
    print("="*70)

    print("\n1. CONFIGURATION BREAKDOWN:")
    print("-"*70)
    print("  - 8 agents: Llama 3.1 8B (Meta, open-source)")
    print("  - 4 agents: Mistral 7B (Mistral AI)")
    print("  - 3 agents: Gemma 2 9B (Google)")
    print()
    print("Why this distribution?")
    print("  ✓ Llama 3.1 8B: Cheapest + most reliable → use as primary")
    print("  ✓ Mistral 7B: Different architecture → adds diversity")
    print("  ✓ Gemma 2 9B: Google's approach → additional variation")

    print("\n2. COST BREAKDOWN (estimated):")
    print("-"*70)
    print("Per 50 questions (N=15 agents):")
    print("  - Answer generation: 15 × 50 × ~200 tokens = 150K tokens")
    print("  - Comparisons: 15 × 14 × 50 × ~300 tokens = 3.15M tokens")
    print("  - Total: ~3.3M tokens")
    print()
    print("Cost estimates:")
    print("  - Llama 3.1 8B: ~$0.15-0.25 (8 agents)")
    print("  - Mistral 7B: ~$0.10-0.15 (4 agents)")
    print("  - Gemma 2 9B: ~$0.05-0.10 (3 agents)")
    print("  - TOTAL: ~$0.30-0.50")

    print("\n3. COMPARISON TO OTHER CONFIGS:")
    print("-"*70)
    configs = [
        ("Original (5 mid + 10 low)", "$2.50-3.50", "Baseline", "❌"),
        ("Cost-optimized (15 low mixed)", "$0.50-1.00", "70-80%", "✅"),
        ("Ultra-optimized (3 models)", "$0.30-0.50", "85-90%", "🏆"),
        ("Extreme (1 model only)", "$0.20-0.40", "90-95%", "⚠️"),
    ]

    print(f"{'Configuration':<35} {'Cost/50Q':<12} {'Savings':<12} {'Rating'}")
    print("-"*70)
    for name, cost, savings, rating in configs:
        print(f"{name:<35} {cost:<12} {savings:<12} {rating}")

    print("\n4. BUDGET SCENARIOS:")
    print("-"*70)
    print("With ultra-optimized config:")
    print()
    print("Budget $1:")
    print("  → 150-200 questions")
    print("  → Excellent statistical power")
    print()
    print("Budget $2:")
    print("  → 300-400 questions")
    print("  → Publication-quality data")
    print()
    print("Budget $5:")
    print("  → 750-1000 questions")
    print("  → Comprehensive analysis")

    print("\n5. SCIENTIFIC VALUE:")
    print("-"*70)
    print("Architecture diversity:")
    print("  ✓ Meta's Llama (decoder-only transformer)")
    print("  ✓ Mistral's architecture (sliding window attention)")
    print("  ✓ Google's Gemma (multi-query attention)")
    print()
    print("Story: 'We tested across 3 major open-source architectures,")
    print("        showing the algorithm limitations are fundamental,")
    print("        not architecture-specific.'")

    print("\n6. PRACTICAL BENEFITS:")
    print("-"*70)
    print("  ✓ 85-90% cost reduction")
    print("  ✓ 3 different model families")
    print("  ✓ Avoids single-model bias")
    print("  ✓ More realistic than 1 model")
    print("  ✓ Strong scientific narrative")
    print("  ✓ Extremely affordable")

    print("\n" + "="*70)
    print("RECOMMENDATION: USE ULTRA-OPTIMIZED CONFIG 🎯")
    print("="*70)


if __name__ == "__main__":
    print_ultra_cost_analysis()
