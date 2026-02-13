"""
Cost-Optimized Agent Configuration: 15 agents, all low-tier.

Strategy: Use persona diversity instead of model diversity to create
variation in answer quality. This provides sufficient signal for
hammer-spammer detection at 70-80% lower cost.

Total cost for 50 questions: ~$0.50-1.00 (vs $2.50-3.50 with mid-tier)
"""

from typing import List, Dict

# All low-tier models with different personas
# Cost: ~$0.50-1.00 for 50 questions (vs $2.50-3.50)
COST_OPTIMIZED_CONFIGS = [
    # === Llama 3.1 8B with different personas (5 agents) ===
    {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "name": "llama-careful",
        "tier": "low",
        "persona": "You are a careful mathematician who shows all work step-by-step and double-checks calculations.",
        "description": "Llama 3.1 8B - Careful persona"
    },
    {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "name": "llama-quick",
        "tier": "low",
        "persona": "You are a quick problem solver who finds shortcuts and patterns.",
        "description": "Llama 3.1 8B - Quick persona"
    },
    {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "name": "llama-creative",
        "tier": "low",
        "persona": "You think creatively and explore multiple approaches before solving.",
        "description": "Llama 3.1 8B - Creative persona"
    },
    {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "name": "llama-formal",
        "tier": "low",
        "persona": "You approach problems methodically using formal mathematical notation.",
        "description": "Llama 3.1 8B - Formal persona"
    },
    {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "name": "llama-intuitive",
        "tier": "low",
        "persona": "You rely on intuition and pattern recognition to solve problems quickly.",
        "description": "Llama 3.1 8B - Intuitive persona"
    },

    # === Mistral 7B with different personas (5 agents) ===
    {
        "model": "mistralai/mistral-7b-instruct",
        "name": "mistral-detailed",
        "tier": "low",
        "persona": "You are a detail-oriented solver who breaks down complex problems into small steps.",
        "description": "Mistral 7B - Detailed persona"
    },
    {
        "model": "mistralai/mistral-7b-instruct",
        "name": "mistral-efficient",
        "tier": "low",
        "persona": "You are an efficient solver who prioritizes speed while maintaining accuracy.",
        "description": "Mistral 7B - Efficient persona"
    },
    {
        "model": "mistralai/mistral-7b-instruct",
        "name": "mistral-visual",
        "tier": "low",
        "persona": "You are a visual thinker who uses diagrams and geometric reasoning when possible.",
        "description": "Mistral 7B - Visual persona"
    },
    {
        "model": "mistralai/mistral-7b-instruct",
        "name": "mistral-logical",
        "tier": "low",
        "persona": "You are a logical reasoner who builds proofs step-by-step.",
        "description": "Mistral 7B - Logical persona"
    },
    {
        "model": "mistralai/mistral-7b-instruct",
        "name": "mistral-practical",
        "tier": "low",
        "persona": "You are a practical solver who tests answers with concrete examples.",
        "description": "Mistral 7B - Practical persona"
    },

    # === Gemma 2 9B with different personas (5 agents) ===
    {
        "model": "google/gemma-2-9b-it",
        "name": "gemma-educator",
        "tier": "low",
        "persona": "You are an experienced educator who explains reasoning clearly for students.",
        "description": "Gemma 2 9B - Educator persona"
    },
    {
        "model": "google/gemma-2-9b-it",
        "name": "gemma-competitive",
        "tier": "low",
        "persona": "You are a competitive problem solver who seeks elegant solutions.",
        "description": "Gemma 2 9B - Competitive persona"
    },
    {
        "model": "google/gemma-2-9b-it",
        "name": "gemma-systematic",
        "tier": "low",
        "persona": "You are a systematic thinker who follows algorithmic approaches.",
        "description": "Gemma 2 9B - Systematic persona"
    },
    {
        "model": "google/gemma-2-9b-it",
        "name": "gemma-heuristic",
        "tier": "low",
        "persona": "You are a heuristic solver who uses estimation and approximation first.",
        "description": "Gemma 2 9B - Heuristic persona"
    },
    {
        "model": "google/gemma-2-9b-it",
        "name": "gemma-conservative",
        "tier": "low",
        "persona": "You are very conservative and verify every step before moving forward.",
        "description": "Gemma 2 9B - Conservative persona"
    },
]

# Verify we have exactly N=15 agents
assert len(COST_OPTIMIZED_CONFIGS) == 15, f"Expected 15 agents, got {len(COST_OPTIMIZED_CONFIGS)}"


def print_cost_comparison():
    """Print cost comparison between original and optimized configs."""
    print("\n" + "="*70)
    print("COST COMPARISON: Mid-tier vs All Low-tier")
    print("="*70)

    print("\n1. ORIGINAL CONFIGURATION (5 mid + 10 low):")
    print("-"*70)
    print("Mid-tier models (5):")
    print("  - GPT-3.5-turbo, Claude Haiku, Gemini Flash, Llama 70B, Qwen 72B")
    print("  - Cost per 50 questions: ~$2.00-3.00")
    print()
    print("Low-tier models (10):")
    print("  - Various 1B-9B models")
    print("  - Cost per 50 questions: ~$0.50")
    print()
    print("TOTAL: ~$2.50-3.50 per 50 questions")

    print("\n2. COST-OPTIMIZED CONFIGURATION (15 low-tier):")
    print("-"*70)
    print("All low-tier with diverse personas (15):")
    print("  - 5x Llama 3.1 8B (different personas)")
    print("  - 5x Mistral 7B (different personas)")
    print("  - 5x Gemma 2 9B (different personas)")
    print("  - Cost per 50 questions: ~$0.50-1.00")
    print()
    print("TOTAL: ~$0.50-1.00 per 50 questions")

    print("\n3. SAVINGS:")
    print("-"*70)
    print("Cost reduction: 70-80%")
    print("Budget for 100 questions:")
    print("  - Original: ~$5-7")
    print("  - Optimized: ~$1-2")
    print("  - Savings: ~$4-5 ✅")

    print("\n4. QUALITY IMPACT:")
    print("-"*70)
    print("Given diagnostic results (correlation ~0.07-0.14):")
    print("  - Algorithms CAN'T distinguish quality tiers anyway")
    print("  - Persona diversity creates sufficient answer variation")
    print("  - No meaningful accuracy loss expected")
    print()
    print("RECOMMENDATION: Use cost-optimized config! 🎯")
    print("="*70)


def get_cost_optimized_configs() -> List[Dict]:
    """Get cost-optimized agent configurations."""
    return COST_OPTIMIZED_CONFIGS


if __name__ == "__main__":
    print_cost_comparison()

    print("\n\nSample Agents:")
    print("-"*70)
    for i, agent in enumerate(COST_OPTIMIZED_CONFIGS[:3], 1):
        print(f"\n{i}. {agent['name']}")
        print(f"   Model: {agent['model']}")
        print(f"   Persona: {agent['persona']}")
