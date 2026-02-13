"""
Agent configuration for N=15 diverse LLM agents.

This module defines the configurations for 15 different agents using
various models to create natural hammer-spammer dynamics.

IMPORTANT: GPT-4o (openai/gpt-4o) is reserved as the Oracle model for
validation and should NOT be included as an agent to avoid bias in
correlation analysis.
"""

from typing import List, Dict

# N=15 agent configurations - COST OPTIMIZED
# Distribution:
# - 5 mid-tier models (Hammers) - 중간 성능 모델
# - 10 low-tier models (Spammers 후보) - 초저렴 모델
#
# Note: spammer는 --epsilon 또는 코드에서 인위적으로 지정

AGENT_CONFIGS = [
    # === MID-TIER MODELS (5) - Hammers (Decent Quality) ===
    {
        "model": "openai/gpt-3.5-turbo",
        "name": "gpt35-turbo",
        "tier": "mid",
        "description": "OpenAI GPT-3.5 Turbo - reliable performance"
    },
    {
        "model": "anthropic/claude-haiku-4.5",
        "name": "claude-haiku",
        "tier": "mid",
        "description": "Anthropic Claude Haiku 4.5 - affordable quality"
    },
    {
        "model": "google/gemini-2.5-flash",
        "name": "gemini-flash",
        "tier": "mid",
        "description": "Google Gemini 2.5 Flash - fast and cheap"
    },
    {
        "model": "meta-llama/llama-3.1-70b-instruct",
        "name": "llama-31-70b",
        "tier": "mid",
        "description": "Meta Llama 3.1 70B - open source"
    },
    {
        "model": "qwen/qwen-2.5-72b-instruct",
        "name": "qwen-72b",
        "tier": "mid",
        "description": "Qwen 2.5 72B - Chinese open-source"
    },

    # === LOW-TIER MODELS (10) - Spammer Candidates (테스트 통과 확인된 모델만!) ===
    {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "name": "llama-31-8b-1",
        "tier": "low",
        "description": "Meta Llama 3.1 8B #1"
    },
    {
        "model": "mistralai/mistral-7b-instruct",
        "name": "mistral-7b-1",
        "tier": "low",
        "description": "Mistral 7B #1"
    },
    {
        "model": "google/gemma-2-9b-it",
        "name": "gemma-9b-1",
        "tier": "low",
        "description": "Google Gemma 2 9B #1"
    },
    {
        "model": "deepseek/deepseek-chat",
        "name": "deepseek-1",
        "tier": "low",
        "description": "DeepSeek Chat #1"
    },
    {
        "model": "meta-llama/llama-3-8b-instruct",
        "name": "llama-3-8b-1",
        "tier": "low",
        "description": "Llama 3 8B #1"
    },
    {
        "model": "meta-llama/llama-3.2-3b-instruct",
        "name": "llama-32-3b-1",
        "tier": "low",
        "description": "Meta Llama 3.2 3B #1"
    },
    {
        "model": "meta-llama/llama-3.2-1b-instruct",
        "name": "llama-32-1b-1",
        "tier": "low",
        "description": "Meta Llama 3.2 1B #1"
    },
    {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "name": "llama-31-8b-2",
        "tier": "low",
        "description": "Meta Llama 3.1 8B #2"
    },
    {
        "model": "mistralai/mistral-7b-instruct",
        "name": "mistral-7b-2",
        "tier": "low",
        "description": "Mistral 7B #2"
    },
    {
        "model": "google/gemma-2-9b-it",
        "name": "gemma-9b-2",
        "tier": "low",
        "description": "Google Gemma 2 9B #2"
    },
]

# Verify we have exactly N=15 agents
assert len(AGENT_CONFIGS) == 15, f"Expected 15 agents, got {len(AGENT_CONFIGS)}"


def get_agent_config(name: str) -> Dict:
    """
    Get configuration for a specific agent by name.

    Args:
        name: Agent name

    Returns:
        Agent configuration dictionary
    """
    for config in AGENT_CONFIGS:
        if config['name'] == name:
            return config
    raise ValueError(f"Agent '{name}' not found in configurations")


def get_agents_by_tier(tier: str) -> List[Dict]:
    """
    Get all agents of a specific tier.

    Args:
        tier: 'high', 'mid', or 'low'

    Returns:
        List of agent configurations
    """
    return [cfg for cfg in AGENT_CONFIGS if cfg['tier'] == tier]


def get_agent_names() -> List[str]:
    """
    Get list of all agent names.

    Returns:
        List of agent names
    """
    return [cfg['name'] for cfg in AGENT_CONFIGS]


def print_agent_summary():
    """Print summary of all configured agents."""
    print(f"\n=== Agent Configuration Summary (N={len(AGENT_CONFIGS)}) ===\n")

    for tier in ['high', 'mid', 'low']:
        tier_agents = get_agents_by_tier(tier)
        tier_name = {
            'high': 'HIGH-QUALITY (Strong Hammers)',
            'mid': 'MID-TIER (Moderate Hammers)',
            'low': 'LOWER-TIER (Potential Spammers)'
        }[tier]

        print(f"\n{tier_name}: {len(tier_agents)} agents")
        print("-" * 60)
        for i, agent in enumerate(tier_agents, 1):
            print(f"{i}. {agent['name']:<20} | {agent['model']}")
            print(f"   {agent['description']}")


# Alternative configuration using personas (fallback if model diversity unavailable)
PERSONA_CONFIGS = [
    "You are a careful mathematician who shows all work step-by-step.",
    "You are a quick problem solver who finds shortcuts and patterns.",
    "You think creatively and explore multiple approaches before solving.",
    "You are very conservative and double-check every calculation.",
    "You rely on intuition and pattern recognition to solve problems.",
    "You are a detail-oriented solver who breaks down complex problems.",
    "You are an efficient solver who prioritizes speed while maintaining accuracy.",
    "You approach problems methodically using formal mathematical notation.",
    "You are a practical solver who tests answers with concrete examples.",
    "You are an experienced educator who explains reasoning clearly.",
    "You are a competitive problem solver who seeks elegant solutions.",
    "You are a systematic thinker who follows algorithmic approaches.",
    "You are a visual thinker who uses diagrams and geometric reasoning.",
    "You are a logical reasoner who builds proofs step-by-step.",
    "You are a heuristic solver who uses estimation and approximation."
]

# Verify we have exactly N=15 personas as backup
assert len(PERSONA_CONFIGS) == 15, f"Expected 15 personas, got {len(PERSONA_CONFIGS)}"


if __name__ == "__main__":
    # Test agent configurations
    print_agent_summary()

    print("\n\n=== Testing Agent Retrieval ===")
    test_name = "gpt4-turbo"
    config = get_agent_config(test_name)
    print(f"\nRetrieved config for '{test_name}':")
    print(f"  Model: {config['model']}")
    print(f"  Tier: {config['tier']}")
    print(f"  Description: {config['description']}")

    print("\n\n=== Agent Names ===")
    names = get_agent_names()
    print(f"All agent names: {', '.join(names)}")
