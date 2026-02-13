"""
Agent configuration for N=10 diverse LLM agents.

This module defines the configurations for 10 different agents using
various models to create natural hammer-spammer dynamics.

Distribution:
- 2 high-quality models (Strong Hammers)
- 3 mid-tier models (Moderate Hammers)
- 5 low-tier models (Potential Spammers)
"""

from typing import List, Dict

# N=10 agent configurations
# Distribution:
# - 2 high-quality models (Strong Hammers) - 최고 성능 모델
# - 3 mid-tier models (Moderate Hammers) - 중간 성능 모델
# - 5 low-tier models (Potential Spammers) - 저성능 모델

AGENT_CONFIGS = [
    # === HIGH-QUALITY MODELS (2) - Strong Hammers ===
    {
        "model": "openai/gpt-4-turbo",
        "name": "gpt4-turbo",
        "tier": "high",
        "description": "OpenAI GPT-4 Turbo - highest quality"
    },
    {
        "model": "anthropic/claude-3.5-sonnet",
        "name": "claude-sonnet",
        "tier": "high",
        "description": "Anthropic Claude 3.5 Sonnet - top tier reasoning"
    },

    # === MID-TIER MODELS (3) - Moderate Hammers ===
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
        "description": "Google Gemini 2.5 Flash - fast and balanced"
    },

    # === LOW-TIER MODELS (5) - Potential Spammers ===
    {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "name": "llama-31-8b",
        "tier": "low",
        "description": "Meta Llama 3.1 8B"
    },
    {
        "model": "mistralai/mistral-7b-instruct",
        "name": "mistral-7b",
        "tier": "low",
        "description": "Mistral 7B"
    },
    {
        "model": "google/gemma-2-9b-it",
        "name": "gemma-9b",
        "tier": "low",
        "description": "Google Gemma 2 9B"
    },
    {
        "model": "meta-llama/llama-3.2-3b-instruct",
        "name": "llama-32-3b",
        "tier": "low",
        "description": "Meta Llama 3.2 3B"
    },
    {
        "model": "meta-llama/llama-3.2-1b-instruct",
        "name": "llama-32-1b",
        "tier": "low",
        "description": "Meta Llama 3.2 1B"
    },
]

# Verify we have exactly N=10 agents
assert len(AGENT_CONFIGS) == 10, f"Expected 10 agents, got {len(AGENT_CONFIGS)}"


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
]

# Verify we have exactly N=10 personas as backup
assert len(PERSONA_CONFIGS) == 10, f"Expected 10 personas, got {len(PERSONA_CONFIGS)}"


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
