"""
Agent configuration for N=10 diverse LLM agents.

This module defines the configurations for 10 different agents using
various models and personas to create natural hammer-spammer dynamics.

IMPORTANT: GPT-4o (openai/gpt-4o) is reserved as the Oracle model for
validation and should NOT be included as an agent to avoid bias in
correlation analysis.

에이전트 구성 직관:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SVD가 작동하려면 E[R̃]에 의미 있는 low-rank 구조가 필요함.
이를 위해 적어도 일부 에이전트의 row가 실제 품질 순서를 반영하는
일관된 패턴을 가져야 함.

- Strong 모델 2개가 있으면, 그 row들이 서로 비슷한 패턴을 공유하면서
  SVD의 top singular vector를 지배함. 이게 "신호"임.
- 약한 모델의 row는 노이즈로 처리됨.

구성: 2 high-tier + 3 mid-tier + 5 low-tier = 10
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from typing import List, Dict

# N=10 agent configurations
# Distribution:
# - 2 high-tier models (Strong Hammers) — 핵심 신호원
# - 3 mid-tier models (Moderate Hammers) — noisy하지만 일관된 패턴
# - 5 low-tier models (Spammer 후보) — 비교 판단이 거의 랜덤
#
# 방법 1 (서로 다른 모델) + 방법 2 (같은 모델 + 다른 페르소나)를 혼합.
# persona 필드로 에이전트 행동을 차별화.

AGENT_CONFIGS = [
    # === HIGH-TIER MODELS (2) - Strong Hammers (Core Signal) ===
    # SVD의 top singular vector를 지배하는 핵심 신호원.
    # persona 없음 — 모델의 자연스러운 능력이 최고 품질의 신호.
    {
        "model": "anthropic/claude-sonnet-4",
        "name": "claude-sonnet",
        "tier": "high",
        "persona": "",
        "description": "Anthropic Claude Sonnet 4 - strong reasoning, reliable comparisons"
    },
    {
        "model": "google/gemini-2.5-pro",
        "name": "gemini-pro",
        "tier": "high",
        "persona": "",
        "description": "Google Gemini 2.5 Pro - strong math, accurate judgments"
    },

    # === MID-TIER MODELS (3) - Moderate Hammers (Noisy Signal) ===
    # β가 낮은 hammer 역할 — SVD에서 중간 크기의 u₁ 가중치.
    # 페르소나로 비교 판단 스타일에 분산 부여.
    {
        "model": "openai/gpt-4o-mini",
        "name": "gpt4o-mini",
        "tier": "mid",
        "persona": "You are a careful mathematician who shows all work step-by-step.",
        "description": "OpenAI GPT-4o Mini - careful step-by-step solver"
    },
    {
        "model": "anthropic/claude-haiku-4.5",
        "name": "claude-haiku",
        "tier": "mid",
        "persona": "You are very conservative and double-check every calculation.",
        "description": "Anthropic Claude Haiku 4.5 - conservative double-checker"
    },
    {
        "model": "google/gemini-2.5-flash",
        "name": "gemini-flash",
        "tier": "mid",
        "persona": "",
        "description": "Google Gemini 2.5 Flash - fast and cheap"
    },

    # === LOW-TIER MODELS (5) - Spammer Candidates (Noise) ===
    # SVD에서 u₁ 가중치 ≈ 0 → 자동 무시.
    # 페르소나로 다양성 부여: "성급한 에이전트", "학생 에이전트" 등.
    {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "name": "llama-31-8b",
        "tier": "low",
        "persona": "Answer quickly and concisely. Do not overthink.",
        "description": "Meta Llama 3.1 8B - quick/hasty solver"
    },
    {
        "model": "mistralai/mistral-7b-instruct",
        "name": "mistral-7b",
        "tier": "low",
        "persona": "You are a student learning math. Try your best.",
        "description": "Mistral 7B - student persona"
    },
    {
        "model": "google/gemma-2-9b-it",
        "name": "gemma-9b",
        "tier": "low",
        "persona": "Solve this creatively, using unconventional approaches.",
        "description": "Google Gemma 2 9B - creative/unconventional"
    },
    {
        "model": "meta-llama/llama-3.2-3b-instruct",
        "name": "llama-32-3b",
        "tier": "low",
        "persona": "You rely on intuition and pattern recognition to solve problems.",
        "description": "Meta Llama 3.2 3B - intuition-based"
    },
    {
        "model": "meta-llama/llama-3-8b-instruct",
        "name": "llama-3-8b",
        "tier": "low",
        "persona": "You are a quick problem solver who finds shortcuts and patterns.",
        "description": "Llama 3 8B - shortcut seeker"
    },
]

N_AGENTS = len(AGENT_CONFIGS)
assert N_AGENTS == 10, f"Expected 10 agents, got {N_AGENTS}"


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
    print(f"\n=== Agent Configuration Summary (N={N_AGENTS}) ===\n")

    for tier in ['high', 'mid', 'low']:
        tier_agents = get_agents_by_tier(tier)
        tier_name = {
            'high': 'HIGH-TIER (Strong Hammers) — Core Signal',
            'mid': 'MID-TIER (Moderate Hammers) — Noisy Signal',
            'low': 'LOW-TIER (Spammer Candidates) — Noise'
        }[tier]

        print(f"\n{tier_name}: {len(tier_agents)} agents")
        print("-" * 60)
        for i, agent in enumerate(tier_agents, 1):
            persona_tag = f' [{agent["persona"][:40]}...]' if agent.get("persona") else ''
            print(f"{i}. {agent['name']:<20} | {agent['model']}{persona_tag}")
            print(f"   {agent['description']}")


# Alternative configuration using personas (fallback if model diversity unavailable)
# 같은 모델에 시스템 프롬프트를 다르게 줘서 다양성을 만드는 방법 2.
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
    "You are a student learning math. Try your best.",
]

assert len(PERSONA_CONFIGS) == N_AGENTS, f"Expected {N_AGENTS} personas, got {len(PERSONA_CONFIGS)}"


if __name__ == "__main__":
    print_agent_summary()

    print("\n\n=== Agent Names ===")
    names = get_agent_names()
    print(f"All agent names: {', '.join(names)}")

    print("\n\n=== Tier Distribution ===")
    for tier in ['high', 'mid', 'low']:
        agents = get_agents_by_tier(tier)
        print(f"  {tier}: {len(agents)} agents")
