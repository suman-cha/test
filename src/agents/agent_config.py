"""
Agent configuration for N=15 diverse LLM agents.

This module defines the configurations for 15 different agents using
various models to create natural hammer-spammer dynamics.

IMPORTANT: GPT-4o (openai/gpt-4o) is reserved as the Oracle model for
validation and should NOT be included as an agent to avoid bias in
correlation analysis.

에이전트 구성 직관:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SVD가 작동하려면 E[R̃]에 의미 있는 low-rank 구조가 필요함.
이를 위해 적어도 일부 에이전트의 row가 실제 품질 순서를 반영하는
일관된 패턴을 가져야 함.

- Strong 모델 2~3개가 있으면, 그 row들이 서로 비슷한 패턴을 공유하면서
  SVD의 top singular vector를 지배함. 이게 "신호"임.
- 약한 모델의 row는 노이즈로 처리됨.
- 전부 약한 모델이면, 어떤 row도 일관된 패턴이 없어서 SVD가 잡을 신호가 없음.

구성: 3 high-tier + 5 mid-tier + 7 low-tier = 15
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from typing import List, Dict

# N=15 agent configurations
# Distribution:
# - 3 high-tier models (Strong Hammers) — 핵심 신호원. 답변 정확도 + 비교 판단 모두 신뢰 가능
# - 5 mid-tier models (Moderate Hammers) — noisy하지만 어느 정도 일관된 패턴 제공
# - 7 low-tier models (Spammer 후보) — 초저렴 모델, 비교 판단이 거의 랜덤에 가까움
#
# Note: spammer는 --epsilon 또는 코드에서 인위적으로 지정

AGENT_CONFIGS = [
    # === HIGH-TIER MODELS (3) - Strong Hammers (Core Signal) ===
    # 이 모델들이 SVD의 top singular vector를 지배하는 "신호"를 만듦.
    # 답변도 정확하고, 비교 판단도 신뢰할 수 있음.
    # API 비용이 들지만 핵심 신호원이므로 반드시 필요.
    {
        "model": "anthropic/claude-sonnet-4",
        "name": "claude-sonnet",
        "tier": "high",
        "description": "Anthropic Claude Sonnet 4 - strong reasoning, reliable comparisons"
    },
    {
        "model": "google/gemini-2.5-pro",
        "name": "gemini-pro",
        "tier": "high",
        "description": "Google Gemini 2.5 Pro - strong math, accurate judgments"
    },
    {
        "model": "openai/gpt-4-turbo",
        "name": "gpt4-turbo",
        "tier": "high",
        "description": "OpenAI GPT-4 Turbo - near GPT-4o quality at lower cost"
    },

    # === MID-TIER MODELS (5) - Moderate Hammers (Noisy Signal) ===
    # 비용 대비 효율이 좋음. 비교 판단이 noisy하지만 완전 랜덤은 아님.
    # β가 낮은 hammer 역할 — SVD에서 중간 크기의 u₁ 가중치를 받음.
    {
        "model": "openai/gpt-4o-mini",
        "name": "gpt4o-mini",
        "tier": "mid",
        "description": "OpenAI GPT-4o Mini - cost-efficient, decent quality"
    },
    {
        "model": "anthropic/claude-haiku-4.5",
        "name": "claude-haiku",
        "tier": "mid",
        "description": "Anthropic Claude Haiku 4.5 - fast and affordable"
    },
    {
        "model": "google/gemini-2.5-flash",
        "name": "gemini-flash",
        "tier": "mid",
        "description": "Google Gemini 2.5 Flash - fast, cheap, good for math"
    },
    {
        "model": "openai/gpt-3.5-turbo",
        "name": "gpt35-turbo",
        "tier": "mid",
        "description": "OpenAI GPT-3.5 Turbo - baseline mid-tier performance"
    },
    {
        "model": "qwen/qwen-2.5-72b-instruct",
        "name": "qwen-72b",
        "tier": "mid",
        "description": "Qwen 2.5 72B - strong open-source alternative"
    },

    # === LOW-TIER MODELS (7) - Spammer Candidates (Noise) ===
    # 비교 판단이 거의 랜덤에 가까움 → SVD에서 u₁ 가중치 ≈ 0.
    # Majority voting에서는 이것들이 strong 모델의 신호를 희석시키지만,
    # SVD 기반에서는 자동으로 무시됨 — 이게 알고리즘의 핵심 장점.
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
        "name": "gemma-9b",
        "tier": "low",
        "description": "Google Gemma 2 9B"
    },
    {
        "model": "deepseek/deepseek-chat",
        "name": "deepseek",
        "tier": "low",
        "description": "DeepSeek Chat"
    },
    {
        "model": "meta-llama/llama-3.2-3b-instruct",
        "name": "llama-32-3b",
        "tier": "low",
        "description": "Meta Llama 3.2 3B - small model"
    },
    {
        "model": "meta-llama/llama-3.2-1b-instruct",
        "name": "llama-32-1b",
        "tier": "low",
        "description": "Meta Llama 3.2 1B - smallest model"
    },
    {
        "model": "meta-llama/llama-3-8b-instruct",
        "name": "llama-3-8b",
        "tier": "low",
        "description": "Llama 3 8B - older generation"
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
            'high': 'HIGH-TIER (Strong Hammers) — Core Signal',
            'mid': 'MID-TIER (Moderate Hammers) — Noisy Signal',
            'low': 'LOW-TIER (Spammer Candidates) — Noise'
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
    print_agent_summary()

    print("\n\n=== Agent Names ===")
    names = get_agent_names()
    print(f"All agent names: {', '.join(names)}")

    print("\n\n=== Tier Distribution ===")
    for tier in ['high', 'mid', 'low']:
        agents = get_agents_by_tier(tier)
        print(f"  {tier}: {len(agents)} agents")
