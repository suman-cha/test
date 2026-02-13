# Hammer-Spammer LLM Agent Experiment
# Package exports for convenient importing

try:
    from .agent_config import AGENT_CONFIGS, get_agent_config, get_agents_by_tier
    from .agent_system import AgentSystem
    from .llm_agent import LLMAgent
except ImportError:
    # Flat file usage
    from agent_config import AGENT_CONFIGS, get_agent_config, get_agents_by_tier
    from agent_system import AgentSystem
    from llm_agent import LLMAgent
