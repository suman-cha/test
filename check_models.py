"""
Check if all configured models are available on OpenRouter.
"""

import os
import sys
import time
from dotenv import load_dotenv

# Add project root to path
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.agents.agent_config import AGENT_CONFIGS
from src.agents.llm_agent import LLMAgent


def test_model(model_id: str, name: str, api_key: str) -> bool:
    """Test if a model is available."""
    try:
        agent = LLMAgent(model_id=model_id, name=name, api_key=api_key)
        result = agent.generate_answer("What is 2+2?")
        
        if result['answer'] and 'ERROR' not in result['answer']:
            return True
        return False
    except Exception as e:
        print(f"  Error: {str(e)[:100]}")
        return False


def main():
    """Test all configured models."""
    print("="*60)
    print("Testing Model Availability on OpenRouter")
    print("="*60)
    print()

    load_dotenv()
    api_key = os.getenv('OPENROUTER_API_KEY')

    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found")
        sys.exit(1)

    available = []
    unavailable = []

    for i, config in enumerate(AGENT_CONFIGS, 1):
        model_id = config['model']
        name = config['name']
        tier = config['tier']

        print(f"{i}/15: {name} ({tier})")
        print(f"     {model_id}")

        is_available = test_model(model_id, name, api_key)

        if is_available:
            print(f"     ✓ Available")
            available.append(name)
        else:
            print(f"     ✗ Unavailable")
            unavailable.append((name, model_id))

        print()
        time.sleep(1)

    print("="*60)
    print(f"Available:   {len(available)}/15")
    print(f"Unavailable: {len(unavailable)}/15")
    print("="*60)

    if unavailable:
        print("\nUnavailable models:")
        for name, model_id in unavailable:
            print(f"  - {name}: {model_id}")
        sys.exit(1)
    else:
        print("\n✓ All models available!")
        print("\nRun experiments:")
        print("  bash experiments/phase1_quick.sh")


if __name__ == "__main__":
    main()
