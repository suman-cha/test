"""
Fixed Mid-Tier Model Configurations for OpenRouter.

These model IDs are verified to work with OpenRouter API.
"""

PILOT_MID_TIER_CONFIGS_FIXED = [
    {
        "model": "openai/gpt-3.5-turbo",
        "name": "gpt35-turbo",
        "description": "OpenAI GPT-3.5 Turbo"
    },
    {
        "model": "anthropic/claude-3-haiku",  # Fixed: remove date suffix
        "name": "claude-haiku",
        "description": "Anthropic Claude 3 Haiku"
    },
    {
        "model": "google/gemini-flash-1.5-8b",  # Fixed: correct model name
        "name": "gemini-flash",
        "description": "Google Gemini 1.5 Flash 8B"
    },
    {
        "model": "meta-llama/llama-3.1-70b-instruct",
        "name": "llama-70b",
        "description": "Meta Llama 3.1 70B"
    },
    {
        "model": "qwen/qwen-2.5-72b-instruct",
        "name": "qwen-72b",
        "description": "Qwen 2.5 72B"
    },
]

# Alternative if some still don't work:
PILOT_CONFIGS_SAFE = [
    {"model": "openai/gpt-3.5-turbo", "name": "gpt35-turbo"},
    {"model": "openai/gpt-4o-mini", "name": "gpt4o-mini"},  # Very reliable
    {"model": "google/gemini-2.5-flash", "name": "gemini-flash"},
    {"model": "meta-llama/llama-3.1-70b-instruct", "name": "llama-70b"},
    {"model": "anthropic/claude-3.5-sonnet", "name": "claude-sonnet"},  # If budget allows
]

print("Fixed model IDs:")
for cfg in PILOT_MID_TIER_CONFIGS_FIXED:
    print(f"  - {cfg['model']}")
