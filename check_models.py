import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# Get available models
response = requests.get(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {api_key}"}
)

if response.status_code == 200:
    models = response.json()['data']
    # Filter for popular/reliable models
    popular = [m for m in models if any(x in m['id'].lower() for x in ['gpt-3.5', 'gpt-4', 'claude', 'gemini', 'llama'])]
    print(f"Total models available: {len(models)}")
    print(f"\nRecommended models for N=15 agents:")
    for i, m in enumerate(popular[:20], 1):
        print(f"{i}. {m['id']}")
else:
    print(f"Error: {response.status_code}")
