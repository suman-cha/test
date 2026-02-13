#!/usr/bin/env python3
"""Test OpenRouter API connection"""

import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    print("❌ OPENROUTER_API_KEY not found in .env")
    exit(1)

print(f"🔑 API Key loaded: {OPENROUTER_API_KEY[:20]}...")

# Test API call
try:
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "openai/gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": "Say 'Hello!' if you can read this."}
            ],
            "max_tokens": 20
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        message = result['choices'][0]['message']['content']
        print(f"\n✅ API Connection Successful!")
        print(f"📝 Response: {message}")
        print(f"\n💰 Usage: {result.get('usage', {})}")
    else:
        print(f"❌ API Error: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"❌ Connection Error: {e}")

