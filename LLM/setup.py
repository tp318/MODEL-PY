import requests
import os

# Set your Aikipedia API key
AIKIPEDIA_API_KEY = "00403043-2c22-42b2-a461-d44e843d0f7a"

# Define the Claude 3 Haiku model and endpoint
API_URL = "https://backend.aikipedia.workers.dev/chat"

def ask_claude(prompt, system_prompt="", history=[]):
    payload = {
        "user_id": AIKIPEDIA_API_KEY,  # API key in user_id field
        "message": prompt,
        "model": "anthropic/claude-3-haiku:beta",
        "history": history,
        "systemPrompt": system_prompt
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(API_URL, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get("response", "No response field in output")
    else:
        return f"❌ Error {response.status_code}: {response.text}"