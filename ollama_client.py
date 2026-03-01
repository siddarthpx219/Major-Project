import os
import requests

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate"
)

def query_llama(prompt: str, temperature: float = 0.3):
    payload = {
        "model": "llama3.1:8b",
        "prompt": prompt,
        "temperature": temperature,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()

    return response.json()["response"]