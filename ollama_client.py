import os
import requests
import time
import json

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate"
)

OLLAMA_HEALTH_URL = os.getenv(
    "OLLAMA_HEALTH_URL",
    "http://localhost:11434"
)


def wait_for_ollama(timeout: int = 120):
    """
    Wait until Ollama server is available.
    """
    print("Waiting for Ollama server...")

    start_time = time.time()

    while True:
        try:
            response = requests.get(OLLAMA_HEALTH_URL, timeout=5)
            if response.status_code == 200:
                print("Ollama server is ready.")
                return
        except requests.RequestException:
            pass

        if time.time() - start_time > timeout:
            raise TimeoutError("Ollama server did not start within timeout.")

        time.sleep(2)


def query_llama(prompt: str, temperature: float = 0.3):

    payload = {
        "model": "llama3.1:8b",
        "prompt": prompt,
        "temperature": temperature,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=(10, 600))
    except requests.exceptions.ConnectionError as e:
        print(f"Connection failed: {e}. Ensure Ollama is running.")
        return None  # Or raise a custom exception

    print("Ollama status:", response.status_code)

    try:
        data = response.json()
    except json.JSONDecodeError:
        print(f"Invalid JSON response: {response.text}")
        return ""

    return data.get("response", "")