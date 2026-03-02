import os
import requests
import time

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate"
)

OLLAMA_HEALTH_URL = os.getenv(
    "OLLAMA_HEALTH_URL",
    "http://localhost:11434"
)


def wait_for_ollama(timeout: int = 60):
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
    """
    Send prompt to local Ollama model.
    """

    payload = {
        "model": "llama3.1:8b",
        "prompt": prompt,
        "temperature": temperature,
        "stream": False
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    return response.json()["response"]