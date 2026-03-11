import numpy as np
import pandas as pd
import json
import re

from config import NUM_MARKET_REGIMES, TAU
from regime import get_regime_labels
from ollama_client import wait_for_ollama, query_llama


# ---------------------------------------------------
# Internal Ollama Call (No HuggingFace, No Pipeline)
# ---------------------------------------------------

def _call_llama_model(prompt: str) -> str:
    """
    Calls local Ollama and returns raw text.
    Does NOT fallback to simulation.
    """
    wait_for_ollama()

    response_text = query_llama(prompt)

    return response_text


# ---------------------------------------------------
# Robust JSON Extraction
# ---------------------------------------------------

def _extract_json(text: str):
    """
    Extract JSON safely from LLM output.
    Handles raw JSON or markdown fenced JSON.
    """
    try:
        text = text.strip()

        # Remove markdown fences if present
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()

        # First attempt: direct parse
        return json.loads(text)

    except Exception:
        pass

    try:
        # Fallback: slice from first { to last }
        start = text.find("{")
        end = text.rfind("}") + 1

        if start == -1 or end == -1:
            return None

        return json.loads(text[start:end])

    except Exception as e:
        print("JSON extraction error:", e)
        return None


# ---------------------------------------------------
# Simulated Fallback (Safe for College Demo)
# ---------------------------------------------------

def _simulate_llama_response_fallback(prompt: str) -> str:
    """
    Simulated fallback if Ollama fails.
    """
    print("Using simulated LLM fallback.")

    num_assets_match = re.search(r"Total number of assets: (\d+)", prompt)
    num_assets = int(num_assets_match.group(1)) if num_assets_match else 4

    P_matrix = np.identity(num_assets).tolist()

    Q_vector = (np.random.randn(num_assets) * 0.02).tolist()
    omega_diag = (np.abs(np.random.randn(num_assets)) * 0.001 + 0.0005).tolist()
    Omega_matrix = np.diag(omega_diag).tolist()

    return json.dumps({
        "P_matrix": P_matrix,
        "Q_vector": Q_vector,
        "Omega_matrix": Omega_matrix
    })


# ---------------------------------------------------
# Main Interface Function
# ---------------------------------------------------

def generate_llama_views_and_confidence(
    current_regime_idx: int,
    market_covariance: np.ndarray,
    log_returns: pd.DataFrame,
    hmm_model,
    scaler,
    tickers: list[str],
    TAU: float = TAU
):

    num_assets = len(tickers)

    regime_labels, sorted_means = get_regime_labels(
        hmm_model.means_,
        NUM_MARKET_REGIMES
    )

    current_regime_label = regime_labels.get(
        current_regime_idx,
        f"Regime {current_regime_idx}"
    )

    recent_returns = log_returns.tail(5)

    recent_performance_str = "\n".join([
        f"{ticker}: {recent_returns[ticker].iloc[-1] * 100:.2f}%"
        for ticker in tickers
    ])

    prompt = f"""
You are a quantitative financial analyst.

Current Market Regime: {current_regime_label}

Recent Asset Performance:
{recent_performance_str}

Assets: {', '.join(tickers)}
Total number of assets: {num_assets}

Return ONLY a JSON object with:
- P_matrix (identity matrix {num_assets}x{num_assets})
- Q_vector (annual expected excess returns)
- Omega_matrix (diagonal uncertainty matrix)

No explanations.
"""

    try:
        raw_response = _call_llama_model(prompt)

        response_data = _extract_json(raw_response)

        print("JSON parsed successfully:", response_data is not None)

    except Exception as e:
        print(f"LLM request failed: {e}")
        response_data = None


    if response_data is None:
        print("Using simulated LLM fallback.")
        response_data = json.loads(
            _simulate_llama_response_fallback(prompt)
        )


    P_matrix = np.array(response_data["P_matrix"])
    Q_vector = np.array(response_data["Q_vector"])
    Omega_matrix = np.array(response_data["Omega_matrix"])

    print(f"Generated LLM views for regime: {current_regime_label}")

    return P_matrix, Q_vector, Omega_matrix