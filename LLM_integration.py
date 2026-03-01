import numpy as np
import pandas as pd
import json
import re

from config import NUM_MARKET_REGIMES, VIEWS_CONFIDENCE
from regime import get_regime_labels
from ollama_client import wait_for_ollama, query_llama


# ---------------------------------------------------
# Internal Ollama Call (No HuggingFace, No Pipeline)
# ---------------------------------------------------

def _call_llama_model(prompt: str) -> str:
    """
    Calls local Ollama (LLaMA 3.1) and returns raw text.
    Falls back to simulation if Ollama fails.
    """
    try:
        wait_for_ollama()
        response_text = query_llama(prompt)
        return response_text
    except Exception as e:
        print(f"Ollama call failed: {e}")
        print("Falling back to simulated response.")
        return _simulate_llama_response_fallback(prompt)


# ---------------------------------------------------
# Robust JSON Extraction
# ---------------------------------------------------

def _extract_json(text: str):
    """
    Extract JSON from LLM response.
    Handles raw JSON or markdown fenced output.
    """
    try:
        text = text.strip()

        # Remove markdown fences if present
        if text.startswith("```"):
            text = re.sub(r"```json|```", "", text).strip()

        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not json_match:
            return None

        return json.loads(json_match.group())

    except Exception as e:
        print(f"JSON extraction error: {e}")
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
    views_confidence: float = VIEWS_CONFIDENCE
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

    raw_response = _call_llama_model(prompt)

    response_data = _extract_json(raw_response)

    if response_data is None:
        response_data = json.loads(
            _simulate_llama_response_fallback(prompt)
        )

    P_matrix = np.array(response_data["P_matrix"])
    Q_vector = np.array(response_data["Q_vector"])
    Omega_matrix = np.array(response_data["Omega_matrix"])

    print(f"Generated LLM views for regime: {current_regime_label}")

    return P_matrix, Q_vector, Omega_matrix