import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import (
    PLOT_SAVE_PATH,
    RISK_FREE_RATE_ANNUAL,
    PERIODS_PER_YEAR
)


def _ensure_path(path):
    os.makedirs(path, exist_ok=True)


# --------------------------------------------------
# Price Plot
# --------------------------------------------------

def plot_stock_prices(prices: pd.DataFrame, save_path: str = PLOT_SAVE_PATH):
    _ensure_path(save_path)

    plt.figure(figsize=(12, 6))
    for col in prices.columns:
        plt.plot(prices.index, prices[col], label=col)

    plt.title("Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc="best", fontsize="small")
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "stock_prices.png"))
    plt.close()


# --------------------------------------------------
# Log Returns Plot
# --------------------------------------------------

def plot_log_returns(log_returns: pd.DataFrame, save_path: str = PLOT_SAVE_PATH):
    _ensure_path(save_path)

    plt.figure(figsize=(12, 6))
    for col in log_returns.columns:
        plt.plot(log_returns.index, log_returns[col], label=col)

    plt.title("Log Returns")
    plt.xlabel("Date")
    plt.ylabel("Log Return")
    plt.legend(loc="best", fontsize="small")
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "log_returns.png"))
    plt.close()


# --------------------------------------------------
# Covariance Heatmap
# --------------------------------------------------

def plot_covariance_heatmap(cov_matrix: pd.DataFrame, save_path: str = PLOT_SAVE_PATH):
    _ensure_path(save_path)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cov_matrix, cmap="Reds", square=True)
    plt.title("Annualized Covariance Heatmap")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "covariance_heatmap.png"))
    plt.close()


# --------------------------------------------------
# LLM Confidence Plot
# --------------------------------------------------

def plot_ticker_confidence(tickers, confidences, save_path: str = PLOT_SAVE_PATH):
    _ensure_path(save_path)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=confidences, y=tickers, palette="viridis")

    plt.xlabel("Confidence (0–1)")
    plt.title("LLM View Confidence")
    plt.xlim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "ticker_confidence.png"))
    plt.close()


# --------------------------------------------------
# Capital Allocation Plot
# --------------------------------------------------

def plot_capital_allocation_map(tickers, weights, save_path: str = PLOT_SAVE_PATH):
    _ensure_path(save_path)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=weights, y=tickers, palette="coolwarm")

    plt.xlabel("Weight")
    plt.title("Capital Allocation")

    for i, w in enumerate(weights):
        plt.text(w + 0.002, i, f"{w:.2%}", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "capital_allocation.png"))
    plt.close()


# --------------------------------------------------
# Portfolio Cumulative Value
# --------------------------------------------------

def plot_portfolio_cumulative_value(
    log_returns: pd.DataFrame,
    weights,
    start_value: float = 1.0,
    save_path: str = PLOT_SAVE_PATH
):
    _ensure_path(save_path)

    weights = np.array(weights)

    portfolio_log_returns = log_returns.dot(weights)
    cumulative = start_value * np.exp(np.cumsum(portfolio_log_returns.fillna(0)))

    plt.figure(figsize=(12, 6))
    plt.plot(cumulative.index, cumulative, label="Portfolio")

    plt.title("Portfolio Cumulative Value")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "portfolio_cumulative_value.png"))
    plt.close()


# --------------------------------------------------
# Risk Metrics
# --------------------------------------------------

def calculate_and_display_risk_metrics(
    log_returns: pd.DataFrame,
    weights,
    risk_free_rate: float = RISK_FREE_RATE_ANNUAL
):
    """
    Computes annualized return, volatility and Sharpe ratio
    using configured PERIODS_PER_YEAR.
    """

    weights = np.array(weights)

    port_log_r = log_returns.dot(weights).dropna()

    mean_log = port_log_r.mean()
    std_log = port_log_r.std()

    annual_log_return = mean_log * PERIODS_PER_YEAR
    annual_return = np.exp(annual_log_return) - 1
    annual_volatility = std_log * np.sqrt(PERIODS_PER_YEAR)

    eps = 1e-12
    sharpe = (
        (annual_return - risk_free_rate) /
        (annual_volatility + eps)
    )

    print("\nPortfolio Risk Metrics:")
    print(f"  Annualized Return: {annual_return:.2%}")
    print(f"  Annualized Volatility: {annual_volatility:.2%}")
    print(f"  Sharpe Ratio: {sharpe:.3f}")

    return {
        "annualized_return": annual_return,
        "annualized_volatility": annual_volatility,
        "sharpe": sharpe
    }