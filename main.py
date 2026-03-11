import os
import numpy as np
import pandas as pd

from config import (
    STOCK_TICKERS,
    START_DATE,
    END_DATE,
    DATA_FREQUENCY,
    PERIODS_PER_YEAR,
    RISK_FREE_RATE_ANNUAL,
    IMPLIED_RISK_AVERSION,
    TAU,
    NUM_MARKET_REGIMES,
    HMM_TRAINING_PERIOD_YEARS,
    PLOT_SAVE_PATH
)

from data_acquisition import get_stock_data
from data_processing import clean_data, calculate_log_returns
from regime import fit_hmm_model, get_current_regime, get_regime_labels
from LLM_integration import generate_llama_views_and_confidence
from blacklitterman import get_portfolio_recommendations
from plotting import (
    plot_stock_prices,
    plot_log_returns,
    plot_covariance_heatmap,
    plot_ticker_confidence,
    plot_capital_allocation_map,
    plot_portfolio_cumulative_value,
    calculate_and_display_risk_metrics
)


def main():
    print("Starting Adaptive Portfolio Optimizer...\n")

    os.makedirs(PLOT_SAVE_PATH, exist_ok=True)

    # --------------------------------------------------
    # 1. Data Acquisition
    # --------------------------------------------------
    print("--- 1. Acquiring Stock Data ---")
    stock_prices = get_stock_data(
        STOCK_TICKERS,
        START_DATE,
        END_DATE,
        DATA_FREQUENCY
    )

    if stock_prices.empty:
        print("No stock data. Exiting.")
        return

    # --------------------------------------------------
    # 2. Data Processing
    # --------------------------------------------------
    print("\n--- 2. Processing Data ---")

    cleaned_prices = clean_data(stock_prices)
    if cleaned_prices.empty:
        print("Cleaned data empty. Exiting.")
        return

    log_returns = calculate_log_returns(cleaned_prices)
    if log_returns.empty:
        print("No log returns. Exiting.")
        return

    # --------------------------------------------------
    # 3. HMM Regime Detection
    # --------------------------------------------------
    print("\n--- 3. Detecting Market Regime ---")

    last_date = log_returns.index[-1]
    hmm_train_start = last_date - pd.DateOffset(years=HMM_TRAINING_PERIOD_YEARS)
    hmm_training_returns = log_returns[log_returns.index >= hmm_train_start]

    if hmm_training_returns.shape[0] < NUM_MARKET_REGIMES * 2:
        print("Insufficient rolling window data. Using full dataset.")
        hmm_training_returns = log_returns

    hmm_model, scaler, regime_means = fit_hmm_model(
        hmm_training_returns,
        n_components=NUM_MARKET_REGIMES
    )

    current_regime_idx  = get_current_regime(hmm_model, scaler, log_returns)
    regime_labels, _ = get_regime_labels(regime_means, NUM_MARKET_REGIMES)

    print(f"Detected Regime: {regime_labels[current_regime_idx]}")

    # --------------------------------------------------
    # 4. Annualization (Critical for BL Consistency)
    # --------------------------------------------------
    print("\n--- 4. Annualizing Data ---")

    annualized_cov_matrix = log_returns.cov() * PERIODS_PER_YEAR
    num_assets = log_returns.shape[1]

    market_cap_weights = np.ones(num_assets) / num_assets
    print(f"Assumed Equal Market Weights: {market_cap_weights.round(3)}")

    # --------------------------------------------------
    # 5. LLM Views
    # --------------------------------------------------
    print("\n--- 5. Generating LLM Views ---")

    P_llm, Q_llm, Omega_llm = generate_llama_views_and_confidence(
        current_regime_idx=current_regime_idx,
        market_covariance=annualized_cov_matrix,
        log_returns=log_returns,
        hmm_model=hmm_model,
        scaler=scaler,
        tickers=log_returns.columns.tolist()
    )

    # Confidence normalization for plotting only 
    omega_diag = np.diag(Omega_llm)
    omega_diag_safe = np.where(omega_diag == 0, 1e-10, omega_diag)
    inv_conf = 1 / omega_diag_safe

    if inv_conf.max() == inv_conf.min():
        llm_confidence_scores = np.ones_like(inv_conf)
    else:
        llm_confidence_scores = (
            (inv_conf - inv_conf.min()) /
            (inv_conf.max() - inv_conf.min())
        )

    # --------------------------------------------------
    # 6. Black–Litterman
    # --------------------------------------------------
    print("\n--- 6. Running Black–Litterman ---")

    pi_excess, posterior_excess, optimal_weights = \
        get_portfolio_recommendations(
            market_cap_weights=market_cap_weights,
            cov_matrix=annualized_cov_matrix.values,
            implied_risk_aversion=IMPLIED_RISK_AVERSION,
            P_llm=P_llm,
            Q_llm=Q_llm,
            Omega_llm=Omega_llm,
            tau=TAU
        )

    print("\nPosterior Expected Excess Returns (Annualized):")
    print(posterior_excess.round(4))

    print("\nOptimal Weights:")
    for asset, weight in zip(log_returns.columns, optimal_weights):
        print(f"{asset}: {weight:.4f}")

    print(f"Weight Sum: {np.sum(optimal_weights):.4f}")

    # --------------------------------------------------
    # 7. Plotting & Risk Metrics
    # --------------------------------------------------
    print("\n--- 7. Generating Plots & Risk Metrics ---")

    plot_stock_prices(cleaned_prices)
    plot_log_returns(log_returns)
    plot_covariance_heatmap(
        pd.DataFrame(
            annualized_cov_matrix,
            index=log_returns.columns,
            columns=log_returns.columns
        )
    )
    plot_ticker_confidence(log_returns.columns.tolist(), llm_confidence_scores)
    plot_capital_allocation_map(log_returns.columns.tolist(), optimal_weights)
    plot_portfolio_cumulative_value(log_returns, optimal_weights)
    calculate_and_display_risk_metrics(log_returns, optimal_weights)

    print("\nAdaptive Portfolio Optimizer completed successfully.")
    print(f"Plots saved to: {os.path.abspath(PLOT_SAVE_PATH)}")


if __name__ == "__main__":
    main()