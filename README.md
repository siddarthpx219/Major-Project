# Adaptive Portfolio Optimizer with AI Driven Insights

This project implements an adaptive portfolio optimizer that leverages quantitative techniques and AI-driven insights to construct and evaluate optimal investment portfolios. The system automates the process of data ingestion, market regime detection, portfolio optimization, and performance evaluation through backtesting and risk analysis.

## High-Level Summary

This project is a quantitative finance tool designed to construct an optimal investment portfolio. It follows a structured, multi-step process:

1.  **Data Gathering**: It starts by fetching over a decade of historical stock price data for a list of 20 prominent Indian companies from Yahoo Finance.
2.  **Market Analysis**: It then analyzes this data to identify distinct "market regimes" (e.g., high-volatility, trending, etc.) using a machine learning model (a Hidden Markov Model).
3.  **Core Optimization**: Using the principles of Modern Portfolio Theory, it calculates the single best "buy-and-hold" portfolio allocation that maximizes expected return for a given level of risk.
4.  **Evaluation**: Finally, it simulates how this optimal portfolio would have performed historically (backtesting) and analyzes its risk profile, particularly the potential for extreme losses (tail risk).

The "AI Driven Insights" part of the title primarily refers to the regime detection step, which uses an unsupervised learning model to classify the market's behavior.

## Detailed Step-by-Step Breakdown

### 1. Configuration (`config.py`)

This file acts as the central control panel for the project, defining key parameters and settings:

*   **`TICKER`**: Defines the investment universe as a list of 20 stocks on the National Stock Exchange of India (e.g., 'RELIANCE.NS', 'HDFCBANK.NS').
*   **`START_DATE`** & **`END_DATE`**: Sets the time frame for the historical data analysis, currently from **January 1, 2013, to January 1, 2024**.
*   **`MAX_WEIGHT`**: Imposes a crucial diversification constraint, ensuring no single stock can make up more than **15%** of the portfolio.
*   **`FREQUENCY`**: Specifies the data frequency (set to "M" for monthly, though `yfinance` typically returns daily and the code uses daily returns implicitly).
*   **`RISK_FREE_RATE`**: The assumed risk-free rate for calculations (currently `0.04`).
*   **`NUM_REGIMES`**: The number of market regimes the HMM model attempts to detect (set to `3`).
*   **`ROLLING_WINDOW`**: The size of the rolling window used for certain calculations, like factor betas (set to `36`).
*   **`TRANSACTION_COST`**: A placeholder for transaction costs (currently `0.001`), though not explicitly used in the current backtesting logic.
*   **`LONG_ONLY`**: A boolean flag indicating whether only long positions are allowed (set to `True`).

### 2. Data Handling (`data_ingestion.py`)

This module is responsible for sourcing and preparing the financial market data:

*   **`load_price_data`**: Utilizes the `yfinance` library to download historical adjusted closing prices for all specified tickers within the defined date range.
*   **`compute_returns`**: Transforms the raw price data into daily logarithmic returns, a common practice in quantitative finance due to their additive properties and ease of interpretation for percentage changes.
*   **`clean_returns`**: Performs data cleaning by handling infinite values (which can arise from zero prices) and filling any missing values using forward-fill, followed by dropping any remaining `NaN` entries.

### 3. Feature Engineering & Regime Detection (`feature_engineering.py`, `regime_detection.py`)

This section extracts meaningful patterns from the market data and classifies market states.

*   **`feature_engineering.py`**:
    *   **`compute_factor_betas`**: (Note: This function is present but not currently utilized in `main.py`). It calculates rolling betas of assets against factor returns, indicating an asset's sensitivity to market-wide factors.
    *   **`build_ml_features`**: Creates two primary features representing the overall market environment:
        *   **`volatility`**: A 12-period rolling standard deviation of returns, averaged across all assets, indicating market risk.
        *   **`momentum`**: A 12-period rolling mean of returns, averaged across all assets, indicating market trend.
*   **`regime_detection.py`**:
    *   **`detect_regimes`**: This is a key "AI" component. It employs a **Gaussian Hidden Markov Model (HMM)** to identify `NUM_REGIMES` (e.g., 3) distinct, unobservable market states based on the `volatility` and `momentum` features. The HMM learns the characteristics of each regime and assigns a probability of the market being in each regime at any given time.

### 4. Modeling Inputs (`return_forecasting.py`, `covariance_model.py`)

This part of the project focuses on estimating the essential inputs for portfolio optimization: expected returns and risks.

*   **`return_forecasting.py`**:
    *   **`train_return_model`**: Trains an **XGBoost Regressor** model to forecast future asset returns using the engineered features. The function includes data cleaning to handle non-finite target values.
    *   **`predict_returns`**: Uses the trained XGBoost model to generate return predictions.
    *   **Note**: While an advanced ML model is trained for return forecasting, the `main.py` script currently utilizes the simpler historical mean returns (`returns.mean()`) as the expected returns for the optimization step. This could be a design choice prioritizing robustness, or an area for future enhancement where ML predictions could be integrated.
*   **`covariance_model.py`**:
    *   **`estimate_covariance`**: Calculates the covariance matrix of asset returns, which is crucial for assessing portfolio risk. Instead of a simple sample covariance, it uses the **Ledoit-Wolf shrinkage estimator** from `sklearn.covariance`. This method provides a more robust and statistically stable estimate, especially in high-dimensional settings, by shrinking the sample covariance towards a structured target.

### 5. Portfolio Optimization (`optimizer.py`)

This module is where the portfolio construction takes place.

*   **`mean_variance_optimizer`**: Implements the classic **Mean-Variance Optimization (MVO)** framework, a cornerstone of Modern Portfolio Theory.
    *   It uses `cvxpy`, a Python-embedded modeling language for convex optimization problems, to mathematically formulate and solve the optimization.
    *   **Objective**: Maximizes the portfolio's expected return while penalizing its variance (risk), essentially seeking the optimal balance between reward and risk.
    *   **Constraints**: Applies practical constraints to the asset weights:
        *   The sum of all weights must equal `1` (fully invested portfolio).
        *   Weights must be non-negative (`w >= 0`), enforcing a **long-only** strategy (no short selling).
        *   Individual asset weights are capped by `MAX_WEIGHT` (e.g., 15%), ensuring diversification and preventing over-concentration in any single stock.

### 6. Evaluation and Analysis (`backtester.py`, `tail_risk.py`, `Plotter.py`)

This final stage assesses the performance and risk characteristics of the optimized portfolio.

*   **`backtester.py`**:
    *   **`backtest`**: Simulates the historical performance of the optimized portfolio using the calculated weights and historical asset returns. It computes the daily portfolio returns and the **Net Asset Value (NAV)**, which represents the growth of an initial investment over time.
*   **`tail_risk.py`**:
    *   **`compute_var_cvar`**: Quantifies potential downside risk using two standard metrics:
        *   **Value at Risk (VaR)**: Estimates the maximum expected loss over a given period at a specified confidence level (e.g., 95%).
        *   **Conditional Value at Risk (CVaR)** (also known as Expected Shortfall): Measures the average loss that would occur in scenarios beyond the VaR threshold, providing a more comprehensive view of extreme losses.
*   **`Plotter.py`**:
    *   **`plot_portfolio_performance`**: Generates visual outputs, including a plot of the portfolio's NAV over time (equity curve) and a histogram showing the distribution of daily portfolio returns.
    *   **`plot_var_cvar`**: Visualizes the portfolio's return distribution along with vertical lines indicating the calculated VaR and CVaR levels, offering a clear picture of tail risk.

## How to Run the Project

1.  **Install Dependencies**: Ensure you have all required Python packages installed. You can install them using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run `main.py`**: Execute the main script to perform the entire analysis:
    ```bash
    python main.py
    ```

The script will download data, perform calculations, print results to the console, and display several plots visualizing the portfolio's performance and risk.
