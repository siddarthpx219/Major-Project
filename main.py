# main.py

from data_ingestion import load_price_data, compute_returns, clean_returns
from feature_engineering import build_ml_features
from regime_detection import detect_regimes
from return_forecasting import train_return_model, predict_returns
from covariance_model import estimate_covariance
from optimizer import mean_variance_optimizer
from backtester import backtest
from tail_risk import compute_var_cvar
from Plotter import plot_portfolio_performance, plot_var_cvar
import config

# 1. Load data
prices = load_price_data(config.TICKER, config.START_DATE, config.END_DATE)
print("unclean prices", prices.head())
returns = clean_returns(compute_returns(prices))
print("clean returns", returns.head())

# 2. Features
features = build_ml_features(returns)

# 3. Regimes
regimes, probs, _ = detect_regimes(features)

# 4. ML Forecast
future_returns = returns.shift(-1).loc[features.index]
model = train_return_model(features, future_returns.mean(axis=1))
# Use historical mean returns per asset for optimization
mu_hat = returns.mean().values

# 5. Covariance
cov = estimate_covariance(returns)

# 6. Optimization
weights = mean_variance_optimizer(mu_hat, cov.values)

# 7. Backtest
nav, port_returns = backtest(weights, returns)
plot_portfolio_performance(nav, port_returns)


# 8. Risk
var, cvar = compute_var_cvar(port_returns)
print("VaR:", var, "CVaR:", cvar)
plot_var_cvar(port_returns, var, cvar)
