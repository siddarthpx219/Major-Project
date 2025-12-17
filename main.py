# main.py

from data.data_ingestion import load_price_data, compute_returns, clean_returns
from features.feature_engineering import build_ml_features
from regimes.regime_detection import detect_regimes
from ml.return_forecasting import train_return_model, predict_returns
from risk.covariance_model import estimate_covariance
from optimization.optimizer import mean_variance_optimizer
from backtest.backtester import backtest
from diagnostics.tail_risk import compute_var_cvar

# 1. Load data
prices = load_price_data("prices.csv")
returns = clean_returns(compute_returns(prices))

# 2. Features
features = build_ml_features(returns)

# 3. Regimes
regimes, probs, _ = detect_regimes(features)

# 4. ML Forecast
future_returns = returns.shift(-1).mean(axis=1).loc[features.index]
model = train_return_model(features, future_returns)
mu_hat = predict_returns(model, features).iloc[-1]

# 5. Covariance
cov = estimate_covariance(returns)

# 6. Optimization
weights = mean_variance_optimizer(mu_hat.values, cov.values)

# 7. Backtest
nav, port_returns = backtest(weights, returns)

# 8. Risk
var, cvar = compute_var_cvar(port_returns)
print("VaR:", var, "CVaR:", cvar)
