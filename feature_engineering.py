#features/feature_engineering

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def compute_factor_betas(asset_returns, factor_returns, window=36):
    betas = {}

    for asset in asset_returns.columns:
        rolling_betas = []
        for i in range(window, len(asset_returns)):
            y = asset_returns[asset].iloc[i-window:i].values
            X = factor_returns.iloc[i-window:i].values
            model = LinearRegression().fit(X, y)
            rolling_betas.append(model.coef_)
        betas[asset] = np.array(rolling_betas)

    return betas

def build_ml_features(returns: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=returns.index)
    features["volatility"] = returns.rolling(12).std().mean(axis=1)
    features["momentum"] = returns.rolling(12).mean().mean(axis=1)
    return features.dropna()
