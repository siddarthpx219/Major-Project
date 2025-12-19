#ml/return_forecasting

import xgboost as xgb
import pandas as pd
import numpy as np

def train_return_model(features, future_returns):
    # Clean target: remove NaN, inf, and extreme values
    mask = np.isfinite(future_returns)
    features_clean = features[mask]
    returns_clean = future_returns[mask]
    
    # Remove rows with any NaN in features
    features_clean = features_clean.dropna()
    returns_clean = returns_clean[features_clean.index]
    
    if len(returns_clean) == 0:
        raise ValueError("No valid data after cleaning (all NaN or inf)")
    
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05
    )
    model.fit(features_clean, returns_clean)
    return model

def predict_returns(model, features):
    preds = model.predict(features)
    return pd.Series(preds, index=features.index)
