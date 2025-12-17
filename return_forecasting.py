#ml/return_forecasting

import xgboost as xgb
import pandas as pd

def train_return_model(features, future_returns):
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05
    )
    model.fit(features, future_returns)
    return model

def predict_returns(model, features):
    preds = model.predict(features)
    return pd.Series(preds, index=features.index)
