#data/data_ingestion

import pandas as pd
import numpy as np

def load_price_data(price_path: str) -> pd.DataFrame:
    prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
    return prices.sort_index()

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns

def clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    returns = returns.replace([np.inf, -np.inf], np.nan)
    return returns.fillna(method="ffill").dropna()
