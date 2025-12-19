#data/data_ingestion

import pandas as pd
import numpy as np
import yfinance as yf

def load_price_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load price data from yfinance and compute returns.
    """
    # Download adjusted close prices from yfinance
    prices = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns

def clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    returns = returns.replace([np.inf, -np.inf], np.nan)
    return returns.fillna(method="ffill").dropna()
