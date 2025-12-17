# backtest

import pandas as pd
import numpy as np

def backtest(weights, returns):
    portfolio_returns = (returns @ weights).dropna()
    nav = (1 + portfolio_returns).cumprod()
    return nav, portfolio_returns
