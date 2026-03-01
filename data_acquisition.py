import pandas as pd
import yfinance as yf
import numpy as np
from config import STOCK_TICKERS, START_DATE, END_DATE, DATA_FREQUENCY


def get_stock_data(
    tickers=STOCK_TICKERS,
    start_date=START_DATE,
    end_date=END_DATE,
    frequency=DATA_FREQUENCY
) -> pd.DataFrame:

    print(f"Fetching stock data...")
    print(f"Tickers: {len(tickers)}")
    print(f"Range: {start_date} → {end_date}")
    print(f"Frequency: {frequency}")

    try:
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            interval=frequency,
            group_by="ticker",
            auto_adjust=False,
            progress=False
        )

        if data.empty:
            raise ValueError("No data fetched from yfinance.")

        # Handle multi-index columns properly
        if isinstance(data.columns, pd.MultiIndex):
            adj_close = pd.DataFrame({
                ticker: data[ticker]["Adj Close"]
                for ticker in tickers
                if ticker in data.columns.get_level_values(0)
            })
        else:
            # Single ticker case
            adj_close = data["Adj Close"].to_frame()
            
        return adj_close

    except Exception as e:
        print(f"Data acquisition error: {e}")
        return pd.DataFrame()