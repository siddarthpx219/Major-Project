import pandas as pd
import numpy as np


# -------------------------------------------------
# Clean Price Data
# -------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans price data with controlled forward fill
    and reasonable filtering.
    """

    print(f"Original data shape: {df.shape}")

    if df.empty:
        return df

    df = df.sort_index()
    df = df[~df.index.duplicated()]

    # Drop columns with too many missing values
    min_valid_obs = int(0.8 * len(df))
    df = df.dropna(axis=1, thresh=min_valid_obs)

    # Forward fill small gaps only
    df = df.ffill(limit=3)

    # Drop rows still containing NaN
    df = df.dropna(how="any")

    print(f"Cleaned data shape: {df.shape}")

    if df.empty:
        print("Warning: DataFrame empty after cleaning.")

    return df


# -------------------------------------------------
# Log Return Calculation
# -------------------------------------------------

def calculate_log_returns(
    df: pd.DataFrame,
    clip_extremes: bool = True
) -> pd.DataFrame:
    """
    Calculates log returns with optional outlier clipping.
    """

    if df.empty:
        return pd.DataFrame()

    # Guard against non-positive prices
    if (df <= 0).any().any():
        raise ValueError("Non-positive price detected. Cannot compute log returns.")

    log_returns = np.log(df / df.shift(1)).dropna()

    if clip_extremes:
        # Clip extreme returns at 5 standard deviations
        z_scores = (log_returns - log_returns.mean()) / log_returns.std()
        log_returns = log_returns.mask(np.abs(z_scores) > 5)
        log_returns = log_returns.dropna()

    print(f"Log returns shape: {log_returns.shape}")

    return log_returns