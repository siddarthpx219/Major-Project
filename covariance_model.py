#risk/covariance_model

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

def estimate_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    lw = LedoitWolf()
    lw.fit(returns.values)
    return pd.DataFrame(
        lw.covariance_,
        index=returns.columns,
        columns=returns.columns
    )
