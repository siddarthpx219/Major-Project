#regimes/regime_detection

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

def detect_regimes(market_features: pd.DataFrame, n_regimes=3):
    model = GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=500)
    model.fit(market_features.values)

    regimes = model.predict(market_features.values)
    regime_probs = model.predict_proba(market_features.values)

    return regimes, regime_probs, model
