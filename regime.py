import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------
# Fit HMM Model
# -------------------------------------------------

def fit_hmm_model(
    log_returns: pd.DataFrame,
    n_components: int = 3,
    random_state: int = 42
):
    """
    Fits a Gaussian Hidden Markov Model to log returns.

    Returns:
        hmm_model,
        scaler,
        regime_means
    """

    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(log_returns.values)

    hmm_model = GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=500,
        random_state=random_state
    )

    hmm_model.fit(scaled_returns)

    regime_means = hmm_model.means_

    return hmm_model, scaler, regime_means


# -------------------------------------------------
# Predict Regime Sequence
# -------------------------------------------------

def predict_regimes(
    hmm_model,
    scaler,
    log_returns: pd.DataFrame
):
    """
    Predicts regime sequence for entire dataset.
    """

    scaled_returns = scaler.transform(log_returns.values)
    regimes = hmm_model.predict(scaled_returns)

    return regimes


# -------------------------------------------------
# Get Current Regime
# -------------------------------------------------

def get_current_regime(
    hmm_model,
    scaler,
    log_returns: pd.DataFrame
):
    """
    Returns the most recent regime index.
    """

    scaled_returns = scaler.transform(log_returns.values)
    regimes = hmm_model.predict(scaled_returns)

    return regimes[-1]


# -------------------------------------------------
# Label Regimes Based on Mean Returns
# -------------------------------------------------

def get_regime_labels(
    regime_means: np.ndarray,
    n_components: int
):
    """
    Labels regimes as Bull, Bear, or Sideways
    based on average mean return of each regime.

    Returns:
        regime_label_dict,
        sorted_means
    """

    # Average across assets for each regime
    mean_returns = regime_means.mean(axis=1)

    # Sort regimes by mean return
    sorted_indices = np.argsort(mean_returns)

    regime_labels = {}

    if n_components == 2:
        regime_labels[sorted_indices[0]] = "Bear Market"
        regime_labels[sorted_indices[1]] = "Bull Market"

    elif n_components >= 3:
        regime_labels[sorted_indices[0]] = "Bear Market"
        regime_labels[sorted_indices[-1]] = "Bull Market"

        for idx in sorted_indices[1:-1]:
            regime_labels[idx] = "Sideways Market"

    else:
        regime_labels[0] = "Single Regime"

    sorted_means = mean_returns[sorted_indices]

    return regime_labels, sorted_means