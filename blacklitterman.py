import numpy as np
from scipy.optimize import minimize


# --------------------------------------------------
# Implied Equilibrium Returns
# --------------------------------------------------

def calculate_implied_equilibrium_returns(
    market_cap_weights: np.ndarray,
    cov_matrix: np.ndarray,
    implied_risk_aversion: float
) -> np.ndarray:
    """
    Pi = delta * Sigma * w
    All inputs assumed ANNUAL.
    """

    w = market_cap_weights.reshape(-1, 1)
    pi = implied_risk_aversion * cov_matrix @ w

    return pi.flatten()


# --------------------------------------------------
# Black–Litterman Posterior
# --------------------------------------------------

def black_litterman_formula(
    pi_excess: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    tau: float,
    cov_matrix: np.ndarray
) -> np.ndarray:

    pi = pi_excess.reshape(-1, 1)
    Q = Q.reshape(-1, 1)

    sigma = cov_matrix

    # Regularization for stability
    eps = 1e-8
    Omega = Omega + np.eye(Omega.shape[0]) * eps

    tau_sigma_inv = np.linalg.inv(tau * sigma)
    omega_inv = np.linalg.inv(Omega)

    middle = tau_sigma_inv + P.T @ omega_inv @ P
    rhs = tau_sigma_inv @ pi + P.T @ omega_inv @ Q

    posterior = np.linalg.solve(middle, rhs)

    return posterior.flatten()


# --------------------------------------------------
# Mean–Variance Optimization
# --------------------------------------------------

def calculate_optimal_weights(
    expected_excess_returns: np.ndarray,
    cov_matrix: np.ndarray,
    implied_risk_aversion: float
) -> np.ndarray:

    num_assets = len(expected_excess_returns)
    sigma = cov_matrix

    def objective(weights):
        w = weights.reshape(-1, 1)
        ret = w.T @ expected_excess_returns.reshape(-1, 1)
        var = w.T @ sigma @ w
        utility = ret - 0.5 * implied_risk_aversion * var
        return -utility.item()

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * num_assets
    init = np.ones(num_assets) / num_assets

    result = minimize(
        objective,
        init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        return init

    return result.x


# --------------------------------------------------
# Full Black–Litterman Pipeline
# --------------------------------------------------

def get_portfolio_recommendations(
    market_cap_weights: np.ndarray,
    cov_matrix: np.ndarray,
    implied_risk_aversion: float,
    P_llm: np.ndarray,
    Q_llm: np.ndarray,
    Omega_llm: np.ndarray,
    tau: float
):

    # Step 1: Prior (Pi)
    pi_excess = calculate_implied_equilibrium_returns(
        market_cap_weights,
        cov_matrix,
        implied_risk_aversion
    )

    # Step 2: Posterior
    posterior_excess = black_litterman_formula(
        pi_excess,
        P_llm,
        Q_llm,
        Omega_llm,
        tau,
        cov_matrix
    )

    # Step 3: Optimal Weights
    optimal_weights = calculate_optimal_weights(
        posterior_excess,
        cov_matrix,
        implied_risk_aversion
    )

    return pi_excess, posterior_excess, optimal_weights