# bl

import numpy as np

def black_litterman(mu_eq, cov, P, Q, omega):
    tau = 0.05
    middle = np.linalg.inv(P @ (tau * cov) @ P.T + omega)
    mu_bl = mu_eq + (tau * cov @ P.T @ middle @ (Q - P @ mu_eq))
    return mu_bl
