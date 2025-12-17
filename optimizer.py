#optimization/optimizer

import cvxpy as cp
import numpy as np

def mean_variance_optimizer(mu, cov, max_weight=0.15):
    n = len(mu)
    w = cp.Variable(n)

    objective = cp.Maximize(mu @ w - 0.5 * cp.quad_form(w, cov))
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_weight
    ]

    cp.Problem(objective, constraints).solve()
    return w.value
