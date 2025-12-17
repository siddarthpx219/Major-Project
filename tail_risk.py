#diagnostics/tail_risk

import numpy as np

def compute_var_cvar(returns, alpha=0.05):
    var = np.percentile(returns, 100 * alpha)
    cvar = returns[returns <= var].mean()
    return var, cvar
