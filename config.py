# -------------------------
# Data Configuration
# -------------------------

START_DATE = '2010-01-01'
END_DATE = '2025-01-01'
DATA_FREQUENCY = '1mo'
PERIODS_PER_YEAR = 12


STOCK_TICKERS = [
    'RELIANCE.NS',
    'TCS.NS',
    'HDFCBANK.NS',
    'ICICIBANK.NS',
]


# -------------------------
# Black–Litterman Parameters
# -------------------------

RISK_FREE_RATE_ANNUAL = 0.03
RISK_FREE_RATE_PERIODIC = RISK_FREE_RATE_ANNUAL / PERIODS_PER_YEAR

IMPLIED_RISK_AVERSION = 2.5

TAU = 0.05  # Prior uncertainty scaling


# -------------------------
# HMM Parameters
# -------------------------

NUM_MARKET_REGIMES = 3
HMM_TRAINING_PERIOD_YEARS = 5


# -------------------------
# Plotting
# -------------------------

PLOT_SAVE_PATH = 'plots/'