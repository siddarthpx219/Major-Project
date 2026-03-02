import yfinance as yf

# -------------------------
# Data Configuration
# -------------------------

START_DATE = '2010-01-01'
END_DATE = '2025-01-01'
DATA_FREQUENCY = '1mo'
PERIODS_PER_YEAR = 12


STOCK_TICKERS = ['BHARTIARTL.NS', 'LTIM.NS', 'HDFCLIFE.NS', 'NTPC.NS', 'MARUTI.NS',
 'NESTLEIND.NS', 'BAJFINANCE.NS', 'KOTAKBANK.NS', 'TATASTEEL.NS',
 'ONGC.NS', 'BAJAJ-AUTO.NS', 'LT.NS', 'ITC.NS', 'TCS.NS', 'BRITANNIA.NS',
 'SHRIRAMFIN.NS', 'ADANIENT.NS', 'CIPLA.NS', 'WIPRO.NS', 'INDUSINDBK.NS',
 'ULTRACEMCO.NS', 'TATACONSUM.NS', 'BAJAJFINSV.NS', 'RELIANCE.NS',
 'HEROMOTOCO.NS', 'COALINDIA.NS', 'TITAN.NS', 'HINDALCO.NS',
 'APOLLOHOSP.NS']


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