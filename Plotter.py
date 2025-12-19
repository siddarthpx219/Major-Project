import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

def plot_portfolio_performance(nav, port_returns):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(nav.index, nav.values, label='NAV', color='blue')
    plt.title('Portfolio Net Asset Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Net Asset Value')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    sns.histplot(port_returns, bins=50, kde=True, color='orange')
    plt.title('Portfolio Returns Distribution')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

def plot_var_cvar(port_returns, var, cvar):
    plt.figure(figsize=(10, 5))
    sns.histplot(port_returns, bins=50, kde=True, color='green')
    plt.axvline(var, color='red', linestyle='--', label=f'VaR: {var:.4f}')
    plt.axvline(cvar, color='purple', linestyle='--', label=f'CVaR: {cvar:.4f}')
    plt.title('Portfolio Returns with VaR and CVaR')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()