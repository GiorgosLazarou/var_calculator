# QuantLib for financial calculations
import QuantLib as ql

# NumPy for numerical operations
import numpy as np

# Pandas for data handling
import pandas as pd

# Matplotlib for visualization
import matplotlib.pyplot as plt

print("✅ Libraries imported successfully!")

# Portfolio weights: 60% Apple, 40% Microsoft
weights = np.array([0.6, 0.4])
print(f"\n📊 Portfolio weights: Apple {weights[0]*100}%, Microsoft {weights[1]*100}%")

# Current portfolio value ($1,000,000)
portfolio_value = 1000000
print(f"💰 Portfolio value: ${portfolio_value:,}")

# Historical daily returns (5 days of sample data)
returns_data = {
    'AAPL': [0.015, -0.022, 0.018, -0.012, 0.028],  # Apple returns
    'MSFT': [0.018, -0.015, 0.012, -0.008, 0.022]    # Microsoft returns
}

# Convert to Pandas DataFrame
returns = pd.DataFrame(returns_data)
print("\n📈 Sample returns data:")
print(returns)

# Measures how the stocks move together
cov_matrix = returns.cov()
print("\n🧮 Covariance matrix:")
print(cov_matrix)

num_simulations = 10000   # Number of simulations
time_horizon = 1          # 1-day VaR
print(f"\n🎲 Running {num_simulations:,} Monte Carlo simulations...")

# 1. Cholesky decomposition to handle correlation
chol = np.linalg.cholesky(cov_matrix)

# 2. Set random seed for reproducibility
np.random.seed(42)

# 3. Generate random numbers
random_numbers = np.random.normal(size=(num_simulations, len(weights)))

# 4. Make them correlated using matrix multiplication
correlated_randoms = random_numbers @ chol

# 5. Convert to log-normal returns
simulated_returns = np.exp(correlated_randoms - 0.5)

# Calculate portfolio returns for each simulation
portfolio_returns = np.dot(simulated_returns, weights)

# Compute future portfolio values
future_portfolio_values = portfolio_value * portfolio_returns

# Sort values from worst to best
sorted_values = np.sort(future_portfolio_values)

# Calculate 95% VaR (5th percentile loss)
confidence_level = 0.95
percentile = 100 * (1 - confidence_level)
var_95 = portfolio_value - np.percentile(future_portfolio_values, percentile)

print("\n📉 Risk Report:")
print(f"1-Day 95% VaR: ${var_95:,.2f}")
print(f"Maximum simulated loss: ${portfolio_value - sorted_values[0]:,.2f}")
print(f"Minimum simulated loss: ${portfolio_value - sorted_values[-1]:,.2f}")

plt.figure(figsize=(12, 7))

# Histogram of portfolio values
plt.hist(future_portfolio_values, bins=50, 
         alpha=0.75, color='skyblue', edgecolor='black')

# Add VaR line
plt.axvline(portfolio_value - var_95, color='red', linestyle='--', 
            linewidth=2, label=f'95% VaR: ${var_95:,.2f}')

# Add current value line
plt.axvline(portfolio_value, color='green', linestyle='-', 
            linewidth=2, label='Current Value ($1M)')

# Add labels and title
plt.xlabel("Portfolio Value ($)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Monte Carlo Simulation: Portfolio Value Distribution", fontsize=14)
plt.legend()
plt.grid(alpha=0.3)

# Format x-axis as currency
plt.gca().xaxis.set_major_formatter('${x:,.0f}')

# Save and show
plt.tight_layout()
plt.savefig('var_simulation.png')
print("\n📊 Visualization saved as 'var_simulation.png'")
plt.show()

print("\n🚀 Project completed successfully!")