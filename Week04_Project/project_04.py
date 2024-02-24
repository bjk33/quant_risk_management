import numpy as np
import matplotlib.pyplot as plt

# Problem 1 #

# Simulating the Price Returns

# Set seed for reproducibility
np.random.seed(47)

# Step 1: Set up parameters
P_0 = 100          # Initial price
sigma = 0.02       # Standard deviation of returns
num_steps = 10000    # Number of time steps in the simulation
num_simulations = 1000  # Number of simulations to perform

# Classical Brownian Motion

# Single Price Path

# Generate random returns for the single simulation
single_returns = np.random.normal(0, sigma, num_steps)
# Calculate the prices for each time step in the single simulation
single_prices_classic = P_0 + np.cumsum(single_returns)
# Calculate the mean of the prices in the single simulation
mean_single_price_classic = np.mean(single_prices_classic)
# Calculate the standard deviation of the prices in the single simulation
std_dev_single_price_classic = np.std(single_prices_classic)

# Multiple Simulations

# Initialize an array to store the final prices from each simulation
final_prices_classic = np.zeros(num_simulations)

# Loop over the number of simulations
for i in range(num_simulations):
    # Generate random returns for each simulation
    returns = np.random.normal(0, sigma, num_steps)
    # Calculate the cumulative sum of returns, representing total price change
    total_price_change_classic = np.cumsum(returns)
    # Calculate and store the final price for each simulation
    final_prices_classic[i] = P_0 + total_price_change_classic[-1]

# Calculate the mean of the final prices across all simulations
mean_final_price_classic = np.mean(final_prices_classic)
# Calculate the standard deviation of the final prices across all simulations
std_dev_final_price_classic = np.std(final_prices_classic)

# Output the results for comparison
print(f"Single Simulation - Mean Price: {mean_single_price_classic}")
print(f"Single Simulation - Standard Deviation: {std_dev_single_price_classic}")
print(f"Multiple Simulations - Mean of Final Prices: {mean_final_price_classic}")
print(f"Multiple Simulations - Standard Deviation of Final Prices: {std_dev_final_price_classic}")

# Plotting the distribution of final prices from the multiple simulations
plt.hist(final_prices_classic, bins=50)
plt.xlabel('Final Price')
plt.ylabel('Frequency')
plt.title('Distribution of Final Prices after Multiple Simulations - Classical Brownian Motion')
plt.show()

# Arithmetic Returns
num_steps_arithmetic = 100
# Single Price Path
single_returns_arithmetic = np.random.normal(0, sigma, num_steps_arithmetic)  # Generate random returns
single_prices_arithmetic = np.zeros(num_steps_arithmetic)  # Initialize array to store prices
single_prices_arithmetic[0] = P_0  # Set the initial price

# Calculate prices for each time step in the single simulation
for t in range(1, num_steps_arithmetic):
    single_prices_arithmetic[t] = single_prices_arithmetic[t-1] * (1 + single_returns[t])

# Calculate the mean and standard deviation for the single simulation
mean_single_price = np.mean(single_prices_arithmetic)
std_dev_single_price = np.std(single_prices_arithmetic)

# Multiple Simulations for Arithmetic Returns
final_prices_arithmetic = np.zeros(num_simulations)  # Initialize array for final prices

# Perform multiple simulations
for i in range(num_simulations):
    returns = np.random.normal(0, sigma, num_steps_arithmetic)  # Generate random returns
    prices = np.zeros(num_steps_arithmetic)  # Initialize array for prices
    prices[0] = P_0  # Set the initial price

    # Calculate prices for each time step in the simulation
    for t in range(1, num_steps_arithmetic):
        prices[t] = prices[t-1] * (1 + returns[t])

    # Store the final price of each simulation
    final_prices_arithmetic[i] = prices[-1]

# Calculate the mean and standard deviation of the final prices across all simulations
mean_final_price_arithmetic = np.mean(final_prices_arithmetic)
std_dev_final_price_arithmetic = np.std(final_prices_arithmetic)

# Output the results
print(f"Arithmetic Returns - Single Simulation - Mean Price: {mean_single_price}")
print(f"Arithmetic Returns - Single Simulation - Standard Deviation: {std_dev_single_price}")
print(f"Arithmetic Returns - Multiple Simulations - Mean of Final Prices: {mean_final_price_arithmetic}")
print(f"Arithmetic Returns - Multiple Simulations - Standard Deviation of Final Prices: {std_dev_final_price_arithmetic}")


# Log Returns A.K.A. Geometric Brownian Motion
num_steps_log = 100
# Single Price Path
single_returns_log = np.random.normal(0, sigma, num_steps_log)
single_prices_log = np.zeros(num_steps_log)  # Initialize array to store prices
single_prices_log[0] = P_0  # Set the initial price

# Calculate prices for each time step in the single simulation using GBM formula
for t in range(1, num_steps_log):
    single_prices_log[t] = single_prices_log[t-1] * np.exp(single_returns_log[t])

# Calculate the mean and standard deviation for the single simulation
mean_single_price_log = np.mean(single_prices_log)
std_dev_single_price_log = np.std(single_prices_log)

# Multiple Simulations for Geometric Brownian Motion
final_prices_log = np.zeros(num_simulations)  # Initialize array for final prices

# Perform multiple simulations
for i in range(num_simulations):
    returns = np.random.normal(0, sigma, num_steps_log)  # Generate random returns
    log_prices = np.zeros(num_steps_log)  # Initialize array for prices
    log_prices[0] = P_0  # Set the initial price

    # Calculate prices for each time step in the simulation using GBM formula
    for t in range(1, num_steps_log):
        log_prices[t] = log_prices[t-1] * np.exp(returns[t])

    # Store the final price of each simulation
    final_prices_log[i] = log_prices[-1]

# Calculate the mean and standard deviation of the final prices across all simulations
mean_final_price_log = np.mean(final_prices_log)
std_dev_final_price_log = np.std(final_prices_log)

# Output the results
print(f"Geometric Brownian Motion - Single Simulation - Mean Price: {mean_single_price}")
print(f"Geometric Brownian Motion - Single Simulation - Standard Deviation: {std_dev_single_price}")
print(f"Geometric Brownian Motion - Multiple Simulations - Mean of Final Prices: {mean_final_price_log}")
print(f"Geometric Brownian Motion - Multiple Simulations - Standard Deviation of Final Prices: {std_dev_final_price_log}")
