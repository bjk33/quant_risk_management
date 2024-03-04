import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t
from scipy.stats import gaussian_kde
import statsmodels.api as sm
from concurrent.futures import ThreadPoolExecutor

# Problem 1 #

# Simulating the Price Returns

# Set seed for reproducibility
np.random.seed(47)

# Set up parameters
P_0 = 100          # Initial price
sigma = 0.02       # Standard deviation of returns
mu = 0    # Mean of returns
num_simulations = 100000  # Number of simulations to perform

# Generate random returns from the normal distribution
returns = np.random.normal(mu, sigma, num_simulations)

# Classical Brownian Motion Returns
P1_brownian = P_0 + returns
print(f"Classical Brownian Mean: {np.mean(P1_brownian)} - Classical Brownian Std: {np.std(P1_brownian)}")

# Arithmetic Returns
P1_arithmetic = P_0 * (1 + returns)
print(f"Arithmetic Mean: {np.mean(P1_arithmetic)} - Arithmetic Std: {np.std(P1_arithmetic)}")

# Log Returns a.k.a. Geometric Brownian Returns
P1_geometric = P_0 * np.exp(returns)
print(f"Geometric Mean: {np.mean(P1_geometric)} - Geometric Std: {np.std(P1_geometric)}")


# Problem 2 #

# Implementing return_calculate() #

# Import Price Data
daily_prices_path = '/Users/brandonkaplan/Desktop/FINTECH545/Week04_Project/DailyPrices.csv'
daily_prices = pd.read_csv(daily_prices_path)


def return_calc(prices, method="DISCRETE", date_column="Date"):
    """
    This function calculate returns for financial data in a DataFrame.

    Parameters:
    :param: prices (DataFrame): DataFrame containing price data and a date column.
    :param: method (str, optional): Method for calculating returns. Options are "DISCRETE" or "LOG". Default is
    "DISCRETE". "DISCRETE" relies on calculating returns using arithmetic returns and "LOG" relies on calculating
    returns using geometric returns
    :param:date_column (str, optional): Column name for dates in the DataFrame. Default is "Date".

    Returns:
    :return: out: DataFrame: A new DataFrame with the calculated returns and corresponding dates.

    Raises:
    ValueError: If the date column is not found in the DataFrame.
    ValueError: If the method is not recognized.
    """

    # Check if the date column is in the DataFrame
    if date_column not in prices.columns:
        raise ValueError(f"dateColumn: {date_column} not in DataFrame.")

    # Selecting columns except the date column
    assets = [col for col in prices.columns if col != date_column]

    # Convert prices to a numpy matrix for calculations
    p = prices[assets].values

    # Calculating the price ratios
    p2 = p[1:] / p[:-1]

    # Applying the selected return calculation method
    if method.upper() == "DISCRETE":
        p2 = p2 - 1
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f"method: {method} must be in (\"LOG\", \"DISCRETE\")")

    # Aligning the dates with the returns

    dates = prices[date_column].iloc[1:]

    # Creating a DataFrame from the returns

    returns_df = pd.DataFrame(p2, columns=assets, index=dates.index)

    # Merging the returns with the dates

    out = pd.concat([prices[date_column], returns_df], axis=1).dropna()

    return out


arithmetic_returns = return_calc(daily_prices)


# Mean Centering
def mean_center_series(df, column_name):
    """
    This function adjusts the specified column in a DataFrame such that its mean is zero.
    Parameters:
    :param: df (DataFrame): DataFrame containing the data.
    :param: column_name (str): Name of the column to be mean-centered.
    Returns:
    :return: df: DataFrame: A new DataFrame with the specified column mean-centered.
    """
    # Check if the column exists in the DataFrame
    if column_name not in df.columns:

        raise ValueError(f"Column {column_name} not found in DataFrame.")

    # Calculate the mean of the specified column

    column_mean = df[column_name].mean()

    # Subtract the mean from the column to mean-center it

    df[column_name] = df[column_name] - column_mean

    return df


meta_mean_centered_returns = mean_center_series(arithmetic_returns, "META")

# Mean Check

# Calculate the mean of the 'META' column
mean_meta = meta_mean_centered_returns['META'].mean()
# Print the mean to verify
print("Mean of 'META' after mean-centering:", mean_meta)

# Calculating VaR 5 Ways
meta_returns = arithmetic_returns['META']
meta_returns = meta_returns - meta_returns.mean() # center mean
alpha = 0.05
confidence_level = 1 - alpha

# (1) Normal Distribution
meta_return_std = meta_returns.std()
z_score = norm.ppf(alpha)
var_normal = -(z_score * meta_return_std)
print("Normal VaR at 95% confidence level:", var_normal)

# (2) Normal Distribution with Exponentially Weighted Variance (lambda = 0.94)
lambda_factor = 0.94
alpha_factor = 1 - lambda_factor
ew_variance = meta_returns.ewm(alpha=alpha_factor, adjust=False).var()
ew_std_dev = np.sqrt(ew_variance.iloc[-1])  # Last value for the most recent standard deviation
z_score_ew = norm.ppf(alpha)
var_ew = -(z_score_ew * ew_std_dev)
print("EWV VaR at 95% confidence level:", var_ew)

# (3) MLE Fitted Student's t-Dist
# Fit the t-distribution to the data
params = t.fit(meta_returns)
df, loc, scale = params  # degrees of freedom, location, and scale
# Calculate the VaR
var_t = t.ppf(alpha, df, loc, scale)
var_t = -var_t
print("MLE fitted t-Dist VaR at 95% confidence level:", var_t)

# (4) Fitted AR(1) Model

# Fit an AR(1) model
ar_model = sm.tsa.ARIMA(meta_returns, order=(1,0,0))
ar_result = ar_model.fit()

# Forecast the next return
forecast = ar_result.params[0] + ar_result.params[1] * meta_returns.iloc[-1]

# Calculate the standard deviation of the residuals
residuals_std = ar_result.resid.std()

# Calculate VaR assuming normality of residuals
z_score_ar = norm.ppf(alpha)
var_ar = -(forecast + z_score_ar * residuals_std)
print("Fitted AR(1) VaR at 95% confidence level:", var_ar)

# (5) Historic Simulation
num_draws = 1000
current_price = daily_prices['META'].iloc[-1]  # This is the most recent META price

# Fit KDE to the returns
kde = gaussian_kde(meta_returns)

# Generate random numbers from the KDE
sampled_returns = kde.resample(num_draws)[0]

# Calculate new prices from the returns
new_prices = (1 + sampled_returns) * current_price

# Calculate VaR
var = np.percentile(new_prices, (alpha) * 100)
var_hist = current_price - var
var_hist_ret = var_hist / current_price

print("Historical VaR with KDE at 95% confidence level:", var_hist_ret)


# Alternative Historic Simulation (no KDE)
def VaR(a, alpha=0.05):
    x = np.sort(a)
    nup = int(np.ceil(len(a) * alpha))
    ndn = int(np.floor(len(a) * alpha))
    v = 0.5 * (x[nup] + x[ndn])

    return -v

print(VaR(meta_returns, alpha=0.05))

# Problem 3 #

# Portfolio VaR

# Load portfolio
portfolio_path = '/Users/brandonkaplan/Desktop/FINTECH545/Week04_Project/portfolio.csv'
portfolio_df = pd.read_csv(portfolio_path)


def ewCovar(x, lambda_):
    """Compute exponentially weighted covariance matrix of a dataframe.
    :param x: a pandas dataframe
    :param lambda_: smoothing parameter
    :return: cov_matrix: a covariance matrix
    """
    m, n = np.shape(x)
    weights = np.zeros(m)

    # Step 1: Remove means
    x_bar = np.mean(x, axis=0)
    x = x - x_bar

    # Step 2: Calculate weights (note we are going from oldest to newest weight)
    for i in range(m):
        weights[i] = (1 - lambda_) * lambda_ ** (m - i - 1)

    weights_mat = np.diag(weights/sum(weights))
    cov_matrix = np.transpose(x.values) @ weights_mat @ x.values
    return cov_matrix


# Mean-centering each stock's returns in the daily_returns_df
mean_centered_returns = arithmetic_returns.iloc[:, 1:] - arithmetic_returns.iloc[:, 1:].mean()
# mean_centered_returns = mean_centered_returns.iloc[:, 1:]  # Exclude 'Date' column

# Calculate EW Covariance Matrix
lambda_factor = 0.94
ew_cov_matrix = ewCovar(mean_centered_returns, lambda_factor)



def calculate_portfolio_values(prices_df, portfolio_df):
    """
    Calculate the total value of each portfolio.

    Parameters:
    :param: prices_df (Pandas DataFrame): Data frame of stock prices
    :param: portfolio_df (Pandas DataFrame): Data frame containing different portfolios and their holdings of the stocks

    Returns:
    :return: pd.Series: Total value of each portfolio.
    """

    # Get the latest prices for each stock
    latest_prices = prices_df.iloc[-1, 1:]  # Skipping the 'Date' column
    latest_prices = latest_prices.astype(float)  # Convert prices to float

    # Pivot the portfolio DataFrame to align with the stock symbols
    restructured_portfolio = (portfolio_df.pivot_table(index='Stock', columns='Portfolio', values='Holding',
                                                       fill_value=0))

    # Calculate the current value of each stock in each portfolio
    portfolio_values = restructured_portfolio.multiply(latest_prices, axis=0)

    # Calculate the total value of each portfolio
    portfolio_totals = portfolio_values.sum(axis=0)

    return portfolio_totals


portfolio_totals = calculate_portfolio_values(daily_prices, portfolio_df)

# Print the portfolio values
print("Current Portfolio Value:")
for portfolio, value in portfolio_totals.items():
    print(f"Portfolio {portfolio}: ${value:,.2f}")


def calculate_portfolio_weights(prices_df, portfolio_df, portfolio_totals):
    """
    Calculate the weights of each stock in each portfolio.

    :param prices_df: DataFrame with stock prices
    :param portfolio_df: DataFrame with portfolio holdings (quantity of shares)
    :param portfolio_totals: Series with total value of each portfolio
    :return: DataFrame with weights of each stock in each portfolio
    """
    # Get the latest prices for each stock
    latest_prices = prices_df.iloc[-1, 1:]  # Assuming the last row contains the latest prices

    # Pivot the portfolio DataFrame to align with the stock symbols
    restructured_portfolio = portfolio_df.pivot_table(index='Stock', columns='Portfolio', values='Holding',
                                                      fill_value=0)

    # Calculate the dollar value of each holding
    dollar_values = restructured_portfolio.multiply(latest_prices, axis=0)

    # Identify the row(s) with NaN values
    nan_rows = dollar_values[dollar_values.isna().all(axis=1)]

    # If there's exactly one row with NaN values, and it's the extra row, remove it
    if len(nan_rows) == 1:
        dollar_values = dollar_values.dropna()
    # Calculate weights
    weights = dollar_values.divide(portfolio_totals)
    return weights


# Calculate the weights for each portfolio
portfolio_weights = calculate_portfolio_weights(daily_prices, portfolio_df, portfolio_totals)

# Placeholder to proceed with the VaR calculation
print(portfolio_weights)


def calculate_var(portfolio_weights, returns_df, lambda_factor, portfolio_totals, z_score):
    """
    Calculate the Delta Normal VaR for each portfolio using an exponentially weighted covariance matrix and a smoothing
    factor of lambda=0.94.

    :param portfolio_weights: DataFrame with weights of each stock in each portfolio
    :param cov_matrix: Exponentially weighted covariance matrix
    :param portfolio_totals: Series with total value of each portfolio
    :param z_score: Z-Score for the desired confidence level (default is 1.65 for 95% confidence)
    :return: Dictionary with VaR for each portfolio and total VaR
    """
    var_values = {}

    for portfolio in portfolio_weights.columns:
        # Filter the returns for the stocks in the portfolio
        portfolio_stocks = portfolio_weights.index[portfolio_weights[portfolio] > 0]
        filtered_returns = returns_df[portfolio_stocks]
        filtered_weights = portfolio_weights.loc[portfolio_stocks, portfolio] # I am now trying to calc portfolio covar

        # Calculate the exponentially weighted covariance matrix for the portfolio
        portfolio_ew_cov_matrix = ewCovar(filtered_returns, lambda_factor)
        #weights = portfolio_weights[portfolio].values # commented this out for filtered weights
        portfolio_variance = np.dot(filtered_weights, np.dot(portfolio_ew_cov_matrix, filtered_weights))
        portfolio_std_dev = np.sqrt(portfolio_variance)
        portfolio_value = portfolio_totals[portfolio]
        print(portfolio_value, portfolio)
        portfolio_var = portfolio_value * z_score * portfolio_std_dev

        var_values[portfolio] = portfolio_var

    return var_values


# Calculate VaR for each portfolio and total VaR
var_results = calculate_var(portfolio_weights, mean_centered_returns,0.94, portfolio_totals, 1.65)
print(var_results)


# implementation of professor's delta normal
def delta_normal_var(portfolio_df, prices_df, returns_df, lambda_factor, z_score):
    """
    Calculate the Delta Normal VaR for each portfolio and total.

    :param portfolio_df: DataFrame with portfolio holdings (quantity of shares)
    :param prices_df: DataFrame with stock prices
    :param returns_df: DataFrame with stock returns
    :param lambda_factor: Lambda factor for exponentially weighted covariance
    :param z_score: Z-Score for the desired confidence level
    :return: Dictionary with VaR for each portfolio
    """
    var_values = {}
    total_holdings = {}
    total_portfolio_value = 0

    for portfolio in portfolio_df['Portfolio'].unique():
        holdings = portfolio_df[portfolio_df['Portfolio'] == portfolio]
        relevant_stocks = holdings['Stock'].tolist()
        current_prices = prices_df[relevant_stocks].iloc[-1]
        filtered_returns = returns_df[relevant_stocks]

        # Calculate portfolio value
        portfolio_value = (sum(holdings[holdings['Stock'] == stock]['Holding'].iloc[0] * current_prices[stock] for
                               stock in relevant_stocks))
        # And total value
        total_portfolio_value += portfolio_value

        # Aggregate holdings for total portfolio calculation
        for stock in relevant_stocks:
            if stock not in total_holdings:
                total_holdings[stock] = 0
            total_holdings[stock] += holdings[holdings['Stock'] == stock]['Holding'].iloc[0] * current_prices[stock]

        # Calculate delta (portfolio weights)
        delta = (np.array([holdings[holdings['Stock'] == stock]['Holding'].iloc[0] * current_prices[stock] /
                           portfolio_value for stock in relevant_stocks]))

        # Calculate covariance matrix
        cov_matrix = ewCovar(filtered_returns, lambda_factor)

        # Portfolio standard deviation
        portfolio_std_dev = np.sqrt(np.dot(delta.T, np.dot(cov_matrix, delta)))

        # Calculate VaR
        var = -portfolio_value * norm.ppf(z_score) * portfolio_std_dev
        var_values[portfolio] = var

    # Total portfolio VaR calculation
    relevant_stocks = list(total_holdings.keys())
    total_delta = np.array([total_holdings[stock] / total_portfolio_value for stock in relevant_stocks])
    total_cov_matrix = ewCovar(returns_df[relevant_stocks], lambda_factor)
    total_portfolio_std_dev = np.sqrt(np.dot(total_delta.T, np.dot(total_cov_matrix, total_delta)))
    total_var = -total_portfolio_value * norm.ppf(z_score) * total_portfolio_std_dev
    var_values['Total'] = total_var

    return var_values


# Usage

z_score = 0.05  # For a 95% confidence level
lambda_factor = 0.94


# Calculate VaR for each portfolio and the total
var_results = delta_normal_var(portfolio_df, daily_prices, mean_centered_returns, lambda_factor, z_score)
print(var_results)


# Now to calculate VaR with Historical Simulation


def calculate_historical_var_kde(value_changes, alpha):
    """
    Calculate Historical VaR using KDE for a series of value changes.

    Parameters:
    value_changes (Series): Pandas Series of value changes.
    confidence_level (float): Confidence level for VaR (e.g., 0.95 for 95%).

    Returns:
    float: The calculated VaR.
    """
    # Fit a KDE to the value changes
    kde = gaussian_kde(value_changes)

    # Generate a range of values (e.g., from -3*std to 3*std)
    range_min = value_changes.min()
    range_max = value_changes.max()
    x_values = np.linspace(range_min, range_max, 10000)

    # Evaluate the cumulative distribution of the KDE
    cdf = np.array([kde.integrate_box_1d(range_min, x) for x in x_values])

    # Find the VaR as the point where the CDF reaches the desired confidence level
    var_index = np.where(cdf >= (alpha))[0][0]
    var = x_values[var_index]
    return -var


# Usage

# First, reshape the daily returns dataframe to merge it with the portfolio holdings
dates = arithmetic_returns.iloc[:, 0]
mean_centered_returns = pd.concat([dates, mean_centered_returns], axis=1)
mean_centered_returns_reshaped = mean_centered_returns.melt(id_vars='Date', var_name='Stock', value_name='Return')

# Merge the portfolio holdings with the reshaped daily returns
portfolio_with_centered_returns = pd.merge(portfolio_df, mean_centered_returns_reshaped, on='Stock')

# Calculate the daily value change for each stock in each portfolio
portfolio_with_centered_returns['DailyValueChange'] = portfolio_with_centered_returns.apply(
    lambda row: row['Holding'] * row['Return'] * current_stock_prices[row['Stock']], axis=1)

# Aggregate daily portfolio values
daily_portfolio_values_mean_centered = (portfolio_with_centered_returns.groupby(['Date', 'Portfolio'])
                                        ['DailyValueChange'].sum().reset_index())

# Calculate total portfolio value over time
total_daily_value_change_mean_centered = daily_portfolio_values_mean_centered.groupby('Date')['DailyValueChange'].sum()

alpha = 0.05
# Calculate Historical VaR with KDE for each portfolio
portfolio_var_hist_kde = (daily_portfolio_values_mean_centered.groupby('Portfolio')['DailyValueChange'].apply
                          (lambda x: calculate_historical_var_kde(x, alpha)))

# Calculate Historical VaR with KDE for the total holdings
total_var_hist_kde = calculate_historical_var_kde(total_daily_value_change_mean_centered, alpha)

# Display the calculated VaR for each portfolio and the total holdings with KDE
print('Portfolios (Historical - KDE):', portfolio_var_hist_kde)
print('Total (Historical - KDE):', total_var_hist_kde)


def calculate_portfolio_var_hist(portfolio_holdings, current_prices, historical_returns, alpha, n_simulations):
    """
    Calculate Historical VaR for a given portfolio.
    Parameters:
    - portfolio_holdings (dict): Dictionary of holdings (quantity of each asset).
    - current_prices (dict): Dictionary of current prices for each asset.
    - historical_returns (DataFrame): DataFrame of historical returns.
    - alpha (float): Confidence level for VaR (e.g., 0.95 for 95% confidence).
    - n_simulations (int): Number of simulations.
    Returns:
    - float: The calculated VaR for the portfolio.
    """
    current_portfolio_value = sum(portfolio_holdings[stock] * current_prices[stock] for stock in portfolio_holdings)
    simulated_values = []
    for _ in range(n_simulations):
        # Sample from historical returns
        sampled_returns = historical_returns.sample(n=1, replace=True)
        # Calculate new portfolio value based on sampled returns
        new_value = sum(
            portfolio_holdings[stock] * current_prices[stock] * (1 + sampled_returns[stock].iloc[0]) for stock in
            portfolio_holdings)
        simulated_values.append(new_value)
    # Find the α% of the simulated portfolio value distribution
    var_value = np.percentile(simulated_values, alpha * 100)
    return current_portfolio_value - var_value


# Usage

alpha = 0.05  # 95% confidence level
n_simulations = 10000
current_stock_prices = daily_prices[portfolio_df['Stock']].iloc[-1]


portfolio_vars = {}
total_holdings = {}
# Calculate VaR for each portfolio
for portfolio in portfolio_df['Portfolio'].unique():
    portfolio_holdings = portfolio_df[portfolio_df['Portfolio'] == portfolio].set_index('Stock')[
        'Holding'].to_dict()
    portfolio_vars[portfolio] = calculate_portfolio_var_hist(portfolio_holdings, current_stock_prices, mean_centered_returns, alpha,
                                                        n_simulations)
    # Aggregate holdings for the total portfolio
    for stock, holding in portfolio_holdings.items():
        if stock not in total_holdings:
            total_holdings[stock] = 0
        total_holdings[stock] += holding
# Calculate VaR for the total portfolio
total_var = calculate_portfolio_var_hist(total_holdings, current_stock_prices, mean_centered_returns, alpha, n_simulations)
# Output
print("Individual Portfolios Historical Simulation VaR:", portfolio_vars)
print("Total Portfolio Historical Simulation VaR:", total_var)


# Now Normal Monte Carlo with Exponentially Weighted Covariance Matrix

# First simulate PCA

def simulate_pca(a, nsim, pctExp=1, mean=None, seed=1234):
    """
    Simulate a multivariate normal distribution using PCA based on a covariance matrix and an optional percentage of
    variance explained (indirectly the number of eigenvalues/principal components to include).
    :param a: The input covariance matrix
    :param nsim: Specifies the number of samples to simulate
    :param pctExp: (optional) The percentage of total variance that should be explained by the principal components. The
    default is 100%
    :param mean: (optional) The mean vector of the covariance matrix. If not provided the default mean is zero
    :param seed: (optional) The seed for random number generation to ensure reproducibility
    :return: out: The matrix of simulated samples
    """
    n = a.shape[0]

    if mean is None:
        _mean = np.zeros(n)
    else:
        _mean = np.array(mean)

    # Eigenvalue decomposition
    vals, vecs = np.linalg.eigh(a)
    vals = np.real(vals)
    vecs = np.real(vecs)
    # Sort values and vectors in descending order
    idx = vals.argsort()[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    tv = np.sum(vals)

    posv = np.where(vals >= 1e-8)[0]
    if pctExp < 1:
        pct = 0.0
        for i in posv:
            pct += vals[i] / tv
            if pct >= pctExp:
                posv = posv[:np.where(posv == i)[0][0] + 1]
                break
    vals = vals[posv]
    vecs = vecs[:, posv]

    # Construct B matrix
    B = vecs @ np.diag(np.sqrt(vals))

    # Generate random samples
    np.random.seed(seed)
    r = np.random.randn(vals.shape[0], nsim)
    print(B.shape, r.shape)
    out = (B @ r).T

    # Add the mean
    for i in range(n):
        out[:, i] += _mean[i]

    return out


# Usage

# Get (mean centered) returns of all stocks in portfolio
portfolio_returns = mean_centered_returns[portfolio_df['Stock']]

# Compute exponentially weighted covariance matrix for portfolio returns
ew_cov_matrix = ewCovar(portfolio_returns, lambda_=0.94)
nsim = 10000
nmc_sim = simulate_pca(ew_cov_matrix, nsim)
sim_returns = pd.DataFrame(nmc_sim, columns=portfolio_df.Stock)

# Cross-joining portfolio and iterations
iterations = pd.DataFrame({'iteration': range(1, nsim + 1)})
values = pd.merge(portfolio_df.assign(key=1), iterations.assign(key=1), on='key').drop('key', axis=1)


# Pricing function
def pricing(row, prices, portfolio):
    """
        Calculate the current and simulated values of a portfolio holding, and its PnL.

        The function takes a row from a DataFrame where each row combines a stock in the portfolio
        with a simulation iteration. It uses the current stock price and simulated returns to
        calculate the current value, simulated value, and PnL for the holding.

        Parameters:
        :param: row (pandas Series): A row from the DataFrame. The row must contain:
            - 'Stock': Stock identifier (e.g., ticker symbol).
            - 'Holding': Quantity of the stock held in the portfolio.
            - 'iteration': Iteration number of the simulation.
          The `current` and `simReturns` data structures should be accessible in the scope
          where this function is called. `current` should be a Series or a DataFrame row
          with the current prices of stocks, and `simReturns` a DataFrame with simulated
          returns for each stock across different iterations.

        Returns:
        :return: current_value (float): The current value of the holding, calculated as Holding * current price.
        :return: simulated_value (float): The simulated value of the holding, adjusted by the simulated return.
        :return:  pnl (float): Profit and Loss for the holding, calculated as the difference between
                       simulated value and current value.
        """

    # All current stock prices
    current_stock_prices = prices[portfolio['Stock']].iloc[-1]

    # Extracting current price for the given stock
    price = current_stock_prices[row['Stock']]

    # Calculating current value of the holding
    current_value = row['Holding'] * price

    # Calculating simulated value of the holding (adjusted by simulated return)
    simulated_value = row['Holding'] * price * (1.0 + sim_returns.loc[row['iteration'] - 1, row['Stock']])

    # Calculating PnL for the holding
    pnl = simulated_value - current_value
    return current_value, simulated_value, pnl


# Applying the pricing function in parallel
with ThreadPoolExecutor() as executor:
    results = list(executor.map(pricing, values.to_dict(orient='records')))

# Extracting results
current_values, simulated_values, pnls = zip(*results)
values['currentValue'] = current_values
values['simulatedValue'] = simulated_values
values['pnl'] = pnls


def calculate_var(series, alpha=0.05):
    """
    Calculate the Value at Risk (VaR) at a specified confidence level.

    Parameters:
    - series (pandas Series): A series of profit and loss (PnL) values.
    - alpha (float): The confidence level (default is 0.05 for 95% confidence).

    Returns:
    - VaR (float): The calculated Value at Risk.
    """
    return -np.percentile(series, 100 * alpha)

# Usage

# Portfolio Level Metrics

# Summing by Portfolio and Iteration
gdf = values.groupby(['Portfolio', 'iteration'])
portfolio_values = gdf.agg({'currentValue': 'sum', 'pnl': 'sum'}).reset_index()

# Calculating Portfolio Level Risk Metrics
gdf = portfolio_values.groupby('Portfolio')
portfolio_risk = gdf.agg({
    'currentValue': 'first',
    'pnl': lambda x: calculate_var(x, alpha=0.05)
}).rename(columns={'currentValue': 'currentValue', 'pnl': 'VaR95'})

# Total Metrics

# Grouping by iteration and summing the values
gdf = values.groupby('iteration')
total_values = gdf.agg({'currentValue': 'sum', 'pnl': 'sum'}).reset_index()

# Separating the aggregation and transformation operations
# Get the first currentValue
total_current_value = total_values['currentValue'].iloc[0]

# Correct calculation of VaR for total risk
total_pnl_var = calculate_var(total_values['pnl'], alpha=0.05)

# Constructing the total_risk DataFrame
total_risk = pd.DataFrame({
    'currentValue': [total_current_value],
    'VaR95': [total_pnl_var],
    'Portfolio': 'Total'
})

# Visualize

# Concatenate the portfolioRisk and totalRisk DataFrames
VaRReport = pd.concat([portfolio_risk, total_risk], ignore_index=True)

# Print the VaRReport DataFrame
print(VaRReport)