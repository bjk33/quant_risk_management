import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t
from scipy.stats import gaussian_kde
import statsmodels.api as sm

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


mean_centered_returns = mean_center_series(arithmetic_returns, "META")

# Mean Check

# Calculate the mean of the 'META' column
mean_meta = mean_centered_returns['META'].mean()
# Print the mean to verify
print("Mean of 'META' after mean-centering:", mean_meta)

# Calculating VaR 5 Ways
meta_returns = mean_centered_returns['META']
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
print("Historical VaR with KDE at 95% confidence level:", var_hist)

# Problem 3 #

# Portfolio VaR


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
    # Step 3: Normalize weights to 1
    weights /= np.sum(weights)

    # Step 4: Compute the covariance matrix: covariance[i,j] = (w dot x)' * x where dot denotes element-wise mult
    weighted_x = x * weights[:, np.newaxis]  # broadcast weights to each row
    cov_matrix = np.dot(weighted_x.T, weighted_x)  # compute the matrix product
    return cov_matrix


def calculate_portfolio_var(portfolio, holdings_df, covariance_matrix, confidence_level):
    """
    Calculate the VaR for a given portfolio using exponentially weighted covariance matrix.

    Parameters:
    portfolio (str): Portfolio identifier.
    holdings_df (DataFrame): DataFrame with portfolio holdings.
    covariance_matrix (ndarray): Exponentially weighted covariance matrix of returns.
    confidence_level (float): Confidence level for VaR (e.g., 0.95 for 95%).

    Returns:
    float: The calculated VaR for the portfolio.
    """
    # Filter holdings for the specific portfolio
    portfolio_holdings = holdings_df[holdings_df['Portfolio'] == portfolio]

    # Map each stock to its index in the covariance matrix
    stock_indices = [centered_returns_data.columns.get_loc(stock) for stock in portfolio_holdings['Stock']]

    # Extract the relevant rows and columns from the covariance matrix
    portfolio_covariance = covariance_matrix[np.ix_(stock_indices, stock_indices)]

    # Calculate portfolio variance
    holdings = portfolio_holdings['Holding'].values
    portfolio_variance = np.dot(holdings, np.dot(portfolio_covariance, holdings))

    # Calculate the portfolio standard deviation (sqrt of variance)
    portfolio_std_dev = np.sqrt(portfolio_variance)

    # Calculate VaR as z-score times standard deviation
    z_score = norm.ppf(1 - confidence_level)
    var = z_score * portfolio_std_dev
    return -var  # VaR is a negative number representing loss


def calculate_total_portfolio_var(holdings_df, covariance_matrix, confidence_level):
    """
    Calculate the VaR for the total holdings across all portfolios.

    Parameters:
    holdings_df (DataFrame): DataFrame with portfolio holdings.
    covariance_matrix (ndarray): Exponentially weighted covariance matrix of returns.
    confidence_level (float): Confidence level for VaR (e.g., 0.95 for 95%).

    Returns:
    float: The calculated VaR for the total holdings.
    """
    # Aggregate holdings across all portfolios
    total_holdings = holdings_df.groupby('Stock')['Holding'].sum()

    # Map each stock to its index in the covariance matrix
    stock_indices = [centered_returns_data.columns.get_loc(stock) for stock in total_holdings.index]

    # Extract the relevant rows and columns from the covariance matrix
    total_covariance = covariance_matrix[np.ix_(stock_indices, stock_indices)]

    # Calculate total portfolio variance
    holdings = total_holdings.values
    total_portfolio_variance = np.dot(holdings, np.dot(total_covariance, holdings))

    # Calculate the total portfolio standard deviation (sqrt of variance)
    total_portfolio_std_dev = np.sqrt(total_portfolio_variance)

    # Calculate VaR as z-score times standard deviation
    z_score = norm.ppf(1 - confidence_level)
    var = z_score * total_portfolio_std_dev
    return -var  # VaR is a negative number representing loss


# Load portfolio
portfolio_path = '/Users/brandonkaplan/Desktop/FINTECH545/Week04_Project/portfolio.csv'
portfolio_df = pd.read_csv(portfolio_path)

# Mean-centering each stock's returns in the daily_returns_df
for asset in arithmetic_returns.columns[1:]:  # Skipping the 'Date' column
    centered_returns = mean_center_series(arithmetic_returns, asset)

# Compute EW Cov matrix
lambda_factor = 0.94
centered_returns_data = centered_returns.iloc[:, 1:]  # Exclude 'Date' column
ew_cov_matrix = ewCovar(centered_returns_data, lambda_factor)

# Calculate VaR for each portfolio and the total holdings
# Calculate VaR for each portfolio
confidence_level = 0.95  # 95% confidence level
portfolios = portfolio_df['Portfolio'].unique()
portfolio_vars = ({portfolio: calculate_portfolio_var(portfolio, portfolio_df, ew_cov_matrix, confidence_level)
                   for portfolio in portfolios})


# Calculate VaR for the total holdings
total_var = calculate_total_portfolio_var(portfolio_df, ew_cov_matrix, confidence_level)

# Display the calculated VaR for each portfolio and the total holdings
print('Portfolios:', portfolio_vars)
print("Total VaR:", total_var)


# Now with Historical VaR

# First, reshape the daily returns dataframe to merge it with the portfolio holdings
centered_returns_reshaped = centered_returns.melt(id_vars='Date', var_name='Stock', value_name='Return')

# Merge the portfolio holdings with the reshaped daily returns
portfolio_with_centered_returns = pd.merge(portfolio_df, centered_returns_reshaped, on='Stock')

# Calculate the daily value change for each stock in each portfolio
portfolio_with_centered_returns['DailyValueChange'] = (portfolio_with_centered_returns['Holding'] *
                                                       portfolio_with_centered_returns['Return'])
# Aggregate daily portfolio values
daily_portfolio_values_mean_centered = (portfolio_with_centered_returns.groupby(['Date', 'Portfolio'])
                                        ['DailyValueChange'].sum().reset_index())

# Calculate total portfolio value over time
total_daily_value_change_mean_centered = daily_portfolio_values_mean_centered.groupby('Date')['DailyValueChange'].sum()


def calculate_historical_var_kde(value_changes, confidence_level):
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
    x_values = np.linspace(range_min, range_max, 1000)

    # Evaluate the cumulative distribution of the KDE
    cdf = np.array([kde.integrate_box_1d(range_min, x) for x in x_values])

    # Find the VaR as the point where the CDF reaches the desired confidence level
    var_index = np.where(cdf >= (1 - confidence_level))[0][0]
    var = x_values[var_index]
    return -var


# Calculate Historical VaR with KDE for each portfolio
portfolio_var_hist_kde = (daily_portfolio_values_mean_centered.groupby('Portfolio')['DailyValueChange'].apply
                          (lambda x: calculate_historical_var_kde(x, confidence_level)))

# Calculate Historical VaR with KDE for the total holdings
total_var_hist_kde = calculate_historical_var_kde(total_daily_value_change_mean_centered, confidence_level)

# Display the calculated VaR for each portfolio and the total holdings with KDE
print('Portfolios (KDE):', portfolio_var_hist_kde)
print('Total (KDE):', total_var_hist_kde)



