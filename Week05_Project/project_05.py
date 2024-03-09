import pandas as pd
import numpy as np
from scipy.stats import norm, t, kurtosis
from scipy.optimize import minimize

# Problem 2 #

# Calculate VaR and Expected Shortfall

returns_path = '/Users/brandonkaplan/Desktop/FINTECH545/Week05_Project/problem1.csv'
returns = pd.read_csv(returns_path)

# Mean Center
returns = returns - returns.mean()


# A) Normal distribution with exponentially weighted variance (lambda = 0.97)


def ewCovar(x, lambda_):
    """Routine for computing exponentially weighted covariance matrix of a dataframe.
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

    # Step 3: Create a diagonal matrix from the normalized weights.
    weights_mat = np.diag(weights / sum(weights))

    # Step 4: Calculate the weighted covariance matrix
    cov_matrix = np.transpose(x.values) @ weights_mat @ x.values

    return cov_matrix


def calculate_var_es_ew_normal(returns_df, lambda_, alpha):
    # Compute EW variance matrix
    variance_matrix = ewCovar(returns_df, lambda_)

    # Extract EW variance
    ew_variance = variance_matrix[0, 0]

    # VaR computation
    ew_std = np.sqrt(ew_variance)
    z_score = norm.ppf(alpha)
    var = -z_score * ew_std

    # ES computation
    es = -np.mean(returns_df) + ew_std * norm.pdf(z_score) / alpha
    return var, es


# Usage

var, es = calculate_var_es_ew_normal(returns, lambda_=0.97, alpha=0.05)

print("Series VaR - Normal EW Variance:", var)
print("Series ES - Normal EW Variance:", es)


# B) Using an MLE fitted Student's t-distribution


def calculate_var_mle_t_dist(returns, alpha):
    """
    Calculate VaR using a MLE fitted t-distribution.

    Parameters:
    :param: returns (Series): Pandas Series of returns.
    :param: alpha (float): The % worst bad day you want to calculate VaR for the returns. Alpha is equal to 1-minus the
    confidence level. Alpha is the % worst case scenario. (E.g., 0.05 for a day worse than 95% of typical days).

    Returns:
    :return: -var: float: The calculated VaR.
    """
    # Fit the t-distribution to the data
    params = t.fit(returns)
    df, loc, scale = params[0], params[1], params[2]  # degrees of freedom, location, and scale

    # Calculate the VaR
    var = t.ppf(alpha, df, loc, scale)

    # Calculate the ES
    t_sim = t.rvs(df, loc, scale, size=10000)
    es = -np.mean(t_sim[t_sim <= var])

    return -var, es


# Usage

var_mle_t, es_mle_t = calculate_var_mle_t_dist(returns, alpha=0.05)
print("Series VaR - MLE Fitted T-dist:", var_mle_t)
print("Series ES - MLE Fitted T-dist:", es_mle_t)


# C) Using Historic Simulation
returns = returns['x'].to_numpy()
def VaR(a, alpha=0.05):
    x = np.sort(a)
    nup = int(np.ceil(len(a) * alpha))
    ndn = int(np.floor(len(a) * alpha))
    v = 0.5 * (x[nup] + x[ndn])

    return -v


def ES(a, alpha=0.05):
    """
    Same as VaR, except that it returns the expected shortfall (a.k.a. conditional VaR) for an input series
    :param a: (array-like): An array of historical financial data (e.g., returns or prices).
    :param alpha: (float, optional): The confidence level, representing the probability of
                             exceeding the VaR. Default value is 0.05 (95% confidence level).
    :return: -es: float: The expected shortfall or expectation of the VaR.
    """
    x = np.sort(a)
    nup = int(np.ceil(len(a) * alpha))
    ndn = int(np.floor(len(a) * alpha))
    v = 0.5 * (x[nup] + x[ndn])
    es = np.mean(x[x <= v])
    return -es


var_hist = VaR(returns)
es_hist = ES(returns)
print("Series VaR - Historic Simulation:", var_hist)
print("Series ES - Historic Simulation:", es_hist)

# Problem 3 #

# Portfolio VaR and ES with copula

prices_path = '/Users/brandonkaplan/Desktop/FINTECH545/Week05_Project/DailyPrices.csv'
portfolio_path = '/Users/brandonkaplan/Desktop/FINTECH545/Week05_Project/portfolio.csv'
prices = pd.read_csv(prices_path)
portfolio = pd.read_csv(portfolio_path)


def return_calc(prices_df, method="DISCRETE", date_column="Date"):
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
    assets = [col for col in prices_df.columns if col != date_column]

    # Convert prices to a numpy matrix for calculations
    p = prices_df[assets].values

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

    dates = prices_df[date_column].iloc[1:]

    # Creating a DataFrame from the returns

    returns_df = pd.DataFrame(p2, columns=assets, index=dates.index)

    # Merging the returns with the dates

    out = pd.concat([prices_df[date_column], returns_df], axis=1).dropna()

    return out


# Compute arithmetic returns
arithmetic_returns = return_calc(prices, method='DISCRETE', date_column='Date')
dates = arithmetic_returns.iloc[:, 0] # if needed

# Assume expected return is zero
arithmetic_returns = arithmetic_returns.iloc[:, 1:] # remove dates
arithmetic_returns = arithmetic_returns - arithmetic_returns.mean()

# arithmetic_returns = pd.concat([dates, arithmetic_returns], axis=1)


# Fit models

# First define fitted model structure in form of a class

class FittedModel:
    def __init__(self, beta, error_model, eval_func, errors, u):
        self.beta = beta
        self.error_model = error_model
        self.eval_func = eval_func
        self.errors = errors
        self.u = u


# Create functions for fitting models

# General t sum ll function
def general_t_ll(mu, s, nu, x):
    """
        Calculate the sum of the logarithms of the probability density function (pdf)
        values of a scaled and shifted t-distribution for a given set of data points.

        Parameters:
        :param: mu (float): The location parameter (mean) of the t-distribution.
        :param: s (float): The scale (sigma) factor applied to the t-distribution.
        :param: nu (int): The degrees of freedom for the t-distribution.
        :param: x (array-like): An array or list of data points to evaluate the t-distribution.

        Returns:
        :return: log_sum (float): The sum of the logarithms of the pdf values for the data points in 'x'.
        """
    # Scale and shift the t-distribution
    scaled_pdf = lambda x_val: t.pdf((x_val - mu) / s, nu) / s
    # Apply the scaled pdf to each element in x and sum their logs
    log_sum = np.sum(np.log(scaled_pdf(x)))
    return log_sum


def fit_general_t(x):
    # Approximate values based on moments
    start_m = np.mean(x)
    start_nu = 6.0 / kurtosis(x, fisher=False) + 4
    start_s = np.sqrt(np.var(x) * (start_nu - 2) / start_nu)

    # Objective function to maximize (log-likelihood)
    def objective(mu, s, nu):
        return -general_t_ll(mu, s, nu, x)  # Negated for minimization

    # Initial parameters
    initial_params = np.array([start_m, start_s, start_nu])

    # Bounds for s and nu
    bounds = [(None, None), (1e-6, None), (2.0001, None)]

    # Optimization
    result = minimize(lambda params: objective(*params), initial_params, bounds=bounds)

    m, s, nu = result.x
    error_model = lambda val: t.pdf(val, nu, loc=m, scale=s)
    errors = x - m
    u = t.cdf(x, nu, loc=m, scale=s)

    # Quantile function
    def eval(u_val):
        return t.ppf(u_val, nu, loc=m, scale=s)

    # Return fitted model and parameters
    fitted_model = FittedModel(None, error_model, eval, errors, u)
    return fitted_model, (m, s, nu, error_model)

# Access
# fitted_model_instance, params_and_dist = fit_general_t(data)
# Access elements of the FittedModel
# error_model = fitted_model_instance.error_model
# errors = fitted_model_instance.errors

# Access the parameters and the distribution object
# m, s, nu, distribution_func = params_and_dist


def fit_normal(x):
    # Calculate mean and standard deviation
    m = np.mean(x)
    s = np.std(x)

    # Create the error model based on the normal distribution
    error_model = lambda val: norm.pdf(val, m, s)

    # Calculate errors and CDF values
    errors = x - m
    u = norm.cdf(x, m, s)

    # Function to evaluate the quantile
    def eval(u_val):
        return norm.ppf(u_val, m, s)

    # Return the FittedModel object
    return FittedModel(None, error_model, eval, errors, u)


# Generalized T for portfolios A and B, normal for C

# First find the stocks in portfolios A and B respectively
stocks_in_portfolio_A = portfolio[portfolio['Portfolio'] == 'A']['Stock'].tolist()
stocks_in_portfolio_B = portfolio[portfolio['Portfolio'] == 'B']['Stock'].tolist()
stocks_in_portfolio_C = portfolio[portfolio['Portfolio'] == 'C']['Stock'].tolist()


# Next we want to extract returns data for each stock
returns_A = arithmetic_returns[stocks_in_portfolio_A]  # for A
returns_B = arithmetic_returns[stocks_in_portfolio_B]  # for B
returns_C = arithmetic_returns[stocks_in_portfolio_C]  # for C

# Method for handling 0 encounters in returns data
epsilon = 1e-10  # define a very small epsilon that shouldn't change our fitted model
returns_A = returns_A.replace(0, epsilon)  # replace 0 with epsilon
returns_B = returns_B.replace(0, epsilon)  # replace 0 with epsilon

# Fit the models accordingly
fitted_models_A = {stock: fit_general_t(returns_A[stock].dropna())[0] for stock in stocks_in_portfolio_A}
fitted_models_B = {stock: fit_general_t(returns_B[stock].dropna())[0] for stock in stocks_in_portfolio_B}
fitted_models_C = {stock: fit_normal(returns_C[stock].dropna()) for stock in stocks_in_portfolio_C}

# Calculate VaR and ES of each portfolio
alpha = 0.05

# Extracting holdings
holdings_A = portfolio[portfolio['Portfolio'] == 'A'][['Stock', 'Holding']]
holdings_B = portfolio[portfolio['Portfolio'] == 'B'][['Stock', 'Holding']]
holdings_C = portfolio[portfolio['Portfolio'] == 'C'][['Stock', 'Holding']]

# Compute total holdings for weight normalization
total_holdings_A = holdings_A['Holding'].sum()
total_holdings_B = holdings_B['Holding'].sum()
total_holdings_C = holdings_C['Holding'].sum()

# Using copula

# (1) Construct the copula

U = pd.DataFrame()  # construct U

# Combine all fitted models from the three portfolios
all_fitted_models = {**fitted_models_A, **fitted_models_B, **fitted_models_C}

# Populate U with the standard normal transformed values
for stock, model in all_fitted_models.items():
    # Transform the CDF values to standard normal using the normal quantile function
    U[stock] = norm.ppf(model.u)

R = U.corr(method='spearman')  # compute Spearman correlation matrix


def is_psd(A, tol=1e-8):
    """
    Returns true if A is a PSD matrix
    :param: A: correlation matrix we want to confirm is PSD
    :param: tol: tolerance to check value of eigenvalues against. If the eigenvalues are all greater than the negative of
    the tolerance, we consider the correlation matrix PSD.
    :returns: Boolean indicating whether A is a PSD matrix
    """
    eigenvalues = np.linalg.eigvalsh(A)
    return np.all(eigenvalues > -tol)


if is_psd(R):
    print('Corr Matrix is PSD')
else:
    print('Corr Matrix is NOT PSD')


# Simulation

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
    out = (B @ r)
    print(out.shape)

    # Add the mean
    for i in range(n):
        out[:, i] += _mean[i]

    return out


nsim = 5000
# Step 1: Simulate standard normals
standard_normals = simulate_pca(R, nsim)
standard_normals = standard_normals.T

# Step 2: Convert Standard Normals to Uniforms
simU = pd.DataFrame(norm.cdf(standard_normals), columns=all_fitted_models.keys())

# Step 3: Transform Uniforms to Simulated Returns
simulatedReturns = pd.DataFrame()

for stock, model in all_fitted_models.items():
    simulatedReturns[stock] = model.eval_func(simU[stock])

# Portfolio Valuation

# Create DataFrame for iterations
iterations_df = pd.DataFrame({'iteration': range(0, nsim)})

# Cross-join portfolio with iterations
# This creates a row for each stock-iteration combination
values = portfolio.assign(key=1).merge(iterations_df.assign(key=1), on='key').drop('key', axis=1)

# Extracting the current prices
current_prices = prices.iloc[-1].drop('Date')  # Assuming the first column is 'Date'

# Calculate current value, simulated value and PnL for each stock-iteration combination
values['currentValue'] = values.apply(lambda row: row['Holding'] * current_prices[row['Stock']], axis=1)
values['simulatedValue'] = (values.apply
                            (lambda row: row['Holding'] * current_prices[row['Stock']] *
                                (1.0 + simulatedReturns.loc[row['iteration'], row['Stock']]), axis=1))
values['pnl'] = values['simulatedValue'] - values['currentValue']


# Calculation of Risk Metrics
def VaR(a, alpha=0.05):
    x = np.sort(a)
    nup = int(np.ceil(len(a) * alpha))
    ndn = int(np.floor(len(a) * alpha))
    v = 0.5 * (x[nup] + x[ndn])

    return -v


def ES(a, alpha=0.05):
    x = np.sort(a)
    nup = int(np.ceil(len(a) * alpha))
    ndn = int(np.floor(len(a) * alpha))
    v = 0.5 * (x[nup] + x[ndn])
    es = np.mean(x[x<=v])
    return -es


# Stock Level

# Group the values DataFrame by 'Stock'
grouped = values.groupby('Stock')

# Calculation of Risk Metrics for each stock
stockRisk = grouped.agg(
    currentValue=('currentValue', lambda x: x.iloc[0]),
    VaR95=('pnl', lambda x: VaR(x, alpha=0.05)),
    ES95=('pnl', lambda x: ES(x, alpha=0.05)),
    # VaR99=('pnl', lambda x: VaR(x, alpha=0.01)),
    # ES99=('pnl', lambda x: ES(x, alpha=0.01)),
    Standard_Dev=('pnl', np.std),
    min=('pnl', 'min'),
    max=('pnl', 'max'),
    mean=('pnl', 'mean')
)

# Portfolio Level
# All stocks

# Group by iteration
grouped_by_iteration = values.groupby('iteration')

# Aggregate totals per simulation iteration
total_values = grouped_by_iteration.agg(
    currentValue=('currentValue', 'sum'),
    simulatedValue=('simulatedValue', 'sum'),
    pnl=('pnl', 'sum')
)


# Function to filter and aggregate values for a specific portfolio
def aggregate_portfolio_values(values, holdings):
    # Filter 'values' to include only stocks in the specific portfolio
    portfolio_values = values[values['Stock'].isin(holdings['Stock'])]

    # Group by 'iteration' and aggregate
    grouped = portfolio_values.groupby('iteration')
    total_values = grouped.agg(
        currentValue=('currentValue', 'sum'),
        simulatedValue=('simulatedValue', 'sum'),
        pnl=('pnl', 'sum')
    )
    return total_values


# Calculate total values for each portfolio
total_values_A = aggregate_portfolio_values(values, holdings_A)
total_values_B = aggregate_portfolio_values(values, holdings_B)
total_values_C = aggregate_portfolio_values(values, holdings_C)


def calculate_total_risk(total_values):
    total_risk = {
        'currentValue': total_values['currentValue'].iloc[0],
        'VaR95': VaR(total_values['pnl'], alpha=0.05),
        'ES95': ES(total_values['pnl'], alpha=0.05),
        # 'VaR99': VaR(total_values['pnl'], alpha=0.01),
        # 'ES99': ES(total_values['pnl'], alpha=0.01),
        'Standard_Dev': np.std(total_values['pnl']),
        'min': np.min(total_values['pnl']),
        'max': np.max(total_values['pnl']),
        'mean': np.mean(total_values['pnl'])
    }
    return total_risk


# Calculate the total risk for each portfolio
total_risk_A = calculate_total_risk(total_values_A)
total_risk_B = calculate_total_risk(total_values_B)
total_risk_C = calculate_total_risk(total_values_C)
totalRisk = calculate_total_risk(total_values)

# Convert risk dictionaries to DataFrames
totalRisk_df = pd.DataFrame([totalRisk])
totalRisk_df['portfolio'] = 'Total'

total_risk_A_df = pd.DataFrame([total_risk_A])
total_risk_A_df['portfolio'] = 'A'

total_risk_B_df = pd.DataFrame([total_risk_B])
total_risk_B_df['portfolio'] = 'B'

total_risk_C_df = pd.DataFrame([total_risk_C])
total_risk_C_df['portfolio'] = 'C'

# Concatenate the DataFrames
riskOut = pd.concat([stockRisk, totalRisk_df, total_risk_A_df, total_risk_B_df, total_risk_C_df])
# Change the index for the last four rows
new_indices = list(riskOut.index[:-4]) + ['Total', 'A', 'B', 'C']
riskOut.index = new_indices