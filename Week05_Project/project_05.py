import pandas as pd
import numpy as np
from scipy.stats import norm, t
from scipy.special import gamma

# Problem 2 #

# Calculate VaR and Expected Shortfall

returns_path = '/Users/brandonkaplan/Desktop/FINTECH545/Week05_Project/problem1.csv'
returns = pd.read_csv(returns_path)

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
    print(params)
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

def VaR(a, alpha=0.05):
    x = np.sort(a)
    nup = int(np.ceil(len(a) * alpha))
    ndn = int(np.floor(len(a) * alpha))
    v = 0.5 * (x[nup] + x[ndn])

    return -v


def calculate_historic_var_es(returns_df, alpha):
    var = VaR(returns_df, alpha=alpha)
    x = np.sort(returns_df)
    es = -np.mean(x[x <= var])

    return var, es


var_hist, es_hist = calculate_historic_var_es(returns, alpha=0.05)
print("Series VaR - Historic Simulation:", var_hist)
print("Series ES - Historic Simulation:", es_hist)
