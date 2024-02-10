import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Problem 1 #

# Routine for calculating an exponentially weighted covariance matrix

# Load and clean data
daily_return_path = '/Users/brandonkaplan/Desktop/Week03_Project/daily_return.csv'
daily_returns = pd.read_csv(daily_return_path)
daily_returns = daily_returns.drop(daily_returns.columns[0], axis=1)  # drop indices

# Create function for calculating exponentially weighted covariance matrix


def ewCovar(x, lambda_):
    """Compute exponentially weighted covariance matrix of a dataframe.
    :param x: a pandas dataframe
    :param lambda_: smoothing parameter
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
    weights /= weights.sum(weights)

    # Step 4: Compute the covariance matrix: covariance[i,j] = (w dot x)' * x where dot denotes element-wise mult
    weighted_x = x * weights[:, np.newaxis]  # broadcast weights to each row
    cov_matrix = np.dot(weighted_x, weighted_x.T)  # compute the matrix product
    return cov_matrix


# Create function to calculate the percentage of variance explained by PCA

def PCA_pctExplained(a):
    """Compute the percentage of variance explained by PCA.
    :param a: an exponentially weighted covariance matrix
    """
