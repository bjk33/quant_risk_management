import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Problem 1 #

# Routine for calculating an exponentially weighted covariance matrix

# Load and clean data
daily_return_path = '/Users/brandonkaplan/Desktop/FINTECH545/Week03_Project/DailyReturn.csv'
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
    weights /= np.sum(weights)

    # Step 4: Compute the covariance matrix: covariance[i,j] = (w dot x)' * x where dot denotes element-wise mult
    weighted_x = x * weights[:, np.newaxis]  # broadcast weights to each row
    cov_matrix = np.dot(weighted_x, weighted_x.T)  # compute the matrix product
    return cov_matrix


# Create function to calculate the percentage of variance explained by PCA

def PCA_pctExplained(a):
    """Compute the percentage of variance explained by PCA.
    :param a: an exponentially weighted covariance matrix
    """
    vals, vecs = np.linalg.eigh(a)  # get eigenvalues and eigenvectors
    vals = np.flip(np.real(vals))  # flip order to descending
    total_vals = np.sum(vals)  # sum eigenvalues
    out = np.cumsum(vals) / total_vals
    return out


# Test functions with different lambda values
lambdas = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
pctExplained = pd.DataFrame()

for lambda_ in lambdas:
    covar = ewCovar(daily_returns.values, lambda_)
    expl = PCA_pctExplained(covar)
    pctExplained[f'λ={lambda_}'] = expl

# Prepare the data for plotting
pctExplained['x'] = range(1, len(expl) + 1)
# Set of distinct colors for better visibility
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

plt.figure(figsize=(12, 8))

# Plotting with distinct colors
for i, lambda_ in enumerate(lambdas):
    plt.plot(pctExplained['x'], pctExplained[f'λ={lambda_}'],
             label=f'λ={lambda_}', color=colors[i % len(colors)])

plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('% Explained by Eigenvalue (Direct Method with Distinct Colors)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

"""
From the plot it is evident that the value of lambda and the number of principal components necessary to explain the 
variance have an *inverse* relationship. A lower lambda thus implies that a greater amount of variance is explained by
the first principal component (eigenvalue) than a higher lambda. This is because more weight is added to more recent
observations (see exponential smoothing model in Week_03 notes). As a lower lambda places greater emphasis on recent
data, the covariance matrix is in turn more influenced by recent trends or fluctuations in prices. Since PCA identifies
the principal components (directions in which the data varies the most) and more weight is given to the most recent
data, any variance here will be more pronounced in the principal components. Thus, because the covariance matrix is
shaped more so by the recent trends and fluctuations, the lower lambda will allow the first principal components to 
explain a greater amount of the variance.
"""