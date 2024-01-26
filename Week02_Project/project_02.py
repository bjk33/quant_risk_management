from cmath import sqrt
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, norm
import statsmodels.api as sm
from scipy.optimize import minimize


# Problem 1 #

# a) Calculate first four moments using normalized formulas
df1 = pd.read_csv('problem1.csv')
data1 = df1[df1.columns[0]]
n = len(data1)

mu_hat = sum(data1) / n
data1_corrected = data1 - mu_hat

sigma_2_hat = sum(data1_corrected ** 2) / (n - 1)
sigma_hat = sqrt(sigma_2_hat)

skew_hat = (n / ((n - 1) * (n - 2))) * sum((data1_corrected / sigma_hat) ** 3)

kurtosis_hat = (
        ((n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) *
         sum((data1_corrected / sigma_hat) ** 4)
         ) - (3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))

print("Formula Mean:", mu_hat)
print("Formula Variance:", sigma_2_hat)
print("Formula Skewness:", skew_hat)
print("Formula Kurtosis:", kurtosis_hat)

# b) Calculate the first four moments again using your chosen statistical package

mean = np.mean(data1)
var_biased = np.var(data1_corrected)  # biased (divide by n)
skewness_biased = skew(data1_corrected) # biased default
kurtosis_biased = kurtosis(data1_corrected)  # biased default

var = np.var(data1_corrected, ddof=1)  # unbiased (divide by n-1)
skewness = skew(data1_corrected, bias=False)  # unbiased
kurtosis = kurtosis(data1_corrected, bias=False)  # unbiased

print("Scipy Mean:", mean)

print("Scipy Biased Variance:", var_biased)
print("Scipy Biased Skewness:", skewness_biased)
print("Scipy Biased Kurtosis:", kurtosis_biased)

print("Scipy Unbiased Variance:", var)
print("Scipy Unbiased Skewness:", skewness)
print("Scipy Unbiased Kurtosis:", kurtosis)

# c) Is your statistical package function biased?

# Difference Between Moments calculated by Formula and Biased Moments Calculate Using Scipy

difference_sigma_2_hat = sigma_2_hat - var_biased
difference_skew_hat = skew_hat - skewness_biased
difference_kurtosis_hat = kurtosis_hat - kurtosis_biased

print("Difference Between Formula and Biased Variance:", difference_sigma_2_hat)
print("Difference Between Formula and Biased Skewness:", difference_skew_hat)
print("Difference Between Formula and Biased Kurtoses:", difference_kurtosis_hat)

# Difference Between Unbiased and Biased Moments Calculated Using Scipy

difference_var = var - var_biased
difference_skewness = skewness - skewness_biased
difference_kurtosis = kurtosis - kurtosis_biased

print("Difference Between Scipy Calculated Variances:", difference_var)
print("Difference Between Scipy Calculated Skewnesses:", difference_skewness)
print("Difference Between Scipy Calculated Kurtoses:", difference_kurtosis)

# Problem 2 #

# a) Fitting Data Using OLS and MLE (Given Normality Assumption) and Comparing Fit

# Read Data
data2 = pd.read_csv("problem2.csv")

# Define Variables
x = data2.iloc[:, 0]  # First column is X
y = data2.iloc[:, 1]  # Second column is Y
x = sm.add_constant(x)

# Fitting OLS
ols_model = sm.OLS(y, x).fit()

print(ols_model.summary()) #summary

# Extracting
ols_betas = ols_model.params
ols_std_err = ols_model.bse

print("OLS Betas:", ols_betas)
print("OLS Standard Error of Coefficients:", ols_std_err)

# Calculate the standard deviation (sigma) of the OLS errors
rss_ols = ols_model.ssr  # Residual sum of squares
n_ols = len(y)  # Number of observations
p_ols = x.shape[1]  # Number of parameters
sigma_ols = np.sqrt(rss_ols / (n_ols - p_ols))

print("RSS:", rss_ols)
print("OLS Sigma:", sigma_ols)  # Standard deviation of errors

# Note, we expect the MLE sigma to be equal the OLS RSS divided by the number of observations (see write up). Let's
# find out...

# Fitting the data using MLE


def mle_fit(x, y):
    # Define the negative log-likelihood function
    def negative_log_likelihood(params):
        beta, sigma = params[:-1], params[-1]
        y_pred = x @ beta
        likelihood = norm.pdf(y, loc=y_pred, scale=sigma)
        return -np.sum(np.log(likelihood))

    # Initial parameter estimates (using OLS estimates for beta and sigma)
    initial_params = np.append(ols_betas, sigma_ols)

    # Minimizing the negative log-likelihood
    result = (
        minimize(negative_log_likelihood, initial_params, method='L-BFGS-B',
                 bounds=[(None, None)] * (len(initial_params) - 1) + [(0.0001, None)]))

    return result.x


# Perform MLE fitting
mle_params = mle_fit(x, y)

# Extracting MLE estimates for coefficients
mle_betas = mle_params[:-1]

# Calculating MLE residuals and re-computing sigma
mle_residuals = y - (x @ mle_betas)
rss_mle = np.sum(mle_residuals**2)
n_mle = len(y)
sigma_mle = np.sqrt(rss_mle / n_mle)

print ("MLE Betas:", mle_betas)
print("MLE RSS:", rss_mle)
print("MLE Sigma:", sigma_mle)






