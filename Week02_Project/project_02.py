from cmath import sqrt
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, norm, t, multivariate_normal
import statsmodels.api as sm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA


# Problem 1 #

# a) Calculate the First Four Moments Using Normalized Formulas
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

# b) Calculate the First Four Moments Again Using Your Chosen Statistical Package

mean = np.mean(data1)
var_biased = np.var(data1_corrected)  # biased (divide by n)
skewness_biased = skew(data1_corrected)  # biased default
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

# c) Is Your Statistical Package Function Biased?

# Difference between moments calculated by formula and biased moments calculated using scipy

difference_sigma_2_hat = sigma_2_hat - var_biased
difference_skew_hat = skew_hat - skewness_biased
difference_kurtosis_hat = kurtosis_hat - kurtosis_biased

print("Difference Between Formula and Biased Variance:", difference_sigma_2_hat)
print("Difference Between Formula and Biased Skewness:", difference_skew_hat)
print("Difference Between Formula and Biased Kurtoses:", difference_kurtosis_hat)

# Difference between unbiased and biased moments calculated using scipy

difference_var = var - var_biased
difference_skewness = skewness - skewness_biased
difference_kurtosis = kurtosis - kurtosis_biased

print("Difference Between Scipy Calculated Variances:", difference_var)
print("Difference Between Scipy Calculated Skewnesses:", difference_skewness)
print("Difference Between Scipy Calculated Kurtoses:", difference_kurtosis)

# Problem 2 #

# a) Fitting Data Using OLS and MLE (Given Normality Assumption) and Comparing Fit

# Read data
data2 = pd.read_csv("problem2.csv")

# Define variables
x = data2.iloc[:, 0]  # first column is X
y = data2.iloc[:, 1]  # second column is Y
x = sm.add_constant(x)

# Fitting OLS
ols_model = sm.OLS(y, x).fit()

print(ols_model.summary())  # summary

# Extracting coefficients and standard errors
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
print("OLS Sigma:", sigma_ols)  # standard deviation of errors

# Note, we expect the MLE sigma to be equal the OLS RSS divided by the number of observations (see write up). Let's
# find out...

# Fitting data using MLE (given normality assumption)


def mle_fit(x, y):
    # Define the negative log-likelihood function
    def negative_log_likelihood(params):
        beta, sigma = params[:-1], params[-1]
        y_pred = x @ beta
        likelihood = norm.pdf(y, loc=y_pred, scale=sigma)
        return -np.sum(np.log(likelihood))

    # Initial parameter estimates (using OLS estimates for betas and sigma)
    initial_params = np.append(ols_betas, sigma_ols)

    # Minimizing the negative log-likelihood
    result = (
        minimize(negative_log_likelihood, initial_params, method='L-BFGS-B',
                 bounds=[(None, None)] * (len(initial_params) - 1) + [(0.0001, None)]))

    return result.x


# Fitting MLE (normal)
mle_params = mle_fit(x, y)

# Extracting MLE estimates for coefficients
mle_betas = mle_params[:-1]

# Calculating MLE residuals and computing sigma
mle_residuals = y - (x @ mle_betas)
rss_mle = np.sum(mle_residuals**2)
n_mle = len(y)
sigma_mle = np.sqrt(rss_mle / n_mle)

print("MLE (normal) Betas:", mle_betas)
print("MLE (normal) RSS:", rss_mle)
print("MLE (normal) Sigma:", sigma_mle)
print("MLS Sigma if Calculated Using OLS RSS:", np.sqrt(rss_ols / n_ols))


# b) Fitting Data Using MLE (Given Student's t-Distribution Assumption)

# Define the negative log-likelihood function for T-distribution
def negative_log_likelihood_t(params, x, y):
    betas_t = params[:x.shape[1]]
    sigma_t = params[-2]
    df = params[-1]

    y_pred = x @ betas_t
    rv = t(df)
    log_likelihood = rv.logpdf((y - y_pred) / sigma_t) - np.log(sigma_t)

    return -np.sum(log_likelihood)  # negate the negative log likelihood to get log likelihood


# Initial parameter estimates: OLS estimates for beta, and initial guesses for sigma and df
initial_beta = ols_model.params.values
initial_sigma = sigma_ols
initial_df = 10  # starting with an arbitrary value for degrees of freedom
initial_params_t = np.append(initial_beta, [initial_sigma, initial_df])

# Minimizing the negative log-likelihood for T-distribution
result_t = minimize(negative_log_likelihood_t, initial_params_t, args=(x, y), method='L-BFGS-B',
                    bounds=[(None, None)] * (len(initial_params_t) - 2) + [(0.0001, None), (2, None)])

# Extracting MLE estimates for T-distribution model
mle_params_t = result_t.x
mle_beta_t = mle_params_t[:-2]
mle_residuals_t = y - (x @ mle_params_t[:-2])
sigma_mle_t = mle_params_t[-2]
mle_df_t = mle_params_t[-1]

print("MLE (t-dist) Betas:", mle_beta_t)
print("MLE (t-dist) Sigma:", sigma_mle_t)
print("MLE (t-dist) Degrees of Freedom:", mle_df_t)

# Comparison of fitted parameters using AIC and BIC

# Function to calculate log-likelihood for MLE with normal distribution of errors


def calculate_log_likelihood_normal(sigma, residuals):
    n = len(residuals)
    log_likelihood = (-n / 2 * np.log(2 * np.pi) -
                      n / 2 * np.log(sigma ** 2) - 1 / (2 * sigma ** 2) * np.sum(mle_residuals ** 2))
    return log_likelihood

# Function to calculate log-likelihood for MLE with T-distribution of errors


def calculate_log_likelihood_t(df, sigma, residuals):
    rv = t(df)
    log_likelihood = np.sum(rv.logpdf(residuals / sigma) - np.log(sigma))
    return log_likelihood


# Calculate log-likelihoods
log_likelihood_normal_value = calculate_log_likelihood_normal(sigma_mle, mle_residuals)
log_likelihood_t_value = calculate_log_likelihood_t(mle_df_t, sigma_mle_t, mle_residuals_t)

# Number of parameters in each model
n_params_normal = len(mle_params)
n_params_t = len(mle_params_t)

# AIC and BIC for MLE with normal distribution
aic_normal_mle = 2 * n_params_normal - 2 * log_likelihood_normal_value
bic_normal_mle = np.log(len(y)) * n_params_normal - 2 * log_likelihood_normal_value

# AIC and BIC for MLE with T-distribution
aic_t_mle = 2 * n_params_t - 2 * log_likelihood_t_value
bic_t_mle = np.log(len(y)) * n_params_t - 2 * log_likelihood_t_value

print("AIC MLE (Normal): ", aic_normal_mle)
print("BIC MLE (Normal): ", bic_normal_mle)
print("AIC MLE (T): ", aic_t_mle)
print("BIC MLE (T): ", bic_t_mle)


# c) Determine Conditional Distribution By Fitting Data Using Multivariate Normal MLE

# Read data
X = pd.read_csv('problem2_x.csv').to_numpy()
X1 = pd.read_csv('problem2_x1.csv').to_numpy()

# Fit a multivariate normal distribution to X
mean = np.mean(X, axis=0)
cov = np.cov(X, rowvar=False)
mvn = multivariate_normal(mean=mean, cov=cov)

# Derive the conditional distribution of X2 given X1
mean_X1 = mean[0]
mean_X2 = mean[1]
cov_X1_X1 = cov[0, 0]
cov_X2_X2 = cov[1, 1]
cov_X1_X2 = cov[0, 1]

# For each observed value of X1, calculate the mean and variance of the conditional distribution of X2
conditional_means = mean_X2 + cov_X1_X2 / cov_X1_X1 * (X1 - mean_X1)
conditional_var = cov_X2_X2 - cov_X1_X2 ** 2 / cov_X1_X1

# Flatten X1 and conditional_means, ensuring it is one-dimensional for plt.errorbar
X1_flat = X1.flatten()
conditional_means_flat = conditional_means.flatten()

# Creating an error array that is the same length as X1_flat, filled with the calculated standard error
error = 1.96 * np.sqrt(conditional_var)  # This is the standard error (multiply by 1.96 for 95% CI)


# Plot the expected value along with the 95% confidence interval for X2 given X1
plt.figure(figsize=(10, 6))
plt.errorbar(X1_flat, conditional_means_flat, yerr=error, fmt='o', label='95% CI for X2')
plt.xlabel('X1')
plt.ylabel('Expected X2')
plt.title('Conditional Expectation of X2 Given X1 with 95% CI')
plt.legend(loc='lower right')
plt.show()


# Problem 3 #

# Fit Data Using AR(1) Through AR(3) and MA(1) Through MA(3) respectively. Which is the Best of Fit?

