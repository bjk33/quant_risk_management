from cmath import sqrt
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import skew, kurtosis, norm, t, multivariate_normal, gaussian_kde
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA


def mle_fit(x, y, ols_betas, sigma_ols):
    """
    Fitting data using MLE given a normality assumption
    :param x: independent variable data column
    :param y: dependent variable data column
    :param ols_betas: model parameters of fitted OLS model
    :param sigma_ols: standard deviation of the OLS model errors
    :return: result.x: fitted data using MLE given a normality assumption
    """
    # Define the negative log-likelihood function
    def negative_log_likelihood(params):
        """
        Computes negative log likelihood given parameters
        :param params: model parameters
        :return: negative log likelihood
        """
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


def mle_fit_t(x, y, initial_betas, initial_sigma, initial_df=10):
    """
    Fitting data using MLE given Student's t-distribution assumption
    :param x: independent variable data column
    :param y: dependent variable data column
    :param initial_betas: model parameters of fitted OLS model
    :param initial_sigma: standard deviation of OLS model errors
    :param initial_df: degrees of freedom (default is 10)
    :return: fitted data using MLE given t-distribution assumption
    """
    # Define the negative log-likelihood function for T-distribution
    def negative_log_likelihood_t(params):
        betas_t = params[:x.shape[1]]
        sigma_t = params[-2]
        df = params[-1]

        y_pred = x @ betas_t
        rv = t(df)
        log_likelihood = rv.logpdf((y - y_pred) / sigma_t) - np.log(sigma_t)

        return -np.sum(log_likelihood)  # negate the log likelihood to get the negative log likelihood

    initial_params_t = np.append(initial_betas, [initial_sigma, initial_df])

    # Minimizing the negative log-likelihood for T-distribution
    result_t = minimize(negative_log_likelihood_t, initial_params_t, args=(x, y), method='L-BFGS-B',
                        bounds=[(None, None)] * (len(initial_params_t) - 2) + [(0.0001, None), (2, None)])
    return result_t.x


def calculate_log_likelihood_normal(sigma, residuals):
    """
    Compute the residual log likelihood for MLE given normal distribution assumption
     for using in goodness of fit comparisons (AIC, BIC).
    :param sigma: MLE of standard deviation of the data under model assumption
    :param residuals: difference between truth (y) and prediction (x * beta_1)
    :return: log_likelihood_normal: the residual log likelihood
    """
    n = len(residuals)
    log_likelihood_normal = (-n / 2 * np.log(2 * np.pi) - n / 2 * np.log(sigma ** 2) - 1 / (2 * sigma ** 2) *
                             np.sum(residuals ** 2))
    return log_likelihood_normal


def calculate_log_likelihood_t(df, sigma, residuals):
    """
    Compute the residual log likelihood for MLE given t-distribution assumption
    :param df: degrees of freedom of fit model
    :param sigma: MLE of standard deviation of the data under model assumption
    :param residuals: difference between truth (y) and prediction (x * beta_1)
    :return: log_likelihood_t: the residual log likelihood
    """
    rv = t(df)
    log_likelihood_t = np.sum(rv.logpdf(residuals / sigma) - np.log(sigma))
    return log_likelihood_t

