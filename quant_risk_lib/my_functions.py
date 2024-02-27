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