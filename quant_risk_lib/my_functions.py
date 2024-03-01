from cmath import sqrt
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import skew, kurtosis, norm, t, multivariate_normal, gaussian_kde
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# Preamble


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


# (1) Covariance estimation techniques


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
    # Step 3: Normalize weights to 1
    weights /= np.sum(weights)

    # Step 4: Compute the covariance matrix: covariance[i,j] = (w dot x)' * x where dot denotes element-wise mult
    weighted_x = x * weights[:, np.newaxis]  # broadcast weights to each row
    cov_matrix = np.dot(weighted_x.T, weighted_x)  # compute the matrix product
    return cov_matrix


# (2) Non-PSD fixes for correlation matrix


def chol_psd(root, a):
    """
    Cholesky factorization of input matrix assuming that matrix is positive semi-definite (PSD).
    :param root: input matrix modified in place to store the result of the Cholesky factorization
    :param a: input matrix assumed to be PSD that is to be Cholesky factorized

    Initialization:

    'n = a.shape[0]': Determines the size of the matrix a, specifically its number of rows. As a is assumed to be
    square, 'n' represents both the number of rows and columns.

    'root.fill(0.0)': Initializes the "root" matrix, which will be used to store the Cholesky factor, with zeros.

    Column-wise Processing:

    The 'for j in range(n)' loop iterates over each column of the matrix.

    Calculation of Diagonal Elements:

    Within the loop, the function first calculates the sum of squares of the elements above the diagonal in the current
    column ('s = np.dot(root[j, :j], root[j, :j])').

    'temp = a[j, j] - s:' Computes the value for the diagonal element in the root matrix by subtracting the sum s from
    the diagonal element of "a" at the current column. This subtraction is a critical step in the Cholesky
    factorization.

    The, 'if 0 >= temp >= -1e-8:' conditional checks and handles numerical stability by setting very small negative
    numbers to zero, which is particularly important for PSD matrices.

    Setting the Diagonal and Off-diagonal Elements:

    'root[j, j] = np.sqrt(temp)': Assigns the diagonal element in the root matrix, which is the square root of temp.
    This is a fundamental operation in Cholesky decomposition.

    The 'if root[j, j] == 0.0': condition checks if the diagonal element is zero and, if so, sets the remaining elements
    in the column to zero. This step is crucial for handling cases where the matrix is not full rank.

    In the else: block, the function updates the off-diagonal elements in the current column. Each element is computed
    as a scaled difference between the corresponding element in a and a dot product
    ('s = np.dot(root[i, :j], root[j, :j])').

    Completion of Cholesky Factorization:

    This process is repeated for each column, gradually building up the root matrix, which is the lower triangular
    Cholesky factor of the input matrix "a."
    """
    n = a.shape[0]
    # Initialize the root matrix with 0 values
    root.fill(0.0)

    # Loop over columns
    for j in range(n):
        s = 0.0
        # If we are not on the first column, calculate the dot product of the preceding row values.
        if j > 0:
            s = np.dot(root[j, :j], root[j, :j])

        # Diagonal element
        temp = a[j, j] - s
        if 0 >= temp >= -1e-8:
            temp = 0.0
        root[j, j] = np.sqrt(temp)

        # Check for the 0 eigenvalue. Just set the column to 0 if we have one.
        if root[j, j] == 0.0:
            root[j, j + 1:n] = 0.0
        else:
            # Update off-diagonal rows of the column
            ir = 1.0 / root[j, j]
            for i in range(j + 1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (a[i, j] - s) * ir


def near_psd(a, epsilon=0.0):
    """
    Creates a near positive semi-definite (PSD) matrix from a non-PSD square matrix
    :param a: a non-PSD square matrix to adjust.
    :param epsilon: a number we want to adjust the eigenvalues of the input matrix to be at least.
    :return: out: a near PSD matrix.

    Initial Setup: The function starts by copying the input matrix "a" to "out." If the diagonal elements of out are not
    all approximately 1 (indicating it might be a covariance matrix), it normalizes out to a correlation matrix using
    the inverse of the square root of the diagonal elements.

    Spectral Decomposition: The function then computes the eigenvalues and eigenvectors of "out" (assuming it's now a
    correlation matrix). The eigenvalues are adjusted to be at least epsilon to ensure non-negativity, which is a key
    property of PSD matrices.

    Reconstruction: The matrix is reconstructed using the adjusted eigenvalues and eigenvectors. This involves scaling
    the eigenvectors by the square root of the reciprocal of their dot product with the adjusted eigenvalues, and then
    by the square root of the eigenvalues. The final matrix out is obtained by multiplying this matrix by its transpose.

    Reverting to Original Scale: If the original matrix "a" was a covariance matrix (indicated by invSD not being None),
    the function scales out back to the original scale of "a."

    Return: The function returns the modified matrix "out," which is now a near PSD matrix.
    """
    n = a.shape[0]

    invSD = None
    out = a.copy()

    # Calculate the correlation matrix if we got a covariance matrix
    if not np.allclose(np.diag(out), 1.0):
        invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = invSD @ out @ invSD

    # SVD, update the eigenvalue and scale
    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    T = 1.0 / (vecs * vecs @ vals)
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = T @ vecs @ l
    out = B @ B.T

    # Add back the variance
    if invSD is not None:
        invSD = np.diag(1.0 / np.diag(invSD))
        out = invSD @ out @ invSD

    return out


# Implement Higham's (2002) nearest PSD correlation function

# Helper Functions
def _getAplus(A):
    """Compute the nearest PSD matrix to A by setting negative eigenvalues to zero.
    :param: A: An NxN numpy matrix assumed to be non-PSD
    :return: An NxN numpy matrix that is a reconstructed "A" with modified (non-negative) eigenvalues and the original
    eigenvectors
    """
    vals, vecs = np.linalg.eigh(A)
    vals[vals < 0] = 0
    return vecs @ np.diag(vals) @ vecs.T


def _getPS(A, W):
    """Perform a weighted adjustment of "A" to make it closer to being PSD. The function computes the nearest PSD matrix
    to 'W05 * A * W05' and then scales it back using the inverse of 'W05.'
    :param: A: An NxN numpy matrix assumed to be non-PSD
    :param: W: An NxN numpy weight matrix
    :return: Scaled nearest PSD matrix to 'W05 * A * W05'
    """
    W05 = np.sqrt(W)
    iW = np.linalg.inv(W05)
    return iW @ _getAplus(W05 @ A @ W05) @ iW


def _getPu(A, W):
    """Adjust the diagonal of A to 1, maintaining the correlation matrix requirement.
    :param: A: An NxN numpy matrix assumed to be non-PSD
    :param: W: An NxN numpy weight matrix
    :return: Aret: Adjusted "A" matrix with 1 on the diagonal.
    """
    Aret = A.copy()
    np.fill_diagonal(Aret, 1)
    return Aret


def wgtNorm(A, W):
    """Compute a weighted norm of matrix A using weight matrix W. This is used to check for convergence in the Higham
    method.
    :param: A: An NxN numpy matrix assumed to be non-PSD
    :param: W: An NxN numpy weight matrix
    :return: Weighted norm of A
    """
    W05 = np.sqrt(W) @ A @ np.sqrt(W)
    return np.sum(W05 * W05)


def higham_nearestPSD(pc, W=None, epsilon=1e-9, maxIter=100, tol=1e-9):
    """Implement Higham's algorithm to find the nearest PSD correlation matrix. The function iteratively adjusts a given
    matrix "pc" to make it a near PSD matrix, using the alternating projection method.
    :param: pc: An NxN numpy matrix assumed to be non-PSD
    :param: W: An NxN weight matrix
    :param: epsilon: The tolerance for negative eigenvalues. Default is 1e-9. Anything smaller (greater in absolute value)
     than -1e-9 is considered a negative. Anything greater (smaller in absolute value) is considered 0 to account for
     rounding errors.
    :param: maxIter: The maximum number of iterations. Default is 100.
    :param: tol: The tolerance for convergence of the algorithm. Default is 1e-9.
    :return: Yk: An NxN numpy matrix representing the nearest PSD correlation matrix to "pc." It is approximate and
    considered the nearest PSD correlation matrix by the criteria of minimizing the Froebenius norm. If the Froebenius
    norm is less than "tol" we consider "Yk" the sufficiently nearest PSD correlation matrix to "pc."
    """
    n = pc.shape[0]
    if W is None:
        W = np.diag(np.ones(n))

    deltaS = 0
    Yk = pc.copy()
    norml = np.finfo(np.float64).max
    i = 1

    while i <= maxIter:
        Rk = Yk - deltaS
        Xk = _getPS(Rk, W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W)
        norm = wgtNorm(Yk - pc, W)
        minEigVal = np.min(np.real(np.linalg.eigvals(Yk)))

        if abs(norm - norml) < tol and minEigVal > -epsilon:
            break
        norml = norm
        i += 1

    if i < maxIter:
        print(f"Converged in {i} iterations.")
    else:
        print("Convergence failed after {} iterations".format(i - 1))
    return Yk


def is_psd(A, tol=1e-9):
    """
    Returns true if A is a PSD matrix
    :param: A: correlation matrix we want to confirm is PSD
    :param: tol: tolerance to check value of eigenvalues against. If the eigenvalues are all greater than the negative of
    the tolerance, we consider the correlation matrix PSD.
    :returns: Boolean indicating whether A is a PSD matrix
    """
    eigenvalues = np.linalg.eigvalsh(A)
    return np.all(eigenvalues > -tol)


# Simulation

# Implementing a multivariate normal simulation directly from covariance matrix and using PCA

# Directly from covariance matrix


def simulate_normal(N, cov, mean=None, seed=1234):
    """
    Simulate a  multivariate normal distribution directly from a covariance matrix. We use chol_psd() to Cholesky
    factorize an input covariance matrix. This is used to transform standard normal variables into variables with the
    desired covariance structure.
    :param N: The number of samples to generate
    :param cov: The covariance matrix based on which the multivariate normal samples are generated
    :param mean: An optional mean vector to use. If not provided the mean is zero
    :param seed: An optional seed to use for the random number generation in order to ensure reproducibility
    :return: out.T: The matrix of generated samples
    """
    n, m = cov.shape
    if n != m:
        raise ValueError(f"Covariance Matrix is not square ({n},{m})")

    if mean is None:
        mean = np.zeros(n)
    elif len(mean) != n:
        raise ValueError(f"Mean ({len(mean)}) is not the size of cov ({n},{n})")

    # Cholesky Decomposition

    # Attempt standard Cholesky decomposition
    try:
        l = np.linalg.cholesky(cov).T  # NumPy returns upper triangular
    except np.linalg.LinAlgError:  # If not PD check PSD and then use chol_psd()
        if not is_psd(cov):
            raise ValueError("Covariance matrix is not positive semi-definite.")
        else:
            l = np.zeros_like(cov)
            chol_psd(l, cov)

    # Generate random standard normals
    np.random.seed(seed)
    d = stats.norm.rvs(size=(n, N))

    # Multiply generated standard normal variables by the Cholesky factor to get variables with desired covariance
    out = np.dot(l, d)

    # Add the mean
    for i in range(n):
        out[i, :] += mean[i]

    return out.T

# Simulate from PCA

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

# (4) VaR calculation


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

# VaR: Five Ways


def calculate_var_normal(returns, alpha):
    """
    Calculate the Value at Risk (VaR) for mean-centered returns using a normal distribution.

    Parameters:
    :param: returns (Series): Pandas Series of mean-centered returns for the asset.
    :param: alpha (float): The % worst bad day you want to calculate VaR for the returns. Alpha is equal to 1-minus the
    confidence level. Alpha is the % worst case scenario. (E.g., 0.05 for a day worse than 95% of typical days)

    Returns:
    :return: var: float: The calculated VaR.
    """
    std_dev = returns.std()

    # Find the z-score for the given confidence level
    z_score = norm.ppf(alpha)

    # Calculate VaR
    var = -(z_score * std_dev)
    return var


def calculate_var_ewm(returns, alpha, lambda_factor):
    """
    Calculate VaR using exponentially weighted variance.

    Parameters:
    :param: returns (Series): Pandas Series of returns.
    :param: alpha (float): The % worst bad day you want to calculate VaR for the returns. Alpha is equal to 1-minus the
    confidence level. Alpha is the % worst case scenario. (E.g., 0.05 for a day worse than 95% of typical days)
    :param: lambda_factor (float): The smoothing parameter for exponentially weighted calculations. Must lie between 0
    and 1 inclusive. A smaller lambda (close to 0) will result in greater weight to more recent observations.
    A larger lambda (close to 1) will result in greater weight to observations in the past. Exponentially weighting
    always gives more weight to more recent observations than standard normal.

    Returns:
    :return: var: float: The calculated VaR.
    """
    alpha_factor = 1 - lambda_factor
    ew_variance = returns.ewm(alpha=alpha_factor, adjust=False).var()
    ew_std_dev = np.sqrt(ew_variance.iloc[-1])  # Last value for the most recent standard deviation

    # Find the z-score for the given confidence level
    z_score = norm.ppf(alpha)

    # Calculate VaR
    var = -(z_score * ew_std_dev)
    return var


def calculate_var_mle(returns, alpha):
    """
    Calculate VaR using a MLE fitted normal distribution.

    Parameters:
    :param: returns (Series): Pandas Series of returns.
    :param: alpha (float): The % worst bad day you want to calculate VaR for the returns. Alpha is equal to 1-minus the
    confidence level. Alpha is the % worst case scenario. (E.g., 0.05 for a day worse than 95% of typical days).

    Returns:
    :return: -var: float: The calculated VaR.
    """
    # Fit the normal distribution using MLE
    mu, std = norm.fit(returns)

    # Calculate the VaR
    var = norm.ppf(alpha, mu, std)
    return -var


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
    df, loc, scale = params  # degrees of freedom, location, and scale

    # Calculate the VaR
    var = t.ppf(alpha, df, loc, scale)
    return -var


def calculate_var_ar1(returns, alpha):
    """
    Calculate VaR using a fitted AR(1) model with ARIMA from statsmodels.

    Parameters:
    :param: returns (Series): Pandas Series of returns.
    :param: alpha (float): The % worst bad day you want to calculate VaR for the returns. Alpha is equal to 1-minus the
    confidence level. Alpha is the % worst case scenario. (E.g., 0.05 for a day worse than 95% of typical days).

    Returns:
    :return: var: float: The calculated VaR.
    """
    # Fit an AR(1) model using ARIMA with d=0 (no differencing)
    ar_model = sm.tsa.ARIMA(returns, order=(1, 0, 0))
    ar_result = ar_model.fit()

    # Forecast the next return
    forecast = ar_result.params[0] + ar_result.params[1] * returns.iloc[-1]

    # Calculate the standard deviation of the residuals
    residuals_std = ar_result.resid.std()

    # Calculate VaR assuming normality of residuals
    z_score = norm.ppf(alpha)
    var = -(forecast + z_score * residuals_std)
    return var
