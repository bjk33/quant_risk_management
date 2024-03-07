from cmath import sqrt
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import skew, kurtosis, norm, t, multivariate_normal, gaussian_kde
from scipy.optimize import minimize
import statsmodels.api as sm


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


def missing_cov(x, skip_miss=True, fun=np.cov):
    n, m = x.shape
    n_miss = x.isnull().sum(axis=0)

    # nothing missing, just calculate it
    if n_miss.sum() == 0:
        return fun(x, rowvar=False)

    idx_missing = [set(x.index[x[col].isnull()]) for col in x.columns]

    if skip_miss:
        # Skipping Missing, get all the rows which have values and calculate the covariance
        rows = set(range(n))
        for c in range(m):
            rows -= idx_missing[c]
        rows = sorted(rows)
        return fun(x.iloc[rows, :], rowvar=False)
    else:
        # Pairwise, for each cell, calculate the covariance
        out = np.empty((m, m))
        for i in range(m):
            for j in range(i + 1):
                rows = set(range(n))
                for c in (i, j):
                    rows -= idx_missing[c]
                rows = sorted(rows)
                sub_matrix = fun(x.iloc[rows, [i, j]], rowvar=False)
                out[i, j] = sub_matrix[0, 1]
                if i != j:
                    out[j, i] = out[i, j]
        return out


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

# (4) VaR and ES calculation

def VaR(a, alpha=0.05):
    """
    Calculate the Value at Risk (VaR) for a given array of financial data. Used for Historic Simulation on a single
    return series in Project_04.

    Parameters:
    :param: a (array-like): An array of historical financial data (e.g., returns or prices).
    :param: alpha (float, optional): The confidence level, representing the probability of
                             exceeding the VaR. Default value is 0.05 (95% confidence level).

    Returns:
    :return: -v: float: The calculated Value at Risk (VaR). The value is returned as a negative
           number, indicating a potential loss in the context of the given confidence level.
    """
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
    if date_column not in prices_df.columns:
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


def calculate_var_hist_KDE(returns, prices, stock, nsim, alpha):
    """
    Calculates historical VaR by sampling from historical returns. Uses a Gaussian Kernel Density Estimator to smooth
    returns data.
    :param returns: Pandas DataFrame: Returns data of a portfolio.
    :param prices: Pandas DataFrame: Price data for stock
    :param stock: str: Stock name from portfolio you want to find current price for
    :param nsim: int: Number of samples from historical returns
    :param alpha: float: Confidence level/probability of VaR break (0.05 implies 95% confidence)
    :return: var_hist: float: The calculated VaR in dollars
    :return: var_hist_return: The calculated VaR expressed as a return
    """
    current_price = prices[stock].iloc[-1]  # This is the most recent META price

    # Fit KDE to the returns
    kde = gaussian_kde(returns)

    # Generate random numbers from the KDE
    sampled_returns = kde.resample(nsim)[0]

    # Calculate new prices from the returns
    new_prices = (1 + sampled_returns) * current_price

    # Calculate VaR
    var = np.percentile(new_prices, (alpha) * 100)
    var_hist = current_price - var
    var_hist_ret = var_hist / current_price
    return var_hist_ret, var_hist


def calculate_portfolio_values(prices_df, portfolio_df):
    """
    Calculate the total value of each portfolio.

    Parameters:
    :param: prices_df (Pandas DataFrame): Data frame of stock prices
    :param: portfolio_df (Pandas DataFrame): Data frame containing different portfolios and their holdings of the stocks

    Returns:
    :return: pd.Series: Total value of each portfolio.
    """

    # Get the latest prices for each stock
    latest_prices = prices_df.iloc[-1, 1:]  # Skipping the 'Date' column
    latest_prices = latest_prices.astype(float)  # Convert prices to float

    # Pivot the portfolio DataFrame to align with the stock symbols
    restructured_portfolio = (portfolio_df.pivot_table(index='Stock', columns='Portfolio', values='Holding',
                                                       fill_value=0))

    # Calculate the current value of each stock in each portfolio
    portfolio_values = restructured_portfolio.multiply(latest_prices, axis=0)

    # Calculate the total value of each portfolio
    portfolio_totals = portfolio_values.sum(axis=0)

    return portfolio_totals


def calculate_portfolio_weights(prices_df, portfolio_df, portfolio_totals):
    """
    Calculate the weights of each stock in each portfolio.

    :param prices_df: DataFrame with stock prices
    :param portfolio_df: DataFrame with portfolio holdings (quantity of shares)
    :param portfolio_totals: Series with total value of each portfolio
    :return: DataFrame with weights of each stock in each portfolio
    """
    # Get the latest prices for each stock
    latest_prices = prices_df.iloc[-1, 1:]  # Assuming the last row contains the latest prices

    # Pivot the portfolio DataFrame to align with the stock symbols
    restructured_portfolio = portfolio_df.pivot_table(index='Stock', columns='Portfolio', values='Holding',
                                                      fill_value=0)

    # Calculate the dollar value of each holding
    dollar_values = restructured_portfolio.multiply(latest_prices, axis=0)

    # Identify the row(s) with NaN values
    nan_rows = dollar_values[dollar_values.isna().all(axis=1)]

    # If there's exactly one row with NaN values, and it's the extra row, remove it
    if len(nan_rows) == 1:
        dollar_values = dollar_values.dropna()
    # Calculate weights
    weights = dollar_values.divide(portfolio_totals)
    return weights


def delta_normal_var(portfolio_df, prices_df, returns_df, lambda_factor, z_score):
    """
    Calculate the Delta Normal VaR for each portfolio and total.

    :param portfolio_df: DataFrame with portfolio holdings (quantity of shares)
    :param prices_df: DataFrame with stock prices
    :param returns_df: DataFrame with stock returns
    :param lambda_factor: Lambda factor for exponentially weighted covariance
    :param z_score: Z-Score for the desired confidence level
    :return: Dictionary with VaR for each portfolio
    """
    var_values = {}
    total_holdings = {}
    total_portfolio_value = 0

    for portfolio in portfolio_df['Portfolio'].unique():
        holdings = portfolio_df[portfolio_df['Portfolio'] == portfolio]
        relevant_stocks = holdings['Stock'].tolist()
        current_prices = prices_df[relevant_stocks].iloc[-1]
        filtered_returns = returns_df[relevant_stocks]

        # Calculate portfolio value
        portfolio_value = (sum(holdings[holdings['Stock'] == stock]['Holding'].iloc[0] * current_prices[stock] for
                               stock in relevant_stocks))
        # And total value
        total_portfolio_value += portfolio_value

        # Aggregate holdings for total portfolio calculation
        for stock in relevant_stocks:
            if stock not in total_holdings:
                total_holdings[stock] = 0
            total_holdings[stock] += holdings[holdings['Stock'] == stock]['Holding'].iloc[0] * current_prices[stock]

        # Calculate delta (portfolio weights)
        delta = (np.array([holdings[holdings['Stock'] == stock]['Holding'].iloc[0] * current_prices[stock] /
                           portfolio_value for stock in relevant_stocks]))

        # Calculate covariance matrix
        cov_matrix = ewCovar(filtered_returns, lambda_factor)

        # Portfolio standard deviation
        portfolio_std_dev = np.sqrt(np.dot(delta.T, np.dot(cov_matrix, delta)))

        # Calculate VaR
        var = -portfolio_value * norm.ppf(z_score) * portfolio_std_dev
        var_values[portfolio] = var

    # Total portfolio VaR calculation
    relevant_stocks = list(total_holdings.keys())
    total_delta = np.array([total_holdings[stock] / total_portfolio_value for stock in relevant_stocks])
    total_cov_matrix = ewCovar(returns_df[relevant_stocks], lambda_factor)
    total_portfolio_std_dev = np.sqrt(np.dot(total_delta.T, np.dot(total_cov_matrix, total_delta)))
    total_var = -total_portfolio_value * norm.ppf(z_score) * total_portfolio_std_dev
    var_values['Total'] = total_var

    return var_values


def calculate_historical_var_kde(value_changes, alpha):
    """
    Calculate Historical VaR using KDE for a series of value changes.

    Parameters:
    :param: value_changes (Series): Pandas Series of value changes.
    :param: alpha (float): Confidence level for VaR (e.g., 0.05 for 95%).

    Returns:
    float: The calculated VaR.
    """
    # Fit a KDE to the value changes
    kde = gaussian_kde(value_changes)

    # Generate a range of values (e.g., from -3*std to 3*std)
    range_min = value_changes.min()
    range_max = value_changes.max()
    x_values = np.linspace(range_min, range_max, 10000)

    # Evaluate the cumulative distribution of the KDE
    cdf = np.array([kde.integrate_box_1d(range_min, x) for x in x_values])

    # Find the VaR as the point where the CDF reaches the desired confidence level
    var_index = np.where(cdf >= (alpha))[0][0]
    var = x_values[var_index]
    return -var


def calculate_portfolio_var_hist(portfolio_holdings, current_prices, historical_returns, alpha, n_simulations):
    """
    Calculate Historical VaR for a given portfolio.
    Parameters:
    - portfolio_holdings (dict): Dictionary of holdings (quantity of each asset).
    - current_prices (dict): Dictionary of current prices for each asset.
    - historical_returns (DataFrame): DataFrame of historical returns.
    - alpha (float): Confidence level for VaR (e.g., 0.95 for 95% confidence).
    - n_simulations (int): Number of simulations.
    Returns:
    - float: The calculated VaR for the portfolio.
    """
    current_portfolio_value = sum(portfolio_holdings[stock] * current_prices[stock] for stock in portfolio_holdings)
    simulated_values = []
    for _ in range(n_simulations):
        # Sample from historical returns
        sampled_returns = historical_returns.sample(n=1, replace=True)
        # Calculate new portfolio value based on sampled returns
        new_value = sum(
            portfolio_holdings[stock] * current_prices[stock] * (1 + sampled_returns[stock].iloc[0]) for stock in
            portfolio_holdings)
        simulated_values.append(new_value)
    # Find the α% of the simulated portfolio value distribution
    var_value = np.percentile(simulated_values, alpha * 100)
    return current_portfolio_value - var_value


def pricing(row, prices, portfolio, sim_returns):
    """
        Calculate the current and simulated values of a portfolio holding, and its PnL.

        The function takes a row from a DataFrame where each row combines a stock in the portfolio
        with a simulation iteration. It uses the current stock price and simulated returns to
        calculate the current value, simulated value, and PnL for the holding.

        Parameters:
        :param: row (pandas Series): A row from the DataFrame. The row must contain:
            - 'Stock': Stock identifier (e.g., ticker symbol).
            - 'Holding': Quantity of the stock held in the portfolio.
            - 'iteration': Iteration number of the simulation.
          The `current` and `simReturns` data structures should be accessible in the scope
          where this function is called. `current` should be a Series or a DataFrame row
          with the current prices of stocks, and `simReturns` a DataFrame with simulated
          returns for each stock across different iterations.
        :param: prices: (pandas DataFrame): A dataframe containing price data
        :param: portfolio: (pandas DataFrame): A data containing portfolio holdings
        :param: sim_returns: (pandas DataFrame): A dataframe containing simulated returns (normal monte carlo w/ PCA and
        covariance matrix) for all stocks

        Returns:
        :return: current_value (float): The current value of the holding, calculated as Holding * current price.
        :return: simulated_value (float): The simulated value of the holding, adjusted by the simulated return.
        :return:  pnl (float): Profit and Loss for the holding, calculated as the difference between
                       simulated value and current value.
        """

    # All current stock prices
    current_stock_prices = prices[portfolio['Stock']].iloc[-1]

    # Extracting current price for the given stock
    price = current_stock_prices[row['Stock']]

    # Calculating current value of the holding
    current_value = row['Holding'] * price

    # Calculating simulated value of the holding (adjusted by simulated return)
    simulated_value = row['Holding'] * price * (1.0 + sim_returns.loc[row['iteration'] - 1, row['Stock']])

    # Calculating PnL for the holding
    pnl = simulated_value - current_value
    return current_value, simulated_value, pnl


def calculate_var(series, alpha=0.05):
    """
    Calculate the Value at Risk (VaR) at a specified confidence level. Use this for portfolio and group metrics for
    Normal Monte Carlo VaR.

    Parameters:
    - series (pandas Series): A series of profit and loss (PnL) values.
    - alpha (float): The confidence level (default is 0.05 for 95% confidence).

    Returns:
    - VaR (float): The calculated Value at Risk.
    """
    return -np.percentile(series, 100 * alpha)


def aggregate_portfolio_values(values, holdings):
    """
    Aggregate values for a specific portfolio. Takes values computed per stock holding and aggregates them in a single
    data frame.
    :param values: pandas DataFrame: Contains current value, simulated value, and PnL (difference) for each stock in
    portfolio
    :param holdings: pandas DataFrame: Contains stock identifier and corresponding holdings for each stock in portfolio.
    Used to filter values DataFrame if values contains stocks in multiple portfolios
    :return: total_values: pandas DataFrame: Aggregated portfolio values based on stocks in portfolio.
    """
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


def calculate_total_risk(total_values):
    """
    Calculate the risk metrics of a portfolio
    :param total_values: pandas DataFrame: Contains aggregated portfolio values based on stocks in portfolio (current
    values, simulated values, and PnL (difference))
    :return: total_risk: pandas DataFrame: Contains risk metrics for portfolio (VaR95, ES95, standard deviation of PnL,
    min PnL, max PnL and average PnL)
    """
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


def calculate_var_mle_t_dist(returns_df, alpha):
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
    params = t.fit(returns_df)
    print(params)
    df, loc, scale = params[0], params[1], params[2]  # degrees of freedom, location, and scale

    # Calculate the VaR
    var = t.ppf(alpha, df, loc, scale)

    # Calculate the ES
    t_sim = t.rvs(df, loc, scale, size=10000)
    es = -np.mean(t_sim[t_sim <= var])

    return -var, es


def calculate_historic_var_es(returns_df, alpha):
    var = VaR(returns_df, alpha=alpha)
    x = np.sort(returns_df)
    es = -np.mean(x[x <= var])

    return var, es


# Fitted Models
class FittedModel:
    def __init__(self, beta, error_model, eval_func, errors, u):
        self.beta = beta
        self.error_model = error_model
        self.eval_func = eval_func
        self.errors = errors
        self.u = u


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
    initial_params = [start_m, start_s, start_nu]

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


def fit_regression_t(y, x):
    """
    Fit a regression models with t-distributed errors
    :param y: 1-D array or similar iterable: The dependent variable
    :param x: 2-D array or similar iterable: The independent variable. Each row represents an observation and each
    column represents a different independent variable.
    :return: fitted_model_instance: Instance of FittedModel class
    """
    n = x.shape[0]

    global __x, __y
    __x = np.hstack((np.ones((n, 1)), x))
    __y = y

    nB = __x.shape[1]

    # Initial values based on moments and OLS
    b_start = np.linalg.inv(__x.T @ __x) @ __x.T @ __y
    e = __y - __x @ b_start
    start_m = np.mean(e)
    start_nu = 6.0 / stats.kurtosis(e, fisher=False) + 4
    start_s = np.sqrt(np.var(e) * (start_nu - 2) / start_nu)

    # Optimization function
    def objective(params):
        m, s, nu = params[:3]
        B = params[3:]
        return -general_t_ll(m, s, nu, __y - __x @ B)

    # Initial parameters for optimization
    initial_params = np.concatenate(([start_m, start_s, start_nu], b_start))

    # Constraints for s and nu
    bounds = [(None, None), (1e-6, None), (2.0001, None)] + [(None, None)] * nB

    # Optimization
    result = minimize(objective, initial_params, bounds=bounds)

    m, s, nu = result.x[:3]
    beta = result.x[3:]

    # Fitted error model
    errorModel = lambda u: t.ppf(u, nu) * s + m

    # Function to evaluate the model for given x and u
    def eval_model(x, u):
        n = x.shape[0]
        _temp = np.hstack((np.ones((n, 1)), x))
        return _temp @ beta + errorModel(u)

    # Calculate the regression errors and their U values
    errors = y - eval_model(x, np.full(x.shape[0], 0.5))
    u = t.cdf(errors, nu) * s + m

    fitted_model_instance = FittedModel(beta, errorModel, eval_model, errors, u)
    return fitted_model_instance


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