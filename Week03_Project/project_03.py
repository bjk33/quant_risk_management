import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy import stats

# Problem 1 #

# Routine for calculating an exponentially weighted covariance matrix

# Load and clean data
daily_return_path = '/Users/brandonkaplan/Desktop/FINTECH545/Week03_Project/DailyReturn.csv'
daily_returns = pd.read_csv(daily_return_path)
daily_returns = daily_returns[daily_returns['SPY'].notna()] # Filter out rows where 'SPY' column is missing
daily_returns = daily_returns.drop(daily_returns.columns[0], axis=1)  # drop indices

for col in daily_returns.columns:
    # Assuming you want to convert all columns except 'SPY'
    if col != 'SPY':
        daily_returns[col] = pd.to_numeric(daily_returns[col], errors='coerce')

# Create function for calculating exponentially weighted covariance matrix


def ewCovar(x, lambda_):
    """Compute exponentially weighted covariance matrix of a dataframe.
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


# Create function to calculate the percentage of variance explained by PCA

def PCA_pctExplained(a):
    """Compute the percentage of variance explained by PCA.
    :param a: an exponentially weighted covariance matrix
    return: out: percentage of variance explained by PCA
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


# Problem 2 #

# Comparing Cholesky Factorization and Higham's (2002) Nearest PSD Correlation

# Copying chol_psd() and near_psd() from course repo

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


# Use near_psd() and Higham's method to fix a non-PSD correlation matrix.

n = 500
sigma = np.full((n, n), 0.9)  # Create an n x n matrix filled with 0.9
for i in range(n):
    sigma[i, i] = 1.0  # Set diagonal elements to 1.0

sigma[0, 1] = 0.7357  # Adjust indices for zero-based indexing
sigma[1, 0] = 0.7357

W = np.diag(np.ones(n))  # Create an identity weight matrix

# Correct sigma matrix

hnpsd = higham_nearestPSD(sigma)
npsd = near_psd(sigma)

# Confirm whether sigma is now PSD


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


is_psd_hnpsd = is_psd(hnpsd)
is_psd_npsd = is_psd(npsd)

print("Is hnpsd PSD:", is_psd_hnpsd)
print("Is npsd PSD:", is_psd_npsd)

norm_hnpsd = wgtNorm(hnpsd-sigma, W)
norm_npsd = wgtNorm(npsd-sigma, W)
print("Distance of near_psd() for n=500:",norm_npsd)
print("Distance of higham_nearestPSd() for n=500:",norm_hnpsd)

# Measuring runtime for near_psd() with n=500
start_time = time.time()
npsd_result_500 = near_psd(sigma)
end_time = time.time()
near_psd_runtime = end_time - start_time

print("Runtime for near_psd() for n=500: {:.6f} seconds".format(near_psd_runtime))

# Measuring runtime for higham_nearestPSD() with n=500
start_time = time.time()
hnpsd_result_500 = higham_nearestPSD(sigma)
end_time = time.time()
higham_nearestPSD_runtime = end_time - start_time

print("Runtime for higham_nearestPSD() for n=500: {:.6f} seconds".format(higham_nearestPSD_runtime))

# Comparing Froebenius norm and runtime when n=1000
n = 1000
sigma_2 = np.full((n, n), 0.9)  # Create an n x n matrix filled with 0.9
for i in range(n):
    sigma_2[i, i] = 1.0  # Set diagonal elements to 1.0

sigma_2[0, 1] = 0.7357  # Adjust indices for zero-based indexing
sigma_2[1, 0] = 0.7357

W = np.diag(np.ones(n))  # Create an identity weight matrix

hnpsd_1000 = higham_nearestPSD(sigma_2)
npsd_1000 = near_psd(sigma_2)

is_psd_hnpsd_1000 = is_psd(hnpsd_1000)
is_psd_npsd_1000 = is_psd(npsd_1000)

print("Is hnpsd_1000 PSD:", is_psd_hnpsd)
print("Is npsd_1000 PSD:", is_psd_npsd)

norm_hnpsd_1000 = wgtNorm(hnpsd_1000-sigma_2, W)
norm_npsd_1000 = wgtNorm(npsd_1000-sigma_2, W)
print("Distance of near_psd() for n=1000:",norm_npsd_1000)
print("Distance of higham_nearestPSd() for n=1000:",norm_hnpsd_1000)

# Measuring runtime for near_psd() with n=1000
start_time = time.time()
npsd_result_1000 = near_psd(sigma_2)
end_time = time.time()
near_psd_runtime_1000 = end_time - start_time

print("Runtime for near_psd() for n=1000: {:.6f} seconds".format(near_psd_runtime_1000))

# Measuring runtime for higham_nearestPSD() with n=1000
start_time = time.time()
hnpsd_result_1000 = higham_nearestPSD(sigma_2)
end_time = time.time()
higham_nearestPSD_runtime_1000 = end_time - start_time

print("Runtime for higham_nearestPSD() for n=1000: {:.6f} seconds".format(higham_nearestPSD_runtime_1000))


"""
The Higham Nearest PSD method is clearly much slower than naive Nearest PSD, but also much more accurate. There is thus
a trade off between computational time and computational precision. In practice for fast calculations where pretty close
is "good enough" then I would use near_psd(). If you have more time and compute, and a more exact approximation is
important use higham_nearestPSD(). Also note that as n increases, there is not much change in accuracy of 
higham_nearestPSD(). It is about as accurate on n=500 and n=1000. On the other hand, the Froebenius norm for
nearest_psd() roughly doubles between n=500 and n=1000. Whether this increase is statistically significant is a question
that deserves further exploration. Moreover, the runtime for nearest_psd() roughly doubles when the size of the matrix
to approximate doubles. However, the runtime for higham_nearestPSD() roughly quadrupled between n=500 and n=1000. 
"""

# Visualizing this trade-off for larger n

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# Example sizes to test
sizes = [500, 1000, 1500, 2000, 2500]  # Adjust this list as needed

# List to store the results
results_data = []

for n in sizes:
    sigma = np.full((n, n), 0.9)
    np.fill_diagonal(sigma, 1.0)
    sigma[0, 1] = sigma[1, 0] = 0.7357

    # Update the size of W to match the size of sigma
    W = np.diag(np.ones(n))

    # Run near_psd
    start_time = time.time()
    npsd_result = near_psd(sigma)
    runtime_npsd = time.time() - start_time
    error_npsd = wgtNorm(npsd_result - sigma, W)

    # Run higham_nearestPSD
    start_time = time.time()
    hnpsd_result = higham_nearestPSD(sigma)
    runtime_hnpsd = time.time() - start_time
    error_hnpsd = wgtNorm(hnpsd_result - sigma, W)

    # Add results to the list
    results_data.append({'Size': n, 'Method': 'near_psd', 'Runtime': runtime_npsd, 'Error': error_npsd})
    results_data.append({'Size': n, 'Method': 'higham_nearestPSD', 'Runtime': runtime_hnpsd, 'Error': error_hnpsd})

# Create DataFrame from the list
results = pd.DataFrame(results_data)

# Plotting
plt.figure(figsize=(12, 6))

# Plot for Runtime
plt.subplot(1, 2, 1)
for method in ['near_psd', 'higham_nearestPSD']:
    subset = results[results['Method'] == method]
    plt.plot(subset['Size'], subset['Runtime'], label=method)
plt.xlabel('Matrix Size')
plt.ylabel('Runtime (seconds)')
plt.title('Runtime vs Matrix Size')
plt.legend()

# Plot for Error
plt.subplot(1, 2, 2)
for method in ['near_psd', 'higham_nearestPSD']:
    subset = results[results['Method'] == method]
    plt.plot(subset['Size'], subset['Error'], label=method)
plt.xlabel('Matrix Size')
plt.ylabel('Error')
plt.title('Error vs Matrix Size')
plt.legend()

plt.tight_layout()
plt.show()

# Problem 3 #

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


# Generate a correlation matrix and variance vector two ways

# #1 Pearson Covariance and Correlation
pearson_cov = daily_returns.cov().values
pearson_std = np.sqrt(np.diag(pearson_cov))
pearson_cor = daily_returns.corr().values

# #2 Exponentially Weighted Covariance and Standard Deviation
ewma_cov = ewCovar(daily_returns.values, 0.97)  # Implement ewCovar function
ewma_std = np.sqrt(np.diag(ewma_cov))
ewma_cor = np.diag(1.0 / ewma_std) @ ewma_cov @ np.diag(1.0 / ewma_std)

print("EWMA Cov Shape:", ewma_cov.shape)
print("Pearson Std Diag Shape:", np.diag(pearson_std).shape)
print("EWMA Cor Shape:", ewma_cor.shape)
print("Pearson Cor Shape:", pearson_cor.shape)
print("EWMA Std Diag Shape:", np.diag(ewma_std).shape)


# Combine to form 4 different covariance matrices
matrix_lookup = {
    "EWMA": ewma_cov,
    "EWMA_COR_PEARSON_STD": np.diag(pearson_std) @ ewma_cor @ np.diag(pearson_std),
    "PEARSON": pearson_cov,
    "PEARSON_COR_EWMA_STD": np.diag(ewma_std) @ pearson_cor @ np.diag(ewma_std)
}

# Simulate draws and compute covariance
matrix_type = ["EWMA", "EWMA_COR_PEARSON_STD", "PEARSON", "PEARSON_COR_EWMA_STD"]
sim_type = ["Direct", "PCA: pctExp=1", "PCA: pctExp=0.75", "PCA: pctExp=0.5"]

results = []

for sim in sim_type:
    for mat in matrix_type:
        cov_matrix = matrix_lookup[mat]
        elapse = 0.0
        s = []

        if sim == "Direct":  # Simulating directly
            st = time.time()
            for _ in range(20):
                s = simulate_normal(25000, cov_matrix)
            elapse = (time.time() - st) / 20
        elif sim == "PCA: pctExp=1":  # Simulating from PCA with 100% variance explained by principal components
            st = time.time()
            for _ in range(20):
                s = simulate_pca(cov_matrix, 25000, pctExp=1)
            elapse = (time.time() - st) / 20
        elif sim == "PCA: pctExp=0.75":  # Simulating from PCA with 75% of variance explained by principal components
            st = time.time()
            for _ in range(20):
                s = simulate_pca(cov_matrix, 25000, pctExp=0.75)
            elapse = (time.time() - st) / 20
        else:  # PCA: pctExp=0.5  # Simulating from PCA with 50% of variance explained by principal components
            st = time.time()
            for _ in range(20):
                s = simulate_pca(cov_matrix, 25000, pctExp=0.5)
            elapse = (time.time() - st) / 20

        simulated_covar = np.cov(s, rowvar=False)
        l2_norm = np.sum((simulated_covar - cov_matrix) ** 2)
        results.append([mat, sim, elapse, l2_norm])

# Create DataFrame for results
out_table = pd.DataFrame(results, columns=["Matrix", "Simulation", "Runtime", "Norm"])



# Runtime Plot

# Correct labels for the x-axis
correct_labels = ['Direct', 'PCA: pctExp=1', 'PCA: pctExp=0.75', 'PCA: pctExp=0.5']

# Sort 'Simulation' column to control the order of the x-axis categories
out_table['Simulation'] = pd.Categorical(out_table['Simulation'], categories=correct_labels, ordered=True)

# Sort the DataFrame based on 'Simulation' to ensure correct grouping
out_table_sorted = out_table.sort_values('Simulation')

# Runtime Plot
plt.figure(figsize=(12, 6))
runtime_plot = sns.barplot(
    x='Simulation',
    y='Runtime',
    hue='Matrix',
    data=out_table_sorted,
    ci=None
)

# Set the positions for the x-ticks at the center of each group
# Set the custom labels for these x-tick positions
plt.xticks(ticks=np.arange(len(correct_labels)), labels=correct_labels, rotation=45)

# Set the title and labels for the plot
runtime_plot.set_title('Simulation Runtime Comparison')
runtime_plot.set_xlabel('Simulation Type')
runtime_plot.set_ylabel('Runtime (seconds)')

plt.legend(title='Matrix Type')
plt.tight_layout()
plt.show()


# Norm Plot

# Accuracy Plot
plt.figure(figsize=(12, 6))
accuracy_plot = sns.barplot(
    x='Simulation',
    y='Norm',
    hue='Matrix',
    data=out_table_sorted,
    ci=None
)

# Set the positions for the x-ticks at the center of each group
# Set the custom labels for these x-tick positions
plt.xticks(ticks=np.arange(len(correct_labels)), labels=correct_labels, rotation=45)

# Set the title and labels for the plot
accuracy_plot.set_title('Simulation Accuracy Comparison')
accuracy_plot.set_xlabel('Simulation Type')
accuracy_plot.set_ylabel('Frobenius Norm')

plt.legend(title='Matrix Type')
plt.tight_layout()
plt.show()


"""
Overall, the direct simulation took the most time to run, followed by the PCA with 100% of the variance explained by
the principal components, followed by the PCA with 75% of the variance explained by the principal components, followed
by the PCA with 50% of the variance explained by the principal components. The direct simulation and the PCA with 100%
explained took significantly longer than the other two PCA simulations. The PCA with 100% explained took slightly faster
than the direct simulation. In terms of accuracy, the direct simulation had the greatest Froebenius norms for each 
matrix. Thus in the case of direct simulation, the longer run time was not a trade off with the accuracy of the
simulation. The direct simulation simply takes the longest and is the least accurate because of its design and 
methodology. Of the three PCA simulations, the PCA with 100% explained was the most accurate, followed by the PCA with
75% explained, followed by the PCA with 50% explained. However, the differences in accuracy between the 100%, 75%, and 
50% PCA might be considered negligible. With respect to the differences in their run times I would prefer the 50% PCA
because it was more efficient than it was less accurate relative to the 75% and 100% PCA. Additionally, it seems like 
the exponentially weighted moving average covariance matrix and the Pearson correlation plus exponentially weight moving 
average standard deviation matrices were simulated most accurately. However, the run time to simulate all four was 
largely the same.
"""