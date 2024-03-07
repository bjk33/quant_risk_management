from quant_risk_lib import my_functions
import pandas as pd
import numpy as np
from scipy.stats import norm, t
from scipy.optimize import minimize

# Test 1 - missing covariance calculations #
x1_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/test1.csv'
x1 = pd.read_csv(x1_path)
out11_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/testout_1.1.csv'
out11 = pd.read_csv(out11_path)
out12_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/testout_1.2.csv'
out12 = pd.read_csv(out12_path)
out13_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/testout_1.3.csv'
out13 = pd.read_csv(out13_path)
out14_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/testout_1.4.csv'
out14 = pd.read_csv(out14_path)

# 1.1 Skip missing rows - covariance
cout_1 = my_functions.missing_cov(x1, skip_miss=True)
check11 = cout_1 - out11
check11 = check11.to_numpy()
check11 = np.linalg.norm(check11)
print(check11)

# 1.2 Skip missing rows - correlation
cout_2 = my_functions.missing_cov(x1, skip_miss=True, fun=np.corrcoef)
check12 = cout_2 - out12
check12 = check12.to_numpy()
check12 = np.linalg.norm(check12)
print(check12)

# 1.3 Pairwise - covariance
cout_3 = my_functions.missing_cov(x1, skip_miss=False)
check13 = cout_3 - out13
check13 = check13.to_numpy()
check13 = np.linalg.norm(check13)
print(check13)

# 1.4 Pairwise - correlation
cout_4 = my_functions.missing_cov(x1, skip_miss=False, fun=np.corrcoef)
check14 = cout_4 - out14
check14 = check14.to_numpy()
check14 = np.linalg.norm(check14)
print(check14)

# Test 2 - EW Covariance #
x2_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/test2.csv'
x2 = pd.read_csv(x2_path)
out21_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/testout_2.1.csv'
out21 = pd.read_csv(out21_path)
out22_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/testout_2.2.csv'
out22 = pd.read_csv(out22_path)
out23_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/testout_2.3.csv'
out23 = pd.read_csv(out23_path)

# 2.1 EW Covar (lambda = 0.97)
cout_21 = my_functions.ewCovar(x2, lambda_=0.97)
check21 = cout_21 - out21
check21 = check21.to_numpy()
check21 = np.linalg.norm(check21)
print(check21)

# 2.2 EW Correlation (lambda = 0.94)
cout_22 = my_functions.ewCovar(x2, lambda_=0.94)
cout_22 = cout_22 / np.sqrt(np.diag(cout_22))
check22 = cout_22 - out22
check22 = check22.to_numpy()
check22 = np.linalg.norm(check22)
print(check22)

# 2.3 EW Covar with EW Var (lambda = 0.97) and EW Correlation (lambda = 0.94)
cout_23 = my_functions.ewCovar(x2, lambda_=0.97)

