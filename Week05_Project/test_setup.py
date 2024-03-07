from quant_risk_lib import my_functions
import pandas as pd
import numpy as np
from scipy.stats import norm, t
from scipy.optimize import minimize

# Test 1 - missing covariance calculations #
x1_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/test1.csv'
x1 = pd.read_csv(x1_path)
out11_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/testout_1.1.csv'
out_11 = pd.read_csv(out11_path)
out12_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/testout_1.2.csv'
out_12 = pd.read_csv(out12_path)
out13_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/testout_1.3.csv'
out_13 = pd.read_csv(out13_path)
out14_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/testout_1.4.csv'
out_14 = pd.read_csv(out14_path)

# 1.1 Skip missing rows - covariance
cout_1 = my_functions.missing_cov(x1, skip_miss=True)
check11 = cout_1 - out_11
check11 = check11.to_numpy()
check11 = np.linalg.norm(check11)
print(check11)

# 1.2 Skip missing rows - correlation
cout_2 = my_functions.missing_cov(x1, skip_miss=True, fun=np.corrcoef)
check12 = cout_2 - out_12
check12 = check12.to_numpy()
check12 = np.linalg.norm(check12)
print(check12)

# 1.3 Pairwise - covariance
cout_3 = my_functions.missing_cov(x1, skip_miss=False)
check13 = cout_3 - out_13
check13 = check13.to_numpy()
check13 = np.linalg.norm(check13)
print(check13)

# 1.4 Pairwise - correlation
cout_4 = my_functions.missing_cov(x1, skip_miss=False, fun=np.corrcoef)
check14 = cout_4 - out_14
check14 = check14.to_numpy()
check14 = np.linalg.norm(check14)
print(check14)

# Test 2 - EW Covariance #
x2_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/test2.csv'
x2 = pd.read_csv(x2_path)
out21_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/testout_2.1.csv'
out_21 = pd.read_csv(out21_path)
out22_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/testout_2.2.csv'
out_22 = pd.read_csv(out22_path)
out23_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/testout_2.3.csv'
out_23 = pd.read_csv(out23_path)

# 2.1 EW Covar (lambda = 0.97)
cout_21 = my_functions.ewCovar(x2, lambda_=0.97)
check21 = cout_21 - out_21
check21 = check21.to_numpy()
check21 = np.linalg.norm(check21)
print(check21)

# 2.2 EW Correlation (lambda = 0.94)
cout_22 = my_functions.ewCovar(x2, lambda_=0.94)
sd = 1 / np.sqrt(np.diag(cout_22))
cout_22 = np.diag(sd) @ cout_22 @ np.diag(sd)
check22 = cout_22 - out_22
check22 = check22.to_numpy()
check22 = np.linalg.norm(check22)
print(check22)

# 2.3 EW Covar with EW Var (lambda = 0.97) and EW Correlation (lambda = 0.94)
cout_23 = my_functions.ewCovar(x2, lambda_=0.97)
sd1 = np.sqrt(np.diag(cout_23))
cout_23 = my_functions.ewCovar(x2, lambda_=0.94)
sd = 1 / np.sqrt(np.diag(cout_23))
cout_23 = np.diag(sd1) @ np.diag(sd) @ cout_23 @ np.diag(sd) @ np.diag(sd1)
check23 = cout_23 - out_23
check23 = check23.to_numpy()
check23 = np.linalg.norm(check23)
print(check23)

# Test 3 - Non-PSD matrices #
cin31_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/testout_1.3.csv'
cin32_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/testout_1.4.csv'
cin33_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/testout_1.3.csv'
cin34_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/testout_1.4.csv'
cin31 = pd.read_csv(cin31_path)
cin32 = pd.read_csv(cin32_path)
cin33 = pd.read_csv(cin33_path)
cin34 = pd.read_csv(cin34_path)

out31_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/testout_3.1.csv'
out32_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/testout_3.2.csv'
out33_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/testout_3.3.csv'
out34_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/testout_3.4.csv'
out_31 = pd.read_csv(out31_path)
out_32 = pd.read_csv(out32_path)
out_33 = pd.read_csv(out33_path)
out_34 = pd.read_csv(out34_path)

# 3.1 near_psd covariance
cout_31 = my_functions.near_psd(cin31)
check31 = cout_31 - out_31
check31 = check31.to_numpy()
check31 = np.linalg.norm(check31)
print(check31)

# 3.2 near_psd correlation
cout_32 = my_functions.near_psd(cin32)
check32 = cout_32 - out_32
check32 = check32.to_numpy()
check32 = np.linalg.norm(check32)
print(check32)

# 3.3 Higham covariance
cin33 = cin33.to_numpy()
out_33 = out_33.to_numpy()
cout_33 = my_functions.higham_nearestPSD(cin33)
check33 = cout_33 - out_33
check33 = np.linalg.norm(check33)
print(check33)

# 3.4 Higham correlation
cin34 = cin34.to_numpy()
out_34 = out_34.to_numpy()
cout_34 = my_functions.higham_nearestPSD(cin34)
check34 = cout_34 - out_34
check34 = np.linalg.norm(check34)
print(check34)

# Test 4 - Cholesky Factorization #
cin4_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/testout_3.1.csv'
cin4 = pd.read_csv(cin4_path)
out4_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/testout_4.1.csv'
out_4 = pd.read_csv(out4_path)

out_4 = out_4.to_numpy()
cin4 = cin4.to_numpy()
n, m = cin4.shape
cout_4 = np.zeros((n, m))
my_functions.chol_psd(cout_4, cin4)
check4 = cout_4 - out_4
check4 = np.linalg.norm(check4)
print(check4)

# Test 5 - Normal Simulation #

cin51_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/test5_1.csv'
cin52_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/test5_2.csv'
cin53_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/test5_3.csv'
cin54_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/test5_3.csv'
cin55_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/test5_2.csv'

cin51 = pd.read_csv(cin51_path)
cin52 = pd.read_csv(cin52_path)
cin53 = pd.read_csv(cin53_path)
cin54 = pd.read_csv(cin54_path)

out51_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/testout_5.1.csv'
out52_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/testout_5.2.csv'
out53_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/testout_5.3.csv'
out54_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/testout_5.4.csv'
out55_path = '/Users/brandonkaplan/Desktop/FINTECH545/tests/test_data/testout_5.5.csv'

out_51 = pd.read_csv(out51_path)
out_52 = pd.read_csv(out52_path)
out_53 = pd.read_csv(out53_path)
out_54 = pd.read_csv(out54_path)
out_55 = pd.read_csv(out55_path)

# 5.1 PD input
out_51 = out_51.to_numpy()
cin51 = cin51.to_numpy()
cout_51 = np.cov(my_functions.simulate_normal(100000, cin51))
check51 = cout_51 - out_51
check51 = np.linalg.norm(check51)
print(check51)

# 5.2 PSD Input
out_52 = out_52.to_numpy()
cin52 = cin52.to_numpy()
cout_52 = np.cov(my_functions.simulate_normal(100000, cin52))
check52 = cout_52 - out_52
check52 = np.linalg.norm(check52)
print(check52)

# 5.3
out_53 = out_53.to_numpy()
cin53 = cin53.to_numpy()
cout_53 = np.cov(my_functions.simulate_normal(100000, cin53, fix_method=my_functions.near_psd()))
check53 = cout_53 - out_53
check53 = np.linalg.norm(check53)
print(check53)
