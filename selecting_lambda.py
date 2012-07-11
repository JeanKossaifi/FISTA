from datasets import fetch_data
from lambda_selection import select_lambda
#import numpy as np


# lambdas = np.arange(10**-2, 10**2, 1)
lambdas = [0.001, 0.01, 0.1, 1, 10, 100]
data = fetch_data()
K = data.kernels.kernel_matrix_pfamdom_exp_cn_3588
y = data.y[:, 12]
result = select_lambda('l11', lambdas, 5, K, y)
