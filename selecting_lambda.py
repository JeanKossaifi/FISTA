from datasets import fetch_data
from lambda_selection import select_lambda
import numpy as np
from time import time

data = fetch_data()

# Settings
name = 'all'
norm = 'l11'
K = data.K
y = data.y[:, 12]
#lambdas = np.array([0.001, 0.006, 0.01, 0.06, 0.1, 0.6, 1, 6, 10, 60, 100])
lambdas = np.linspace(10**-2, 10**2, 10)

# Selection
t1 = time()
#lambdas = [0.001, 0.01, 0.1, 1, 10, 100]
result = select_lambda(norm, lambdas, 5, K, y, name)
exec_time = time() - t1
result['execution_time'] = exec_time
np.save('cross_val_%s_kernel__%s_norm__%d_values.npy' % (name, norm, len(lambdas)), result)
print "exectution time : %f", exec_time
