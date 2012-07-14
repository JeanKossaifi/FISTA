from datasets import fetch_data
from cross_val import select_lambda
#from cross_val import cross_val
import numpy as np
from time import time

data = fetch_data()

# Settings
name = 'all'
norm = 'l11'
K = data.K
y = data.y[:, 12]

lambdas = [10**i for i in range(-4, 4)]
#lambdas = 0.5

# Selection
t1 = time()
result = select_lambda(norm, lambdas, 5, K, y, name)
exec_time = time() - t1
result['execution_time'] = exec_time
np.save('cross_val_%s_kernel__%s_norm__%d_values.npy' % (name, norm, len(lambdas)), result)
print "exectution time : %f", exec_time
