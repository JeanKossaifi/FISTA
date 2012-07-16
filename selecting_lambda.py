from datasets import fetch_data
from cross_val import cross_val
#from cross_val import cross_val
import numpy as np
from time import time

data = fetch_data()

# Settings
name = 'all'
norm = 'l11'
K = data.K
y = data.y[:, 12]

#lambdas = [10**i for i in range(-4, 4)]
lambdas = 0.5

# Selection
t1 = time()
result = cross_val(norm, lambdas, 5, K, y, name)
exec_time = time() - t1
result['execution_time'] = exec_time
np.save('cross_val_%s_kernel__%s_norm__%f_values.npy' % (name, norm, lambdas), result)
print "exectution time : %f", exec_time


# Selection
norm = 'l22'
lambdas = 0.005
K = data.K

t1 = time()
result = cross_val(norm, lambdas, 5, K, y, name)
exec_time = time() - t1
result['execution_time'] = exec_time
np.save('cross_val_%s_kernel__%s_norm__%f_values.npy' % (name, norm, lambdas), result)
print "exectution time : %f", exec_time



# Selection
norm = 'l12'
lambdas = 0.5
K = data.K

t1 = time()
result = cross_val(norm, lambdas, 5, K, y, name)
exec_time = time() - t1
result['execution_time'] = exec_time
np.save('cross_val_%s_kernel__%s_norm__%f_values.npy' % (name, norm, lambdas), result)
print "exectution time : %f", exec_time



# Selection
norm = 'l21'
lambdas = 0.5
K = data.K

t1 = time()
result = cross_val(norm, lambdas, 5, K, y, name)
exec_time = time() - t1
result['execution_time'] = exec_time
np.save('cross_val_%s_kernel__%s_norm__%f_values.npy' % (name, norm, lambdas), result)
print "exectution time : %f", exec_time
