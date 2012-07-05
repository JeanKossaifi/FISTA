from fista import Fista
from sklearn.datasets import load_iris
import numpy as np

data = load_iris()

y = data.target
X = data.data

X = X[y<2]
y = y[y<2]
y[y==0] = -1

K1 = X[:, 0]
K2 = X[:, 1]
K3 = X[:, 2]
K4 = X[:, 3]

K1 = np.dot(K1[:, np.newaxis], K1[:, np.newaxis].transpose())
K2 = np.dot(K2[:, np.newaxis], K2[:, np.newaxis].transpose())
K3 = np.dot(K3[:, np.newaxis], K3[:, np.newaxis].transpose())
K4 = np.dot(K4[:, np.newaxis], K4[:, np.newaxis].transpose())
K = np.concatenate((K1, K2, K3, K4), axis=1)

fista = Fista(loss='hinge', penalty='l11', lambda_=0.1, n_iter=50)
fista.fit(K, y)
print "pourcentage de bonne prediction avec l11: %d " % fista.prediction_score(K, y)
fista.penalty='l12'
fista.fit(K, y)
print "pourcentage de bonne prediction avec l12: %d " % fista.prediction_score(K, y)
fista.penalty='l21'
fista.fit(K, y)
print "pourcentage de bonne prediction avec l21: %d " % fista.prediction_score(K, y)
fista.penalty='l22'
fista.fit(K, y)
print "pourcentage de bonne prediction avec l22: %d " % fista.prediction_score(K, y)

