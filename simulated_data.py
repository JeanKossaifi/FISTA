from fista import Fista
import numpy as np

fista = Fista(lambda_=0.5, loss='hinge', penalty='l11', n_iter=1000)
X = np.random.normal(size=(10, 40))
y = np.sign(np.random.normal(size=10))
fista.fit(X, y)
print "taux de bonne prediction with l11: %f " % fista.prediction_score(X, y)
fista.penalty='l12'
fista.fit(X, y)
print "taux de bonne prediction with l12: %f " % fista.prediction_score(X, y)
fista.penalty='l21'
fista.fit(X, y)
print "taux de bonne prediction with l21: %f " % fista.prediction_score(X, y)
fista.penalty='l22'
fista.fit(X, y)
print "taux de bonne prediction with l22: %f " % fista.prediction_score(X, y)

