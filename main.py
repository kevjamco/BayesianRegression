"""
Kevin Cotton
Gradient decent on Bayesian Model
(Bayes by Backprop)


See paper here: http://proceedings.mlr.press/v37/blundell15.pdf

"""
import matplotlib.pyplot as plt
import numpy as np
from models.RidgeGradientDescent import Ridge
from models.BayesianRidgeGradientDecent import Bayesian

coefs = [ 1, -1.3, -0.98, 1.14, 2.32]
X_orig = np.linspace(0,2)
p = np.poly1d(coefs)
y_orig = p(X_orig)

# stochasisity
eps1 = 0.04
eps2 = 0.4
X = np.random.rand(100, 1) * 1
X.sort(axis=0)
p = np.poly1d(coefs)
y = p(X) + np.random.randn(100, 1) * eps1 + ((1/3 < X) & (X < 2/3)) * np.random.randn(100,1) * eps2

# pyRidge code
# rig = Ridge(num_iters=10000, alpha=0.01, beta=0.5, deg=6, normalization=False)
rig = Bayesian(num_iters=10000, alpha=0.01, beta=0.5, deg=6, normalization=False)

rig.fit(X, y[:,0])
if hasattr(rig, 'mu'):
    print(f'mu:  {rig.mu}')
if hasattr(rig, 'rho'):
    print(f'rho: {rig.rho}')

#eps2 non-locality
X_new = np.expand_dims(np.linspace(0,2), axis=1)
samples = 100
y_predict = np.zeros([samples, np.size(X_new)])
for sample in range(samples):
    pred = rig.predict(X_new, fixed=False)
    y_predict[sample,:] = pred[:,0]

confidence_interval = 80
lower_p = (100 - confidence_interval) / 2
upper_p = 100 - lower_p
lower = np.percentile(y_predict, lower_p, axis=0)
upper = np.percentile(y_predict, upper_p, axis=0)


sample_mean = np.mean(y_predict, axis=0)
#epistemic uncertainty
epistemic = np.diag(np.dot((y_predict - sample_mean).T,(y_predict - sample_mean))) / samples
# this def. of aleatoric uncertainty only works with logistic outputs, as we'll see
# aleatoric = np.sum(y_predict - np.diag(np.dot(y_predict.T,y_predict)), axis=0 ) / samples

sig = np.log(1 + np.exp(rig.rho))
aleatoric_scalar = np.sum(np.square(sig))
print(f'Aleatoric uncertainty scalar: {aleatoric_scalar}')

plt.plot(rig.J_all)
plt.show()

plt.figure()
# plt.plot(X_new, y_predict, "r-")
plt.fill_between(X_new[:,0], lower, upper, facecolor='red', interpolate=True)
# plt.plot(X_orig, y_orig, "g-")
# plt.plot(X_new[:,0], epistemic, "b-")
plt.plot(X_new, p(X_new), "g-")
plt.plot(X, y, "b.")
plt.show()