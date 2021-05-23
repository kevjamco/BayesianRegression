"""
Kevin J Cotton


"""

import numpy as np
from models.math_tools import normal_pdf, ln_pdf_dx, ln_pdf_dmu, ln_pdf_dsig


class BayesianSplit():
    """Ridge Regression using Gradient Descent

    Using beta for lambda to avoid python conflict
    """

    def __init__(self, num_iters=2000, alpha=0.1, beta=0.1, deg=2, normalization=False):
        """
        alpha: train speed
        beta: regularization constant
        """
        self.num_iters = num_iters
        self.alpha = alpha
        self.beta = beta
        self.deg = deg
        self.normalization = normalization

        self.prior_mu = 0
        self.prior_sig = 4
        self.init_post_mu = 0.1
        self.init_post_rho = 0.1

    def _evaluate_f(self, X, w, chi, normalize=False):
        Xn = np.ndarray.copy(X).astype('float64')

        if normalize:
            Xn -= self.X_mean
            Xn /= self.X_std
        Xs = np.hstack((np.ones(Xn.shape[0])[np.newaxis].T, Xn, Xn ** 2))
        for power in range(2, self.deg):
            Xs = np.hstack((Xs, Xn ** power))

        y_out = Xs.dot(w)
        sig_out = Xs.dot(chi)
        return y_out, sig_out

    def _compute_cost(self, X, y, mu, rho, w, beta):
        """
        Compute the value of cost function
        X: data
        y: target
        mu: mean
        rho:
        w: sampled weight
        beta:
        """
        m = X.shape[0]

        # evaluate f
        y_pred = X.dot(w)

        # compute KL divergence
        sig = np.log(1 + np.exp(rho))
        q_w = normal_pdf(w, mu, sig)
        p_w = normal_pdf(w, self.prior_mu, self.prior_sig)
        # todo: not sure what sigma should be for P(D|w)
        # Calculate P(D|w)
        # todo: This seems to be a gaussian
        # "For
        # regression Y is R and P(y|x, w) is a Gaussian distribution
        # – this corresponds to a squared loss."
        p_D_w = normal_pdf(np.expand_dims(y, axis=1), y_pred, 1/np.sqrt(2*np.pi))
        # todo the size of p(D|w) does not make sense..
        KL = np.sum(np.log(q_w)) - np.sum(np.log(p_w)) - np.sum(np.log(p_D_w))
        return KL

    def _compute_gradient(self, X, y, mu, rho, mu2, rho2, eps, w, eps2, w2):
        """
        X: data
        y: target
        m, rho: weight parameters (before update)
        eps: sampled noise
        w: sampled weights
        beta: regularization term

        mu2, rho2 describe s=log(sig_i^2), where sig_i is the learned heteroscedastic noise.
        """
        # f(w, θ) = log q(w|θ) − log P(w) - log P(D|w).
        m = X.shape[0]
        y_pred = X.dot(w)
        s_i = X.dot(w2)
        sig = np.log(1 + np.exp(rho))
        sig2 = np.log(1 + np.exp(rho2))
        # gamma: regularization for loss
        # gamma = 1 / np.sqrt(2 * np.pi)
        # lnP_Dw = ln_pdf_dmu(np.expand_dims(y, axis=1), y_pred, gamma)
        lnP_Ds = 1/2*np.exp(-s_i) * (np.expand_dims(y, axis=1) - y_pred)**2 - 1/2
        lnP_Dy = np.exp(-s_i) * (np.expand_dims(y, axis=1) - y_pred)
        dfdw = ln_pdf_dx(w, mu, sig) - ln_pdf_dx(w, self.prior_mu, self.prior_sig) - 1/m*np.dot(X.T, lnP_Dy)
        dfdm = ln_pdf_dmu(w, mu, sig)
        dfdr = ln_pdf_dsig(w, mu, sig)

        dfdw2 = ln_pdf_dx(w, mu2, sig2) - ln_pdf_dx(w2, self.prior_mu, self.prior_sig) - 1/m*np.dot(X.T, lnP_Ds)
        dfdm2 = ln_pdf_dmu(w, mu2, sig2)
        dfdr2 = ln_pdf_dsig(w, mu2, sig2)

        del_mu = dfdw + dfdm
        del_sig = dfdw * (eps / 1 + np.exp(-rho)) + dfdr

        del_mu2 = dfdw2 + dfdm2
        del_sig2 = dfdw2 * (eps2 / 1 + np.exp(-rho2)) + dfdr2

        return del_mu, del_sig, del_mu2, del_sig2

    def _sample(self, mu, rho, mu2, rho2, fixed=False):
        size = rho.shape
        if fixed:
            eps = np.zeros(size)
            w = mu
        else:
            eps = np.random.normal(size=size)
            sig = np.log(1 + np.exp(rho))
            w = mu + sig * eps

        size = rho2.shape
        if fixed:
            eps2 = np.zeros(size)
            w2 = mu2
        else:
            eps2 = np.random.normal(size=size)
            sig = np.log(1 + np.exp(rho2))
            w2 = mu2 + sig * eps

        return eps, w, eps2, w2

    def _gradient_descent(self, X, y, mu, rho, mu2, rho2, num_iterations, alpha, beta):
        """Performs Gradient Descent.
        The threshold is set by num_iters, instead of some value in this implementation
        X: data
        y: target
        mu: weight parameter
        rho: weight parameter
        num_iterations
        alpha: training speed
        beta: regularization constant
        """
        m = X.shape[0]
        # Keep a history of Costs (for visualisation)
        J_all = np.zeros((num_iterations, 1))

        # perform gradient descent
        for i in range(num_iterations):
            eps, w, eps2, w2 = self._sample(mu, rho, mu2, rho2)
            J_all[i] = self._compute_cost(X, y, mu, rho, w, self.beta)
            del_mu, del_sig, del_mu2, del_sig2 = self._compute_gradient(X, y, mu, rho, mu2, rho2, eps, w, eps2, w2)

            mu = mu - (alpha / m) * del_mu
            rho = rho - (alpha / m) * del_sig
            mu2 = mu2 - (alpha / m) * del_mu2
            rho2 = rho2 - (alpha / m) * del_sig2

        return mu, rho, mu2, rho2, J_all

    def fit(self, X, y):
        """Fit the model
        """
        Xn = np.ndarray.copy(X).astype('float64')
        yn = np.ndarray.copy(y).astype('float64')

        # initialise w params for linear model, from w0 to w_num_features
        num_weights = Xn.shape[1] + self.deg
        mu_init = np.full((num_weights, 1), fill_value=self.init_post_mu)
        rho_init = np.full((num_weights, 1), fill_value=self.init_post_rho)

        if self.normalization:
            self.X_mean = np.mean(Xn, axis=0)
            self.X_std = np.std(Xn, axis=0)
            Xn -= self.X_mean
            self.X_std[self.X_std == 0] = 1
            Xn /= self.X_std
            self.y_mean = yn.mean(axis=0)
            yn -= self.y_mean

        # add ones for intercept term
        Xs = np.hstack((np.ones(Xn.shape[0])[np.newaxis].T, Xn))
        for power in range(2, self.deg+1):
            Xs = np.hstack((Xs, Xn ** power))

        self.mu, self.rho, self.mu2, self.rho2, self.J_all = self._gradient_descent(
            Xs, yn, mu_init, rho_init, mu_init, rho_init, self.num_iters, self.alpha, self.beta)

    def predict(self, X, fixed=False):
        """Predict values for given X
        """
        eps, w, eps2, w2 = self._sample(self.mu, self.rho, self.mu2, self.rho2, fixed)
        y_pred, sig_pred = self._evaluate_f(X, w, w2, normalize=self.normalization)

        if self.normalization:
            return y_pred + self.y_mean, sig_pred
        else:
            return y_pred, sig_pred

