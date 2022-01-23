import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder


class SVM(BaseEstimator, ClassifierMixin):

    def __init__(self, C=0.01, max_iteration=1000, tol=0.001,
                 random_state=None):
        self.C = C
        self.max_iteration = max_iteration
        self.tol = tol,
        self.random_state = random_state
        self.dual_coef = None
        self.coef = None
        self.label_encoder = LabelEncoder()

    def projection(self, v, z=1):
        n_features = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - z
        ind = np.arange(n_features) + 1
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        w = np.maximum(v - theta, 0)
        return w

    def _partial_gradient(self, X, y, i):
        g = np.dot(X[i], self.coef.T) + 1
        g[y[i]] -= 1
        return g

    def _violation(self, g, y, i):
        min_value = np.inf
        for k in range(g.shape[0]):
            if k == y[i] and self.dual_coef[k, i] >= self.C:
                continue
            elif k != y[i] and self.dual_coef[k, i] >= 0:
                continue
            min_value = min(min_value, g[k])
        return g.max() - min_value

    def solve_problem(self, g, y, norms, i):
        Ci = np.zeros(g.shape[0])
        Ci[y[i]] = self.C
        beta_hat = norms[i] * (Ci - self.dual_coef[:, i]) + g / norms[i]
        z = self.C * norms[i]
        beta = self.projection(beta_hat, z)
        return Ci - self.dual_coef[:, i] - beta / norms[i]

    def fit(self, X, Y):
        sample_count, feature_count = X.shape
        Y = self.label_encoder.fit_transform(Y)
        class_count = len(self.label_encoder.classes_)
        self.dual_coef = np.zeros((class_count, sample_count), dtype=np.float64)
        self.coef = np.zeros((class_count, feature_count))
        norms = np.sqrt(np.sum(X ** 2, axis=1))
        rs = check_random_state(self.random_state)
        ind = np.arange(sample_count)
        rs.shuffle(ind)

        violation_init = None
        for iteration in range(self.max_iteration):
            violation_sum = 0

            for ii in range(sample_count):
                i = ind[ii]
                if norms[i] == 0:
                    continue
                g = self._partial_gradient(X, Y, i)
                v = self._violation(g, Y, i)
                violation_sum += v
                if v < 1e-12:
                    continue
                delta = self.solve_problem(g, Y, norms, i)
                self.coef += (delta * X[i][:, np.newaxis]).T
                self.dual_coef[:, i] += delta

            if iteration == 0:
                violation_init = violation_sum
            ratio = violation_sum / violation_init
            if ratio < self.tol:
                break

        return self

    def predict(self, X):
        decision = np.dot(X, self.coef.T)
        pred = decision.argmax(axis=1)
        return self.label_encoder.inverse_transform(pred)
