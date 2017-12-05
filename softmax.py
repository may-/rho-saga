###########################################################
#
# Softmax Regression
#
###########################################################


from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import scipy.sparse as sp

from sklearn.utils.extmath import safe_sparse_dot, softmax
from sklearn.metrics import zero_one_loss
from sklearn.base import BaseEstimator, ClassifierMixin


class SoftmaxRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, method='full'):
        self.method = method

    def _gradient(self, x, y):
        n_classes = y.shape[0]

        if not isinstance(x, sp.csr.csr_matrix):
            x = np.expand_dims(x, axis=0) # cast shape=(n,) -> shape=(1,n)

        p = self.predict_proba(x)
        y_hat = self.inference(p)
        p = p.reshape((n_classes,))

        grad = np.zeros_like(self.W_)
        if self.method == 'full':
            for j in range(n_classes):
                g = (p[j] - y[j]) * x
                if isinstance(g, sp.csr.csr_matrix):
                    g = g.toarray()
                grad[j] = g

        elif self.method == 'bandit':
            y_tilde = np.random.multinomial(1, p)
            feedback = (y.argmax() != y_tilde.argmax())
            for j in range(n_classes):
                g = feedback * (y_tilde[j] - p[j]) * x
                if isinstance(g, sp.csr.csr_matrix):
                    g = g.toarray()
                grad[j] = g

        elif self.method == 'cv':
            y_tilde = np.random.multinomial(1, p)
            feedback = (y.argmax() != y_tilde.argmax())
            self.avg_feedback += (feedback - self.avg_feedback)/(self.n_seen + 1.0)
            for j in range(n_classes):
                g = (feedback - self.avg_feedback) * (y_tilde[j] - p[j]) * x
                if isinstance(g, sp.csr.csr_matrix):
                    g = g.toarray()
                grad[j] = g

        else:
            raise ValueError('method %s not found. Abort.' % self.method)

        return (y.argmax() != y_hat), grad

    def evaluate(self, X_dev, Y_dev, grad):
        self.train_log.append(self.cumulative_loss / self.n_seen)
        self.dev_log.append(self.score(X_dev, Y_dev))
        self.norm_log.append(self.avg_norm)
        self.variance_log.append(self.avg_variance)

        report = '[%8d] train loss: %.5f; dev loss: %.5f; norm: %.5f; var: %.5f'
        print(report % (self.n_seen, self.train_log[-1], self.dev_log[-1],
                        self.norm_log[-1], self.variance_log[-1]))
        sys.stdout.flush()

        # check optimality
        if self.dev_log[-1] >= self.best_score:
            self.best_score = self.dev_log[-1]
            self.best_W = self.W_
            self.stop_point = self.n_seen

        # save weights & gradients
        if self.cache_path:
            w = self.W_
            g = grad
            if isinstance(grad, sp.csr.csr_matrix):
                w = sp.csr_matrix(self.W_)
                g = sp.csr_matrix(grad)
            filename = self.cache_path + '_' + str(self.n_seen)
            np.savez_compressed(filename, w=w, g=g)

    def inference(self, p):
        y_hat = p.argmax() # Only the first occurrence is returned
        maxP = p.max()
        ind = np.where(p == maxP)[0]
        if len(ind) > 1:
            y_hat = np.random.choice(ind)
        return y_hat

    def decision_function(self, X):
        return safe_sparse_dot(X, self.W_.T)

    def predict(self, X):
        return [self.inference(p) for p in self.predict_proba(X)]

    def predict_proba(self, X):
        return softmax(self.decision_function(X))

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def score(self, X, Y):  # thus is equal to (1 - accuracy)
        Y_pred = self.predict(X)
        return zero_one_loss(Y.argmax(axis=1), Y_pred)

    def norm(self, g):
        return np.linalg.norm(g) ** 2

    def gradients_norm(self, grad):
        self.avg_norm += (self.norm(grad) - self.avg_norm) / self.n_seen

    def variance(self, grad):
        self.avg_gradient += (grad - self.avg_gradient) / self.n_seen
        variance = self.norm(grad - self.avg_gradient)
        self.avg_variance += (variance - self.avg_variance) / self.n_seen


class SGD(SoftmaxRegression):
    def __init__(self, eta=0.1, epochs=10, method='full',
                 eval_every=1000, cache_path=None):
        SoftmaxRegression.__init__(self, method)
        self.eta = eta
        self.epochs = epochs
        self.eval_every = eval_every
        self.cache_path = cache_path

    def fit(self, X_train, Y_train, X_dev, Y_dev):
        self.n_seen = 0

        n_samples, n_features = X_train.shape
        n_classes = Y_train.shape[1]
        self.W_ = np.zeros((n_classes, n_features))  # initialize weights
        grad = np.zeros((n_classes, n_features))

        self.cumulative_loss = 0.0
        self.train_log = []
        self.dev_log = []
        self.norm_log = []
        self.variance_log = []

        self.avg_gradient = np.zeros_like(self.W_)  # only for variance calc.
        self.avg_norm = 0.0                         # (avg over T, not N)
        self.avg_variance = 0.0                     # (avg over T, not N)

        if self.method == 'cv':
            self.avg_feedback = 0.0

        self.best_W = np.zeros((n_classes, n_features))
        self.best_score = 0.0
        self.stop_point = 0

        # train loop
        for epoch in range(self.epochs):
            # shuffle
            perm = np.random.permutation(n_samples)
            for i in perm:
                # compute avg gradients
                loss, grad = self._gradient(X_train[i], Y_train[i])

                # update
                self.W_ -= self.eta * grad          # update weights

                self.cumulative_loss += loss
                self.n_seen += 1                    # update number of seen examples
                self.gradients_norm(grad)           # update avg_norm
                self.variance(grad)                 # update avg_variance

                # evaluate
                if self.n_seen % self.eval_every == 0:
                    self.evaluate(X_dev, Y_dev, grad)

        # final results
        self.evaluate(X_dev, Y_dev, grad)

        return self


class SAGA(SoftmaxRegression):
    def __init__(self, eta=0.1, epochs=10, method='full',
                 eval_every=1000, cache_path=None):
        SoftmaxRegression.__init__(self, method)
        self.eta = eta
        self.epochs = epochs
        self.eval_every = eval_every
        self.cache_path = cache_path

    def fit(self, X_train, Y_train, X_dev, Y_dev):
        self.n_seen = 0

        n_samples, n_features = X_train.shape
        n_classes = Y_train.shape[1]
        self.W_ = np.zeros((n_classes, n_features))
        self.G_ = np.zeros((n_samples, n_classes, n_features))
        grad = np.zeros((n_classes, n_features))
        avg_G = np.zeros((n_classes, n_features))

        self.cumulative_loss = 0.0
        self.train_log = []
        self.dev_log = []
        self.norm_log = []
        self.variance_log = []

        self.avg_gradient = np.zeros_like(self.W_)  # only for variance calc.
        self.avg_norm = 0.0                         # (avg over T, not N)
        self.avg_variance = 0.0                     # (avg over T, not N)

        self.best_W = np.zeros((n_classes, n_features))
        self.best_score = 0.0
        self.stop_point = 0

        # train loop
        for epoch in range(self.epochs):
            # take average
            if epoch == 1:
                avg_G = np.mean(self.G_, axis=0)
                assert avg_G.shape == self.W_.shape

            # shuffle
            perm = np.random.permutation(n_samples)
            for i in perm:
                # compute gradients
                loss, grad = self._gradient(X_train[i], Y_train[i])

                if epoch == 0:                      # sgd update
                    v = grad
                else:                               # saga update
                    v = grad - self.G_[i] + avg_G   # v = new - old + avg
                    avg_G -= (1.0/n_samples) * (self.G_[i] - v)

                # update
                self.W_ -= self.eta * v             # update weights
                self.G_[i] = v                      # update history table

                self.cumulative_loss += loss
                self.n_seen += 1                    # update number of seen examples
                self.gradients_norm(v)              # update avg_norm
                self.variance(v)                    # update avg_variance

                # evaluate
                if self.n_seen % self.eval_every == 0:
                    self.evaluate(X_dev, Y_dev, grad)

        # final results
        self.evaluate(X_dev, Y_dev, grad)

        return self
