###########################################################
#
# Softmax Regression
#
#   author: mayumi ohta <ohta@cl.uni-heidelberg.de>
#   last update: 12. 12. 2017
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
    """Softmax Regression (a.k.a. multinomial regression)

    Parameters
    ----------
    method : str, {'full', 'greedy', 'bandit', 'cv'}, default 'bandit'
        Method name, specifies how to compute the gradients.

    """
    def __init__(self, method='bandit'):
        self.method = method

    def _gradient(self, x, y, idx=None):
        """Computes gradient

        Parameters
        ----------
        x : {array-like, sparse matrix}, shape = [n_features,] or [1, n_features]
            Set of dev samples used to determine the stopping point.

        y : ndarray, shape = [n_classes,]
            Corresponding gold labels.

        Returns
        -------
        loss : bool
            Zero-one loss (y.argmax() != y_hat)

        grad : (dense) ndarray, shape = [n_classes, n_features]
            Gradients.

        """

        # cast shape=(n,) -> shape=(1,n)
        if not isinstance(x, sp.csr.csr_matrix):
            x = np.expand_dims(x, axis=0)

        # probability
        p = self.predict_proba(x)
        y_hat = self.inference(p)
        p = p.reshape((self.n_classes,))

        # gradients
        grad = np.zeros_like(self.W_)   # placeholder
        if self.method == 'full':
            for j in range(self.n_classes):
                g = (p[j] - y[j]) * x
                if isinstance(g, sp.csr.csr_matrix):
                    g = g.toarray()
                grad[j] = g

        elif self.method == 'greedy':   # It's buggy!! Don't use this!!
            for j in range(self.n_classes):
                y_hat_ = 1 if j == y_hat else 0   # one-hot encoding (same as y_hat[j])
                g = (p[j] - y_hat_) * x
                if isinstance(g, sp.csr.csr_matrix):
                    g = g.toarray()
                grad[j] = g

        elif self.method == 'bandit':
            y_tilde = np.random.multinomial(1, p)
            feedback = int(y.argmax() != y_tilde.argmax())
            for j in range(self.n_classes):
                g = feedback * (y_tilde[j] - p[j]) * x
                if isinstance(g, sp.csr.csr_matrix):
                    g = g.toarray()
                grad[j] = g

        elif self.method == 'cv':
            y_tilde = np.random.multinomial(1, p)
            feedback = int(y.argmax() != y_tilde.argmax())
            #if not np.isnan(self.feedback_history[idx]):    # if the table cell is filled already
            if self.avg_feedback != 0.0:
                self.avg_feedback += (feedback - self.feedback_history[idx])/self.n_samples # update avg
            self.feedback_history[idx] = feedback           # update history table
            for j in range(self.n_classes):
                g = (feedback - self.avg_feedback) * (y_tilde[j] - p[j]) * x

                if isinstance(g, sp.csr.csr_matrix):
                    g = g.toarray()
                grad[j] = g

        else:
            raise ValueError('Method %s not found. Abort.' % self.method)

        return int(y.argmax() != y_hat), grad

    def evaluate(self, X_dev, Y_dev, grad):
        """Computes loss on the dev set
            and stores the weights, gradients and function values(= losses)

        Parameters
        ----------
        X_dev : {array-like, sparse matrix}, shape (n_samples, n_features)
            Set of dev samples used to determine the stopping point.

        Y_dev : array-like, shape (n_samples, n_classes)
            Corresponding gold labels.

        grad :
            Gradients (updates).

        """
        self.train_log.append(self.cumulative_loss / self.n_seen)
        self.dev_log.append(self.score(X_dev, Y_dev))
        self.norm_log.append(self.avg_norm)
        self.variance_log.append(self.avg_variance)

        report = '[%8d] train loss: %.5f; dev loss: %.5f; norm: %.5f; var: %.5f'
        print(report % (self.n_seen, self.train_log[-1], self.dev_log[-1],
                        self.norm_log[-1], self.variance_log[-1]))
        sys.stdout.flush()

        # check optimality
        if self.dev_log[-1] <= self.best_score:
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
        """MAP Inference of the label
            If the prob is uniform, one class index will be randomly chosen

        Parameters
        ----------
        p : ndarray, shape = [n_classes,]
            Classwise probability

        Returns
        -------
        y_hat : int
            Argmax (class index) of the one-hot encoded MAP label

        """
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

    def score(self, X, Y):
        """Computes task score: zero-one loss (= 1 - accuracy)
        """
        Y_pred = self.predict(X)
        return zero_one_loss(Y.argmax(axis=1), Y_pred)

    def norm(self, g):
        """Defines squared norm.
        """
        return np.linalg.norm(g) ** 2

    def gradients_norm(self, grad):
        """Computes gradient norm.
        """
        self.avg_norm += (self.norm(grad) - self.avg_norm) / self.n_seen

    def variance(self, grad):
        """Computes variance.
        """
        self.avg_gradient += (grad - self.avg_gradient) / self.n_seen
        variance = self.norm(grad - self.avg_gradient)
        self.avg_variance += (variance - self.avg_variance) / self.n_seen


class SGD(SoftmaxRegression):
    """SGD solver for Softmax regression

    Parameters
    ----------
    eta : float, default 0.1
        Constant learning rate value.

    epochs : int, default 10
        Number of epochs.

    method : str, {'full', 'greedy', 'bandit', 'cv'}, default 'bandit'
        Method name.

    eval_every : int, default 1000
        Evaluate in every -- iterations.

    cache_path : str, default None
        Path to save weights & gradients. If ``None``, not saved.

    """
    def __init__(self, eta=0.1, epochs=10, method='bandit',
                 start_at=2, eval_every=1000, cache_path=None):
        SoftmaxRegression.__init__(self, method)
        self.eta = eta
        self.epochs = epochs
        self.start_at = start_at
        self.eval_every = eval_every
        self.cache_path = cache_path

    def fit(self, X_train, Y_train, X_dev, Y_dev):
        """Optimize weights.

        Parameters
        ----------
        X_train : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Set of train samples, where n_samples is the number of samples and
            n_features is the number of features.

        Y_train : array-like, shape = [n_samples, n_classes]
            Corresponding gold labels, where n_samples is the number of samples and
            n_classes is the number of classes.

        X_dev : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Set of dev samples used to determine the stopping point.

        Y_dev : array-like, shape (n_samples, n_classes)
            Corresponding gold labels.

        Attributes
        ----------
        W_ : ndarray, shape = [n_classes, n_features]
            Current weights.

        n_seen : int
            Number of examples seen so far.

        cumulative_loss : float
            Cumulative train loss.

        train_log : list of float
            List of train loss(= 1 - accuracy) in each iteration so far.

        dev_log : list of float
            List of dev loss(= 1 - accuracy) in each iteration so far.

        norm_log : list of float
            List of gradient norm in each iteration so far.

        variance_log : list of float
            List of variance in each iteration so far.

        best_W : ndarray, shape = [n_classes, n_features]
            Weights at the stopping point.

        best_score : float
            Loss value at the stopping point

        stop_point : int
            Iteration index which returns the lowest loss value

        Returns
        -------
        self : object
            Returns self.

        """
        self.n_seen = 1

        self.n_samples, self.n_features = X_train.shape
        self.n_classes = Y_train.shape[1]
        self.W_ = np.zeros((self.n_classes, self.n_features))  # initialize weights
        grad = np.zeros((self.n_classes, self.n_features))

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
            self.feedback_history = np.empty((self.n_samples,))
            self.feedback_history[:] = np.nan

        self.best_W = np.zeros((self.n_classes, self.n_features))
        self.best_score = 1.0
        self.stop_point = 0

        # train loop
        for epoch in range(self.epochs):
            if self.method == 'cv' and epoch == self.start_at:
                self.avg_feedback = np.mean(self.feedback_history)

            # shuffle
            perm = np.random.permutation(self.n_samples)
            for i in perm:
                # compute avg gradients
                loss, grad = self._gradient(X_train[i], Y_train[i], idx=i)

                # update
                self.W_ -= self.eta * grad          # update weights

                self.cumulative_loss += loss
                self.gradients_norm(grad)           # update avg_norm
                self.variance(grad)                 # update avg_variance

                # evaluate
                if self.n_seen % self.eval_every == 0:
                    self.evaluate(X_dev, Y_dev, grad)

                self.n_seen += 1                    # update number of seen examples

        # final results
        self.evaluate(X_dev, Y_dev, grad)

        return self


class SAGA(SoftmaxRegression):
    """SAGA solver for Softmax regression

    Parameters
    ----------
    eta : float, default 0.1
        Constant learning rate value.

    epochs : int, default 10
        Number of epochs.

    method : str, {'full', 'greedy', 'bandit'}, default 'bandit'
        Method name, specifies how to compute the gradients.

    start_at : int, default 2
        Epoch index when to start SAGA update. If set to 0, it starts
        computing average from the partially filled table.

    eval_every : int, default 1000
        Evaluate in every -- iterations.

    cache_path : str, default None
        Path to save weights & gradients. If ``None``, not saved.

    """
    def __init__(self, eta=0.1, epochs=10, method='bandit', start_at=2,
                 eval_every=1000, cache_path=None):
        SoftmaxRegression.__init__(self, method)
        self.eta = eta
        self.epochs = epochs
        self.start_at = start_at
        self.eval_every = eval_every
        self.cache_path = cache_path

    def fit(self, X_train, Y_train, X_dev, Y_dev):
        """Optimize weights.

        Parameters
        ----------
        X_train : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Set of train samples, where n_samples is the number of samples and
            n_features is the number of features.

        Y_train : array-like, shape = [n_samples, n_classes]
            Corresponding gold labels, where n_samples is the number of samples and
            n_classes is the number of classes.

        X_dev : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Set of dev samples used to determine the stopping point.

        Y_dev : array-like, shape (n_samples, n_classes)
            Corresponding gold labels.

        Attributes
        ----------
        W_ : ndarray, shape = [n_classes, n_features]
            Current weights.

        n_seen : int
            Number of examples seen so far.

        cumulative_loss : float
            Cumulative train loss.

        train_log : list of float
            List of train loss(= 1 - accuracy) in each iteration so far.

        dev_log : list of float
            List of dev loss(= 1 - accuracy) in each iteration so far.

        norm_log : list of float
            List of gradient norm in each iteration so far.

        variance_log : list of float
            List of variance in each iteration so far.

        best_W : ndarray, shape = [n_classes, n_features]
            Weights at the stopping point.

        best_score : float
            Loss value at the stopping point

        stop_point : int
            Iteration index which returns the lowest loss value

        Returns
        -------
        self : object
            Returns self.

        """
        self.n_seen = 1

        self.n_samples, self.n_features = X_train.shape
        self.n_classes = Y_train.shape[1]
        self.W_ = np.zeros((self.n_classes, self.n_features))
        self.G_ = np.zeros((self.n_samples, self.n_classes, self.n_features))
        v = np.zeros((self.n_classes, self.n_features))
        avg_G = np.zeros((self.n_classes, self.n_features))

        self.cumulative_loss = 0.0
        self.train_log = []
        self.dev_log = []
        self.norm_log = []
        self.variance_log = []

        self.avg_gradient = np.zeros_like(self.W_)  # only for variance calc.
        self.avg_norm = 0.0                         # (avg over T, not N)
        self.avg_variance = 0.0                     # (avg over T, not N)

        self.best_W = np.zeros((self.n_classes, self.n_features))
        self.best_score = 1.0
        self.stop_point = 0

        # train loop
        for epoch in range(self.epochs):
            if epoch == self.start_at:
                avg_G = np.mean(self.G_, axis=0)

            # shuffle
            perm = np.random.permutation(self.n_samples)
            for i in perm:
                # compute gradients
                loss, grad = self._gradient(X_train[i], Y_train[i])

                if epoch < self.start_at:               # sgd update
                    v = grad
                else:                                   # saga update
                    v = grad - self.G_[i] + avg_G       # v = new - old + avg
                    n = self.n_seen if self.n_seen < self.n_samples else self.n_samples # =min(n_seen, n_samples)
                    avg_G += (grad - self.G_[i]) / n    # update avg_G

                # update
                self.W_ -= self.eta * v             # update weights
                self.G_[i] = grad                   # update history table

                self.cumulative_loss += loss
                self.gradients_norm(v)              # update avg_norm
                self.variance(v)                    # update avg_variance

                # evaluate
                if self.n_seen % self.eval_every == 0:
                    self.evaluate(X_dev, Y_dev, v)

                self.n_seen += 1                    # update number of seen examples

        # final results
        self.evaluate(X_dev, Y_dev, v)

        return self
