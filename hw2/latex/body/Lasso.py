# Implement Lasso for Problems 5-6.

import numpy as np

class Lasso:
    def __init__(self, regularized_lambda):
        self.rl = regularized_lambda

    def train(self, X, y, w, delta=1E-3, verbose=False):
        ''' Train the lasso using coordinate descent. '''
        n, d = X.shape
        self.w = np.zeros(d) if w is None else w

        # Let history hold the history of loss measures.
        self.history = []

        # Precompute a to speed things up considerably.
        a = 2 * np.sum(X ** 2, axis=0)

        # Let i be the current iteration.
        i = 0

        # while not converged
        w_change = float('inf')
        while w_change >= delta:
            # For measuring iteration differences.
            w_prev = np.copy(self.w)

            # b <- \frac{1}{n} \sum_{i=1}^n ( y_i - \sum_{j=1}^d w_j * x_{i,j} )
            self.b = np.mean(y - X.dot(self.w))

            # for k in {1, 2, ... d} do
            for k in range(d):
                # Access precomputed a.
                a_k = a[k]

                # Used in computing c_k
                not_k = np.arange(d) != k

                # c_k <- 2 * \sum_{i=1}^n x_{i,k} * (y_i - (b + \sum_{j \neq k} w_j * x_{i,j}))
                c_k = 2 * np.sum(X[:, k] * (y - (self.b + X[:, not_k].dot(self.w[not_k]))), axis=0)

                # w_k is a piecewise assignment.
                # w_k <- (c_k + \lambda) / a_k  if c_k < -\lambda
                # w_k <- (c_k - \lambda) / a_k  if c_k > \lambda
                # w_k <- 0                      if c_k \in [-\lambda, \lambda]
                condlist = [c_k < -self.rl, c_k > self.rl, ]
                funclist = [(c_k + self.rl) / a_k, (c_k - self.rl) / a_k, 0]
                self.w[k] = np.float(np.piecewise(c_k, condlist, funclist))

            # Compute the loss.
            loss = self.loss(X, y)
            self.history.append(loss)

            # Output progress.
            if verbose:
                print(f'\t{i}\tLoss: {loss}')

            # Update the change in w with the new values.
            w_change = np.linalg.norm(self.w - w_prev, ord=np.inf)

            # Increment the current iteration.
            i += 1

        return self.history

    def loss(self, X, y):
        ''' Compute the lasso loss. '''
        return (np.linalg.norm(X.dot(self.w) + self.b - y)) ** 2 + self.rl * np.linalg.norm(self.w, ord=1)

    def predict(self, X):
        ''' Predict y_hat using the trained model. '''
        return X.dot(self.w) + self.b
