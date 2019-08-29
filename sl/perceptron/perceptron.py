# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 22:08:29 2019

@author: IMRANAX
"""

import numpy as np

class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        #print("#, w0, w1, w2, Sepal (L), Petal (L), target, Netinput + w0, class Prediction, w udpated")
        for _ in range(self.n_iter):
            errors = 0
            i =1
            for xi, target in zip(X, y):

                t_input = self.net_input(xi) + self.w_[0]

                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)


                #print("%s, %s, %s, %s, %s, %s, %s, %s" % (_, i, self.w_, xi, target, t_input, target, update))

                if i == 99:
                    i = 0
                i += 1
            self.errors_.append(errors)
            print("Errors, %s" % self.errors_)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)