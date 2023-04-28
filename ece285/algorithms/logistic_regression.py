"""
Logistic regression model
"""

import numpy as np
import math
from tqdm import tqdm


class Logistic(object):
    def __init__(self, n_class: int, lr: float, epochs: int, weight_decay: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.threshold = 0.5  # To threshold the sigmoid
        self.weight_decay = weight_decay

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        # pass
        return 1/(1 + np.exp(-z))

    def train(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.
        Train a logistic regression classifier for each class i to predict the probability that y=i

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        N, D = X_train.shape
        self.w = weights

        # TODO: implement me
        # one-hot encoding of labels
        y_onehot = np.zeros((N, 10))
        y_onehot[np.arange(N), y_train] = 1
        
        for epoch in range(self.epochs):
            # compute softmax probabilities
            y_pred = np.dot(X_train, self.w.T)
            y_pred = self.sigmoid(y_pred)

            # compute loss and gradient
            log_loss = -1 / N * np.sum(y_onehot * np.log(y_pred) + (1 - y_onehot) * np.log(1 - y_pred))
            l2_loss = 1 / N * 0.5 * self.weight_decay * np.sum(np.square(self.w))
            
            loss = log_loss + l2_loss
           
            grad = 1 / N * (np.dot(X_train.T, y_pred - y_onehot).T + self.weight_decay * self.w)
            
            # update weights
            self.w -= self.lr * grad

        return self.w

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        # pass
        y_pred = np.dot(X_test, self.w.T)
        y_pred = np.argmax(y_pred, axis=1)

        return y_pred