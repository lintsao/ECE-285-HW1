"""
Linear Regression model
"""

import numpy as np


class Linear(object):
    def __init__(self, n_class: int, lr: float, epochs: int, weight_decay: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # Initialize in train
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.weight_decay = weight_decay

    def train(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Train the classifier.

        Use the linear regression update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        N, D = X_train.shape
        self.w = weights

        # TODO: implement me
        loss = 0
        y_onehot = np.zeros((N, self.n_class))
        y_onehot[np.arange(N), y_train] = 1
            
        for epoch in range(self.epochs):
            # Calculate the predictions
            y_pred = np.dot(X_train, self.w.T)
#             print("y1", y_pred.shape)

            # Calculate the mean squared error loss
            mse_loss = np.mean((y_pred - y_onehot) ** 2)

            # Add L2 regularization to the loss
#             print(self.w[:, :].shape)
#             print(self.w[:, :-1].shape)
            l2_loss = 0.5 * self.weight_decay * np.sum(self.w ** 2)
            loss = mse_loss + l2_loss

            # Calculate the gradient of the loss with respect to the weights
#             print(np.dot(X_train.T, y_pred - y_train).shape)
            grad = np.dot(X_train.T, y_pred - y_onehot).T + self.weight_decay * self.w

            # Update the weights using the gradient and the learning rate
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
        y_pred = X_test.dot(self.w.T)
        y_pred = np.argmax(y_pred, axis=1)
        
        return y_pred