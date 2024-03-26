import numpy as np


class LogisticRegression:
    def __init__(
        self, optimizer, eta: float = 0.05, epochs: int = 1000, random_seed: int = 1
    ) -> None:
        """
        Initialize the logistic regression model.
        Args:
            optimizer: The optimizer to use for updating the weights (class from optimizers folder)
            eta: Learning rate
            epochs: Number of passes over the training set
            random_seed: Seed for initializing random weights and shuffling the dataset
        """
        self.eta = eta
        self.epochs = epochs
        self.random_seed = random_seed
        self.optimizer = optimizer

    def initialize_weights(self, n_features: int) -> None:
        """
        Initialize weights randomly.
        Args:
            n_features: Number of features in the training set
        """
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        # Initialize weights between -1 and 1
        self.w = 2 * np.random.random(n_features) - 1

    def sigmoid(self, z: np.array) -> np.array:
        """
        Sigmoid activation function.
        Args:
            z: The input to the activation function
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.array, y: np.array) -> None:
        """
        Fit the logistic regression model.
        Args:
            X: The training input samples
            y: The target values
        """
        n_samples, n_features = X.shape
        self.initialize_weights(n_features)
        for _ in range(self.epochs):
            for i in range(n_samples):
                # Compute the negative gradient of the log-likelihood
                # for the ith training sample
                z = np.dot(X[i], self.w)
                y_pred = self.sigmoid(z)
                error = y_pred - y[i]
                grad_wrt_w = X[i] * error
                # Update the weights
                self.w = self.optimizer.update(self.w, grad_wrt_w)

    def predict(self, X: np.array) -> np.array:
        """
        Predict binary labels.
        Args:
            X: The input samples
        """
        # Compute the output of the neuron
        z = np.dot(X, self.w)
        y_pred = np.round(self.sigmoid(z)).astype(int)
        return y_pred
