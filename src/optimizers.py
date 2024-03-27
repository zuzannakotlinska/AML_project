import numpy as np


class Adam:
    def __init__(
        self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999
    ) -> None:
        """
        Initialize the optimizer.
        Args:
            learning_rate: Learning rate
            beta1: Decay rate 1 for the moving average of the gradient (must be between 0 and 1)
            beta2: Decay rate 2 for the moving average of the gradient (must be between 0 and 1)
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = 0
        self.v = 0
        self.t = 0

    def update(self, w: np.array, grad_wrt_w: np.array) -> np.array:
        """
        Update the weights according to the Adaptive moment estimation update rule.
        Args:
            w: The weights
            grad_wrt_w: The gradient of the loss with respect to the weights
        """
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_wrt_w
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad_wrt_w**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        w -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
        return w
    
class SGD:
    def __init__(self, learning_rate: float) -> None:
        """
        Initialize the optimizer.
        Args:
            learning_rate: Learning rate
        """
        self.learning_rate = learning_rate

    def update(self, w: np.array, grad_wrt_w: np.array) -> np.array:
        """
        Update the weights using the stochastic gradient descent update rule.
        Args:
            w: The weights
            grad_wrt_w: The gradient of the loss with respect to the weights
        """
        w -= self.learning_rate * grad_wrt_w
        return w
    
class IRLS:
    def __init__(self, max_iter=100, tol=1e-4):
        """
        Initialize the IRLS optimizer.
        Args:
            max_iter: Maximum number of iterations
            tol: Tolerance for the change in weights
        """
        self.max_iter = max_iter
        self.tol = tol

    def update(self, w: np.array, grad_wrt_w: np.array) -> np.array:
        """
        Update the weights using the Newton-Raphson method.
        Args:
            w: The current weights
            grad_wrt_w: The gradient of the loss with respect to the weights
        """
        hessian = np.dot(grad_wrt_w.T, grad_wrt_w)
        grad = np.dot(grad_wrt_w.T, grad_wrt_w)
        w_new = w - np.dot(np.linalg.pinv(hessian), grad)
        return w_new
