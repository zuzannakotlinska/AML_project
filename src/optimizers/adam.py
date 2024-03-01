import numpy as np


class Adam:
    def __init__(self, eta: float, beta1: float, beta2: float) -> None:
        """
        Initialize the optimizer.
        Args:
            eta: Learning rate
            beta1: Decay rate 1 for the moving average of the gradient (must be between 0 and 1)
            beta2: Decay rate 2 for the moving average of the gradient (must be between 0 and 1)
        """
        self.eta = eta
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
        w -= self.eta * m_hat / (np.sqrt(v_hat) + 1e-8)
        return w
