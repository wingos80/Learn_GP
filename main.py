import jax.numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import logging
from jax import Array
class COLOR:
    PURPLE = "\033[1;35;48m"
    CYAN = "\033[1;36;48m"
    BOLD = "\033[1;37;48m"
    BLUE = "\033[1;34;48m"
    GREEN = "\033[1;32;48m"
    YELLOW = "\033[1;33;48m"
    RED = "\033[1;31;48m"
    BLACK = "\033[1;30;48m"
    UNDERLINE = "\033[4;37;48m"
    END = "\033[1;37;0m"

# Defining the default logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create handler
handler = logging.StreamHandler()

# Custom formatter to apply color to log statements
class ColorFormatter(logging.Formatter):
    def format(self, record):
        message = super().format(record)
        if record.levelname == "INFO":
            return f"{COLOR.GREEN}{message}{COLOR.END}"
        elif record.levelname == "WARNING":
            return f"{COLOR.YELLOW}{message}{COLOR.END}"
        elif record.levelname == "ERROR":
            return f"{COLOR.RED}{message}{COLOR.END}"

        return message


formatter = ColorFormatter(
    fmt="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d >>> %(message)s"
)
handler.setFormatter(formatter)

# Avoid adding multiple handlers if logger is re-used (important!)
if not logger.hasHandlers():
    logger.addHandler(handler)

class GaussianProcess:
    def __init__(self, kernel_func: Callable, noise: float = 1e-5, mean_prior: float = 0):
        """Initialize GP with kernel function, noise level, and mean prior"""
        self.kernel_func = kernel_func
        self.noise = noise
        self.mean_prior = mean_prior
        self.X_train = None
        self.y_train = None
        self.K_inv = None
    
    def update(self, X_new: Array, y_new: Array):
        """Add new data points to the GP"""
        # make sure the domain and range are column vectors
        X_new = np.array(X_new).reshape(-1, 1)
        y_new = np.array(y_new).reshape(-1, 1) - self.mean_prior
        
        if self.X_train is None:
            self.X_train = X_new
            self.y_train = y_new
        else:
            self.X_train = np.vstack([self.X_train, X_new])
            self.y_train = np.vstack([self.y_train, y_new])
        
        # Update inverse kernel matrix efficiently (would be better with rank-1 update)
        self.K_inv = None
    
    def _compute_kernel_inverse(self):
        """Compute inverse of kernel matrix with noise"""
        K = self.kernel_func(self.X_train, self.X_train)
        self.K_inv = np.linalg.inv(K + self.noise * np.eye(len(self.X_train)))
    
    def predict(self, X_test):
        """Make predictions at new test points"""
        X_test = np.array(X_test).reshape(-1, 1)
        
        if self.X_train is None:
            # Prior if no data
            mean = np.zeros(len(X_test)) + self.mean_prior
            cov = self.kernel_func(X_test, X_test)
            return mean, np.diag(cov)
        
        if self.K_inv is None:
            self._compute_kernel_inverse()
        
        # Compute kernel matrices
        K_s = self.kernel_func(self.X_train, X_test)
        K_ss = self.kernel_func(X_test, X_test)
        
        # Compute predictive mean and covariance
        mean = self.mean_prior + K_s.T @ self.K_inv @ self.y_train
        cov = K_ss - K_s.T @ self.K_inv @ K_s
        
        return mean.flatten(), np.diag(cov)
    
    def plot(self, X_test=None):
        """Plot the GP with training data and uncertainty"""
        xlims = (0, 10)
        if X_test is None:
            try:
                X_test = np.linspace(np.min(self.X_train)-1, np.max(self.X_train)+1, 100)
            except Exception as e:
                logger.warning(f"Got error: {e}")
                logger.info("Using default range [0, 10) for X_test.")
                X_test = np.linspace(xlims[0], xlims[1], 100)
        
        mean, var = self.predict(X_test)
        std = np.sqrt(var)
        
        plt.figure(figsize=(10, 6))
        if self.X_train is not None:
            plt.plot(self.X_train, self.y_train + self.mean_prior, 'k*', markersize=10, label='Data')
        plt.plot(X_test, mean, 'b-', label='Mean prediction')
        plt.fill_between(X_test, mean-2*std, mean+2*std, alpha=0.2, color='blue', label='95% CI')
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.xlim(xlims)
        plt.legend()
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Define a simple RBF kernel, ||x - y||^2 = (x - y)T(x - y) = xTx + yTy - 2xTy
    def rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)
    
    # Create GP with mean prior of 0.5
    gp = GaussianProcess(lambda x1, x2: rbf_kernel(x1, x2, 1.0, 1.0), noise=1e-5, mean_prior=0.5)
    
    # Initial data
    X_data = np.array([1, 3, 5, 9])
    y_data = np.array([1.2, 0.5, 1.8, 1.8])
    gp.update(X_data, y_data)
    gp.plot()
    
    # Add new data point
    logger.info("\nAdding new data point at x=4, y=1.5")
    gp.update([4], [1.5])
    gp.plot()
    
    # Add another data point
    logger.info("\nAdding new data point at x=2, y=0.8")
    gp.update([2], [0.8])
    gp.plot()

    logger.warning("\nAdding new data point at x=(6, 6.7), y=(2.0, 6.4)")
    gp.update(np.array([6, 6.7]), np.array([2.0, 6.4]))
    gp.plot()