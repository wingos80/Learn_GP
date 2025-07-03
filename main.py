import numpy as np
import matplotlib.pyplot as plt

class GaussianProcess:
    def __init__(self, kernel_func, noise=1e-5, mean_prior=0):
        """Initialize GP with kernel function, noise level, and mean prior"""
        self.kernel_func = kernel_func
        self.noise = noise
        self.mean_prior = mean_prior
        self.X_train = None
        self.y_train = None
        self.K_inv = None
    
    def update(self, X_new, y_new):
        """Add new data points to the GP"""
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
        if X_test is None:
            X_test = np.linspace(np.min(self.X_train)-1, np.max(self.X_train)+1, 100)
        
        mean, var = self.predict(X_test)
        std = np.sqrt(var)
        
        plt.figure(figsize=(10, 6))
        if self.X_train is not None:
            plt.plot(self.X_train, self.y_train + self.mean_prior, 'k*', markersize=10, label='Data')
        plt.plot(X_test, mean, 'b-', label='Mean prediction')
        plt.fill_between(X_test, mean-2*std, mean+2*std, alpha=0.2, color='blue', label='95% CI')
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.legend()
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Define a simple RBF kernel
    def rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)
    
    # Create GP with mean prior of 0.5
    gp = GaussianProcess(lambda x1, x2: rbf_kernel(x1, x2, 1.0, 1.0), mean_prior=0.5)
    
    # Initial data
    X_data = np.array([1, 3, 5])
    y_data = np.array([1.2, 0.5, 1.8])
    gp.update(X_data, y_data)
    gp.plot()
    
    # Add new data point
    print("\nAdding new data point at x=4, y=1.5")
    gp.update(4, 1.5)
    gp.plot()
    
    # Add another data point
    print("\nAdding new data point at x=2, y=0.8")
    gp.update(2, 0.8)
    gp.plot()