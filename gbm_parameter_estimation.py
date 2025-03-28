import numpy as np
from scipy.optimize import minimize
import pandas as pd
from scipy.stats import norm

class GBMParameterEstimator:
    """
    Geometric Brownian Motion Parameter Estimator
    GBM form: dS_t = μ·S_t·dt + σ·S_t·dW_t
    Estimating both drift parameter μ and volatility parameter σ
    """
    
    def __init__(self, stock_prices, dt, risk_free_rate=None):
        """
        Initialize estimator
        
        Parameters:
        - stock_prices: Stock price time series data (numpy array)
        - dt: Time step
        - risk_free_rate: Risk-free rate r (annualized), used as initial guess for μ parameter if provided
        """
        self.stock_prices = stock_prices
        self.dt = dt
        self.risk_free_rate = risk_free_rate
        self.n = len(stock_prices)
        
        # Calculate log returns for parameter estimation
        self.log_returns = np.diff(np.log(stock_prices))
    
    def simulate_gbm(self, mu, sigma, n_steps=None, initial_price=None, random_seed=None):
        """
        Simulate Geometric Brownian Motion path
        
        Parameters:
        - mu: Drift parameter
        - sigma: Volatility parameter
        - n_steps: Number of simulation steps, defaults to length of price series at initialization
        - initial_price: Initial price, defaults to first price at initialization
        - random_seed: Random seed for reproducibility
        
        Returns:
        - simulated_prices: Simulated price series
        """
        if n_steps is None:
            n_steps = self.n
        
        if initial_price is None:
            initial_price = self.stock_prices[0]
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize price series
        simulated_prices = np.zeros(n_steps)
        simulated_prices[0] = initial_price
        
        # Use Euler-Maruyama method for discretization
        for t in range(1, n_steps):
            drift = mu * simulated_prices[t-1] * self.dt
            diffusion = sigma * simulated_prices[t-1] * np.sqrt(self.dt) * np.random.normal()
            simulated_prices[t] = simulated_prices[t-1] + drift + diffusion
        
        return simulated_prices
    
    def log_likelihood(self, params):
        """
        Calculate log-likelihood function for Geometric Brownian Motion
        
        Parameters:
        - params: Array containing [mu, sigma] parameters
        
        Returns:
        - log_likelihood: Log-likelihood value
        """
        mu, sigma = params
        
        # Log returns of GBM follow normal distribution
        # μ_r = (mu - 0.5*σ²)*dt, σ_r = σ*√dt
        mu_r = (mu - 0.5 * sigma**2) * self.dt
        variance = sigma**2 * self.dt
        
        # Calculate sum of log probability density for each log return
        log_likelihood = np.sum(norm.logpdf(self.log_returns, mu_r, np.sqrt(variance)))
        
        return log_likelihood
    
    def negative_log_likelihood(self, params):
        """
        Calculate negative log-likelihood function for minimization
        
        Parameters:
        - params: Array containing [mu, sigma] parameters
        
        Returns:
        - negative_ll: Negative log-likelihood value
        """
        mu, sigma = params
        
        # Parameter constraint check
        if sigma <= 0:
            return 1e10  # Return large penalty value
        
        return -self.log_likelihood(params)
    
    def estimate_parameters(self, initial_params=None, bounds=None):
        """
        Estimate parameters for Geometric Brownian Motion model
        
        Parameters:
        - initial_params: Initial [mu, sigma] estimates, uses sample statistics if not provided
        - bounds: Bounds for parameters, uses default bounds if not provided
        
        Returns:
        - estimated_params: Estimated [mu, sigma] parameters
        - log_likelihood: Maximum log-likelihood value
        """
        # If initial parameters not provided, estimate from sample statistics
        if initial_params is None:
            # Estimate mu and sigma from log returns
            if self.risk_free_rate is not None:
                initial_mu = self.risk_free_rate
            else:
                # Estimate mu from average log returns
                initial_mu = np.mean(self.log_returns) / self.dt + 0.5 * np.var(self.log_returns) / self.dt
            
            # Estimate annualized volatility from log returns
            initial_sigma = np.std(self.log_returns) / np.sqrt(self.dt)
            
            initial_params = [initial_mu, initial_sigma]
        
        # If bounds not provided, set default bounds
        if bounds is None:
            bounds = [(-0.5, 0.5), (0.001, 2.0)]  # Reasonable ranges for mu and sigma
        
        # Use L-BFGS-B algorithm for optimization (supports bound constraints)
        result = minimize(
            self.negative_log_likelihood,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'disp': True}
        )
        
        # Check optimization result
        if result.success:
            estimated_params = result.x
            print(f"Optimization successful: {result.message}")
        else:
            print(f"Optimization failed: {result.message}")
            estimated_params = initial_params
        
        # Calculate maximum log-likelihood value
        max_log_likelihood = self.log_likelihood(estimated_params)
        
        return estimated_params, max_log_likelihood
    
    def calculate_confidence_interval(self, params, confidence=0.95):
        """
        Calculate confidence interval for parameter estimates
        
        Parameters:
        - params: Estimated [mu, sigma] parameters
        - confidence: Confidence level, default 0.95 for 95% confidence interval
        
        Returns:
        - confidence_intervals: Dictionary with confidence intervals for each parameter
        """
        mu, sigma = params
        n = len(self.log_returns)
        
        # Use inverse of Fisher information matrix to estimate parameter variance
        # For GBM, the Fisher information matrix has entries related to mu and sigma
        var_mu = sigma**2 / (n * self.dt)
        var_sigma = sigma**2 / (2 * n)
        
        # Calculate z-score
        z = norm.ppf((1 + confidence) / 2)
        
        # Calculate confidence intervals
        mu_lower = mu - z * np.sqrt(var_mu)
        mu_upper = mu + z * np.sqrt(var_mu)
        sigma_lower = sigma - z * np.sqrt(var_sigma)
        sigma_upper = sigma + z * np.sqrt(var_sigma)
        
        return {
            'mu': (mu_lower, mu_upper),
            'sigma': (sigma_lower, sigma_upper)
        }
    
    def plot_results(self, estimated_params, title="Geometric Brownian Motion Parameter Estimation Results", n_simulations=5):
        """
        Plot results, including original data and simulated paths
        
        Parameters:
        - estimated_params: Estimated [mu, sigma] parameters
        - title: Chart title
        - n_simulations: Number of simulation paths
        """
        import matplotlib.pyplot as plt
        
        mu, sigma = estimated_params
        
        plt.figure(figsize=(10, 6))
        
        # Plot original data
        plt.plot(self.stock_prices, 'b-', label='Observed Prices')
        
        # Plot several simulation paths
        for i in range(n_simulations):
            simulated_prices = self.simulate_gbm(mu, sigma)
            plt.plot(simulated_prices, 'r-', alpha=0.3, label='Simulated Path' if i == 0 else "")
        
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Display estimated parameters
        textstr = f'Estimated Parameters:\n$\mu$ = {mu:.4f}\n$\sigma$ = {sigma:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.show()
        
        return plt.gcf() 