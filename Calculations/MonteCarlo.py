import numpy as np

def monte_carlo_var(portfolio_returns, num_simulations, confidence_level):
    simulated_returns = np.random.choice(portfolio_returns, size=num_simulations, replace=True)
    
    losses = -simulated_returns
    
    var = np.percentile(losses, (1 - confidence_level) * 100)
    
    return var

portfolio_returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, -0.02, 0.04, -0.01])

num_simulations = 10000
confidence_level = 0.95

var_estimate = monte_carlo_var(portfolio_returns, num_simulations, confidence_level)

print(f"Estimated Value at Risk (VaR) at {confidence_level*100}% confidence level: {var_estimate:.4f}")