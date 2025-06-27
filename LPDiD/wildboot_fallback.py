"""
Fallback wild bootstrap implementation for when wildboottest is not available
"""

import numpy as np
from scipy import stats
from typing import Optional, Tuple


def wild_bootstrap_ols(X: np.ndarray, 
                      y: np.ndarray, 
                      cluster: np.ndarray,
                      param_idx: int,
                      B: int = 999,
                      weights: Optional[np.ndarray] = None,
                      seed: Optional[int] = None,
                      alpha: float = 0.05) -> dict:
    """
    Simple wild cluster bootstrap for OLS
    
    Parameters
    ----------
    X : array
        Design matrix (n x k)
    y : array
        Response variable (n x 1)
    cluster : array
        Cluster identifiers (n x 1)
    param_idx : int
        Index of parameter to test
    B : int
        Number of bootstrap iterations
    weights : array, optional
        Weights for weighted regression
    seed : int, optional
        Random seed
    alpha : float
        Significance level
        
    Returns
    -------
    dict
        Bootstrap results with keys: se, t_stat, p_value, CI
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = len(y)
    k = X.shape[1]
    
    # Get unique clusters
    unique_clusters = np.unique(cluster)
    n_clusters = len(unique_clusters)
    
    # Estimate original model
    if weights is not None:
        W = np.diag(np.sqrt(weights))
        X_w = W @ X
        y_w = W @ y
        beta_hat = np.linalg.lstsq(X_w.T @ X_w, X_w.T @ y_w, rcond=None)[0]
        residuals = y - X @ beta_hat
        residuals_w = W @ residuals
    else:
        beta_hat = np.linalg.lstsq(X.T @ X, X.T @ y, rcond=None)[0]
        residuals = y - X @ beta_hat
        residuals_w = residuals
    
    # Original test statistic
    t_original = beta_hat[param_idx]
    
    # Bootstrap
    t_boot = np.zeros(B)
    
    for b in range(B):
        # Rademacher weights at cluster level
        eta = np.random.choice([-1, 1], size=n_clusters)
        
        # Map to observation level
        eta_obs = np.zeros(n)
        for i, c in enumerate(unique_clusters):
            eta_obs[cluster == c] = eta[i]
        
        # Create bootstrap residuals
        u_boot = residuals * eta_obs
        
        # New y
        y_boot = X @ beta_hat + u_boot
        
        # Re-estimate
        if weights is not None:
            y_boot_w = W @ y_boot
            beta_boot = np.linalg.lstsq(X_w.T @ X_w, X_w.T @ y_boot_w, rcond=None)[0]
        else:
            beta_boot = np.linalg.lstsq(X.T @ X, X.T @ y_boot, rcond=None)[0]
        
        t_boot[b] = beta_boot[param_idx] - beta_hat[param_idx]
    
    # Compute p-value (two-sided)
    p_value = np.mean(np.abs(t_boot) >= np.abs(t_original))
    
    # Compute confidence interval
    q_low = np.percentile(t_boot, alpha/2 * 100)
    q_high = np.percentile(t_boot, (1 - alpha/2) * 100)
    
    ci_low = beta_hat[param_idx] - q_high
    ci_high = beta_hat[param_idx] - q_low
    
    # Bootstrap standard error
    se_boot = np.std(t_boot)
    
    # T-statistic (using bootstrap SE)
    t_stat = beta_hat[param_idx] / se_boot if se_boot > 0 else np.nan
    
    return {
        'se': se_boot,
        't_stat': t_stat,
        'p_value': p_value,
        'CI': (ci_low, ci_high),
        'beta': beta_hat[param_idx]
    }


# Wrapper function to match wildboottest interface
def wildboottest(X, y, cluster, B, param, weights=None, seed=None):
    """Wrapper to match wildboottest package interface"""
    return wild_bootstrap_ols(
        X=X,
        y=y,
        cluster=cluster,
        param_idx=param,
        B=B,
        weights=weights,
        seed=seed
    )