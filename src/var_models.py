"""
VaR Models Module
Implements Historical, Parametric, and Monte Carlo VaR calculation methods.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class VaRResult:
    """Container for VaR calculation results."""
    method: str
    var_95: float
    var_99: float
    es_95: float  # Expected Shortfall (CVaR)
    es_99: float
    
    def __str__(self):
        return (
            f"{self.method} VaR Results:\n"
            f"  VaR (95%): {self.var_95:.4%}\n"
            f"  VaR (99%): {self.var_99:.4%}\n"
            f"  ES (95%):  {self.es_95:.4%}\n"
            f"  ES (99%):  {self.es_99:.4%}"
        )


def historical_var(
    returns: pd.Series,
    confidence_levels: Tuple[float, float] = (0.95, 0.99)
) -> VaRResult:
    """
    Calculate Historical VaR using percentile method.
    
    Historical VaR is non-parametric - it makes no assumptions about
    the distribution of returns. It simply uses the empirical percentile.
    
    Parameters:
    -----------
    returns : pd.Series
        Historical returns
    confidence_levels : Tuple[float, float]
        Confidence levels for VaR calculation
    
    Returns:
    --------
    VaRResult
        VaR and Expected Shortfall at specified confidence levels
    """
    sorted_returns = np.sort(returns)
    
    # VaR is the percentile corresponding to (1 - confidence)
    var_95 = np.percentile(sorted_returns, (1 - confidence_levels[0]) * 100)
    var_99 = np.percentile(sorted_returns, (1 - confidence_levels[1]) * 100)
    
    # Expected Shortfall (CVaR) - average of losses beyond VaR
    es_95 = sorted_returns[sorted_returns <= var_95].mean()
    es_99 = sorted_returns[sorted_returns <= var_99].mean()
    
    return VaRResult(
        method="Historical",
        var_95=var_95,
        var_99=var_99,
        es_95=es_95,
        es_99=es_99
    )


def parametric_var(
    returns: pd.Series,
    confidence_levels: Tuple[float, float] = (0.95, 0.99)
) -> VaRResult:
    """
    Calculate Parametric VaR (Variance-Covariance method).
    
    Assumes returns are normally distributed. Uses mean and standard
    deviation to calculate VaR at specified confidence levels.
    
    VaR = μ - z_α * σ
    
    Parameters:
    -----------
    returns : pd.Series
        Historical returns
    confidence_levels : Tuple[float, float]
        Confidence levels for VaR calculation
    
    Returns:
    --------
    VaRResult
        VaR and Expected Shortfall at specified confidence levels
    """
    mu = returns.mean()
    sigma = returns.std()
    
    # Z-scores for confidence levels
    z_95 = stats.norm.ppf(1 - confidence_levels[0])
    z_99 = stats.norm.ppf(1 - confidence_levels[1])
    
    # VaR calculation
    var_95 = mu + z_95 * sigma
    var_99 = mu + z_99 * sigma
    
    # Expected Shortfall for normal distribution
    # ES = μ - σ * φ(z) / (1-α)
    # where φ is the standard normal pdf
    es_95 = mu - sigma * stats.norm.pdf(z_95) / (1 - confidence_levels[0])
    es_99 = mu - sigma * stats.norm.pdf(z_99) / (1 - confidence_levels[1])
    
    return VaRResult(
        method="Parametric",
        var_95=var_95,
        var_99=var_99,
        es_95=es_95,
        es_99=es_99
    )


def monte_carlo_var(
    returns: pd.Series,
    confidence_levels: Tuple[float, float] = (0.95, 0.99),
    n_simulations: int = 10000,
    time_horizon: int = 1,
    random_seed: Optional[int] = 42
) -> VaRResult:
    """
    Calculate Monte Carlo VaR using Geometric Brownian Motion simulation.
    
    Simulates future price paths using GBM:
    S_t = S_0 * exp((μ - σ²/2)t + σ√t * Z)
    where Z ~ N(0,1)
    
    Parameters:
    -----------
    returns : pd.Series
        Historical returns (used to estimate μ and σ)
    confidence_levels : Tuple[float, float]
        Confidence levels for VaR calculation
    n_simulations : int
        Number of Monte Carlo simulations
    time_horizon : int
        Forecast horizon in days
    random_seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    VaRResult
        VaR and Expected Shortfall at specified confidence levels
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Estimate parameters from historical returns
    mu = returns.mean()
    sigma = returns.std()
    
    # Generate random shocks
    Z = np.random.standard_normal(n_simulations)
    
    # Simulate returns using GBM
    # For daily returns: r = (μ - σ²/2) + σ * Z
    drift = mu - 0.5 * sigma ** 2
    simulated_returns = drift * time_horizon + sigma * np.sqrt(time_horizon) * Z
    
    # Calculate VaR from simulated distribution
    var_95 = np.percentile(simulated_returns, (1 - confidence_levels[0]) * 100)
    var_99 = np.percentile(simulated_returns, (1 - confidence_levels[1]) * 100)
    
    # Expected Shortfall
    es_95 = simulated_returns[simulated_returns <= var_95].mean()
    es_99 = simulated_returns[simulated_returns <= var_99].mean()
    
    return VaRResult(
        method="Monte Carlo",
        var_95=var_95,
        var_99=var_99,
        es_95=es_95,
        es_99=es_99
    )


def calculate_all_var(
    returns: pd.Series,
    confidence_levels: Tuple[float, float] = (0.95, 0.99),
    n_simulations: int = 10000
) -> dict:
    """
    Calculate VaR using all three methods.
    
    Parameters:
    -----------
    returns : pd.Series
        Historical returns
    confidence_levels : Tuple[float, float]
        Confidence levels
    n_simulations : int
        Number of Monte Carlo simulations
    
    Returns:
    --------
    dict
        Dictionary with results from all methods
    """
    results = {
        'historical': historical_var(returns, confidence_levels),
        'parametric': parametric_var(returns, confidence_levels),
        'monte_carlo': monte_carlo_var(returns, confidence_levels, n_simulations)
    }
    
    return results


def var_summary_table(results: dict) -> pd.DataFrame:
    """
    Create a summary table of VaR results.
    
    Parameters:
    -----------
    results : dict
        Dictionary with VaR results from calculate_all_var()
    
    Returns:
    --------
    pd.DataFrame
        Summary table
    """
    data = []
    for method, result in results.items():
        data.append({
            'Method': result.method,
            'VaR (95%)': f"{result.var_95:.2%}",
            'VaR (99%)': f"{result.var_99:.2%}",
            'ES (95%)': f"{result.es_95:.2%}",
            'ES (99%)': f"{result.es_99:.2%}"
        })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    sample_returns = pd.Series(np.random.normal(0.0005, 0.015, 500))
    
    print("Testing VaR Models with Sample Data")
    print("=" * 50)
    
    results = calculate_all_var(sample_returns)
    
    for method, result in results.items():
        print(f"\n{result}")
    
    print("\n" + "=" * 50)
    print("\nSummary Table:")
    print(var_summary_table(results).to_string(index=False))
