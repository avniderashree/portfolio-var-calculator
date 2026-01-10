"""
Backtesting Module
Validates VaR models against historical data.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class BacktestResult:
    """Container for backtesting results."""
    method: str
    n_observations: int
    n_breaches: int
    breach_rate: float
    expected_breaches: float
    kupiec_statistic: float
    kupiec_pvalue: float
    traffic_light: str  # Green, Yellow, Red (Basel framework)
    
    def __str__(self):
        return (
            f"{self.method} Backtest Results:\n"
            f"  Observations: {self.n_observations}\n"
            f"  Breaches: {self.n_breaches} ({self.breach_rate:.1%})\n"
            f"  Expected: {self.expected_breaches:.1f}\n"
            f"  Kupiec p-value: {self.kupiec_pvalue:.4f}\n"
            f"  Traffic Light: {self.traffic_light}"
        )


def count_var_breaches(
    returns: pd.Series,
    var_series: pd.Series
) -> Tuple[int, pd.Series]:
    """
    Count the number of times actual losses exceed VaR.
    
    A breach occurs when the actual return is more negative than the VaR.
    
    Parameters:
    -----------
    returns : pd.Series
        Actual portfolio returns
    var_series : pd.Series
        Rolling VaR estimates
    
    Returns:
    --------
    Tuple[int, pd.Series]
        Number of breaches and boolean series of breach indicators
    """
    # Align indices
    common_idx = returns.index.intersection(var_series.index)
    returns_aligned = returns.loc[common_idx]
    var_aligned = var_series.loc[common_idx]
    
    # Breach when actual return < VaR (VaR is negative)
    breaches = returns_aligned < var_aligned
    
    return breaches.sum(), breaches


def kupiec_pof_test(
    n_observations: int,
    n_breaches: int,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Kupiec's Proportion of Failures (POF) test.
    
    Tests whether the observed breach rate is consistent with the
    expected rate based on the confidence level.
    
    H0: The model is correctly specified (breach rate = 1 - confidence)
    
    Parameters:
    -----------
    n_observations : int
        Total number of observations
    n_breaches : int
        Number of VaR breaches
    confidence_level : float
        VaR confidence level (e.g., 0.95)
    
    Returns:
    --------
    Tuple[float, float]
        Test statistic and p-value
    """
    p_expected = 1 - confidence_level  # Expected breach probability
    p_observed = n_breaches / n_observations if n_observations > 0 else 0
    
    # Handle edge cases
    if n_breaches == 0 or n_breaches == n_observations:
        # Cannot compute log(0)
        return 0.0, 1.0
    
    # Likelihood ratio test statistic
    # LR = -2 * ln[(1-p)^(n-x) * p^x / (1-p_hat)^(n-x) * p_hat^x]
    n = n_observations
    x = n_breaches
    
    lr_stat = -2 * (
        (n - x) * np.log(1 - p_expected) + x * np.log(p_expected) -
        (n - x) * np.log(1 - p_observed) - x * np.log(p_observed)
    )
    
    # LR follows chi-square with 1 df under H0
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
    
    return lr_stat, p_value


def basel_traffic_light(
    n_observations: int,
    n_breaches: int,
    confidence_level: float = 0.99
) -> str:
    """
    Basel Committee Traffic Light approach for VaR backtesting.
    
    Based on 250 trading days at 99% confidence:
    - Green: 0-4 breaches (model is acceptable)
    - Yellow: 5-9 breaches (requires scrutiny)
    - Red: 10+ breaches (model is flawed)
    
    Parameters:
    -----------
    n_observations : int
        Number of observations (typically 250)
    n_breaches : int
        Number of VaR breaches
    confidence_level : float
        VaR confidence level
    
    Returns:
    --------
    str
        Traffic light color
    """
    # Scale thresholds based on observation count
    scale_factor = n_observations / 250
    
    yellow_threshold = int(5 * scale_factor)
    red_threshold = int(10 * scale_factor)
    
    if n_breaches < yellow_threshold:
        return "🟢 Green"
    elif n_breaches < red_threshold:
        return "🟡 Yellow"
    else:
        return "🔴 Red"


def rolling_var_backtest(
    returns: pd.Series,
    var_func,
    window: int = 252,
    confidence_level: float = 0.95
) -> Tuple[pd.Series, BacktestResult]:
    """
    Perform rolling window VaR backtest.
    
    Parameters:
    -----------
    returns : pd.Series
        Full return series
    var_func : callable
        VaR calculation function
    window : int
        Rolling window size
    confidence_level : float
        VaR confidence level
    
    Returns:
    --------
    Tuple[pd.Series, BacktestResult]
        Rolling VaR series and backtest results
    """
    var_values = []
    var_dates = []
    
    # Always pass two confidence levels to var_func
    confidence_levels = (0.95, 0.99)
    
    for i in range(window, len(returns)):
        # Use past 'window' days to estimate VaR
        historical = returns.iloc[i-window:i]
        var_result = var_func(historical, confidence_levels)
        
        # VaR for the next day
        var_values.append(var_result.var_95 if confidence_level == 0.95 else var_result.var_99)
        var_dates.append(returns.index[i])
    
    var_series = pd.Series(var_values, index=var_dates, name='VaR')
    
    # Count breaches
    test_returns = returns.loc[var_dates]
    n_breaches, breach_indicators = count_var_breaches(test_returns, var_series)
    n_observations = len(test_returns)
    
    # Statistical tests
    kupiec_stat, kupiec_pval = kupiec_pof_test(n_observations, n_breaches, confidence_level)
    traffic_light = basel_traffic_light(n_observations, n_breaches, confidence_level)
    
    result = BacktestResult(
        method=var_func.__name__.replace('_', ' ').title(),
        n_observations=n_observations,
        n_breaches=n_breaches,
        breach_rate=n_breaches / n_observations if n_observations > 0 else 0,
        expected_breaches=n_observations * (1 - confidence_level),
        kupiec_statistic=kupiec_stat,
        kupiec_pvalue=kupiec_pval,
        traffic_light=traffic_light
    )
    
    return var_series, breach_indicators, result


def backtest_summary_table(results: list) -> pd.DataFrame:
    """
    Create summary table from backtest results.
    
    Parameters:
    -----------
    results : list
        List of BacktestResult objects
    
    Returns:
    --------
    pd.DataFrame
        Summary table
    """
    data = []
    for r in results:
        data.append({
            'Method': r.method,
            'Breaches': f"{r.n_breaches}/{r.n_observations}",
            'Rate': f"{r.breach_rate:.1%}",
            'Expected': f"{r.expected_breaches:.0f}",
            'p-value': f"{r.kupiec_pvalue:.3f}",
            'Status': r.traffic_light
        })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Test with sample data
    from var_models import historical_var, parametric_var
    
    np.random.seed(42)
    # Generate sample returns with some fat tails
    sample_returns = pd.Series(
        np.random.standard_t(df=5, size=600) * 0.015,
        index=pd.date_range(start='2022-01-01', periods=600, freq='B')
    )
    
    print("Backtesting VaR Models")
    print("=" * 50)
    
    results = []
    
    # Test Historical VaR
    var_series, breaches, result = rolling_var_backtest(
        sample_returns, historical_var, window=252, confidence_level=0.95
    )
    results.append(result)
    print(f"\n{result}")
    
    # Test Parametric VaR
    var_series, breaches, result = rolling_var_backtest(
        sample_returns, parametric_var, window=252, confidence_level=0.95
    )
    results.append(result)
    print(f"\n{result}")
    
    print("\n" + "=" * 50)
    print("\nSummary:")
    print(backtest_summary_table(results).to_string(index=False))
