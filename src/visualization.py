"""
Visualization Module
Charts and plots for VaR analysis and backtesting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_return_distribution(
    returns: pd.Series,
    var_results: Dict,
    confidence_level: float = 0.95,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot return distribution with VaR thresholds.
    
    Parameters:
    -----------
    returns : pd.Series
        Portfolio returns
    var_results : Dict
        Dictionary with VaR results from calculate_all_var()
    confidence_level : float
        Which confidence level to show
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    ax.hist(returns, bins=50, density=True, alpha=0.7, color='steelblue', 
            edgecolor='white', label='Returns')
    
    # Plot KDE
    returns.plot(kind='kde', ax=ax, color='navy', linewidth=2)
    
    # Add VaR lines
    colors = {'historical': '#e74c3c', 'parametric': '#2ecc71', 'monte_carlo': '#9b59b6'}
    linestyles = {'historical': '-', 'parametric': '--', 'monte_carlo': '-.'}
    
    var_key = 'var_95' if confidence_level == 0.95 else 'var_99'
    
    for method, result in var_results.items():
        var_value = getattr(result, var_key)
        ax.axvline(
            x=var_value, 
            color=colors[method], 
            linestyle=linestyles[method],
            linewidth=2.5,
            label=f'{result.method}: {var_value:.2%}'
        )
    
    ax.set_xlabel('Daily Return', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Return Distribution with VaR ({int(confidence_level*100)}% Confidence)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_var_breaches(
    returns: pd.Series,
    var_series: pd.Series,
    breach_indicators: pd.Series,
    method_name: str = "VaR",
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot returns with VaR threshold and breach highlights.
    
    Parameters:
    -----------
    returns : pd.Series
        Portfolio returns
    var_series : pd.Series
        Rolling VaR estimates
    breach_indicators : pd.Series
        Boolean series indicating breaches
    method_name : str
        Name of VaR method for title
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    
    # Top plot: Returns and VaR
    ax1 = axes[0]
    
    # Plot returns
    common_idx = returns.index.intersection(var_series.index)
    returns_plot = returns.loc[common_idx]
    var_plot = var_series.loc[common_idx]
    
    ax1.fill_between(returns_plot.index, returns_plot, 0, 
                     where=returns_plot >= 0, color='green', alpha=0.4, label='Positive Returns')
    ax1.fill_between(returns_plot.index, returns_plot, 0, 
                     where=returns_plot < 0, color='red', alpha=0.4, label='Negative Returns')
    
    # Plot VaR line
    ax1.plot(var_plot.index, var_plot, color='darkred', linewidth=2, 
             label=f'{method_name} Threshold', linestyle='--')
    
    # Highlight breaches
    breach_dates = breach_indicators[breach_indicators].index
    breach_returns = returns_plot.loc[returns_plot.index.isin(breach_dates)]
    ax1.scatter(breach_returns.index, breach_returns, color='black', s=50, 
                zorder=5, marker='x', label=f'Breaches ({len(breach_dates)})')
    
    ax1.set_ylabel('Daily Return', fontsize=12)
    ax1.set_title(f'{method_name} Backtest - Returns vs Threshold', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Cumulative breaches
    ax2 = axes[1]
    
    cumulative_breaches = breach_indicators.cumsum()
    cumulative_breaches.plot(ax=ax2, color='darkred', linewidth=2)
    
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Cumulative Breaches', fontsize=12)
    ax2.set_title('Cumulative VaR Breaches Over Time', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_var_comparison(
    var_results: Dict,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create bar chart comparing VaR across methods.
    
    Parameters:
    -----------
    var_results : Dict
        Dictionary with VaR results
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    methods = []
    var_95 = []
    var_99 = []
    es_95 = []
    es_99 = []
    
    for method, result in var_results.items():
        methods.append(result.method)
        var_95.append(abs(result.var_95) * 100)
        var_99.append(abs(result.var_99) * 100)
        es_95.append(abs(result.es_95) * 100)
        es_99.append(abs(result.es_99) * 100)
    
    x = np.arange(len(methods))
    width = 0.2
    
    bars1 = ax.bar(x - 1.5*width, var_95, width, label='VaR 95%', color='#3498db')
    bars2 = ax.bar(x - 0.5*width, var_99, width, label='VaR 99%', color='#2980b9')
    bars3 = ax.bar(x + 0.5*width, es_95, width, label='ES 95%', color='#e74c3c')
    bars4 = ax.bar(x + 1.5*width, es_99, width, label='ES 99%', color='#c0392b')
    
    ax.set_ylabel('Risk (% of Portfolio)', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title('VaR and Expected Shortfall Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_portfolio_performance(
    prices: pd.DataFrame,
    portfolio_returns: pd.Series,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot portfolio performance overview.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Asset prices
    portfolio_returns : pd.Series
        Portfolio returns
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Normalize prices for comparison
    normalized_prices = prices / prices.iloc[0] * 100
    
    # Plot 1: Normalized prices
    ax1 = axes[0, 0]
    normalized_prices.plot(ax=ax1, linewidth=1.5)
    ax1.set_title('Normalized Asset Prices (Base=100)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=10)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative portfolio returns
    ax2 = axes[0, 1]
    cumulative_returns = (1 + portfolio_returns).cumprod()
    cumulative_returns.plot(ax=ax2, color='navy', linewidth=2)
    ax2.set_title('Cumulative Portfolio Returns', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Growth of $1', fontsize=10)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Rolling volatility
    ax3 = axes[1, 0]
    rolling_vol = portfolio_returns.rolling(window=21).std() * np.sqrt(252) * 100
    rolling_vol.plot(ax=ax3, color='red', linewidth=1.5)
    ax3.set_title('Rolling 21-Day Annualized Volatility', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Volatility (%)', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Return distribution
    ax4 = axes[1, 1]
    portfolio_returns.hist(bins=50, ax=ax4, color='steelblue', edgecolor='white', alpha=0.7)
    ax4.axvline(x=portfolio_returns.mean(), color='green', linestyle='--', 
                linewidth=2, label=f'Mean: {portfolio_returns.mean():.2%}')
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax4.set_title('Return Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Daily Return', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# Import Tuple for type hints
from typing import Tuple


if __name__ == "__main__":
    print("Visualization module loaded successfully.")
    print("Available functions:")
    print("  - plot_return_distribution()")
    print("  - plot_var_breaches()")
    print("  - plot_var_comparison()")
    print("  - plot_portfolio_performance()")
