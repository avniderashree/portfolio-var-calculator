#!/usr/bin/env python3
"""
Portfolio VaR Calculator
========================
Main execution script demonstrating VaR calculation and backtesting.

Author: Avni Derashree
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import fetch_stock_data, calculate_returns, calculate_portfolio_returns, get_sample_portfolio
from src.var_models import calculate_all_var, var_summary_table, historical_var, parametric_var, monte_carlo_var
from src.backtesting import rolling_var_backtest, backtest_summary_table
from src.visualization import (
    plot_return_distribution, 
    plot_var_breaches, 
    plot_var_comparison,
    plot_portfolio_performance
)


def print_header(text: str, char: str = "="):
    """Print formatted section header."""
    print(f"\n{char * 60}")
    print(f" {text}")
    print(f"{char * 60}")


def main():
    """Main execution function."""
    
    print_header("PORTFOLIO VALUE AT RISK (VaR) CALCULATOR", "=")
    print("\nThis analysis calculates VaR using three methods:")
    print("  1. Historical VaR (Percentile-based)")
    print("  2. Parametric VaR (Variance-Covariance)")
    print("  3. Monte Carlo VaR (GBM Simulation)")
    
    # =========================================================================
    # STEP 1: Load Portfolio Data
    # =========================================================================
    print_header("STEP 1: Loading Portfolio Data", "-")
    
    prices, portfolio_returns, tickers = get_sample_portfolio(period="2y")
    
    print(f"\nPortfolio Composition: {', '.join(tickers)} (Equal-Weighted)")
    print(f"\nPortfolio Statistics:")
    print(f"  • Mean Daily Return:     {portfolio_returns.mean():.4%}")
    print(f"  • Daily Volatility:      {portfolio_returns.std():.4%}")
    print(f"  • Annualized Volatility: {portfolio_returns.std() * np.sqrt(252):.2%}")
    print(f"  • Skewness:              {portfolio_returns.skew():.3f}")
    print(f"  • Kurtosis:              {portfolio_returns.kurtosis():.3f}")
    
    if portfolio_returns.kurtosis() > 0:
        print("\n  ⚠️  Positive kurtosis indicates fat tails - Parametric VaR may underestimate risk")
    
    # =========================================================================
    # STEP 2: Calculate VaR Using All Methods
    # =========================================================================
    print_header("STEP 2: Calculating Value at Risk", "-")
    
    var_results = calculate_all_var(portfolio_returns, n_simulations=10000)
    
    print("\nVaR Results (Daily, for a $1M portfolio):\n")
    print("-" * 55)
    print(f"{'Method':<15} {'VaR 95%':>12} {'VaR 99%':>12} {'ES 95%':>12}")
    print("-" * 55)
    
    portfolio_value = 1_000_000
    for method, result in var_results.items():
        var_95_dollar = abs(result.var_95) * portfolio_value
        var_99_dollar = abs(result.var_99) * portfolio_value
        es_95_dollar = abs(result.es_95) * portfolio_value
        print(f"{result.method:<15} ${var_95_dollar:>10,.0f} ${var_99_dollar:>10,.0f} ${es_95_dollar:>10,.0f}")
    
    print("-" * 55)
    print(f"\nInterpretation (99% VaR):")
    print(f"  With 99% confidence, the maximum daily loss should not exceed")
    print(f"  the VaR threshold. On average, this will be breached ~2.5 days/year.")
    
    # =========================================================================
    # STEP 3: Backtest VaR Models
    # =========================================================================
    print_header("STEP 3: Backtesting VaR Models", "-")
    
    print("\nRunning rolling 252-day backtest at 95% confidence...")
    
    backtest_results = []
    var_series_dict = {}
    breach_dict = {}
    
    for var_func, name in [(historical_var, 'Historical'), 
                            (parametric_var, 'Parametric'), 
                            (monte_carlo_var, 'Monte Carlo')]:
        var_series, breaches, result = rolling_var_backtest(
            portfolio_returns, 
            var_func, 
            window=252, 
            confidence_level=0.95
        )
        backtest_results.append(result)
        var_series_dict[name] = var_series
        breach_dict[name] = breaches
    
    print("\nBacktest Results:")
    print(backtest_summary_table(backtest_results).to_string(index=False))
    
    print("\nBasel Traffic Light Framework:")
    for result in backtest_results:
        print(f"  • {result.method:<15} {result.traffic_light}")
    
    # =========================================================================
    # STEP 4: Generate Visualizations
    # =========================================================================
    print_header("STEP 4: Generating Visualizations", "-")
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    print("\nSaving charts to ./output/ directory...")
    
    # Chart 1: Return Distribution
    fig1 = plot_return_distribution(portfolio_returns, var_results, confidence_level=0.95)
    fig1.savefig('output/return_distribution.png', dpi=150, bbox_inches='tight')
    print("  ✓ return_distribution.png")
    
    # Chart 2: VaR Comparison
    fig2 = plot_var_comparison(var_results)
    fig2.savefig('output/var_comparison.png', dpi=150, bbox_inches='tight')
    print("  ✓ var_comparison.png")
    
    # Chart 3: Portfolio Performance
    fig3 = plot_portfolio_performance(prices, portfolio_returns)
    fig3.savefig('output/portfolio_performance.png', dpi=150, bbox_inches='tight')
    print("  ✓ portfolio_performance.png")
    
    # Chart 4: Backtest Breaches (for Historical VaR)
    fig4 = plot_var_breaches(
        portfolio_returns, 
        var_series_dict['Historical'], 
        breach_dict['Historical'],
        method_name='Historical VaR (95%)'
    )
    fig4.savefig('output/backtest_breaches.png', dpi=150, bbox_inches='tight')
    print("  ✓ backtest_breaches.png")
    
    plt.close('all')
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("ANALYSIS COMPLETE", "=")
    
    print("\n📊 Key Findings:")
    print(f"  • Portfolio analyzed: {', '.join(tickers)}")
    print(f"  • Time period: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
    print(f"  • Daily VaR (95%, Historical): {var_results['historical'].var_95:.2%}")
    print(f"  • Daily VaR (99%, Historical): {var_results['historical'].var_99:.2%}")
    
    hist_result = backtest_results[0]
    param_result = backtest_results[1]
    
    print(f"\n📈 Backtest Performance:")
    print(f"  • Historical VaR breaches: {hist_result.breach_rate:.1%} (expected: 5.0%)")
    print(f"  • Parametric VaR breaches: {param_result.breach_rate:.1%} (expected: 5.0%)")
    
    if param_result.breach_rate > hist_result.breach_rate:
        print("\n💡 Insight: Parametric VaR has more breaches, suggesting")
        print("   the normal distribution assumption underestimates tail risk.")
    
    print("\n📁 Output files saved to ./output/")
    print("\nDone! ✅")


if __name__ == "__main__":
    main()
