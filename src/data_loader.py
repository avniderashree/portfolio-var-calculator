"""
Data Loader Module
Fetches and prepares market data for VaR analysis.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Optional, Tuple
from datetime import datetime, timedelta


def fetch_stock_data(
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "2y"
) -> pd.DataFrame:
    """
    Fetch historical adjusted close prices for given tickers.
    
    Parameters:
    -----------
    tickers : List[str]
        List of stock tickers (e.g., ['SPY', 'AAPL', 'MSFT'])
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format
    period : str
        Period to fetch if dates not specified (default: '2y')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with adjusted close prices
    """
    if start_date and end_date:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    else:
        data = yf.download(tickers, period=period, progress=False)
    
    # Handle both old ('Adj Close') and new ('Close') yfinance API
    if 'Adj Close' in data.columns or ('Adj Close' in str(data.columns)):
        price_col = 'Adj Close'
    else:
        price_col = 'Close'
    
    # Handle single ticker case
    if len(tickers) == 1:
        if isinstance(data.columns, pd.MultiIndex):
            prices = data[price_col].to_frame()
            prices.columns = tickers
        else:
            prices = data[price_col].to_frame()
            prices.columns = tickers
    else:
        # For multiple tickers with MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            prices = data[price_col]
        else:
            prices = data[[price_col]]
            prices.columns = tickers
    
    return prices


def calculate_returns(
    prices: pd.DataFrame,
    method: str = 'log'
) -> pd.DataFrame:
    """
    Calculate returns from price data.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame with price data
    method : str
        'log' for log returns, 'simple' for arithmetic returns
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with returns
    """
    if method == 'log':
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()
    
    return returns.dropna()


def calculate_portfolio_returns(
    returns: pd.DataFrame,
    weights: Optional[List[float]] = None
) -> pd.Series:
    """
    Calculate portfolio returns given asset returns and weights.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame with individual asset returns
    weights : List[float], optional
        Portfolio weights (equal-weighted if None)
    
    Returns:
    --------
    pd.Series
        Portfolio returns
    """
    n_assets = returns.shape[1]
    
    if weights is None:
        weights = np.ones(n_assets) / n_assets
    
    weights = np.array(weights)
    
    if len(weights) != n_assets:
        raise ValueError(f"Number of weights ({len(weights)}) must match number of assets ({n_assets})")
    
    if not np.isclose(weights.sum(), 1.0):
        weights = weights / weights.sum()
    
    portfolio_returns = (returns * weights).sum(axis=1)
    
    return portfolio_returns


def get_sample_portfolio(period: str = "2y") -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Get a sample diversified portfolio for demonstration.
    
    Returns:
    --------
    Tuple containing:
        - prices: DataFrame of adjusted close prices
        - portfolio_returns: Series of portfolio returns
        - tickers: List of ticker symbols
    """
    tickers = ['SPY', 'AAPL', 'MSFT', 'GOOGL']
    
    print(f"Fetching data for: {', '.join(tickers)}")
    prices = fetch_stock_data(tickers, period=period)
    
    print("Calculating returns...")
    returns = calculate_returns(prices, method='log')
    
    # Equal-weighted portfolio
    portfolio_returns = calculate_portfolio_returns(returns)
    
    print(f"Data loaded: {len(prices)} trading days")
    print(f"Date range: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
    
    return prices, portfolio_returns, tickers


if __name__ == "__main__":
    # Test the module
    prices, portfolio_returns, tickers = get_sample_portfolio()
    print("\nPortfolio Statistics:")
    print(f"  Mean daily return: {portfolio_returns.mean():.4%}")
    print(f"  Std deviation: {portfolio_returns.std():.4%}")
    print(f"  Annualized volatility: {portfolio_returns.std() * np.sqrt(252):.2%}")
