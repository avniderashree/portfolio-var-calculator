# 🧪 Codelab: Build a Portfolio Value-at-Risk (VaR) Calculator from Scratch

**Estimated time:** 2–3 hours · **Difficulty:** Intermediate · **Language:** Python 3.9+

---

## What You'll Build

By the end of this codelab, you'll have a fully working **Portfolio VaR Calculator** that:

- Downloads real stock market data from Yahoo Finance
- Calculates risk using **3 industry-standard methods** (Historical, Parametric, Monte Carlo)
- Computes **Expected Shortfall (CVaR)** — the "beyond worst case" measure
- **Backtests** the models to see if they actually work in practice
- Runs **Kupiec's statistical test** and the **Basel Traffic Light** regulatory check
- Generates **4 publication-quality charts**

The final project structure will look like this:

```
portfolio-var-calculator/
├── main.py                  # Entry point — runs everything
├── requirements.txt         # Dependencies
├── src/
│   ├── __init__.py          # Makes src a Python package
│   ├── data_loader.py       # Fetches market data, computes returns
│   ├── var_models.py        # The 3 VaR methods + Expected Shortfall
│   ├── backtesting.py       # Rolling backtest, Kupiec test, Basel light
│   └── visualization.py     # 4 professional charts
├── tests/
│   └── test_var.py          # Unit tests
└── output/                  # Where charts get saved
```

---

## Prerequisites

You only need:

- Python 3.9+ installed
- Basic familiarity with Python (functions, classes, imports)
- A terminal / command line

**No finance knowledge required.** We'll explain every concept before we code it.

---

---

# PART 1: THE CONCEPTS (What & Why)

Before writing a single line of code, let's understand the problem we're solving. This section is all theory — no coding yet. Read through it; it'll make every line of code click later.

---

## 1.1 What Is Value at Risk (VaR)?

Imagine you have a portfolio worth **$1,000,000** invested in stocks like Apple, Google, Microsoft, and the S&P 500. You go to sleep tonight. How much money could you lose *by tomorrow*?

That's the question VaR answers:

> **VaR** tells you the *maximum loss* you should expect over a time period, at a given confidence level.

**Concrete example:**

"The 1-day 95% VaR of this portfolio is **$19,000**."

This means:
- On **95 out of 100** trading days, your loss will be **less than** $19,000.
- On **5 out of 100** days, you might lose **more than** $19,000.

**Why 95% and not 100%?** Because 100% confidence would mean "what's the absolute worst that can ever happen" — and that's unbounded. Markets can crash indefinitely. So we pick a practical confidence level (95% or 99%) and accept a small probability of exceeding it.

**Why does VaR matter?**
- Banks are **legally required** to calculate and report it (Basel regulations)
- Fund managers use it to decide position sizes
- Risk teams use it to set stop-losses and capital reserves

---

## 1.2 The Three Methods (and Why We Need All Three)

There's no single "best" way to compute VaR. Each method makes different trade-offs:

### Method 1: Historical VaR

**Idea:** Look at what *actually happened* in the past and assume the future will be similar.

**How it works:**
1. Collect the last ~500 daily returns of your portfolio
2. Sort them from worst to best
3. The 5th percentile (for 95% confidence) is your VaR

**Analogy:** It's like checking the weather forecast by looking at what the temperature was on this date for the past 500 years. No theory, just raw data.

**Formula:**
```
VaR_95% = the 5th percentile of historical daily returns
```

**Strengths:** No assumptions about how returns are distributed. If the market has fat tails (extreme losses happen more often than a bell curve predicts), Historical VaR catches that.

**Weaknesses:** It only knows about the past. If a brand-new type of crisis happens (like COVID in 2020), historical data won't have captured it.

---

### Method 2: Parametric VaR (Variance-Covariance)

**Idea:** Assume returns follow a **normal distribution** (bell curve), then use a formula.

**How it works:**
1. Calculate the **mean (μ)** and **standard deviation (σ)** of daily returns
2. Use the z-score for your confidence level
3. Plug into the formula

**Formula:**
```
VaR = μ - z × σ

Where:
  μ = average daily return (e.g., 0.05%)
  σ = standard deviation of daily returns (e.g., 1.2%)
  z = z-score (1.645 for 95%, 2.326 for 99%)
```

**Worked example (95% confidence):**
```
μ = 0.0005  (0.05% daily return)
σ = 0.012   (1.2% daily volatility)
z = 1.645

VaR = 0.0005 - 1.645 × 0.012 = -0.0192 = -1.92%

On a $1,000,000 portfolio: $19,200 loss
```

**Analogy:** It's like saying "heights of adults follow a bell curve, so I can mathematically calculate how tall the shortest 5% of people are."

**Strengths:** Fast. One formula. Works with tiny amounts of data.

**Weaknesses:** Real stock returns are NOT normally distributed. They have "fat tails" — extreme events happen more often than a bell curve predicts. So Parametric VaR typically **underestimates** the true risk.

---

### Method 3: Monte Carlo VaR

**Idea:** Simulate **10,000 possible futures** using random number generation, then look at how bad things got.

**How it works:**
1. Estimate μ and σ from historical data
2. Generate 10,000 random "tomorrow" returns using Geometric Brownian Motion (GBM)
3. Sort the 10,000 simulated returns
4. The 5th percentile is your VaR

**The GBM formula:**
```
S_tomorrow = S_today × exp((μ - σ²/2) × t + σ × √t × Z)

Where:
  S = stock price
  μ = drift (average return)
  σ = volatility
  t = time step (1 day = 1/252 years)
  Z = random number from N(0,1) — a standard normal distribution
```

**Why `μ - σ²/2`?** This is called the "drift correction" or "Itô correction." When you convert from arithmetic returns to geometric (log) returns, you need to subtract half the variance. Without it, your simulation would systematically overestimate future prices. It's a mathematical consequence of working with log-normal distributions.

**Analogy:** Instead of looking at 500 past days, you *invent* 10,000 alternate realities using randomness calibrated to real market behavior, then check how bad things got across all realities.

**Strengths:** Extremely flexible. Can model complex portfolios, options, and non-linear payoffs. Can generate more scenarios than history provides.

**Weaknesses:** Slow (need 10,000+ simulations). Quality depends on parameter estimates (garbage in, garbage out).

---

## 1.3 Expected Shortfall (CVaR) — Beyond VaR

VaR has a big limitation: it tells you the *threshold* but not *how bad things get beyond that threshold*.

Example: "Your 95% VaR is $19,000" — but if you *do* lose more than $19,000, will you lose $20,000 or $200,000?

**Expected Shortfall (ES)**, also called **Conditional VaR (CVaR)**, answers this:

> **ES** is the *average loss* in the worst cases that exceed VaR.

**Formula:**
```
ES_95% = average of all returns in the worst 5%
```

**Example:** If the worst 5% of your 500 daily returns are:
```
[-3.5%, -3.2%, -3.0%, -2.8%, -2.7%, ... , -2.0%]
```
Then ES = average of these values = roughly -2.8%.

**Why ES > VaR (always):** VaR is the "boundary" of the worst 5%. ES is the *average* within that worst 5%. Since ES considers the most extreme losses, it's always a larger number.

**Why ES matters:** Basel III (the international banking regulation) prefers ES over VaR precisely because it captures the *severity* of tail losses, not just where the tail begins.

---

## 1.4 Backtesting — Does the Model Actually Work?

Calculating VaR is useless if the model is wrong. **Backtesting** checks model accuracy against reality.

**How it works:**
1. Pick a window (e.g., 252 trading days = 1 year)
2. On day 253: use days 1–252 to predict VaR for day 253. Compare against actual return.
3. On day 254: use days 2–253 to predict VaR for day 254. Compare.
4. Repeat until you run out of data.
5. Count **breaches** — days where the actual loss *exceeded* the predicted VaR.

**What's a "breach"?**
If you predicted "95% VaR = -1.5%" and the actual return was -2.3%, that's a breach. The model said "only a 5% chance of losing more than 1.5%" — and it happened.

### Kupiec's Proportion of Failures (POF) Test

This is a statistical hypothesis test:
- **H₀ (null hypothesis):** The model is correct (breach rate = expected rate)
- **H₁ (alternative):** The model is wrong

For a 95% VaR model over 250 days, you'd expect about 12.5 breaches (5% × 250). If you see 30 breaches, something's wrong.

The test uses a **likelihood ratio** that follows a chi-squared distribution with 1 degree of freedom:

```
LR = -2 × ln[(1-p)^(n-x) × p^x] + 2 × ln[(1-x/n)^(n-x) × (x/n)^x]

Where:
  p = expected failure rate (0.05 for 95% VaR)
  n = number of observations  
  x = number of breaches
```

If the p-value < 0.05, reject the model.

### Basel Traffic Light System

Banks use a simpler, color-coded system. For 250 trading days at 99% confidence:

| Zone   | Breaches | Meaning                   |
|--------|----------|---------------------------|
| 🟢 Green  | 0–4      | Model is fine             |
| 🟡 Yellow | 5–9      | Model needs investigation |
| 🔴 Red    | 10+      | Model is broken           |

---

---

# PART 2: PROJECT SETUP (Step 0)

Now let's code. We'll build everything from scratch, file by file.

---

## Step 0.1: Create the Project Folder

Open your terminal and run:

```bash
mkdir portfolio-var-calculator
cd portfolio-var-calculator

# Create the folder structure
mkdir -p src tests output notebooks
```

---

## Step 0.2: Create `requirements.txt`

This file lists every Python library our project needs. Create the file:

**File: `requirements.txt`**
```
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
yfinance>=0.2.18
matplotlib>=3.7.0
seaborn>=0.12.0
```

**What each library does:**

| Library      | Purpose                                                     |
|-------------|-------------------------------------------------------------|
| `numpy`     | Fast math on arrays (mean, std, percentile, random numbers) |
| `pandas`    | DataFrames for time-series data (stock prices, returns)     |
| `scipy`     | Statistical functions (z-scores, chi-squared test)          |
| `yfinance`  | Downloads stock data from Yahoo Finance for free            |
| `matplotlib`| Plotting engine (creates the PNG charts)                    |
| `seaborn`   | Makes matplotlib plots look professional with one line      |

Now install them:

```bash
pip install -r requirements.txt
```

---

## Step 0.3: Create `src/__init__.py`

This empty file tells Python that `src/` is a package so we can do `from src.data_loader import ...`.

**File: `src/__init__.py`**
```python
"""
Portfolio VaR Calculator
========================
A multi-method Value at Risk calculator with backtesting.

Modules:
    data_loader    - Market data fetching & portfolio return computation
    var_models     - VaR calculation (Historical, Parametric, Monte Carlo)
    backtesting    - Model validation (Kupiec test, Basel traffic light)
    visualization  - Publication-quality charts
"""
```

---

---

# PART 3: DATA LOADER (Step 1)

The data loader is responsible for fetching real stock market data and computing portfolio returns. It's the foundation — everything else depends on it.

---

## Step 1.1: Understand What This Module Does

Before we code, here's the flow:

```
Yahoo Finance API
      │
      ▼
┌─────────────┐     ┌───────────────────┐     ┌─────────────────────┐
│ Raw Prices  │ --> │ Individual Stock   │ --> │ Weighted Portfolio   │
│ (Adj Close) │     │ Daily Returns (%)  │     │ Daily Returns (%)    │
└─────────────┘     └───────────────────┘     └─────────────────────┘
     AAPL: $150        AAPL: +1.2%              Portfolio: +0.9%
     MSFT: $380        MSFT: +0.5%              (weighted average)
     GOOGL: $140       GOOGL: +0.8%
     SPY: $450         SPY: +0.6%
```

**Key concept — Daily Returns:**

A "return" is the percentage change from one day to the next:

```
return_today = (price_today - price_yesterday) / price_yesterday
```

In pandas, this is simply `prices.pct_change()`.

**Key concept — Portfolio Returns:**

If you own 4 stocks equally weighted (25% each), your portfolio return on a given day is:

```
portfolio_return = 0.25 × AAPL_return + 0.25 × MSFT_return + 0.25 × GOOGL_return + 0.25 × SPY_return
```

In linear algebra: `portfolio_return = weights · returns` (a dot product).

---

## Step 1.2: Write the Code

Create the file `src/data_loader.py` and follow along. We'll build it function by function.

**File: `src/data_loader.py`**

```python
"""
data_loader.py — Market Data Fetching & Portfolio Return Computation
====================================================================

This module handles all data acquisition and preprocessing:
  1. Download adjusted closing prices from Yahoo Finance
  2. Compute individual stock daily returns
  3. Combine into a single portfolio return series using weights

Why "Adjusted Close"?
  Stock prices are affected by splits and dividends. A 2-for-1 split
  halves the price overnight, but you didn't lose money — you just
  have twice as many shares. "Adjusted Close" corrects for these
  events so the return calculation is accurate.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Tuple, List, Optional


def fetch_stock_data(
    tickers: List[str],
    period: str = '2y'
) -> pd.DataFrame:
    """
    Download adjusted closing prices from Yahoo Finance.

    Parameters
    ----------
    tickers : list of str
        Stock symbols, e.g. ['AAPL', 'MSFT', 'GOOGL', 'SPY']
    period : str
        How far back to look. Options: '1y', '2y', '5y', '10y', 'max'
        Default '2y' gives ~504 trading days — enough for robust statistics.

    Returns
    -------
    pd.DataFrame
        DataFrame with dates as index and one column per ticker,
        containing adjusted closing prices.

    Example
    -------
    >>> prices = fetch_stock_data(['AAPL', 'MSFT'], period='1y')
    >>> prices.head()
                    AAPL    MSFT
    Date
    2024-01-02    185.64  374.58
    2024-01-03    184.25  373.12
    ...
    """
    print(f"Downloading data for {tickers}...")

    # yf.download() fetches data for multiple tickers at once.
    # auto_adjust=True gives us the "Adjusted Close" directly as the 'Close' column.
    data = yf.download(
        tickers,
        period=period,
        auto_adjust=True,    # Adjust for splits and dividends
        progress=False       # Don't show a progress bar
    )

    # yf.download returns a MultiIndex DataFrame when multiple tickers are passed.
    # We want just the 'Close' prices.
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        # Single ticker case — data.columns is flat
        prices = data[['Close']]
        prices.columns = tickers

    # Drop any rows with missing data (e.g., if one stock was listed later than others).
    prices = prices.dropna()

    print(f"  Downloaded {len(prices)} trading days "
          f"({prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')})")

    return prices


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert prices to daily percentage returns.

    The formula is: return_t = (price_t - price_{t-1}) / price_{t-1}

    We drop the first row because it will be NaN (no previous day to compare to).

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted closing prices (output of fetch_stock_data).

    Returns
    -------
    pd.DataFrame
        Daily percentage returns for each stock.

    Example
    -------
    >>> returns = calculate_returns(prices)
    >>> returns.head()
                    AAPL      MSFT
    Date
    2024-01-03   -0.0075   -0.0039
    2024-01-04    0.0120    0.0085
    ...
    """
    # pct_change() computes (current - previous) / previous for each cell.
    returns = prices.pct_change().dropna()

    return returns


def calculate_portfolio_returns(
    returns: pd.DataFrame,
    weights: Optional[np.ndarray] = None
) -> pd.Series:
    """
    Combine individual stock returns into a single portfolio return series.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns per stock (output of calculate_returns).
    weights : np.ndarray or None
        Portfolio weight for each stock. Must sum to 1.0.
        If None, equal weights are used (1/N for N stocks).

    Returns
    -------
    pd.Series
        Daily portfolio returns (a weighted average of stock returns).

    How it works
    ------------
    On each day, the portfolio return is the dot product of weights and stock returns:

        portfolio_return_t = w₁×r₁_t + w₂×r₂_t + ... + wₙ×rₙ_t

    Example
    -------
    >>> weights = np.array([0.25, 0.25, 0.25, 0.25])
    >>> port_returns = calculate_portfolio_returns(returns, weights)
    """
    n_assets = returns.shape[1]

    if weights is None:
        # Equal weight: if 4 stocks, each gets 25%
        weights = np.array([1.0 / n_assets] * n_assets)

    # Validate weights sum to 1 (with a small tolerance for floating-point errors)
    assert abs(sum(weights) - 1.0) < 1e-6, \
        f"Weights must sum to 1.0, got {sum(weights):.6f}"

    # Dot product: multiply each stock's return by its weight, then sum across stocks.
    # returns.values is a (days × stocks) matrix; weights is a (stocks,) vector.
    # The @ operator is matrix multiplication — each row gets dot-producted with weights.
    portfolio_returns = returns @ weights

    # Convert from DataFrame (single column) to Series and give it a name
    portfolio_returns = pd.Series(
        portfolio_returns,
        index=returns.index,
        name='Portfolio'
    )

    return portfolio_returns


def get_sample_portfolio(
    period: str = '2y'
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Convenience function: download a default portfolio and return everything.

    Uses: SPY (S&P 500 ETF), AAPL, MSFT, GOOGL with equal weights.

    This is the "one function to rule them all" that main.py will call.

    Parameters
    ----------
    period : str
        Lookback period (default '2y').

    Returns
    -------
    tuple of (prices, portfolio_returns, tickers)
        - prices: DataFrame of adjusted closing prices
        - portfolio_returns: Series of daily portfolio returns
        - tickers: list of ticker symbols used
    """
    tickers = ['SPY', 'AAPL', 'MSFT', 'GOOGL']
    weights = np.array([0.25, 0.25, 0.25, 0.25])  # Equal weight

    prices = fetch_stock_data(tickers, period=period)
    returns = calculate_returns(prices)
    portfolio_returns = calculate_portfolio_returns(returns, weights)

    print(f"\nPortfolio Summary:")
    print(f"  Assets: {tickers}")
    print(f"  Weights: {weights}")
    print(f"  Trading days: {len(portfolio_returns)}")
    print(f"  Mean daily return: {portfolio_returns.mean():.4%}")
    print(f"  Daily volatility: {portfolio_returns.std():.4%}")

    return prices, portfolio_returns, tickers
```

---

## Step 1.3: What You Just Built

Let's trace through what happens when you call `get_sample_portfolio()`:

1. **`fetch_stock_data(['SPY','AAPL','MSFT','GOOGL'], '2y')`** — Downloads ~504 days of adjusted closing prices from Yahoo Finance. Returns a DataFrame with 4 columns and ~504 rows.

2. **`calculate_returns(prices)`** — Computes `pct_change()` on each column. Row 1 becomes NaN and gets dropped, so you have ~503 rows of daily returns.

3. **`calculate_portfolio_returns(returns, [0.25, 0.25, 0.25, 0.25])`** — On each day, computes `0.25 × AAPL_return + 0.25 × MSFT_return + ...`. Returns a single Series of 503 portfolio returns.

**Quick test** — you can already run this in a Python shell:

```python
from src.data_loader import get_sample_portfolio

prices, port_returns, tickers = get_sample_portfolio()
print(port_returns.describe())
```

---

---

# PART 4: VaR MODELS (Step 2)

This is the core of the project — the three VaR calculation methods plus Expected Shortfall.

---

## Step 2.1: Design Decisions

We want every VaR method to return results in the same format. We'll use Python's `dataclass` for this:

```python
@dataclass
class VaRResult:
    var_95: float      # 95% VaR (as a negative percentage, e.g., -0.018)
    var_99: float      # 99% VaR
    es_95: float       # 95% Expected Shortfall
    es_99: float       # 99% Expected Shortfall
    method: str        # 'Historical', 'Parametric', or 'Monte Carlo'
```

This way, no matter which method we use, downstream code (backtesting, visualization) can always access `.var_95`, `.var_99`, etc.

---

## Step 2.2: Write the Code

**File: `src/var_models.py`**

```python
"""
var_models.py — Value at Risk Calculation Engine
=================================================

Implements three VaR methodologies:
  1. Historical Simulation (non-parametric)
  2. Parametric / Variance-Covariance (assumes normality)
  3. Monte Carlo Simulation (GBM-based)

Each method also computes Expected Shortfall (CVaR).

All functions return a VaRResult dataclass for consistent access.
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Dict


# ─── Result Container ────────────────────────────────────────────

@dataclass
class VaRResult:
    """
    Holds the output of any VaR calculation method.

    Attributes
    ----------
    var_95 : float
        95% VaR as a decimal (negative number, e.g., -0.018 means -1.8%)
    var_99 : float
        99% VaR as a decimal
    es_95 : float
        95% Expected Shortfall (average loss beyond VaR)
    es_99 : float
        99% Expected Shortfall
    method : str
        Name of the method used ('Historical', 'Parametric', 'Monte Carlo')
    """
    var_95: float
    var_99: float
    es_95: float
    es_99: float
    method: str


# ─── Method 1: Historical VaR ────────────────────────────────────

def historical_var(returns: pd.Series) -> VaRResult:
    """
    Historical Simulation VaR.

    This is the simplest approach: sort actual past returns and read off
    the percentile. No distribution assumptions needed.

    How it works
    ------------
    For 95% VaR with 500 data points:
      - Sort all 500 returns from worst to best
      - The 25th worst return (5th percentile) is the VaR
      - The average of those 25 worst returns is the Expected Shortfall

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.

    Returns
    -------
    VaRResult
        Contains var_95, var_99, es_95, es_99.
    """
    returns_array = returns.values  # Convert pandas Series to numpy array for speed

    # ── VaR Calculation ──
    # np.percentile(data, 5) gives the value below which 5% of data falls.
    # This is our 95% VaR — the loss we'd exceed only 5% of the time.
    var_95 = np.percentile(returns_array, 5)
    var_99 = np.percentile(returns_array, 1)

    # ── Expected Shortfall ──
    # ES = average of all returns WORSE than the VaR threshold.
    # Think of it as: "When things go bad (beyond VaR), how bad on average?"
    es_95 = returns_array[returns_array <= var_95].mean()
    es_99 = returns_array[returns_array <= var_99].mean()

    return VaRResult(
        var_95=var_95,
        var_99=var_99,
        es_95=es_95,
        es_99=es_99,
        method='Historical'
    )


# ─── Method 2: Parametric VaR ────────────────────────────────────

def parametric_var(returns: pd.Series) -> VaRResult:
    """
    Parametric (Variance-Covariance) VaR.

    Assumes returns are normally distributed and uses the analytical formula:
        VaR = μ - z × σ

    Where z is the z-score for the desired confidence level.

    Visual intuition
    ----------------
    Imagine a bell curve of daily returns:

                         ┌────────┐
                       ╱            ╲
                     ╱                ╲
                   ╱                    ╲
        ─────────╱──────────────────────╲───────────
             ▲  │         μ              │
             │  │                        │
           VaR  │    (z × σ from μ)      │

    The VaR is at the left tail, z standard deviations below the mean.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.

    Returns
    -------
    VaRResult
    """
    mu = returns.mean()     # Average daily return
    sigma = returns.std()   # Standard deviation of daily returns

    # ── Z-scores ──
    # stats.norm.ppf(0.05) gives the z-score where 5% of the bell curve is to the left.
    # This is approximately -1.645.
    # stats.norm.ppf(0.01) ≈ -2.326.
    z_95 = stats.norm.ppf(0.05)  # = -1.6449
    z_99 = stats.norm.ppf(0.01)  # = -2.3263

    # ── VaR = μ + z × σ ──
    # Note: z is negative (it's in the left tail), so this produces a negative number.
    # Equivalent to: VaR = μ - |z| × σ
    var_95 = mu + z_95 * sigma
    var_99 = mu + z_99 * sigma

    # ── Expected Shortfall for Normal Distribution ──
    # For a normal distribution, there's an analytical formula for ES:
    #   ES = μ - σ × φ(z) / α
    # Where φ(z) is the standard normal PDF evaluated at the z-score,
    # and α is the tail probability (0.05 or 0.01).
    #
    # Intuition: φ(z)/α is the "average depth" into the tail.
    es_95 = mu - sigma * stats.norm.pdf(z_95) / 0.05
    es_99 = mu - sigma * stats.norm.pdf(z_99) / 0.01

    return VaRResult(
        var_95=var_95,
        var_99=var_99,
        es_95=es_95,
        es_99=es_99,
        method='Parametric'
    )


# ─── Method 3: Monte Carlo VaR ───────────────────────────────────

def monte_carlo_var(
    returns: pd.Series,
    n_simulations: int = 10_000,
    seed: int = 42
) -> VaRResult:
    """
    Monte Carlo Simulation VaR using Geometric Brownian Motion (GBM).

    Instead of relying solely on historical data, we generate thousands
    of possible "tomorrow" scenarios using calibrated randomness.

    How it works
    ------------
    1. Estimate μ (drift) and σ (volatility) from historical data.
    2. For each of 10,000 simulations:
       - Draw a random number Z from N(0,1)
       - Compute simulated return: r = exp((μ - σ²/2) + σ × Z) - 1
    3. Collect all 10,000 simulated returns.
    4. Take the 5th and 1st percentile as VaR.

    Why GBM?
    --------
    GBM models stock prices as: S_t = S_0 × exp((μ - σ²/2)t + σ√t × Z)

    The (μ - σ²/2) term is the "drift correction" — it ensures that the
    expected value of the geometric return matches the arithmetic mean.
    Without it, simulations would systematically overestimate growth.

    Parameters
    ----------
    returns : pd.Series
        Historical daily portfolio returns.
    n_simulations : int
        Number of scenarios to simulate (default 10,000).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    VaRResult
    """
    np.random.seed(seed)  # Makes results reproducible

    mu = returns.mean()
    sigma = returns.std()

    # ── Generate random scenarios ──
    # Z is a vector of 10,000 random numbers from a standard normal distribution.
    Z = np.random.standard_normal(n_simulations)

    # ── Simulate returns using GBM ──
    # For a single time step (t=1 day), the formula simplifies to:
    #   simulated_return = exp((μ - σ²/2) + σ × Z) - 1
    #
    # Breaking this down:
    #   (μ - σ²/2)  = drift-adjusted mean (Itô correction)
    #   σ × Z       = random shock scaled by volatility
    #   exp(...)     = convert log-return to simple return
    #   - 1          = convert price ratio to percentage return
    simulated_returns = np.exp(
        (mu - 0.5 * sigma**2) + sigma * Z
    ) - 1

    # ── VaR from simulated distribution ──
    var_95 = np.percentile(simulated_returns, 5)
    var_99 = np.percentile(simulated_returns, 1)

    # ── Expected Shortfall ──
    es_95 = simulated_returns[simulated_returns <= var_95].mean()
    es_99 = simulated_returns[simulated_returns <= var_99].mean()

    return VaRResult(
        var_95=var_95,
        var_99=var_99,
        es_95=es_95,
        es_99=es_99,
        method='Monte Carlo'
    )


# ─── Convenience Functions ────────────────────────────────────────

def calculate_all_var(returns: pd.Series) -> Dict[str, VaRResult]:
    """
    Run all three VaR methods and return a dictionary of results.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.

    Returns
    -------
    dict
        Keys: 'Historical', 'Parametric', 'Monte Carlo'
        Values: VaRResult objects
    """
    results = {
        'Historical': historical_var(returns),
        'Parametric': parametric_var(returns),
        'Monte Carlo': monte_carlo_var(returns),
    }

    return results


def var_summary_table(
    results: Dict[str, VaRResult],
    portfolio_value: float = 1_000_000
) -> pd.DataFrame:
    """
    Create a formatted summary table of all VaR results.

    Shows both percentage VaR and dollar VaR for a given portfolio value.

    Parameters
    ----------
    results : dict
        Output of calculate_all_var().
    portfolio_value : float
        Portfolio value in dollars (default $1,000,000).

    Returns
    -------
    pd.DataFrame
        Summary table with rows per method and columns for each metric.
    """
    rows = []
    for method_name, result in results.items():
        rows.append({
            'Method': result.method,
            'VaR 95% (%)': f"{result.var_95:.2%}",
            'VaR 99% (%)': f"{result.var_99:.2%}",
            'ES 95% (%)': f"{result.es_95:.2%}",
            'ES 99% (%)': f"{result.es_99:.2%}",
            'VaR 95% ($)': f"${abs(result.var_95) * portfolio_value:,.0f}",
            'VaR 99% ($)': f"${abs(result.var_99) * portfolio_value:,.0f}",
            'ES 95% ($)': f"${abs(result.es_95) * portfolio_value:,.0f}",
            'ES 99% ($)': f"${abs(result.es_99) * portfolio_value:,.0f}",
        })

    return pd.DataFrame(rows).set_index('Method')
```

---

## Step 2.3: What You Just Built

You now have three complete VaR engines. Let's trace through each one with an example:

**Given:** 500 daily portfolio returns, mean = 0.05%, std = 1.2%

| Method | What it does | VaR 95% (approx) |
|---|---|---|
| Historical | Sorts the 500 real returns, picks the 25th worst | -1.82% |
| Parametric | `0.0005 + (-1.645 × 0.012)` | -1.92% |
| Monte Carlo | Generates 10,000 random returns, picks 500th worst | -1.91% |

**Notice:** Parametric and Monte Carlo give similar answers (both assume normality). Historical gives a different answer because it uses the *actual* distribution, which may have fat tails.

---

---

# PART 5: BACKTESTING (Step 3)

Now we build the validation framework that answers: "Can we trust these VaR numbers?"

---

## Step 3.1: The Rolling Window Concept

Imagine a sliding window of 252 days (1 trading year) moving through your data:

```
Day 1─────────Day 252   Day 253 ← predict VaR for this day
     [training window]   [test]

  Day 2─────────Day 253   Day 254 ← predict VaR for this day
       [training window]   [test]

    Day 3─────────Day 254   Day 255 ← predict VaR for this day
         [training window]   [test]

... and so on until the end of the data.
```

Each day, we ask: "Based on the last year of data, what should VaR be?" Then we check if the actual return that day was worse than our prediction.

---

## Step 3.2: Write the Code

**File: `src/backtesting.py`**

```python
"""
backtesting.py — VaR Model Validation Framework
================================================

Validates whether a VaR model's predictions match reality by:
  1. Rolling window backtest — predict VaR, check against actuals
  2. Kupiec's POF test — statistical test of breach frequency
  3. Basel Traffic Light — regulatory compliance check

Why backtest?
  A VaR model that says "95% confidence" should see breaches
  about 5% of the time. If breaches happen 15% of the time,
  the model is dangerously underestimating risk. If they happen
  0.5% of the time, the model is too conservative (wasting capital).
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass
class BacktestResult:
    """
    Holds the complete output of a VaR backtest.

    Attributes
    ----------
    n_observations : int
        Number of days tested (total days minus the training window).
    n_breaches : int
        Number of days where actual loss exceeded predicted VaR.
    breach_rate : float
        n_breaches / n_observations (should be close to 1 - confidence).
    expected_rate : float
        The expected breach rate (0.05 for 95% confidence).
    kupiec_statistic : float
        The likelihood ratio test statistic.
    kupiec_p_value : float
        p-value of the Kupiec test. If < 0.05, the model is rejected.
    kupiec_reject : bool
        True if the model is statistically rejected (p < 0.05).
    traffic_light : str
        Basel traffic light zone: 'Green', 'Yellow', or 'Red'.
    """
    n_observations: int
    n_breaches: int
    breach_rate: float
    expected_rate: float
    kupiec_statistic: float
    kupiec_p_value: float
    kupiec_reject: bool
    traffic_light: str


def kupiec_test(
    n_observations: int,
    n_breaches: int,
    confidence_level: float = 0.95
) -> Tuple[float, float, bool]:
    """
    Kupiec's Proportion of Failures (POF) Test.

    Tests whether the observed number of VaR breaches is statistically
    consistent with the expected failure rate.

    The Math
    --------
    Under H₀ (model is correct), breaches follow a Binomial(n, p) distribution
    where p = 1 - confidence_level.

    The likelihood ratio statistic is:

        LR = -2 × [ln(L₀) - ln(L₁)]

    Where:
        L₀ = (1-p)^(n-x) × p^x            (likelihood under H₀)
        L₁ = (1-x/n)^(n-x) × (x/n)^x      (likelihood under MLE)

    Under H₀, LR follows a χ²(1) distribution.

    Parameters
    ----------
    n_observations : int
        Total number of test days.
    n_breaches : int
        Number of VaR breaches observed.
    confidence_level : float
        The confidence level used for VaR (e.g., 0.95).

    Returns
    -------
    tuple of (statistic, p_value, reject)
        - statistic: the LR test statistic
        - p_value: probability of seeing this result if model is correct
        - reject: True if p_value < 0.05 (model should be rejected)
    """
    p = 1 - confidence_level  # Expected failure rate (e.g., 0.05 for 95% VaR)
    x = n_breaches            # Observed failures
    n = n_observations        # Total observations

    # Handle edge cases
    if x == 0 or x == n:
        # Can't compute log(0), so the test is inconclusive.
        # With 0 breaches, the model is almost certainly too conservative.
        # With n breaches, the model is catastrophically wrong.
        return 0.0, 1.0, False

    # ── Likelihood under H₀: breach rate = p (the expected rate) ──
    # L₀ = (1-p)^(n-x) × p^x
    # Taking log: ln(L₀) = (n-x) × ln(1-p) + x × ln(p)
    log_L0 = (n - x) * np.log(1 - p) + x * np.log(p)

    # ── Likelihood under H₁: breach rate = x/n (the observed rate, MLE) ──
    # L₁ = (1 - x/n)^(n-x) × (x/n)^x
    observed_rate = x / n
    log_L1 = (n - x) * np.log(1 - observed_rate) + x * np.log(observed_rate)

    # ── Likelihood Ratio ──
    # LR = -2 × (ln(L₀) - ln(L₁))
    lr_statistic = -2 * (log_L0 - log_L1)

    # ── P-value from χ²(1) distribution ──
    # sf = survival function = 1 - CDF = probability of getting a MORE extreme statistic
    p_value = stats.chi2.sf(lr_statistic, df=1)

    # ── Decision ──
    reject = p_value < 0.05

    return lr_statistic, p_value, reject


def basel_traffic_light(
    n_breaches: int,
    n_observations: int = 250,
    confidence_level: float = 0.99
) -> str:
    """
    Basel Committee Traffic Light system for VaR model validation.

    The Basel framework uses a simple zone system for 250 trading days
    at 99% confidence. We scale the thresholds proportionally for other
    parameters.

    Standard thresholds (250 days, 99% confidence):
        Green:  0–4 breaches    → Model is acceptable
        Yellow: 5–9 breaches    → Model needs investigation
        Red:    10+ breaches    → Model is rejected

    Parameters
    ----------
    n_breaches : int
        Observed number of breaches.
    n_observations : int
        Number of trading days in the backtest period.
    confidence_level : float
        Confidence level of the VaR model.

    Returns
    -------
    str
        'Green', 'Yellow', or 'Red'
    """
    # Scale thresholds based on actual observation count and confidence level
    # The standard is 250 days at 99% (expected breaches = 2.5)
    expected = n_observations * (1 - confidence_level)

    # Define zones relative to expected breaches
    # Green: up to ~1.6x expected
    # Yellow: 1.6x to ~4x expected
    # Red: above 4x expected
    green_threshold = max(4, int(expected * 1.6))
    yellow_threshold = max(9, int(expected * 3.6))

    if n_breaches <= green_threshold:
        return 'Green'
    elif n_breaches <= yellow_threshold:
        return 'Yellow'
    else:
        return 'Red'


def rolling_var_backtest(
    returns: pd.Series,
    var_function: Callable,
    window: int = 252,
    confidence_level: float = 0.95
) -> Tuple[pd.Series, pd.Series, BacktestResult]:
    """
    Perform a rolling-window VaR backtest.

    For each day after the initial training window:
      1. Estimate VaR using the trailing `window` days
      2. Compare predicted VaR to the actual return on the next day
      3. Record whether a breach occurred

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns (full history).
    var_function : callable
        A VaR function (e.g., historical_var) that takes a returns Series
        and returns a VaRResult.
    window : int
        Number of trailing days to use for each VaR estimate (default 252 = 1 year).
    confidence_level : float
        Confidence level (default 0.95).

    Returns
    -------
    tuple of (var_series, breaches, backtest_result)
        - var_series: pd.Series of predicted VaR values for each test day
        - breaches: pd.Series of boolean flags (True = breach occurred)
        - backtest_result: BacktestResult with summary statistics
    """
    n = len(returns)
    var_values = []   # Will store predicted VaR for each test day
    breach_flags = [] # Will store True/False for each test day
    test_dates = []   # Will store the date of each test day

    # Choose VaR attribute based on confidence level
    var_attr = f"var_{int(confidence_level * 100)}"  # e.g., 'var_95'

    print(f"\nRunning backtest (window={window}, confidence={confidence_level:.0%})...")

    # ── Rolling loop ──
    # Start at index `window` (first day we have enough history)
    # End at the last day (n-1)
    for i in range(window, n):
        # Training data: the `window` days immediately before day i
        train_returns = returns.iloc[i - window:i]

        # Predict VaR using the training data
        result = var_function(train_returns)
        predicted_var = getattr(result, var_attr)

        # Actual return on day i (the day we're testing against)
        actual_return = returns.iloc[i]

        # A breach occurs when the actual return is WORSE (more negative)
        # than the predicted VaR.
        is_breach = actual_return < predicted_var

        var_values.append(predicted_var)
        breach_flags.append(is_breach)
        test_dates.append(returns.index[i])

    # Convert to pandas Series
    var_series = pd.Series(var_values, index=test_dates, name='VaR')
    breaches = pd.Series(breach_flags, index=test_dates, name='Breach')

    # ── Compute summary statistics ──
    n_obs = len(breach_flags)
    n_breaches = sum(breach_flags)
    breach_rate = n_breaches / n_obs if n_obs > 0 else 0

    # Run Kupiec test
    kupiec_stat, kupiec_p, kupiec_reject = kupiec_test(
        n_obs, n_breaches, confidence_level
    )

    # Determine Basel traffic light
    traffic = basel_traffic_light(
        n_breaches, n_obs, confidence_level
    )

    result = BacktestResult(
        n_observations=n_obs,
        n_breaches=n_breaches,
        breach_rate=breach_rate,
        expected_rate=1 - confidence_level,
        kupiec_statistic=kupiec_stat,
        kupiec_p_value=kupiec_p,
        kupiec_reject=kupiec_reject,
        traffic_light=traffic
    )

    print(f"  Observations: {n_obs}")
    print(f"  Breaches: {n_breaches} ({breach_rate:.1%})")
    print(f"  Expected: {1 - confidence_level:.1%}")
    print(f"  Kupiec p-value: {kupiec_p:.4f} ({'REJECT' if kupiec_reject else 'PASS'})")
    print(f"  Basel Traffic Light: {traffic}")

    return var_series, breaches, result
```

---

## Step 3.3: What You Just Built

The backtesting module answers three questions:

1. **Rolling backtest:** "How often did reality exceed our VaR prediction?" — Produces a time series of VaR predictions and breach flags.

2. **Kupiec test:** "Is the breach rate statistically consistent with our confidence level?" — If you claim 95% confidence but breach 12% of the time, the Kupiec test will return p < 0.05, meaning "reject this model."

3. **Basel traffic light:** "Would a bank regulator approve this model?" — Simple color-coded pass/fail based on breach count.

---

---

# PART 6: VISUALIZATION (Step 4)

Time to make everything visual. This module generates four charts.

---

## Step 4.1: Write the Code

**File: `src/visualization.py`**

```python
"""
visualization.py — Publication-Quality Financial Charts
========================================================

Generates four charts:
  1. Return distribution with VaR thresholds
  2. VaR method comparison (bar chart)
  3. Portfolio performance overview (4-panel)
  4. Backtest breach timeline

All charts use seaborn styling for a professional look and
are saved to the output/ directory as PNG files.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, Optional

# Use seaborn's clean style globally
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)

# Import our types
from src.var_models import VaRResult


def plot_return_distribution(
    returns: pd.Series,
    results: Dict[str, VaRResult],
    save_path: str = 'output/return_distribution.png'
) -> None:
    """
    Chart 1: Histogram of portfolio returns with VaR threshold lines.

    This is the most intuitive chart — it shows the actual distribution
    of daily returns and marks where each VaR method draws the "danger line."

    Visual layout:
        ┌──────────────────────────────────────┐
        │           Distribution of             │
        │          Daily Returns                │
        │                                       │
        │    ▎  ┌─────────┐                     │
        │    ▎  │         │                     │
        │    ▎  │         │ ┌──┐                │
        │  VaR──│─────────│─│──│────────────    │
        │  lines│         │ │  │  ┌──┐          │
        │    ▎  │         │ │  │  │  │          │
        └──────────────────────────────────────┘
              -3%  -2%  -1%   0%  +1%  +2%
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the histogram of actual returns
    ax.hist(
        returns, bins=50, density=True,
        alpha=0.6, color='steelblue', edgecolor='white',
        label='Daily Returns'
    )

    # Overlay a KDE (smooth curve) for visual clarity
    returns.plot.kde(ax=ax, linewidth=2, color='navy', label='KDE')

    # ── Draw VaR lines for each method ──
    colors = {'Historical': '#e74c3c', 'Parametric': '#2ecc71', 'Monte Carlo': '#f39c12'}
    linestyles = {'Historical': '-', 'Parametric': '--', 'Monte Carlo': '-.'}

    for method_name, result in results.items():
        ax.axvline(
            result.var_95,
            color=colors[method_name],
            linestyle=linestyles[method_name],
            linewidth=2,
            label=f'{method_name} VaR 95%: {result.var_95:.2%}'
        )

    ax.set_title('Portfolio Return Distribution with VaR Thresholds', fontsize=14, fontweight='bold')
    ax.set_xlabel('Daily Return', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_var_comparison(
    results: Dict[str, VaRResult],
    portfolio_value: float = 1_000_000,
    save_path: str = 'output/var_comparison.png'
) -> None:
    """
    Chart 2: Bar chart comparing VaR and ES across all methods.

    Shows side-by-side bars for each method at both confidence levels.
    Uses dollar values for immediate impact.
    """
    methods = list(results.keys())
    x = np.arange(len(methods))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate dollar values (take absolute value since VaR is negative)
    var_95_vals = [abs(results[m].var_95) * portfolio_value for m in methods]
    var_99_vals = [abs(results[m].var_99) * portfolio_value for m in methods]
    es_95_vals = [abs(results[m].es_95) * portfolio_value for m in methods]
    es_99_vals = [abs(results[m].es_99) * portfolio_value for m in methods]

    # Draw grouped bars
    bars1 = ax.bar(x - 1.5*width, var_95_vals, width, label='VaR 95%', color='#3498db', alpha=0.85)
    bars2 = ax.bar(x - 0.5*width, var_99_vals, width, label='VaR 99%', color='#2c3e50', alpha=0.85)
    bars3 = ax.bar(x + 0.5*width, es_95_vals, width, label='ES 95%', color='#e74c3c', alpha=0.85)
    bars4 = ax.bar(x + 1.5*width, es_99_vals, width, label='ES 99%', color='#c0392b', alpha=0.85)

    # Add value labels on top of each bar
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'${height:,.0f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom', fontsize=7
            )

    ax.set_title(f'VaR & Expected Shortfall Comparison (Portfolio: ${portfolio_value:,.0f})',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Dollar Value at Risk', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'${val:,.0f}'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_portfolio_performance(
    prices: pd.DataFrame,
    returns: pd.Series,
    save_path: str = 'output/portfolio_performance.png'
) -> None:
    """
    Chart 3: Four-panel portfolio overview.

    Panel layout:
    ┌──────────────────┬──────────────────┐
    │ Normalized Prices │ Cumulative Return│
    │ (all stocks)      │ (portfolio)      │
    ├──────────────────┼──────────────────┤
    │ Rolling 30-day   │ Return Histogram  │
    │ Volatility       │                   │
    └──────────────────┴──────────────────┘
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Panel 1: Normalized Prices ──
    # Divide each stock's price series by its first value to show relative performance
    normalized = prices / prices.iloc[0] * 100  # Start at 100
    normalized.plot(ax=axes[0, 0], linewidth=1.5)
    axes[0, 0].set_title('Normalized Stock Prices (Base = 100)', fontweight='bold')
    axes[0, 0].set_ylabel('Index Value')
    axes[0, 0].legend(fontsize=8)

    # ── Panel 2: Cumulative Portfolio Return ──
    cumulative = (1 + returns).cumprod() - 1  # Compound returns over time
    cumulative.plot(ax=axes[0, 1], linewidth=2, color='navy')
    axes[0, 1].set_title('Cumulative Portfolio Return', fontweight='bold')
    axes[0, 1].set_ylabel('Cumulative Return')
    axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    axes[0, 1].axhline(y=0, color='gray', linewidth=0.8, linestyle='--')

    # ── Panel 3: Rolling Volatility ──
    # Calculate standard deviation over a rolling 30-day window, annualize it
    rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
    rolling_vol.plot(ax=axes[1, 0], linewidth=1.5, color='darkorange')
    axes[1, 0].set_title('Rolling 30-Day Annualized Volatility', fontweight='bold')
    axes[1, 0].set_ylabel('Volatility')
    axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # ── Panel 4: Return Distribution ──
    axes[1, 1].hist(returns, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    axes[1, 1].set_title('Daily Return Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Daily Return')
    axes[1, 1].set_ylabel('Density')

    # Add summary statistics as text
    stats_text = (
        f"Mean: {returns.mean():.4%}\n"
        f"Std: {returns.std():.4%}\n"
        f"Skew: {returns.skew():.2f}\n"
        f"Kurt: {returns.kurtosis():.2f}"
    )
    axes[1, 1].text(
        0.02, 0.98, stats_text,
        transform=axes[1, 1].transAxes,
        verticalalignment='top',
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.suptitle('Portfolio Performance Overview', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_backtest_results(
    returns: pd.Series,
    var_series: pd.Series,
    breaches: pd.Series,
    confidence_level: float = 0.95,
    method_name: str = 'Historical',
    save_path: str = 'output/backtest_breaches.png'
) -> None:
    """
    Chart 4: Backtest timeline showing VaR predictions vs actual returns.

    Top panel: Actual returns with VaR line; breach days highlighted.
    Bottom panel: Cumulative breach count over time.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})

    # Align returns to backtest period
    test_returns = returns.loc[var_series.index]

    # ── Top Panel: Returns vs VaR ──
    axes[0].plot(test_returns.index, test_returns.values,
                 linewidth=0.8, color='steelblue', alpha=0.7, label='Actual Returns')
    axes[0].plot(var_series.index, var_series.values,
                 linewidth=1.5, color='red', label=f'{method_name} VaR {confidence_level:.0%}')

    # Highlight breach days with red dots
    breach_dates = breaches[breaches].index
    breach_returns = test_returns.loc[breach_dates]
    axes[0].scatter(breach_dates, breach_returns, color='red', s=20, zorder=5,
                    label=f'Breaches (n={len(breach_dates)})')

    axes[0].set_title(f'VaR Backtest — {method_name} Method ({confidence_level:.0%} Confidence)',
                      fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Daily Return', fontsize=11)
    axes[0].legend(loc='lower left', fontsize=9)
    axes[0].axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

    # ── Bottom Panel: Cumulative Breaches ──
    cumulative_breaches = breaches.astype(int).cumsum()
    axes[1].fill_between(cumulative_breaches.index, cumulative_breaches.values,
                         color='red', alpha=0.3)
    axes[1].plot(cumulative_breaches.index, cumulative_breaches.values,
                 color='red', linewidth=1.5)
    axes[1].set_title('Cumulative VaR Breaches', fontweight='bold')
    axes[1].set_ylabel('Breach Count', fontsize=11)
    axes[1].set_xlabel('Date', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
```

---

---

# PART 7: MAIN SCRIPT (Step 5)

This is the entry point that ties everything together.

---

## Step 5.1: Write the Code

**File: `main.py`**

```python
"""
main.py — Portfolio VaR Calculator Entry Point
================================================

Orchestrates the complete analysis pipeline:
  1. Download market data
  2. Calculate VaR using all three methods
  3. Backtest Historical VaR model
  4. Generate all visualizations

Run with: python main.py
"""

import os
import warnings
from datetime import datetime

# Suppress yfinance and pandas warnings for cleaner output
warnings.filterwarnings('ignore')

# ── Import our modules ──
from src.data_loader import get_sample_portfolio, calculate_returns
from src.var_models import calculate_all_var, var_summary_table, historical_var
from src.backtesting import rolling_var_backtest
from src.visualization import (
    plot_return_distribution,
    plot_var_comparison,
    plot_portfolio_performance,
    plot_backtest_results,
)


def main():
    """Run the complete VaR analysis pipeline."""

    print("=" * 60)
    print("  PORTFOLIO VALUE AT RISK (VaR) CALCULATOR")
    print("=" * 60)
    print(f"  Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ── Step 1: Load Data ──
    print("─" * 40)
    print("STEP 1: Loading Market Data")
    print("─" * 40)

    prices, portfolio_returns, tickers = get_sample_portfolio(period='2y')

    # ── Step 2: Calculate VaR ──
    print()
    print("─" * 40)
    print("STEP 2: Calculating Value at Risk")
    print("─" * 40)

    results = calculate_all_var(portfolio_returns)

    # Display the summary table
    portfolio_value = 1_000_000
    summary = var_summary_table(results, portfolio_value)
    print(f"\nVaR Results (Portfolio: ${portfolio_value:,.0f}):")
    print(summary.to_string())

    # Print key insights
    print(f"\n  Key Insights:")
    print(f"    Skewness: {portfolio_returns.skew():.2f} (0 = symmetric)")
    print(f"    Kurtosis: {portfolio_returns.kurtosis():.2f} (0 = normal distribution)")

    if portfolio_returns.kurtosis() > 3:
        print(f"    ⚠ Fat tails detected — Historical VaR is likely more accurate than Parametric")

    # ── Step 3: Backtest ──
    print()
    print("─" * 40)
    print("STEP 3: Backtesting VaR Model")
    print("─" * 40)

    var_series, breaches, backtest_result = rolling_var_backtest(
        portfolio_returns,
        historical_var,
        window=252,
        confidence_level=0.95
    )

    # ── Step 4: Visualizations ──
    print()
    print("─" * 40)
    print("STEP 4: Generating Visualizations")
    print("─" * 40)

    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)

    print("  Generating charts...")
    plot_return_distribution(portfolio_returns, results)
    plot_var_comparison(results, portfolio_value)
    plot_portfolio_performance(prices, portfolio_returns)
    plot_backtest_results(
        portfolio_returns, var_series, breaches,
        confidence_level=0.95, method_name='Historical'
    )

    # ── Summary ──
    print()
    print("=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Charts saved to: output/")
    print(f"  Files generated:")
    for f in os.listdir('output'):
        print(f"    - {f}")
    print()


if __name__ == '__main__':
    main()
```

---

---

# PART 8: UNIT TESTS (Step 6)

Good code has tests. Let's write tests that verify our VaR calculations are correct.

---

## Step 6.1: Write the Tests

**File: `tests/test_var.py`**

```python
"""
test_var.py — Unit Tests for VaR Models
========================================

Tests verify:
  1. VaR values are negative (they represent losses)
  2. 99% VaR is more extreme than 95% VaR
  3. Expected Shortfall is more extreme than VaR
  4. All three methods produce results in the right ballpark
  5. Kupiec test works correctly

Run with: python -m pytest tests/test_var.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.var_models import historical_var, parametric_var, monte_carlo_var, calculate_all_var
from src.backtesting import kupiec_test


# ─── Test Fixture ──────────────────────────────────────────────

@pytest.fixture
def sample_returns():
    """
    Create a realistic sample of 500 daily returns for testing.
    
    Uses a normal distribution with mean=0.0005 (0.05%) and std=0.012 (1.2%).
    These are typical values for a diversified equity portfolio.
    """
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.012, 500)
    return pd.Series(returns, name='Portfolio')


# ─── Tests for Historical VaR ──────────────────────────────────

class TestHistoricalVar:

    def test_var_is_negative(self, sample_returns):
        """VaR should be negative because it represents a loss threshold."""
        result = historical_var(sample_returns)
        assert result.var_95 < 0, "95% VaR should be negative"
        assert result.var_99 < 0, "99% VaR should be negative"

    def test_var_99_more_extreme_than_95(self, sample_returns):
        """99% VaR should be a bigger loss (more negative) than 95% VaR."""
        result = historical_var(sample_returns)
        assert result.var_99 < result.var_95, \
            "99% VaR should be more negative than 95% VaR"

    def test_es_more_extreme_than_var(self, sample_returns):
        """Expected Shortfall should always be worse than VaR."""
        result = historical_var(sample_returns)
        assert result.es_95 < result.var_95, \
            "ES 95% should be more negative than VaR 95%"
        assert result.es_99 < result.var_99, \
            "ES 99% should be more negative than VaR 99%"

    def test_method_name(self, sample_returns):
        """The method name should be set correctly."""
        result = historical_var(sample_returns)
        assert result.method == 'Historical'


# ─── Tests for Parametric VaR ──────────────────────────────────

class TestParametricVar:

    def test_var_is_negative(self, sample_returns):
        result = parametric_var(sample_returns)
        assert result.var_95 < 0
        assert result.var_99 < 0

    def test_var_99_more_extreme(self, sample_returns):
        result = parametric_var(sample_returns)
        assert result.var_99 < result.var_95

    def test_es_more_extreme(self, sample_returns):
        result = parametric_var(sample_returns)
        assert result.es_95 < result.var_95
        assert result.es_99 < result.var_99

    def test_method_name(self, sample_returns):
        result = parametric_var(sample_returns)
        assert result.method == 'Parametric'


# ─── Tests for Monte Carlo VaR ─────────────────────────────────

class TestMonteCarloVar:

    def test_var_is_negative(self, sample_returns):
        result = monte_carlo_var(sample_returns)
        assert result.var_95 < 0
        assert result.var_99 < 0

    def test_reproducibility(self, sample_returns):
        """Same seed should give same results."""
        result1 = monte_carlo_var(sample_returns, seed=42)
        result2 = monte_carlo_var(sample_returns, seed=42)
        assert result1.var_95 == result2.var_95, \
            "Same seed should produce identical results"

    def test_different_seeds_differ(self, sample_returns):
        """Different seeds should give different results."""
        result1 = monte_carlo_var(sample_returns, seed=42)
        result2 = monte_carlo_var(sample_returns, seed=123)
        assert result1.var_95 != result2.var_95

    def test_method_name(self, sample_returns):
        result = monte_carlo_var(sample_returns)
        assert result.method == 'Monte Carlo'


# ─── Tests for Calculate All ────────────────────────────────────

class TestCalculateAll:

    def test_returns_all_three_methods(self, sample_returns):
        results = calculate_all_var(sample_returns)
        assert 'Historical' in results
        assert 'Parametric' in results
        assert 'Monte Carlo' in results

    def test_all_methods_agree_on_direction(self, sample_returns):
        """All methods should agree that VaR is negative."""
        results = calculate_all_var(sample_returns)
        for method_name, result in results.items():
            assert result.var_95 < 0, f"{method_name} VaR 95% should be negative"


# ─── Tests for Kupiec Test ──────────────────────────────────────

class TestKupiecTest:

    def test_perfect_model_passes(self):
        """A model with exactly the expected breach rate should pass."""
        # 250 obs, 95% confidence → expect 12.5 breaches → round to 12 or 13
        stat, p_value, reject = kupiec_test(250, 12, 0.95)
        assert not reject, "A well-calibrated model should pass Kupiec test"

    def test_bad_model_fails(self):
        """A model with way too many breaches should fail."""
        # 250 obs, 95% confidence → expect 12.5, but we have 40 breaches
        stat, p_value, reject = kupiec_test(250, 40, 0.95)
        assert reject, "A badly calibrated model should fail Kupiec test"

    def test_zero_breaches_does_not_crash(self):
        """Edge case: 0 breaches should not raise an error."""
        stat, p_value, reject = kupiec_test(250, 0, 0.95)
        # Should complete without error


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

---

# PART 9: RUN IT! (Step 7)

---

## Step 7.1: Run the Full Pipeline

Make sure you're in the project root directory, then:

```bash
python main.py
```

You should see output like:

```
============================================================
  PORTFOLIO VALUE AT RISK (VaR) CALCULATOR
============================================================
  Run time: 2026-02-08 14:30:00

────────────────────────────────────
STEP 1: Loading Market Data
────────────────────────────────────
Downloading data for ['SPY', 'AAPL', 'MSFT', 'GOOGL']...
  Downloaded 504 trading days (2024-02-08 to 2026-02-06)

Portfolio Summary:
  Assets: ['SPY', 'AAPL', 'MSFT', 'GOOGL']
  Weights: [0.25 0.25 0.25 0.25]
  Trading days: 503
  Mean daily return: 0.0700%
  Daily volatility: 1.2000%

────────────────────────────────────
STEP 2: Calculating Value at Risk
────────────────────────────────────
VaR Results (Portfolio: $1,000,000):
             VaR 95%    VaR 99%   ES 95%    ES 99% ...
Historical   -1.82%     -3.53%   -2.81%    -4.49%
Parametric   -1.93%     -2.76%   -2.44%    -3.18%
Monte Carlo  -1.95%     -2.76%   -2.46%    -3.21%

────────────────────────────────────
STEP 3: Backtesting VaR Model
────────────────────────────────────
  Observations: 251
  Breaches: 14 (5.6%)
  Expected: 5.0%
  Kupiec p-value: 0.6523 (PASS)
  Basel Traffic Light: Green

────────────────────────────────────
STEP 4: Generating Visualizations
────────────────────────────────────
  Saved: output/return_distribution.png
  Saved: output/var_comparison.png
  Saved: output/portfolio_performance.png
  Saved: output/backtest_breaches.png

============================================================
  ANALYSIS COMPLETE
============================================================
```

---

## Step 7.2: Run the Tests

```bash
pip install pytest
python -m pytest tests/test_var.py -v
```

Expected output:

```
tests/test_var.py::TestHistoricalVar::test_var_is_negative PASSED
tests/test_var.py::TestHistoricalVar::test_var_99_more_extreme_than_95 PASSED
tests/test_var.py::TestHistoricalVar::test_es_more_extreme_than_var PASSED
tests/test_var.py::TestHistoricalVar::test_method_name PASSED
tests/test_var.py::TestParametricVar::test_var_is_negative PASSED
... (all tests PASSED)
```

---

---

# PART 10: HOW TO READ THE RESULTS

---

## 10.1: Interpreting the VaR Table

| Method | VaR 95% | VaR 99% | What it means |
|---|---|---|---|
| Historical | -1.82% | -3.53% | Based on real past data; captures fat tails |
| Parametric | -1.93% | -2.76% | Assumes bell curve; underestimates 99% tail |
| Monte Carlo | -1.95% | -2.76% | Simulated; similar to Parametric since both assume normality |

**Key insight:** Look at the 99% VaR gap between Historical (-3.53%) and Parametric (-2.76%). That 0.77% gap is the **fat tail effect** — real markets have more extreme losses than a bell curve predicts.

---

## 10.2: Interpreting the Backtest

- **Breach rate of 5.6%** vs expected 5.0% — Very close! The model is well-calibrated.
- **Kupiec p-value of 0.6523** — Well above 0.05, so we cannot reject the model. In other words, the model passes.
- **Green traffic light** — A bank regulator would approve this model.

---

## 10.3: Interpreting the Charts

**Chart 1 (Return Distribution):** If Historical VaR line is further left than Parametric, the real distribution has fatter tails than normal.

**Chart 2 (VaR Comparison):** ES bars should always be taller than VaR bars (ES captures more risk). If Historical 99% is much taller than Parametric 99%, the market is not normally distributed.

**Chart 3 (Performance Overview):** Look at the Kurtosis number in the bottom-right panel. Values above 3 confirm fat tails. Skewness < 0 means more extreme losses than gains.

**Chart 4 (Backtest):** Red dots (breaches) should be roughly evenly scattered, not clustered. Clusters of breaches suggest the model fails during volatile periods.

---

---

# PART 11: QUICK REFERENCE CARD

## Project Structure at a Glance

```
main.py                     → Runs everything
src/data_loader.py          → fetch_stock_data(), calculate_returns(), 
                               calculate_portfolio_returns(), get_sample_portfolio()
src/var_models.py           → historical_var(), parametric_var(), monte_carlo_var(),
                               calculate_all_var(), var_summary_table()
src/backtesting.py          → rolling_var_backtest(), kupiec_test(), 
                               basel_traffic_light()
src/visualization.py        → plot_return_distribution(), plot_var_comparison(),
                               plot_portfolio_performance(), plot_backtest_results()
tests/test_var.py           → 14 unit tests across 6 test classes
```

## Key Formulas

| Concept | Formula |
|---|---|
| Daily Return | `(price_today - price_yesterday) / price_yesterday` |
| Portfolio Return | `Σ(weight_i × return_i)` |
| Historical VaR (95%) | `5th percentile of returns` |
| Parametric VaR | `μ - z × σ` (z=1.645 for 95%) |
| Monte Carlo Return | `exp((μ - σ²/2) + σ×Z) - 1` |
| Expected Shortfall | `mean(returns where return ≤ VaR)` |
| Kupiec LR | `-2 × [ln(L₀) - ln(L₁)]` ~ χ²(1) |

## Dependencies

```
numpy      → Arrays, percentiles, random numbers
pandas     → DataFrames, time series
scipy      → Z-scores (norm.ppf), chi-squared test (chi2.sf)
yfinance   → Yahoo Finance data download
matplotlib → Chart creation
seaborn    → Chart styling
pytest     → Test runner
```
