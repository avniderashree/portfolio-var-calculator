# Portfolio Value at Risk (VaR) Calculator

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-ready Python implementation of Value at Risk (VaR) using three industry-standard methodologies, complete with backtesting validation and professional visualizations.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Sample Results](#-sample-results)
- [Usage Examples](#-usage-examples)
- [Backtesting Framework](#-backtesting-framework)
- [Visualizations](#-visualizations)
- [Technical Skills Demonstrated](#-technical-skills-demonstrated)
- [References](#-references)

---

## 📊 Overview

**Value at Risk (VaR)** is the cornerstone of financial risk management. It answers a critical question:

> *"What is the maximum loss I can expect over a given time period, at a specific confidence level?"*

For example, a **1-day 95% VaR of -1.65%** means:
- With 95% confidence, the portfolio will **not** lose more than 1.65% in a single day
- Equivalently, there's a 5% chance of losing **more** than 1.65%

This project implements **three complementary approaches** to VaR calculation, allowing risk managers to compare methodologies and choose the most appropriate for their use case.

---

## ✨ Features

### VaR Calculation Methods

| Method | Approach | Assumptions | Best For |
|--------|----------|-------------|----------|
| **Historical** | Percentile of actual returns | None (non-parametric) | Fat-tailed distributions |
| **Parametric** | Variance-covariance | Normal distribution | Quick estimates |
| **Monte Carlo** | GBM simulation (10,000 paths) | Log-normal prices | Complex portfolios |

### Additional Capabilities

- ✅ **Expected Shortfall (CVaR)** — Average loss beyond VaR threshold
- ✅ **Rolling Backtesting** — Validates model accuracy over time
- ✅ **Kupiec's POF Test** — Statistical validation of breach rate
- ✅ **Basel Traffic Light** — Regulatory framework compliance
- ✅ **Professional Visualizations** — Publication-ready charts
- ✅ **Real Market Data** — Live data via Yahoo Finance API

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/avniderashree/portfolio-var-calculator.git
cd portfolio-var-calculator

# Install dependencies
pip install -r requirements.txt
```

### Run the Analysis

```bash
python main.py
```

This will:
1. Download 2 years of market data for SPY, AAPL, MSFT, GOOGL
2. Calculate VaR using all three methods
3. Backtest models against historical returns
4. Generate visualizations in `output/` directory

### Interactive Exploration

```bash
jupyter notebook notebooks/var_analysis.ipynb
```

---

## 📁 Project Structure

```
portfolio-var-calculator/
│
├── main.py                     # Main execution script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── src/                        # Core modules
│   ├── __init__.py
│   ├── data_loader.py          # Market data fetching & portfolio returns
│   ├── var_models.py           # VaR calculation (Historical, Parametric, MC)
│   ├── backtesting.py          # Validation framework (Kupiec, Basel)
│   └── visualization.py        # Charts and plots
│
├── notebooks/
│   └── var_analysis.ipynb      # Interactive Jupyter walkthrough
│
├── tests/
│   └── test_var.py             # Unit tests
│
├── output/                     # Generated visualizations
│   ├── return_distribution.png
│   ├── var_comparison.png
│   ├── portfolio_performance.png
│   └── backtest_breaches.png
│
└── data/                       # Downloaded market data (gitignored)
```

---

## 🧮 Methodology

### 1. Historical VaR (Non-Parametric)

Uses the empirical distribution of past returns without assuming any distribution shape.

```python
VaR_α = Percentile(returns, (1 - α) × 100)

# Example: 95% VaR
VaR_95 = 5th percentile of historical returns
```

**Advantages:**
- No distribution assumptions
- Captures fat tails and skewness
- Easy to explain to stakeholders

**Limitations:**
- Requires sufficient historical data
- Assumes past patterns continue

---

### 2. Parametric VaR (Variance-Covariance)

Assumes returns follow a normal distribution and uses analytical formulas.

```python
VaR_α = μ - z_α × σ

# Where:
#   μ = mean daily return
#   σ = standard deviation of returns
#   z_α = z-score for confidence level (1.645 for 95%, 2.326 for 99%)
```

**Advantages:**
- Computationally efficient
- Works with small datasets
- Easy to scale to large portfolios

**Limitations:**
- Underestimates tail risk (fat tails)
- Misses skewness in returns

---

### 3. Monte Carlo VaR (Simulation-Based)

Simulates future price paths using Geometric Brownian Motion (GBM).

```python
# Simulate 10,000 possible future returns
S_t = S_0 × exp((μ - σ²/2)t + σ√t × Z)

# Where Z ~ N(0,1)
# VaR = Percentile of simulated returns
```

**Advantages:**
- Flexible for complex instruments
- Can model non-linear payoffs
- Handles path-dependent options

**Limitations:**
- Computationally intensive
- Dependent on parameter estimates

---

### 4. Expected Shortfall (CVaR)

Also known as Conditional VaR, this measures the **average loss** when losses exceed VaR.

```python
ES_α = E[Loss | Loss > VaR_α]

# For a 95% ES: average of the worst 5% of losses
```

ES is preferred by Basel III because it:
- Captures tail risk magnitude (not just threshold)
- Is a coherent risk measure
- Better for extreme scenarios

---

## 📈 Sample Results

### Portfolio Tested
- **Assets:** SPY, AAPL, MSFT, GOOGL (equal-weighted)
- **Period:** 2 years of daily data
- **Portfolio Value:** $1,000,000

### VaR Results

| Method | VaR (95%) | VaR (99%) | ES (95%) | ES (99%) |
|--------|-----------|-----------|----------|----------|
| Historical | $18,212 | $35,320 | $28,113 | $44,857 |
| Parametric | $19,257 | $27,613 | $24,381 | $31,842 |
| Monte Carlo | $19,455 | $27,617 | $24,611 | $32,105 |

### Key Insights

1. **Fat Tails Present:** Portfolio kurtosis of 12.93 (vs. 0 for normal)
2. **Parametric Underestimates:** 99% Parametric VaR is 22% lower than Historical
3. **ES Captures More Risk:** ES (95%) is ~50% larger than VaR (95%)

---

## 💻 Usage Examples

### Basic VaR Calculation

```python
from src.data_loader import get_sample_portfolio
from src.var_models import calculate_all_var, var_summary_table

# Load portfolio data
prices, portfolio_returns, tickers = get_sample_portfolio(period='2y')

# Calculate VaR using all methods
results = calculate_all_var(portfolio_returns)

# Display summary
print(var_summary_table(results))
```

### Custom Portfolio

```python
from src.data_loader import fetch_stock_data, calculate_returns, calculate_portfolio_returns

# Define your portfolio
tickers = ['NVDA', 'META', 'AMZN', 'TSLA']
weights = [0.3, 0.25, 0.25, 0.2]  # Custom weights

# Fetch data and calculate returns
prices = fetch_stock_data(tickers, period='1y')
returns = calculate_returns(prices)
portfolio_returns = calculate_portfolio_returns(returns, weights)

# Calculate VaR
from src.var_models import historical_var
result = historical_var(portfolio_returns)
print(f"95% VaR: {result.var_95:.2%}")
```

### Backtesting

```python
from src.backtesting import rolling_var_backtest
from src.var_models import historical_var

# Run backtest
var_series, breaches, result = rolling_var_backtest(
    portfolio_returns, 
    historical_var, 
    window=252,  # 1-year rolling window
    confidence_level=0.95
)

print(f"Breaches: {result.n_breaches}/{result.n_observations}")
print(f"Status: {result.traffic_light}")
```

---

## 🔍 Backtesting Framework

### Validation Approach

The backtest uses a **rolling window** approach:
1. At each day `t`, use the past 252 days to estimate VaR
2. Compare predicted VaR against actual return on day `t+1`
3. Count "breaches" where actual loss exceeds VaR

### Statistical Tests

#### Kupiec's Proportion of Failures (POF) Test

Tests whether the observed breach rate matches the expected rate.

- **H₀:** Model is correctly calibrated (breach rate = 1 - confidence)
- **Test Statistic:** Likelihood ratio (χ² distributed with 1 df)
- **Decision:** Reject H₀ if p-value < 0.05

#### Basel Traffic Light System

For 250 trading days at 99% confidence:

| Zone | Breaches | Interpretation |
|------|----------|----------------|
| 🟢 Green | 0-4 | Model is acceptable |
| 🟡 Yellow | 5-9 | Requires scrutiny |
| 🔴 Red | 10+ | Model is flawed |

---

## 📊 Visualizations

The project generates four publication-ready charts:

### 1. Return Distribution with VaR Thresholds
Shows the empirical distribution of returns with VaR lines from all three methods.

### 2. VaR Method Comparison
Bar chart comparing VaR and ES across methods at both confidence levels.

### 3. Portfolio Performance Overview
Four-panel view: normalized prices, cumulative returns, rolling volatility, return histogram.

### 4. Backtest Breach Timeline
Visual timeline of VaR breaches and cumulative breach count.

---

## 🎓 Technical Skills Demonstrated

| Category | Skills |
|----------|--------|
| **Quantitative Finance** | VaR, CVaR, Monte Carlo simulation, volatility modeling |
| **Statistics** | Hypothesis testing, confidence intervals, distribution analysis |
| **Python** | pandas, numpy, scipy, yfinance, matplotlib, seaborn |
| **Software Engineering** | Modular architecture, type hints, docstrings, unit testing |
| **Data Visualization** | Publication-quality charts, multi-panel layouts |

---

## 📚 References

1. Jorion, P. (2006). *Value at Risk: The New Benchmark for Managing Financial Risk*. McGraw-Hill.
2. Basel Committee on Banking Supervision. *Supervisory Framework for the Use of Backtesting*.
3. Kupiec, P. (1995). *Techniques for Verifying the Accuracy of Risk Measurement Models*. Journal of Derivatives.
4. Hull, J.C. (2018). *Risk Management and Financial Institutions*. Wiley.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Avni Derashree**  
Quantitative Risk Analyst | Python | Machine Learning

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/avniderashree/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/avniderashree)

---

*This project is part of a quantitative finance portfolio. See my other projects on [GitHub](https://github.com/avniderashree).*
