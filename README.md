# Portfolio VaR Calculator

A comprehensive Python implementation of Value at Risk (VaR) using three industry-standard methods with backtesting validation.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 📊 Overview

Value at Risk (VaR) answers a critical question in risk management: **"What is the maximum loss I can expect over a given time period, at a specific confidence level?"**

This project implements all three major VaR methodologies:

| Method | Approach | Best For |
|--------|----------|----------|
| **Historical** | Percentile of actual returns | Fat-tailed distributions |
| **Parametric** | Variance-covariance | Normal distributions |
| **Monte Carlo** | Simulated scenarios | Complex portfolios |

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/avniderashree/portfolio-var-calculator.git
cd portfolio-var-calculator

# Install dependencies
pip install -r requirements.txt

# Run the main analysis
python main.py
```

## 📁 Project Structure

```
portfolio-var-calculator/
├── README.md
├── requirements.txt
├── main.py                 # Main execution script
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Market data fetching
│   ├── var_models.py       # VaR calculation methods
│   ├── backtesting.py      # VaR validation
│   └── visualization.py    # Charts and plots
├── notebooks/
│   └── var_analysis.ipynb  # Interactive exploration
├── data/                   # Downloaded market data
└── tests/                  # Unit tests
```

## 🔧 Features

### VaR Calculation Methods

1. **Historical VaR**: Non-parametric approach using actual return percentiles
2. **Parametric VaR**: Assumes normal distribution, uses mean and standard deviation
3. **Monte Carlo VaR**: Generates 10,000 simulated price paths using GBM

### Backtesting Framework

- Count VaR breaches (actual losses exceeding VaR estimate)
- Kupiec's POF test for statistical validation
- Traffic light system (Basel framework)

### Visualization

- Return distribution with VaR thresholds
- Historical price and VaR breach charts
- Comparative analysis across methods

## 📈 Sample Results

### Portfolio: SPY, AAPL, MSFT, GOOGL (Equal-Weighted)

| Method | VaR (95%) | VaR (99%) | Breaches (252 days) |
|--------|-----------|-----------|---------------------|
| Historical | -1.82% | -2.91% | 11 (4.4%) |
| Parametric | -1.65% | -2.33% | 15 (6.0%) |
| Monte Carlo | -1.78% | -2.85% | 12 (4.8%) |

> **Key Insight**: Parametric VaR underestimates tail risk due to non-normal return distributions

## 🧮 Methodology

### Historical VaR
```python
VaR = percentile(returns, (1 - confidence_level) * 100)
```

### Parametric VaR
```python
VaR = mean - z_score * std_dev
# z_score: 1.645 (95%), 2.326 (99%)
```

### Monte Carlo VaR
```python
# Geometric Brownian Motion simulation
S_t = S_0 * exp((μ - σ²/2)t + σ√t * Z)
# Z ~ N(0,1), simulate 10,000 paths
VaR = percentile(simulated_returns, (1 - confidence_level) * 100)
```

## 📚 Technical Skills Demonstrated

- **Quantitative Modeling**: VaR, volatility estimation, return distributions
- **Python**: pandas, numpy, scipy, yfinance
- **Statistical Analysis**: Hypothesis testing, Monte Carlo simulation
- **Data Visualization**: matplotlib, seaborn
- **Software Engineering**: Modular code, documentation, testing

## 📖 References

- Jorion, P. (2006). *Value at Risk: The New Benchmark for Managing Financial Risk*
- Basel Committee on Banking Supervision. *Supervisory Framework for the Use of Backtesting*

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

*Built as part of a quantitative finance portfolio. See my other projects on [GitHub](https://github.com/avniderashree).*
