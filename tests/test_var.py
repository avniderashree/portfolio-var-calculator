"""
Unit Tests for VaR Calculator
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.var_models import historical_var, parametric_var, monte_carlo_var, calculate_all_var
from src.backtesting import count_var_breaches, kupiec_pof_test, basel_traffic_light


class TestVaRModels:
    """Test VaR calculation methods."""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns for testing."""
        np.random.seed(42)
        return pd.Series(np.random.normal(0.0005, 0.015, 500))
    
    def test_historical_var_returns_correct_structure(self, sample_returns):
        """Test that historical VaR returns correct result structure."""
        result = historical_var(sample_returns)
        
        assert hasattr(result, 'var_95')
        assert hasattr(result, 'var_99')
        assert hasattr(result, 'es_95')
        assert hasattr(result, 'es_99')
        assert result.method == "Historical"
    
    def test_var_95_less_than_var_99(self, sample_returns):
        """Test that 99% VaR is more extreme than 95% VaR."""
        result = historical_var(sample_returns)
        
        # VaR values are negative, so 99% should be more negative
        assert result.var_99 < result.var_95
    
    def test_expected_shortfall_more_extreme_than_var(self, sample_returns):
        """Test that ES is more extreme than VaR at same confidence."""
        result = historical_var(sample_returns)
        
        assert result.es_95 < result.var_95
        assert result.es_99 < result.var_99
    
    def test_parametric_var_symmetry(self):
        """Test parametric VaR with symmetric normal returns."""
        np.random.seed(42)
        symmetric_returns = pd.Series(np.random.normal(0, 0.01, 1000))
        
        result = parametric_var(symmetric_returns)
        
        # For symmetric distribution centered at 0, VaR should be approximately -1.645*sigma at 95%
        expected_var_95 = -1.645 * symmetric_returns.std()
        assert abs(result.var_95 - expected_var_95) < 0.001
    
    def test_monte_carlo_reproducibility(self, sample_returns):
        """Test Monte Carlo VaR is reproducible with same seed."""
        result1 = monte_carlo_var(sample_returns, random_seed=42)
        result2 = monte_carlo_var(sample_returns, random_seed=42)
        
        assert result1.var_95 == result2.var_95
        assert result1.var_99 == result2.var_99
    
    def test_calculate_all_var_returns_all_methods(self, sample_returns):
        """Test that calculate_all_var returns results for all methods."""
        results = calculate_all_var(sample_returns)
        
        assert 'historical' in results
        assert 'parametric' in results
        assert 'monte_carlo' in results


class TestBacktesting:
    """Test backtesting functions."""
    
    def test_count_var_breaches(self):
        """Test breach counting logic."""
        returns = pd.Series([-0.05, -0.02, 0.01, -0.03, 0.02], 
                           index=pd.date_range('2024-01-01', periods=5))
        var_series = pd.Series([-0.025] * 5, 
                               index=pd.date_range('2024-01-01', periods=5))
        
        n_breaches, breach_indicators = count_var_breaches(returns, var_series)
        
        # Returns < -0.025: -0.05, -0.03 = 2 breaches
        assert n_breaches == 2
        assert breach_indicators.sum() == 2
    
    def test_kupiec_test_perfect_model(self):
        """Test Kupiec test for a perfectly calibrated model."""
        # 250 observations, 12-13 breaches expected at 95% (5%)
        stat, pval = kupiec_pof_test(n_observations=250, n_breaches=12, confidence_level=0.95)
        
        # Should not reject null hypothesis
        assert pval > 0.05
    
    def test_kupiec_test_poor_model(self):
        """Test Kupiec test for a poorly calibrated model."""
        # Way too many breaches
        stat, pval = kupiec_pof_test(n_observations=250, n_breaches=50, confidence_level=0.95)
        
        # Should reject null hypothesis
        assert pval < 0.05
    
    def test_basel_traffic_light_green(self):
        """Test Basel traffic light - green zone."""
        result = basel_traffic_light(n_observations=250, n_breaches=3)
        assert "Green" in result
    
    def test_basel_traffic_light_yellow(self):
        """Test Basel traffic light - yellow zone."""
        result = basel_traffic_light(n_observations=250, n_breaches=7)
        assert "Yellow" in result
    
    def test_basel_traffic_light_red(self):
        """Test Basel traffic light - red zone."""
        result = basel_traffic_light(n_observations=250, n_breaches=15)
        assert "Red" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
