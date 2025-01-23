import pandas as pd
import pytest
from src.components.factor_analysis import FactorAnalyzer

def test_factor_analysis_initialization():
    fa = FactorAnalyzer()
    assert fa is not None

def test_load_data():
    fa = FactorAnalyzer()
    data = fa.load_data("data/sample_data.xlsx")
    assert isinstance(data, pd.DataFrame)
    assert not data.empty

def test_perform_factor_analysis():
    fa = FactorAnalyzer()
    data = fa.load_data("data/sample_data.xlsx")
    fa.fit(data)
    assert hasattr(fa, 'loadings')
    assert fa.loadings.shape[0] > 0

def test_get_eigenvalues():
    fa = FactorAnalyzer()
    data = fa.load_data("data/sample_data.xlsx")
    fa.fit(data)
    eigenvalues = fa.get_eigenvalues()
    assert isinstance(eigenvalues, list)
    assert len(eigenvalues) > 0