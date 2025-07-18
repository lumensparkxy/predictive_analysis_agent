"""
Simple test to verify the testing framework works.
"""

import pytest
from unittest.mock import Mock
import pandas as pd
import numpy as np


def test_basic_functionality():
    """Test basic functionality of the testing framework."""
    # Test basic assertions
    assert 1 + 1 == 2
    assert "test" in "testing"
    assert len([1, 2, 3]) == 3


def test_mock_functionality():
    """Test mock functionality works correctly."""
    mock_obj = Mock()
    mock_obj.method.return_value = "test_result"
    
    result = mock_obj.method()
    assert result == "test_result"
    mock_obj.method.assert_called_once()


def test_pandas_numpy_integration():
    """Test pandas and numpy integration works."""
    # Create sample data
    df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6]
    })
    
    # Test basic operations
    assert len(df) == 3
    assert df['a'].sum() == 6
    assert df['b'].mean() == 5.0
    
    # Test numpy operations
    arr = np.array([1, 2, 3, 4, 5])
    assert arr.sum() == 15
    assert arr.mean() == 3.0


def test_fixtures_work(sample_dataframe, test_config):
    """Test that fixtures from conftest.py work."""
    # Test sample dataframe fixture
    assert sample_dataframe is not None
    assert len(sample_dataframe) == 100
    assert 'timestamp' in sample_dataframe.columns
    assert 'cost' in sample_dataframe.columns
    
    # Test config fixture
    assert test_config is not None
    assert test_config['testing'] is True
    assert test_config['debug'] is True


def test_external_dependencies_mocked(mock_external_dependencies):
    """Test that external dependencies are properly mocked."""
    # Test mocked dependencies
    assert mock_external_dependencies is not None
    assert 'snowflake' in mock_external_dependencies
    assert 'openai' in mock_external_dependencies
    assert 'redis' in mock_external_dependencies
    assert 'requests' in mock_external_dependencies


@pytest.mark.parametrize("input_value,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
    (4, 8),
])
def test_parametrized_functionality(input_value, expected):
    """Test parametrized test functionality."""
    result = input_value * 2
    assert result == expected