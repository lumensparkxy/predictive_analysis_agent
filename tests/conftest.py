"""
Pytest configuration and fixtures for Snowflake Analytics Agent tests.

Provides shared test fixtures, database setup/teardown,
and testing utilities for the test suite.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

# Add src to Python path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="snowflake_analytics_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def test_db(test_data_dir):
    """Create a test SQLite database."""
    try:
        from snowflake_analytics.storage.sqlite_store import SQLiteStore
        
        db_path = test_data_dir / "test.db"
        store = SQLiteStore(str(db_path))
        yield store
        
        # Cleanup
        if db_path.exists():
            db_path.unlink()
    except ImportError:
        # Return a mock store if the module doesn't exist
        mock_store = Mock()
        mock_store.query.return_value = []
        mock_store.save.return_value = True
        yield mock_store


@pytest.fixture(scope="function")
def test_file_store(test_data_dir):
    """Create a test file store."""
    try:
        from snowflake_analytics.storage.file_store import FileStore
        
        store = FileStore(str(test_data_dir))
        yield store
    except ImportError:
        # Return a mock store if the module doesn't exist
        mock_store = Mock()
        mock_store.load.return_value = {}
        mock_store.save.return_value = True
        yield mock_store


@pytest.fixture(scope="function")
def test_cache(test_data_dir):
    """Create a test cache store."""
    try:
        from snowflake_analytics.storage.cache_store import CacheStore
        
        cache_dir = test_data_dir / "cache"
        cache = CacheStore(str(cache_dir), size_limit=1_000_000)  # 1MB for tests
        yield cache
    except ImportError:
        # Return a mock cache if the module doesn't exist
        mock_cache = Mock()
        mock_cache.get.return_value = None
        mock_cache.set.return_value = True
        yield mock_cache


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)  # For reproducible tests
    
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
        'value': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'metric': np.random.uniform(0, 100, 100),
        'cost': np.random.uniform(10, 1000, 100),
        'usage': np.random.uniform(0, 10, 100)
    })


@pytest.fixture
def sample_cost_data():
    """Create sample cost data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    return pd.DataFrame({
        'date': dates,
        'warehouse_cost': np.random.uniform(100, 1000, 30),
        'storage_cost': np.random.uniform(50, 500, 30),
        'compute_cost': np.random.uniform(200, 2000, 30),
        'total_cost': np.random.uniform(350, 3500, 30),
        'credits_used': np.random.uniform(10, 100, 30)
    })


@pytest.fixture
def sample_usage_data():
    """Create sample usage data for testing."""
    np.random.seed(42)
    
    timestamps = pd.date_range('2024-01-01', periods=168, freq='H')  # 1 week
    return pd.DataFrame({
        'timestamp': timestamps,
        'query_count': np.random.poisson(50, 168),
        'active_users': np.random.poisson(10, 168),
        'warehouse_usage': np.random.uniform(0, 100, 168),
        'data_processed_gb': np.random.uniform(1, 1000, 168),
        'query_duration_ms': np.random.uniform(100, 10000, 168)
    })


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    from unittest.mock import Mock
    
    settings = Mock()
    settings.app = Mock()
    settings.app.name = "Test Analytics Agent"
    settings.app.version = "0.0.1"
    settings.app.debug = True
    settings.app.log_level = "DEBUG"
    
    settings.snowflake = Mock()
    settings.snowflake.account = "test_account"
    settings.snowflake.user = "test_user"
    settings.snowflake.password = "test_password"
    settings.snowflake.database = "test_db"
    settings.snowflake.warehouse = "test_warehouse"
    
    settings.llm = Mock()
    settings.llm.openai_api_key = "test_key"
    settings.llm.model = "gpt-4"
    settings.llm.temperature = 0.7
    
    return settings


@pytest.fixture
def mock_snowflake_connection():
    """Create a mock Snowflake connection."""
    mock_conn = Mock()
    mock_cursor = Mock()
    
    # Mock cursor methods
    mock_cursor.execute.return_value = None
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = None
    mock_cursor.description = []
    
    mock_conn.cursor.return_value = mock_cursor
    mock_conn.close.return_value = None
    
    return mock_conn


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "Test response"
    
    mock_client.chat.completions.create.return_value = mock_response
    
    return mock_client


@pytest.fixture
def mock_prediction_model():
    """Create a mock prediction model."""
    mock_model = Mock()
    mock_model.is_trained = True
    mock_model.predict.return_value = np.array([100, 200, 300])
    mock_model.fit.return_value = None
    mock_model.score.return_value = 0.85
    
    return mock_model


@pytest.fixture
def mock_anomaly_detector():
    """Create a mock anomaly detector."""
    mock_detector = Mock()
    mock_detector.detect.return_value = [False, False, True, False]
    mock_detector.fit.return_value = None
    mock_detector.score.return_value = 0.95
    
    return mock_detector


@pytest.fixture
def mock_notification_client():
    """Create a mock notification client."""
    mock_client = Mock()
    mock_client.send_email.return_value = True
    mock_client.send_slack.return_value = True
    mock_client.send_sms.return_value = True
    
    return mock_client


@pytest.fixture
def test_config():
    """Create test configuration."""
    return {
        "testing": True,
        "debug": True,
        "log_level": "DEBUG",
        "database_url": "sqlite:///:memory:",
        "cache_size": 1000,
        "batch_size": 100,
        "max_retries": 3,
        "timeout": 30
    }


@pytest.fixture(autouse=True)
def mock_external_dependencies():
    """Mock external dependencies for all tests."""
    # Mock the modules themselves
    mock_snowflake = Mock()
    mock_snowflake.connector = Mock()
    mock_snowflake.connector.connect = Mock()
    
    mock_openai = Mock()
    mock_redis = Mock()
    mock_requests = Mock()
    
    # Mock sys.modules to prevent import errors
    with patch.dict('sys.modules', {
        'snowflake': mock_snowflake,
        'snowflake.connector': mock_snowflake.connector,
        'openai': mock_openai,
        'redis': mock_redis,
        'requests': mock_requests
    }):
        yield {
            'snowflake': mock_snowflake,
            'openai': mock_openai,
            'redis': mock_redis,
            'requests': mock_requests
        }


@pytest.fixture
def performance_test_data():
    """Create large dataset for performance testing."""
    np.random.seed(42)
    
    size = 10000
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=size, freq='min'),
        'metric1': np.random.randn(size),
        'metric2': np.random.uniform(0, 100, size),
        'metric3': np.random.poisson(5, size),
        'cost': np.random.uniform(1, 1000, size),
        'category': np.random.choice(['A', 'B', 'C', 'D'], size),
        'warehouse': np.random.choice(['WH1', 'WH2', 'WH3'], size)
    })


@pytest.fixture
def test_ml_models():
    """Create test ML models for testing."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    
    return {
        'random_forest': RandomForestRegressor(n_estimators=10, random_state=42),
        'linear_regression': LinearRegression(),
        'mock_model': Mock()
    }


@pytest.fixture
def test_alerts():
    """Create test alert configurations."""
    return [
        {
            'id': 'alert_1',
            'name': 'Cost Spike Alert',
            'type': 'cost_anomaly',
            'threshold': 1000,
            'severity': 'high',
            'enabled': True
        },
        {
            'id': 'alert_2',
            'name': 'Usage Anomaly Alert',
            'type': 'usage_anomaly',
            'threshold': 0.95,
            'severity': 'medium',
            'enabled': True
        }
    ]


@pytest.fixture
def test_automation_actions():
    """Create test automation actions."""
    return [
        {
            'id': 'action_1',
            'name': 'Scale Down Warehouse',
            'type': 'warehouse_scaling',
            'parameters': {'size': 'SMALL'},
            'safety_checks': True
        },
        {
            'id': 'action_2',
            'name': 'Pause Warehouse',
            'type': 'warehouse_pause',
            'parameters': {},
            'safety_checks': True
        }
    ]
