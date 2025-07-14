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
    from snowflake_analytics.storage.sqlite_store import SQLiteStore
    
    db_path = test_data_dir / "test.db"
    store = SQLiteStore(str(db_path))
    yield store
    
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture(scope="function")
def test_file_store(test_data_dir):
    """Create a test file store."""
    from snowflake_analytics.storage.file_store import FileStore
    
    store = FileStore(str(test_data_dir))
    yield store


@pytest.fixture(scope="function")
def test_cache(test_data_dir):
    """Create a test cache store."""
    from snowflake_analytics.storage.cache_store import CacheStore
    
    cache_dir = test_data_dir / "cache"
    cache = CacheStore(str(cache_dir), size_limit=1_000_000)  # 1MB for tests
    yield cache


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    import pandas as pd
    import numpy as np
    
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
        'value': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'metric': np.random.uniform(0, 100, 100)
    })


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    from snowflake_analytics.config.settings import Settings, AppConfig
    
    return Settings(
        app=AppConfig(
            name="Test Analytics Agent",
            version="0.0.1",
            debug=True,
            log_level="DEBUG"
        )
    )
