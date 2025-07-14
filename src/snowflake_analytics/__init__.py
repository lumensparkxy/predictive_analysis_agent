"""
Snowflake Analytics Agent

A file-based predictive analytics system for Snowflake data with zero external dependencies.
Provides data collection, ML modeling, alerting, and dashboard capabilities.
"""

__version__ = "1.0.0"
__author__ = "Snowflake Analytics Team"
__email__ = "analytics@yourcompany.com"

# Core modules
from .config.settings import get_settings
from .storage.sqlite_store import SQLiteStore
from .utils.logger import setup_logging

__all__ = [
    "get_settings",
    "SQLiteStore", 
    "setup_logging",
]
