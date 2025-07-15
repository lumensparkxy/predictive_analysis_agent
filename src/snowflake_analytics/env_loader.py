"""
Environment Early Loader

This module MUST be imported before any other snowflake_analytics modules
to ensure environment variables are loaded before configuration parsing.
"""

import os
from pathlib import Path

def load_env():
    """Load environment variables from .env file."""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value

# Automatically load environment variables when this module is imported
load_env()
