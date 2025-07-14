#!/usr/bin/env python3
"""
Minimal test for JWT authentication with the core Snowflake client.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
def load_env():
    """Load environment variables from .env file."""
    env_file = project_root / '.env'
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip('"').strip("'")
                    os.environ[key] = value

load_env()

# Test the core components directly
try:
    from src.snowflake_analytics.config.settings import load_snowflake_config, SnowflakeSettings
    from src.snowflake_analytics.connectors.snowflake_client import SnowflakeClient
    
    print("🔐 Testing JWT Authentication with Core Components")
    print("=" * 60)
    
    # Load connection config from environment
    print("📋 Loading Snowflake configuration...")
    connection_config = load_snowflake_config()
    if not connection_config:
        print("❌ Failed to load Snowflake configuration from environment variables")
        sys.exit(1)
    
    print(f"✓ Configuration loaded:")
    print(f"  Account: {connection_config.account}")
    print(f"  User: {connection_config.user}")
    print(f"  Authenticator: {connection_config.authenticator}")
    print(f"  Private Key Path: {connection_config.private_key_path}")
    
    # Create settings from connection config
    settings = SnowflakeSettings.from_connection_config(connection_config)
    print(f"✓ Settings created from configuration")
    
    # Initialize client
    print(f"\n🔌 Initializing Snowflake client...")
    client = SnowflakeClient(settings)
    print(f"✓ Client initialized")
    
    # Test connection
    print(f"\n🌐 Testing connection...")
    connection = client.connect()
    print(f"✓ Connected successfully!")
    
    # Test query execution
    print(f"\n📊 Testing query execution...")
    result = client.execute_query("SELECT CURRENT_VERSION(), CURRENT_USER(), CURRENT_ROLE()")
    if result and len(result) > 0:
        row = result[0]
        print(f"✓ Query executed successfully:")
        for key, value in row.items():
            print(f"  {key}: {value}")
    else:
        print(f"⚠️  Query executed but no results returned")
    
    # Test warehouse info
    print(f"\n🏗️  Testing warehouse information...")
    warehouse_result = client.execute_query("SELECT CURRENT_WAREHOUSE()")
    if warehouse_result and len(warehouse_result) > 0:
        row = warehouse_result[0]
        warehouse = list(row.values())[0]  # Get first value
        print(f"✓ Current Warehouse: {warehouse}")
    else:
        print(f"⚠️  Could not retrieve warehouse information")
    
    # Clean up
    client.close()
    print(f"\n✅ All tests passed! JWT authentication is working correctly.")
    
except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
