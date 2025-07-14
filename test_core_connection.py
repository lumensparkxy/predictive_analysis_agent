#!/usr/bin/env python3
"""
Simple connection test using the core components.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Load environment variables from .env file
def load_env():
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip('"').strip("'")
                    os.environ[key] = value

load_env()

# Test the core connection functionality
try:
    print("🔌 Testing Core Snowflake Connection")
    print("=" * 50)
    
    # Import core components
    from src.snowflake_analytics.config.settings import load_snowflake_config, SnowflakeSettings, get_settings
    from src.snowflake_analytics.connectors.snowflake_client import SnowflakeClient
    from src.snowflake_analytics.utils.logger import setup_logging
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Load settings
    print("📋 Loading application settings...")
    app_settings = get_settings()
    print(f"✅ App settings loaded: {app_settings.app.name} v{app_settings.app.version}")
    
    # Load Snowflake configuration
    print("\n🔧 Loading Snowflake configuration...")
    snowflake_config = load_snowflake_config()
    if not snowflake_config:
        print("❌ Failed to load Snowflake configuration")
        sys.exit(1)
    
    print("✅ Snowflake configuration loaded:")
    print(f"  Account: {snowflake_config.account}")
    print(f"  User: {snowflake_config.user}")
    print(f"  Authenticator: {snowflake_config.authenticator}")
    print(f"  Warehouse: {snowflake_config.warehouse}")
    print(f"  Database: {snowflake_config.database}")
    print(f"  Schema: {snowflake_config.schema}")
    
    # Create settings from config
    print("\n⚙️  Creating Snowflake settings...")
    snowflake_settings = SnowflakeSettings.from_connection_config(snowflake_config)
    print("✅ Snowflake settings created")
    
    # Initialize client
    print("\n🔌 Initializing Snowflake client...")
    client = SnowflakeClient(snowflake_settings)
    print("✅ Snowflake client initialized")
    
    # Test connection
    print("\n🌐 Testing connection to Snowflake...")
    connection = client.connect()
    print("✅ Connection established successfully!")
    
    # Test queries
    print("\n📊 Running test queries...")
    
    # Basic info query
    result = client.execute_query("SELECT CURRENT_VERSION(), CURRENT_USER(), CURRENT_ROLE(), CURRENT_WAREHOUSE()")
    if result and len(result) > 0:
        row = result[0]
        print("✅ System information retrieved:")
        for key, value in row.items():
            print(f"  {key}: {value}")
    
    # Account info query
    print("\n🏢 Getting account information...")
    account_result = client.execute_query("SELECT CURRENT_ACCOUNT(), CURRENT_REGION()")
    if account_result and len(account_result) > 0:
        row = account_result[0]
        print("✅ Account information:")
        for key, value in row.items():
            print(f"  {key}: {value}")
    
    # Test warehouse status
    print("\n🏗️  Checking warehouse status...")
    warehouse_result = client.execute_query("SHOW WAREHOUSES")
    if warehouse_result:
        print(f"✅ Found {len(warehouse_result)} warehouse(s) in account")
        # Show current warehouse info
        current_wh_result = client.execute_query("SELECT CURRENT_WAREHOUSE() as warehouse")
        if current_wh_result:
            current_warehouse = current_wh_result[0]['WAREHOUSE']
            print(f"  Current warehouse: {current_warehouse}")
    else:
        print("⚠️  No warehouse information available")
    
    # Clean up
    client.close()
    print(f"\n🎉 All tests completed successfully!")
    print("✅ JWT authentication is working correctly")
    print("✅ Snowflake Data Collection System is ready to use")
    
except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
