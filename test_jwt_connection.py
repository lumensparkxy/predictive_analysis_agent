#!/usr/bin/env python3
"""
Test script to validate JWT authentication setup for Snowflake connection.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.snowflake_analytics.config.settings import SnowflakeSettings
from src.snowflake_analytics.connectors.snowflake_client import SnowflakeClient


def test_jwt_setup():
    """Test JWT authentication setup and configuration."""
    print("🔐 Testing JWT Authentication Setup...")
    
    try:
        # Load settings
        settings = SnowflakeSettings()
        print(f"✓ Settings loaded successfully")
        print(f"  Account: {settings.account}")
        print(f"  User: {settings.user}")
        print(f"  Authenticator: {settings.authenticator}")
        print(f"  Private Key Path: {settings.private_key_path}")
        
        # Check if private key file exists
        if settings.private_key_path:
            key_path = Path(settings.private_key_path)
            if key_path.exists():
                print(f"✓ Private key file found: {key_path}")
                print(f"  File size: {key_path.stat().st_size} bytes")
            else:
                print(f"✗ Private key file not found: {key_path}")
                return False
        else:
            print("✗ No private key path configured")
            return False
            
        # Initialize client
        client = SnowflakeClient(settings)
        print(f"✓ SnowflakeClient initialized")
        
        # Check if JWT auth is detected
        if client._is_jwt_auth():
            print(f"✓ JWT authentication detected")
        else:
            print(f"✗ JWT authentication not detected")
            return False
            
        # Try to load private key
        if hasattr(client, '_private_key') and client._private_key:
            print(f"✓ Private key loaded successfully")
        else:
            print(f"✗ Private key not loaded")
            return False
            
        print(f"\n🎉 JWT setup validation completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error during JWT setup validation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_connection():
    """Test actual connection to Snowflake."""
    print("\n🌐 Testing Snowflake Connection...")
    
    try:
        settings = SnowflakeSettings()
        client = SnowflakeClient(settings)
        
        # Try to connect
        connection = client.connect()
        print(f"✓ Successfully connected to Snowflake!")
        
        # Test a simple query
        result = client.execute_query("SELECT CURRENT_VERSION()")
        if result:
            version = result[0][0] if result else "Unknown"
            print(f"✓ Snowflake version: {version}")
        
        client.disconnect()
        print(f"✓ Connection closed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Connection test failed: {e}")
        return False


if __name__ == "__main__":
    print("Snowflake JWT Authentication Test")
    print("=" * 50)
    
    # Test JWT setup
    jwt_ok = test_jwt_setup()
    
    if jwt_ok:
        # Test actual connection
        conn_ok = test_connection()
        
        if conn_ok:
            print(f"\n🎉 All tests passed! JWT authentication is working correctly.")
            sys.exit(0)
        else:
            print(f"\n⚠️  JWT setup is correct but connection failed. Check your Snowflake credentials.")
            sys.exit(1)
    else:
        print(f"\n❌ JWT setup validation failed. Please check your configuration.")
        sys.exit(1)
