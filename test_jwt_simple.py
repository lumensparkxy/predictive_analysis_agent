#!/usr/bin/env python3
"""
Simple test script to validate JWT authentication setup for Snowflake connection.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
def load_env():
    """Load environment variables from .env file."""
    env_file = project_root / '.env'
    if env_file.exists():
        print(f"📁 Loading environment from {env_file}")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip('"').strip("'")
                    os.environ[key] = value
        print(f"✓ Environment variables loaded")
    else:
        print(f"⚠️  No .env file found at {env_file}")

load_env()

def test_imports():
    """Test that required libraries can be imported."""
    print("📦 Testing imports...")
    
    try:
        import snowflake.connector
        print("✓ snowflake-connector-python imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import snowflake-connector-python: {e}")
        return False
        
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.serialization import load_pem_private_key
        print("✓ cryptography library imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import cryptography: {e}")
        return False
        
    try:
        from pydantic import BaseModel
        print("✓ pydantic imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import pydantic: {e}")
        return False
        
    return True


def test_env_variables():
    """Test that required environment variables are set."""
    print("\n🔧 Testing environment variables...")
    
    required_vars = [
        'SNOWFLAKE_ACCOUNT',
        'SNOWFLAKE_USER', 
        'SNOWFLAKE_WAREHOUSE',
        'SNOWFLAKE_DATABASE',
        'SNOWFLAKE_SCHEMA',
        'SNOWFLAKE_AUTHENTICATOR',
        'SNOWFLAKE_PRIVATE_KEY_FILE'
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            if var == 'SNOWFLAKE_PRIVATE_KEY_FILE':
                print(f"✓ {var}: {value}")
            else:
                print(f"✓ {var}: {value}")
        else:
            print(f"✗ {var}: Not set")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n❌ Missing required environment variables: {missing_vars}")
        return False
    
    return True


def test_private_key_file():
    """Test that the private key file exists and is readable."""
    print("\n🔐 Testing private key file...")
    
    key_path = os.getenv('SNOWFLAKE_PRIVATE_KEY_FILE')
    if not key_path:
        print("✗ SNOWFLAKE_PRIVATE_KEY_FILE not set")
        return False
        
    key_file = Path(key_path)
    if not key_file.exists():
        print(f"✗ Private key file does not exist: {key_file}")
        return False
        
    print(f"✓ Private key file exists: {key_file}")
    print(f"  File size: {key_file.stat().st_size} bytes")
    
    # Test reading the file
    try:
        with open(key_file, 'rb') as f:
            key_data = f.read()
        print(f"✓ Private key file is readable ({len(key_data)} bytes)")
        
        # Test parsing the key
        from cryptography.hazmat.primitives.serialization import load_pem_private_key
        private_key_obj = load_pem_private_key(key_data, password=None)
        print(f"✓ Private key parsed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to read/parse private key: {e}")
        return False


def test_connection_params():
    """Test building connection parameters."""
    print("\n⚙️  Testing connection parameters...")
    
    try:
        # Build connection parameters manually
        params = {
            'account': os.getenv('SNOWFLAKE_ACCOUNT'),
            'user': os.getenv('SNOWFLAKE_USER'),
            'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE'),
            'database': os.getenv('SNOWFLAKE_DATABASE'),
            'schema': os.getenv('SNOWFLAKE_SCHEMA'),
            'authenticator': os.getenv('SNOWFLAKE_AUTHENTICATOR'),
            'client_session_keep_alive': True,
        }
        
        if os.getenv('SNOWFLAKE_ROLE'):
            params['role'] = os.getenv('SNOWFLAKE_ROLE')
            
        print("✓ Connection parameters built successfully:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # Load private key
        key_path = os.getenv('SNOWFLAKE_PRIVATE_KEY_FILE')
        with open(key_path, 'rb') as f:
            key_data = f.read()
        
        from cryptography.hazmat.primitives.serialization import load_pem_private_key
        from cryptography.hazmat.primitives import serialization
        
        private_key_obj = load_pem_private_key(key_data, password=None)
        private_key_der = private_key_obj.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        params['private_key'] = private_key_der
        print("✓ Private key added to connection parameters")
        
        return params
        
    except Exception as e:
        print(f"✗ Failed to build connection parameters: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_snowflake_connection(params):
    """Test actual connection to Snowflake."""
    print("\n🌐 Testing Snowflake connection...")
    
    try:
        import snowflake.connector
        
        print("Attempting to connect to Snowflake...")
        conn = snowflake.connector.connect(**params)
        print("✓ Successfully connected to Snowflake!")
        
        # Test a simple query
        cursor = conn.cursor()
        cursor.execute("SELECT CURRENT_VERSION()")
        result = cursor.fetchone()
        version = result[0] if result else "Unknown"
        print(f"✓ Snowflake version: {version}")
        
        cursor.close()
        conn.close()
        print("✓ Connection closed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False


if __name__ == "__main__":
    print("Snowflake JWT Authentication Test")
    print("=" * 50)
    
    # Test each component
    tests = [
        ("imports", test_imports),
        ("environment variables", test_env_variables), 
        ("private key file", test_private_key_file),
        ("connection parameters", test_connection_params),
    ]
    
    params = None
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing {test_name}...")
        result = test_func()
        
        if test_name == "connection parameters":
            params = result
            result = params is not None
            
        if not result:
            print(f"❌ {test_name} test failed")
            all_passed = False
            break
        else:
            print(f"✅ {test_name} test passed")
    
    if all_passed and params:
        # Test actual connection
        conn_ok = test_snowflake_connection(params)
        
        if conn_ok:
            print(f"\n🎉 All tests passed! JWT authentication is working correctly.")
            sys.exit(0)
        else:
            print(f"\n⚠️  Setup is correct but connection failed. Check your Snowflake credentials.")
            sys.exit(1)
    else:
        print(f"\n❌ Setup validation failed. Please check your configuration.")
        sys.exit(1)
