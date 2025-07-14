#!/usr/bin/env python3
"""
Test connection command for the main application.
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

# Test the service directly
try:
    from src.snowflake_analytics.data_collection_service import DataCollectionService
    
    print("🔌 Testing Snowflake Data Collection Service")
    print("=" * 50)
    
    # Create service
    service = DataCollectionService()
    
    # Test initialization
    print("📋 Initializing service...")
    if service.initialize():
        print("✅ Service initialized successfully")
    else:
        print("❌ Service initialization failed")
        sys.exit(1)
    
    # Test connection
    print("\n🌐 Testing Snowflake connection...")
    if service.test_connection():
        print("✅ Connection test successful!")
    else:
        print("❌ Connection test failed!")
        sys.exit(1)
    
    # Show status
    print("\n📊 Service status:")
    status = service.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Clean up
    service.shutdown()
    print("\n🎉 All tests passed! Service is working correctly.")
    
except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
