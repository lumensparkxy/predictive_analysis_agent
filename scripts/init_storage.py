#!/usr/bin/env python3
"""
Database and storage initialization script.

Initializes SQLite database with schema and creates required
directory structure for the analytics system.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from snowflake_analytics.storage.sqlite_store import SQLiteStore
from snowflake_analytics.storage.file_store import FileStore
from snowflake_analytics.config.settings import get_settings


def initialize_storage():
    """Initialize all storage components."""
    print("🔧 Initializing Snowflake Analytics storage...")
    
    try:
        # Initialize SQLite database
        print("  → Initializing SQLite database...")
        sqlite_store = SQLiteStore()
        print("  ✅ SQLite database initialized")
        
        # Initialize file storage
        print("  → Setting up file storage...")
        file_store = FileStore()
        print("  ✅ File storage initialized")
        
        # Create required directories
        print("  → Creating directory structure...")
        directories = [
            "data/raw", "data/processed", "data/models", "data/exports",
            "cache", "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        print("  ✅ Directory structure created")
        
        # Test configuration loading
        print("  → Testing configuration...")
        settings = get_settings()
        print(f"  ✅ Configuration loaded: {settings.app.name}")
        
        # Get database statistics
        stats = sqlite_store.get_database_stats()
        print(f"  📊 Database stats: {stats}")
        
        print("\n🎉 Storage initialization completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Storage initialization failed: {e}")
        return False


if __name__ == "__main__":
    success = initialize_storage()
    sys.exit(0 if success else 1)
