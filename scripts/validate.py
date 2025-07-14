#!/usr/bin/env python3
"""
Project validation and status check script.

Validates the project structure, configuration files,
and provides a summary of the current setup status.
"""

import sys
from pathlib import Path
import json
from typing import Dict, List, Tuple


def check_file_exists(file_path: str) -> Tuple[bool, str]:
    """Check if a file exists and return status."""
    path = Path(file_path)
    if path.exists():
        if path.is_file():
            size = path.stat().st_size
            return True, f"✅ {file_path} ({size} bytes)"
        else:
            return False, f"❌ {file_path} exists but is not a file"
    else:
        return False, f"❌ {file_path} missing"


def check_directory_exists(dir_path: str) -> Tuple[bool, str]:
    """Check if a directory exists and return status."""
    path = Path(dir_path)
    if path.exists() and path.is_dir():
        file_count = len(list(path.iterdir()))
        return True, f"✅ {dir_path}/ ({file_count} items)"
    else:
        return False, f"❌ {dir_path}/ missing"


def validate_json_file(file_path: str) -> Tuple[bool, str]:
    """Validate JSON file syntax."""
    try:
        with open(file_path, 'r') as f:
            json.load(f)
        return True, f"✅ {file_path} (valid JSON)"
    except FileNotFoundError:
        return False, f"❌ {file_path} not found"
    except json.JSONDecodeError as e:
        return False, f"❌ {file_path} invalid JSON: {e}"
    except Exception as e:
        return False, f"❌ {file_path} error: {e}"


def validate_project_structure():
    """Validate the complete project structure."""
    print("🔍 Validating Snowflake Analytics Agent Project Structure")
    print("=" * 60)
    
    # Required files
    required_files = [
        "README.md",
        "requirements.txt", 
        "requirements-dev.txt",
        ".env.example",
        ".gitignore",
        "main.py",
    ]
    
    print("\n📄 Core Files:")
    files_status = []
    for file_path in required_files:
        success, message = check_file_exists(file_path)
        files_status.append((success, message))
        print(f"  {message}")
    
    # Configuration files
    print("\n⚙️  Configuration Files:")
    config_files = [
        "config/settings.json",
        "config/snowflake.json"
    ]
    
    config_status = []
    for file_path in config_files:
        # Check existence
        exists, exists_msg = check_file_exists(file_path)
        config_status.append((exists, exists_msg))
        print(f"  {exists_msg}")
        
        # Validate JSON if exists
        if exists:
            valid, valid_msg = validate_json_file(file_path)
            if not valid:
                print(f"    {valid_msg}")
    
    # Script files
    print("\n🔧 Script Files:")
    script_files = [
        "scripts/setup.py",
        "scripts/init_storage.py"
    ]
    
    script_status = []
    for file_path in script_files:
        success, message = check_file_exists(file_path)
        script_status.append((success, message))
        print(f"  {message}")
    
    # Source code structure
    print("\n📦 Source Code Structure:")
    source_dirs = [
        "src/snowflake_analytics",
        "src/snowflake_analytics/config",
        "src/snowflake_analytics/storage",
        "src/snowflake_analytics/data_collection",
        "src/snowflake_analytics/data_processing",
        "src/snowflake_analytics/models",
        "src/snowflake_analytics/utils"
    ]
    
    source_status = []
    for dir_path in source_dirs:
        success, message = check_directory_exists(dir_path)
        source_status.append((success, message))
        print(f"  {message}")
    
    # Data directories
    print("\n💾 Data Directories:")
    data_dirs = [
        "data/raw",
        "data/processed", 
        "data/models",
        "data/exports",
        "cache",
        "logs"
    ]
    
    data_status = []
    for dir_path in data_dirs:
        success, message = check_directory_exists(dir_path)
        data_status.append((success, message))
        print(f"  {message}")
    
    # Test structure
    print("\n🧪 Test Structure:")
    test_dirs = [
        "tests",
        "tests/test_storage",
        "tests/test_data_collection"
    ]
    
    test_status = []
    for dir_path in test_dirs:
        success, message = check_directory_exists(dir_path)
        test_status.append((success, message))
        print(f"  {message}")
    
    # Summary
    print("\n📊 Summary:")
    print("=" * 30)
    
    all_statuses = files_status + config_status + script_status + source_status + data_status + test_status
    total_checks = len(all_statuses)
    passed_checks = sum(1 for success, _ in all_statuses if success)
    
    print(f"Total checks: {total_checks}")
    print(f"Passed: {passed_checks}")
    print(f"Failed: {total_checks - passed_checks}")
    print(f"Success rate: {passed_checks/total_checks*100:.1f}%")
    
    if passed_checks == total_checks:
        print("\n🎉 Project structure validation PASSED!")
        print("Ready for setup and development.")
    else:
        print(f"\n⚠️  Project structure validation has {total_checks - passed_checks} issues.")
        print("Some components may need attention.")
    
    return passed_checks == total_checks


def show_next_steps():
    """Show next steps for users."""
    print("\n🚀 Next Steps:")
    print("=" * 20)
    print("1. Run the automated setup:")
    print("   python3 scripts/setup.py")
    print()
    print("2. Configure Snowflake credentials:")
    print("   cp .env.example .env")
    print("   # Edit .env with your Snowflake details")
    print()
    print("3. Test the installation:")
    print("   python3 main.py status")
    print()
    print("4. Start the application:")
    print("   python3 main.py serve")
    print()
    print("5. Open the dashboard:")
    print("   http://localhost:8000")


def show_project_info():
    """Show project information."""
    print("📋 Project Information:")
    print("=" * 25)
    print("Name: Snowflake Predictive Analytics Agent")
    print("Version: 1.0.0")
    print("Architecture: File-based (SQLite + Parquet + JSON)")
    print("Dependencies: Minimal (see requirements.txt)")
    print("Platform: Cross-platform (Python 3.9+)")
    print()
    print("Key Features:")
    print("• Automated Snowflake data collection")
    print("• ML-based predictive analytics")
    print("• Real-time monitoring dashboard")
    print("• Intelligent alerting system")
    print("• Zero external service dependencies")


def main():
    """Main validation function."""
    print("Snowflake Analytics Agent - Project Validator")
    print("=" * 50)
    
    # Show project info
    show_project_info()
    print()
    
    # Validate structure
    is_valid = validate_project_structure()
    
    # Show next steps
    show_next_steps()
    
    print("\n" + "=" * 50)
    if is_valid:
        print("✅ Project validation completed successfully!")
        return 0
    else:
        print("⚠️  Project validation completed with issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
