#!/usr/bin/env python3
"""
Automated setup script for Snowflake Analytics Agent.

Provides one-command setup with virtual environment creation,
dependency installation, and system initialization.
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path
import venv
import json


class SnowflakeAnalyticsSetup:
    """Automated setup manager for the analytics system."""
    
    def __init__(self):
        """Initialize setup manager."""
        self.project_root = Path(__file__).parent.parent
        self.venv_path = self.project_root / "venv"
        self.python_executable = None
        self.setup_log = []
    
    def log(self, message: str, level: str = "INFO"):
        """Log a setup message."""
        log_entry = f"[{level}] {message}"
        self.setup_log.append(log_entry)
        print(log_entry)
    
    def check_python_version(self) -> bool:
        """Check if Python version is 3.9 or higher."""
        if sys.version_info < (3, 9):
            self.log(f"Python 3.9+ required. Current version: {sys.version}", "ERROR")
            return False
        
        self.log(f"Python version check passed: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        return True
    
    def create_virtual_environment(self) -> bool:
        """Create a virtual environment for the project."""
        try:
            if self.venv_path.exists():
                self.log("Virtual environment already exists, removing...")
                shutil.rmtree(self.venv_path)
            
            self.log("Creating virtual environment...")
            venv.create(self.venv_path, with_pip=True)
            
            # Determine Python executable path
            if sys.platform == "win32":
                self.python_executable = self.venv_path / "Scripts" / "python.exe"
                pip_executable = self.venv_path / "Scripts" / "pip.exe"
            else:
                self.python_executable = self.venv_path / "bin" / "python"
                pip_executable = self.venv_path / "bin" / "pip"
            
            if not self.python_executable.exists():
                raise Exception("Python executable not found in virtual environment")
            
            self.log("Virtual environment created successfully")
            return True
            
        except Exception as e:
            self.log(f"Failed to create virtual environment: {e}", "ERROR")
            return False
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies."""
        try:
            requirements_file = self.project_root / "requirements.txt"
            if not requirements_file.exists():
                self.log("requirements.txt not found", "ERROR")
                return False
            
            self.log("Installing production dependencies...")
            result = subprocess.run([
                str(self.python_executable), "-m", "pip", "install", 
                "-r", str(requirements_file)
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode != 0:
                self.log(f"Failed to install dependencies: {result.stderr}", "ERROR")
                return False
            
            # Install development dependencies if file exists
            dev_requirements = self.project_root / "requirements-dev.txt"
            if dev_requirements.exists():
                self.log("Installing development dependencies...")
                result = subprocess.run([
                    str(self.python_executable), "-m", "pip", "install",
                    "-r", str(dev_requirements)
                ], capture_output=True, text=True, cwd=self.project_root)
                
                if result.returncode != 0:
                    self.log(f"Warning: Failed to install dev dependencies: {result.stderr}", "WARNING")
            
            self.log("Dependencies installed successfully")
            return True
            
        except Exception as e:
            self.log(f"Failed to install dependencies: {e}", "ERROR")
            return False
    
    def create_directory_structure(self) -> bool:
        """Create required directory structure."""
        try:
            directories = [
                "data/raw", "data/processed", "data/models", "data/exports",
                "cache", "logs", "config",
                "src/snowflake_analytics/storage",
                "src/snowflake_analytics/data_collection",
                "src/snowflake_analytics/data_processing",
                "src/snowflake_analytics/models",
                "src/snowflake_analytics/utils",
                "tests", "scripts", "notebooks"
            ]
            
            self.log("Creating directory structure...")
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                
                # Add .gitkeep files to empty directories
                if directory.startswith("data/") or directory in ["cache", "logs"]:
                    gitkeep = dir_path / ".gitkeep"
                    if not gitkeep.exists():
                        gitkeep.write_text("# Placeholder for directory\n")
            
            self.log("Directory structure created successfully")
            return True
            
        except Exception as e:
            self.log(f"Failed to create directory structure: {e}", "ERROR")
            return False
    
    def initialize_database(self) -> bool:
        """Initialize SQLite database."""
        try:
            self.log("Initializing database...")
            
            # Run the database initialization
            result = subprocess.run([
                str(self.python_executable), "-c",
                "import sys; sys.path.insert(0, 'src'); "
                "from snowflake_analytics.storage.sqlite_store import SQLiteStore; "
                "store = SQLiteStore(); "
                "print('Database initialized successfully')"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode != 0:
                self.log(f"Database initialization failed: {result.stderr}", "ERROR")
                return False
            
            self.log("Database initialized successfully")
            return True
            
        except Exception as e:
            self.log(f"Failed to initialize database: {e}", "ERROR")
            return False
    
    def generate_configuration_files(self) -> bool:
        """Generate sample configuration files."""
        try:
            self.log("Generating configuration files...")
            
            # Check if .env already exists
            env_file = self.project_root / ".env"
            if not env_file.exists():
                env_example = self.project_root / ".env.example"
                if env_example.exists():
                    shutil.copy(env_example, env_file)
                    self.log("Created .env file from template")
                else:
                    # Create basic .env file
                    env_content = """# Snowflake Connection Settings
SNOWFLAKE_ACCOUNT=your_account_identifier
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema
SNOWFLAKE_ROLE=your_role

# Application Settings
LOG_LEVEL=INFO
DEBUG=false
API_HOST=0.0.0.0
API_PORT=8000
"""
                    env_file.write_text(env_content)
                    self.log("Created basic .env file")
            
            # Verify configuration files exist
            config_files = ["config/settings.json", "config/snowflake.json"]
            for config_file in config_files:
                config_path = self.project_root / config_file
                if not config_path.exists():
                    self.log(f"Warning: Configuration file missing: {config_file}", "WARNING")
            
            self.log("Configuration files ready")
            return True
            
        except Exception as e:
            self.log(f"Failed to generate configuration: {e}", "ERROR")
            return False
    
    def run_basic_tests(self) -> bool:
        """Run basic system tests to verify setup."""
        try:
            self.log("Running basic system tests...")
            
            # Test configuration loading
            result = subprocess.run([
                str(self.python_executable), "-c",
                "import sys; sys.path.insert(0, 'src'); "
                "from snowflake_analytics.config.settings import get_settings; "
                "settings = get_settings(); "
                "print(f'Configuration loaded: {settings.app.name}')"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode != 0:
                self.log(f"Configuration test failed: {result.stderr}", "ERROR")
                return False
            
            # Test health check
            result = subprocess.run([
                str(self.python_executable), "-c",
                "import sys; sys.path.insert(0, 'src'); "
                "from snowflake_analytics.utils.health_check import quick_health_check; "
                "healthy = quick_health_check(); "
                "print('Health check:', 'PASSED' if healthy else 'FAILED')"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode != 0:
                self.log(f"Health check test failed: {result.stderr}", "WARNING")
                # Don't fail setup for health check issues
            
            self.log("Basic tests completed")
            return True
            
        except Exception as e:
            self.log(f"Basic tests failed: {e}", "WARNING")
            return True  # Don't fail setup for test issues
    
    def generate_activation_script(self) -> bool:
        """Generate activation script for easy environment management."""
        try:
            if sys.platform == "win32":
                script_name = "activate.bat"
                script_content = f"""@echo off
call "{self.venv_path}\\Scripts\\activate.bat"
echo Snowflake Analytics Agent environment activated
echo Run "python main.py --help" to see available commands
"""
            else:
                script_name = "activate.sh"
                script_content = f"""#!/bin/bash
source "{self.venv_path}/bin/activate"
echo "Snowflake Analytics Agent environment activated"
echo "Run 'python main.py --help' to see available commands"
"""
            
            script_path = self.project_root / script_name
            script_path.write_text(script_content)
            
            if not sys.platform == "win32":
                os.chmod(script_path, 0o755)
            
            self.log(f"Created activation script: {script_name}")
            return True
            
        except Exception as e:
            self.log(f"Failed to create activation script: {e}", "WARNING")
            return True  # Don't fail setup for this
    
    def run_setup(self) -> bool:
        """Run the complete setup process."""
        self.log("Starting Snowflake Analytics Agent setup...")
        self.log("=" * 50)
        
        # Setup steps
        steps = [
            ("Checking Python version", self.check_python_version),
            ("Creating virtual environment", self.create_virtual_environment),
            ("Installing dependencies", self.install_dependencies),
            ("Creating directory structure", self.create_directory_structure),
            ("Initializing database", self.initialize_database),
            ("Generating configuration", self.generate_configuration_files),
            ("Running basic tests", self.run_basic_tests),
            ("Creating activation script", self.generate_activation_script),
        ]
        
        for step_name, step_function in steps:
            self.log(f"\n➤ {step_name}...")
            if not step_function():
                self.log(f"Setup failed at step: {step_name}", "ERROR")
                return False
        
        return True
    
    def print_success_message(self):
        """Print setup success message with next steps."""
        self.log("\n" + "=" * 50)
        self.log("🎉 Snowflake Analytics Agent setup completed successfully!")
        self.log("=" * 50)
        
        self.log("\nNext steps:")
        self.log("1. Activate the virtual environment:")
        if sys.platform == "win32":
            self.log("   activate.bat")
        else:
            self.log("   source activate.sh")
        
        self.log("\n2. Configure your Snowflake credentials:")
        self.log("   Edit the .env file with your Snowflake account details")
        
        self.log("\n3. Verify the setup:")
        self.log("   python main.py status")
        
        self.log("\n4. Start the application:")
        self.log("   python main.py serve")
        
        self.log("\n5. Access the dashboard:")
        self.log("   Open http://localhost:8000 in your browser")
        
        self.log(f"\nProject root: {self.project_root}")
        self.log(f"Virtual environment: {self.venv_path}")
        
        self.log("\n📖 For more information, see README.md")
    
    def save_setup_log(self):
        """Save setup log to file."""
        try:
            log_file = self.project_root / "logs" / "setup.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(log_file, 'w') as f:
                f.write("Snowflake Analytics Agent Setup Log\n")
                f.write("=" * 40 + "\n\n")
                for entry in self.setup_log:
                    f.write(entry + "\n")
            
            self.log(f"Setup log saved to: {log_file}")
        except Exception as e:
            self.log(f"Failed to save setup log: {e}", "WARNING")


def main():
    """Main setup function."""
    print("🚀 Snowflake Analytics Agent - Automated Setup")
    print("=" * 50)
    
    setup = SnowflakeAnalyticsSetup()
    
    try:
        success = setup.run_setup()
        
        if success:
            setup.print_success_message()
        else:
            setup.log("\n❌ Setup failed. Check the error messages above.", "ERROR")
            setup.log("You can run this script again after fixing the issues.")
            sys.exit(1)
        
    except KeyboardInterrupt:
        setup.log("\n\nSetup interrupted by user.", "WARNING")
        sys.exit(1)
    except Exception as e:
        setup.log(f"\nUnexpected error during setup: {e}", "ERROR")
        sys.exit(1)
    finally:
        setup.save_setup_log()


if __name__ == "__main__":
    main()
