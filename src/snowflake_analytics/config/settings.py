"""
Settings and configuration management for Snowflake Analytics Agent.

This module provides centralized configuration loading with validation
using Pydantic models and support for environment variable overrides.
"""


# Load environment variables BEFORE any other imports
try:
    from ..env_loader import load_env
except ImportError:
    # Fallback env loading
    import os
    from pathlib import Path
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, validator


class AppConfig(BaseModel):
    """Application configuration."""
    name: str = Field(default="Snowflake Analytics Agent")
    version: str = Field(default="1.0.0")
    description: str = Field(default="Predictive analytics system for Snowflake data")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()


class DataCollectionConfig(BaseModel):
    """Data collection configuration."""
    interval_seconds: int = Field(default=3600, ge=60)
    batch_size: int = Field(default=10000, ge=100)
    max_rows_per_query: int = Field(default=100000, ge=1000)
    retry_attempts: int = Field(default=3, ge=1)
    retry_delay_seconds: int = Field(default=60, ge=10)
    timeout_seconds: int = Field(default=300, ge=30)


class StorageConfig(BaseModel):
    """Storage configuration."""
    data_retention_days: int = Field(default=30, ge=1)
    cleanup_interval_hours: int = Field(default=24, ge=1)
    max_file_size_mb: int = Field(default=100, ge=1)
    compression_enabled: bool = Field(default=True)


class CacheConfig(BaseModel):
    """Cache configuration."""
    ttl_seconds: int = Field(default=3600, ge=60)
    size_limit_gb: float = Field(default=1.0, ge=0.1)
    cleanup_threshold: float = Field(default=0.8, ge=0.1, le=1.0)


class ModelsConfig(BaseModel):
    """ML models configuration."""
    training_interval_hours: int = Field(default=24, ge=1)
    validation_split: float = Field(default=0.2, ge=0.1, le=0.5)
    test_split: float = Field(default=0.1, ge=0.05, le=0.3)
    random_state: int = Field(default=42)
    n_jobs: int = Field(default=-1)


class AlertsConfig(BaseModel):
    """Alerts and notifications configuration."""
    enabled: bool = Field(default=True)
    check_interval_minutes: int = Field(default=15, ge=1)
    cost_threshold_percent: float = Field(default=20.0, ge=1.0)
    usage_threshold_percent: float = Field(default=80.0, ge=1.0, le=100.0)
    error_rate_threshold: float = Field(default=0.1, ge=0.01, le=1.0)


class APIConfig(BaseModel):
    """API server configuration."""
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1000, le=65535)
    workers: int = Field(default=1, ge=1)
    cors_origins: List[str] = Field(default=["*"])
    request_timeout_seconds: int = Field(default=30, ge=10)


class DashboardConfig(BaseModel):
    """Dashboard configuration."""
    refresh_interval_seconds: int = Field(default=300, ge=60)
    max_data_points: int = Field(default=1000, ge=100)
    chart_types: List[str] = Field(default=["line", "bar", "scatter", "heatmap"])


class SnowflakeConnectionConfig(BaseModel):
    """Snowflake connection configuration."""
    account: str
    user: str
    password: Optional[str] = None
    warehouse: str
    database: str
    schema: str
    role: Optional[str] = None
    private_key_path: Optional[str] = None
    private_key_passphrase: Optional[str] = None
    authenticator: Optional[str] = None
    network_timeout: int = Field(default=60, ge=10)  # Network timeout in seconds
    login_timeout: int = Field(default=60, ge=10)    # Login timeout in seconds  
    retry_delay_base: float = Field(default=1.0, ge=0.1)  # Base delay for exponential backoff
    client_session_keep_alive: bool = Field(default=True)  # Keep session alive

    @validator('account', 'user', 'warehouse', 'database', 'schema')
    def validate_required_fields(cls, v):
        if not v or not v.strip():
            raise ValueError("Required Snowflake connection field cannot be empty")
        return v.strip()


class SnowflakeSettings(BaseModel):
    """Snowflake settings for the client and connectors."""
    account: str
    user: str
    password: Optional[str] = None
    warehouse: str
    database: str
    schema: str
    role: Optional[str] = None
    private_key_path: Optional[str] = None
    private_key_passphrase: Optional[str] = None
    authenticator: Optional[str] = None
    
    # Connection settings
    network_timeout: int = Field(default=60)
    login_timeout: int = Field(default=60)
    retry_delay_base: int = Field(default=1)
    
    @classmethod
    def from_connection_config(cls, config: SnowflakeConnectionConfig) -> 'SnowflakeSettings':
        """Create SnowflakeSettings from SnowflakeConnectionConfig."""
        return cls(
            account=config.account,
            user=config.user,
            password=config.password,
            warehouse=config.warehouse,
            database=config.database,
            schema=config.schema,
            role=config.role,
            private_key_path=config.private_key_path,
            private_key_passphrase=config.private_key_passphrase,
            authenticator=config.authenticator
        )


class Settings(BaseModel):
    """Main settings configuration."""
    app: AppConfig = Field(default_factory=AppConfig)
    data_collection: DataCollectionConfig = Field(default_factory=DataCollectionConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    snowflake: Optional[SnowflakeConnectionConfig] = None

    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration as dictionary."""
        return {
            'database_path': 'storage.db',
            'options': {
                'data_retention_days': self.storage.data_retention_days,
                'cleanup_interval_hours': self.storage.cleanup_interval_hours,
                'max_file_size_mb': self.storage.max_file_size_mb,
                'compression_enabled': self.storage.compression_enabled
            }
        }

    def get_snowflake_config(self) -> Optional[SnowflakeConnectionConfig]:
        """Get Snowflake connection configuration."""
        return self.snowflake

    def get_connection_pool_config(self) -> Dict[str, Any]:
        """Get connection pool configuration."""
        return {
            'max_connections': 5,
            'connection_timeout': 300,
            'retry_attempts': 3,
            'retry_delay': 60
        }

    def get_scheduler_config(self) -> Dict[str, Any]:
        """Get scheduler configuration."""
        return {
            'collection_interval': self.data_collection.interval_seconds,
            'batch_size': self.data_collection.batch_size,
            'max_rows_per_query': self.data_collection.max_rows_per_query,
            'retry_attempts': self.data_collection.retry_attempts,
            'retry_delay_seconds': self.data_collection.retry_delay_seconds,
            'timeout_seconds': self.data_collection.timeout_seconds,
            'job_queue': {
                'max_workers': 3,
                'max_queue_size': 100,
                'default_timeout': self.data_collection.timeout_seconds
            },
            'retry_handler': {
                'max_retries': self.data_collection.retry_attempts,
                'backoff_factor': 2.0
            },
            'status_monitor': {
                'check_interval': 60,
                'alert_threshold': 5
            }
        }


def load_json_config(file_path: Path) -> Dict:
    """Load configuration from JSON file with environment variable substitution."""
    if not file_path.exists():
        return {}
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace environment variables in format ${VAR_NAME}
    import re
    def replace_env_var(match):
        var_name = match.group(1)
        return os.getenv(var_name, match.group(0))
    
    content = re.sub(r'\$\{([^}]+)\}', replace_env_var, content)
    
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")


def load_snowflake_config() -> Optional[SnowflakeConnectionConfig]:
    """Load Snowflake connection configuration."""
    # Try to load from environment variables first
    env_config = {}
    env_mapping = {
        'account': 'SNOWFLAKE_ACCOUNT',
        'user': 'SNOWFLAKE_USER', 
        'password': 'SNOWFLAKE_PASSWORD',
        'warehouse': 'SNOWFLAKE_WAREHOUSE',
        'database': 'SNOWFLAKE_DATABASE',
        'schema': 'SNOWFLAKE_SCHEMA',
        'role': 'SNOWFLAKE_ROLE',
        'private_key_path': 'SNOWFLAKE_PRIVATE_KEY_FILE',  # Updated to match .env
        'private_key_passphrase': 'SNOWFLAKE_PRIVATE_KEY_PASSPHRASE',
        'authenticator': 'SNOWFLAKE_AUTHENTICATOR'
    }
    
    for key, env_var in env_mapping.items():
        value = os.getenv(env_var)
        if value:
            env_config[key] = value
    
    # Try to load from JSON config
    config_path = Path("config/snowflake.json")
    json_config = load_json_config(config_path)
    
    # Merge configurations (environment variables take precedence)
    connection_config = json_config.get('connection', {})
    connection_config.update(env_config)
    
    if not connection_config:
        return None
    
    try:
        return SnowflakeConnectionConfig(**connection_config)
    except Exception as e:
        raise ValueError(f"Invalid Snowflake configuration: {e}")


@lru_cache()
def get_settings() -> Settings:
    """Get application settings (cached)."""
    # Load from JSON configuration files
    settings_path = Path("config/settings.json")
    config_data = load_json_config(settings_path)
    
    # Create settings with overrides from environment variables
    settings = Settings(**config_data)
    
    # Override with environment variables where applicable
    if os.getenv("DEBUG"):
        settings.app.debug = os.getenv("DEBUG").lower() in ("true", "1", "yes")
    
    if os.getenv("LOG_LEVEL"):
        settings.app.log_level = os.getenv("LOG_LEVEL")
    
    if os.getenv("API_HOST"):
        settings.api.host = os.getenv("API_HOST")
    
    if os.getenv("API_PORT"):
        settings.api.port = int(os.getenv("API_PORT"))
    
    # Load Snowflake configuration
    settings.snowflake = load_snowflake_config()
    
    return settings


def reload_settings() -> Settings:
    """Force reload of settings (clears cache)."""
    get_settings.cache_clear()
    return get_settings()


def validate_configuration() -> Dict[str, bool]:
    """Validate the complete configuration and return status."""
    status = {
        "settings_loaded": False,
        "snowflake_configured": False,
        "directories_exist": False,
        "permissions_ok": False
    }
    
    try:
        # Test settings loading
        settings = get_settings()
        status["settings_loaded"] = True
        
        # Test Snowflake configuration
        if settings.snowflake:
            status["snowflake_configured"] = True
        
        # Test directory structure
        required_dirs = ["data/raw", "data/processed", "data/models", "data/exports", "cache", "logs"]
        for directory in required_dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)
        status["directories_exist"] = True
        
        # Test write permissions
        test_file = Path("logs/config_test.tmp")
        test_file.write_text("test")
        test_file.unlink()
        status["permissions_ok"] = True
        
    except Exception:
        pass
    
    return status


if __name__ == "__main__":
    # Test configuration loading
    try:
        settings = get_settings()
        print(f"✅ Configuration loaded successfully")
        print(f"App: {settings.app.name} v{settings.app.version}")
        
        if settings.snowflake:
            print(f"✅ Snowflake: {settings.snowflake.account}")
        else:
            print("⚠️  Snowflake configuration not found")
        
        validation = validate_configuration()
        print("\nValidation Results:")
        for check, status in validation.items():
            icon = "✅" if status else "❌"
            print(f"{icon} {check.replace('_', ' ').title()}")
    
    except Exception as e:
        print(f"❌ Configuration error: {e}")
