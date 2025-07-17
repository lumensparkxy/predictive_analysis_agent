"""
Snowflake Analytics Agent

A comprehensive data collection and analytics system for Snowflake with automated
collection, validation, monitoring, and predictive capabilities.
"""

__version__ = "1.0.0"
__author__ = "Snowflake Analytics Team"
__email__ = "analytics@yourcompany.com"

# Import main service and components
try:
    from .data_collection_service import DataCollectionService, main
except ImportError as e:
    print(f"Warning: DataCollectionService import failed: {e}")
    DataCollectionService = None
    main = None

try:
    from .connectors import ConnectionPool, SnowflakeClient
except ImportError as e:
    print(f"Warning: Connectors import failed: {e}")
    try:
        from .connectors.snowflake_client import SnowflakeClient
        from .connectors.connection_pool import ConnectionPool
    except ImportError:
        SnowflakeClient = None
        ConnectionPool = None

try:
    from .data_collection import (
        UsageCollector, QueryMetricsCollector, WarehouseMetricsCollector,
        UserActivityCollector, CostCollector
    )
except ImportError as e:
    print(f"Warning: Data collection components import failed: {e}")
    UsageCollector = None
    QueryMetricsCollector = None
    WarehouseMetricsCollector = None
    UserActivityCollector = None
    CostCollector = None

try:
    from .scheduler import (
        CollectionScheduler, JobQueue, RetryHandler, StatusMonitor
    )
except ImportError as e:
    print(f"Warning: Scheduler components import failed: {e}")
    CollectionScheduler = None
    JobQueue = None
    RetryHandler = None
    StatusMonitor = None

try:
    from .validation import (
        SchemaValidator, DataQualityChecker, AnomalyDetector, DataLineageTracker
    )
except ImportError as e:
    print(f"Warning: Validation components import failed: {e}")
    SchemaValidator = None
    DataQualityChecker = None
    AnomalyDetector = None
    DataLineageTracker = None

try:
    from .llm import (
        LLMService, get_llm_service, create_llm_service, reset_llm_service,
        OpenAIClient, AzureOpenAIClient, QueryInterface, InsightGenerator
    )
except ImportError as e:
    print(f"Warning: LLM components import failed: {e}")
    LLMService = None
    get_llm_service = None
    create_llm_service = None
    reset_llm_service = None
    OpenAIClient = None
    AzureOpenAIClient = None
    QueryInterface = None
    InsightGenerator = None

# Core modules (always available)
from .config.settings import get_settings, Settings
from .storage.sqlite_store import SQLiteStore
from .storage import FileStore, CacheStore
from .utils.logger import setup_logging, get_logger
from .utils.health_check import HealthChecker

# Main service instance for easy access
_service_instance = None


def create_service(config_path: str = "config"):
    """Create a new data collection service instance."""
    try:
        return DataCollectionService(config_path=config_path)
    except NameError:
        raise ImportError("DataCollectionService not available - check dependencies")


def get_service():
    """Get the global service instance, creating it if needed."""
    global _service_instance
    if _service_instance is None:
        _service_instance = create_service()
    return _service_instance


def start_service(config_path: str = "config") -> bool:
    """Start the global data collection service."""
    global _service_instance
    _service_instance = create_service(config_path)
    return _service_instance.start()


def stop_service():
    """Stop the global data collection service."""
    global _service_instance
    if _service_instance:
        _service_instance.shutdown()
        _service_instance = None


__all__ = [
    # Main service functions
    "create_service",
    "get_service", 
    "start_service",
    "stop_service",
    
    # Core always-available components
    "get_settings",
    "Settings",
    "SQLiteStore",
    "FileStore", 
    "CacheStore",
    "setup_logging",
    "get_logger",
    "HealthChecker",
    
    # Main service (if available)
    "DataCollectionService",
    "main",
    
    # Data collection components (if available)
    "ConnectionPool",
    "SnowflakeClient", 
    "UsageCollector",
    "QueryMetricsCollector", 
    "WarehouseMetricsCollector",
    "UserActivityCollector",
    "CostCollector",
    
    # Scheduling components (if available)
    "CollectionScheduler",
    "JobQueue",
    "RetryHandler",
    "StatusMonitor",
    
    # Validation components (if available)
    "SchemaValidator",
    "DataQualityChecker",
    "AnomalyDetector",
    "DataLineageTracker",
    
    # LLM components (if available)
    "LLMService",
    "get_llm_service",
    "create_llm_service", 
    "reset_llm_service",
    "OpenAIClient",
    "AzureOpenAIClient",
    "QueryInterface",
    "InsightGenerator",
]
