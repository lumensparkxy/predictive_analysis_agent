"""
Logging configuration and setup for Snowflake Analytics Agent.

Provides centralized logging configuration with file rotation,
structured logging, and performance monitoring.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = "logs/application.log",
    max_bytes: int = 10_000_000,  # 10MB
    backup_count: int = 5,
    format_string: Optional[str] = None
) -> None:
    """
    Setup comprehensive logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None for console only)
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        format_string: Custom log format string
    """
    
    # Ensure logs directory exists
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Define log format
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)-20s | "
            "%(funcName)-15s | %(lineno)-4d | %(message)s"
        )
    
    # Create formatter
    formatter = logging.Formatter(
        fmt=format_string,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation (if log_file is specified)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels for third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("snowflake.connector").setLevel(logging.WARNING)
    logging.getLogger("pandas").setLevel(logging.WARNING)
    
    # Log the initialization
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)


class PerformanceLogger:
    """Context manager for performance logging."""
    
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        """Initialize performance logger."""
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
    
    def __enter__(self):
        """Start timing the operation."""
        import time
        self.start_time = time.time()
        self.logger.debug(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log the operation completion time."""
        import time
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.info(f"Completed operation: {self.operation_name} in {duration:.3f}s")
        else:
            self.logger.error(f"Failed operation: {self.operation_name} after {duration:.3f}s - {exc_val}")


def log_function_calls(logger: Optional[logging.Logger] = None):
    """Decorator to log function entry and exit."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_logger = logger or logging.getLogger(func.__module__)
            func_logger.debug(f"Entering function: {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                func_logger.debug(f"Exiting function: {func.__name__}")
                return result
            except Exception as e:
                func_logger.error(f"Function {func.__name__} failed: {e}")
                raise
        return wrapper
    return decorator


def log_dataframe_info(df, name: str = "DataFrame", logger: Optional[logging.Logger] = None):
    """Log DataFrame information for debugging."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.debug(f"{name} info: shape={df.shape}, memory={df.memory_usage(deep=True).sum()/1024/1024:.1f}MB")
    logger.debug(f"{name} columns: {list(df.columns)}")
    logger.debug(f"{name} dtypes: {dict(df.dtypes)}")


if __name__ == "__main__":
    # Test logging setup
    setup_logging(log_level="DEBUG", log_file="test_logs/test.log")
    
    logger = get_logger(__name__)
    
    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test performance logger
    with PerformanceLogger("test_operation", logger):
        import time
        time.sleep(0.1)  # Simulate work
    
    # Test function decorator
    @log_function_calls(logger)
    def test_function(x, y):
        return x + y
    
    result = test_function(1, 2)
    logger.info(f"Function result: {result}")
    
    print("✅ Logging test completed successfully")
    
    # Cleanup test logs
    import shutil
    shutil.rmtree("test_logs", ignore_errors=True)
