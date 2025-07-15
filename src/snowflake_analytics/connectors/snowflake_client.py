"""
Snowflake Client - Main connection manager with secure credential handling and retry logic.

This module provides the primary interface for connecting to Snowflake with automatic
connection management, credential security, and robust error handling. Supports both
password and JWT (private key) authentication methods.
"""

import time
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import snowflake.connector
from snowflake.connector import SnowflakeConnection
from snowflake.connector.errors import (
    DatabaseError, 
    OperationalError, 
    ProgrammingError,
    InterfaceError
)

# Import for JWT key handling
try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

from ..config.settings import SnowflakeSettings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SnowflakeClient:
    """
    Primary Snowflake connection manager with retry logic and secure credential handling.
    
    Features:
    - Secure credential management from configuration
    - Support for both password and JWT (private key) authentication
    - Exponential backoff retry logic
    - Connection timeout handling
    - Query execution with automatic retries
    - Connection health monitoring
    """
    
    def __init__(self, settings: Optional[SnowflakeSettings] = None):
        """
        Initialize Snowflake client with configuration.
        
        Args:
            settings: Snowflake configuration settings. If None, loads from default config.
        """
        self.settings = settings or SnowflakeSettings()
        self._connection: Optional[SnowflakeConnection] = None
        self._connection_params = self._build_connection_params()
        self._private_key = None
        
        # Connection status tracking
        self.is_in_use = False
        self.is_healthy = True
        
        # Load private key if using JWT authentication
        if self._is_jwt_auth():
            self._load_private_key()
        
    def _is_jwt_auth(self) -> bool:
        """Check if JWT authentication is configured."""
        return (
            self.settings.authenticator and 
            self.settings.authenticator.upper() == 'SNOWFLAKE_JWT' and
            self.settings.private_key_path
        )
        
    def _load_private_key(self):
        """Load and parse the private key for JWT authentication."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError(
                "cryptography library is required for JWT authentication. "
                "Install it with: pip install cryptography"
            )
            
        if not self.settings.private_key_path:
            raise ValueError("private_key_path is required for JWT authentication")
            
        key_path = Path(self.settings.private_key_path)
        if not key_path.exists():
            raise FileNotFoundError(f"Private key file not found: {key_path}")
            
        try:
            with open(key_path, 'rb') as key_file:
                private_key_data = key_file.read()
            
            # Load the private key with optional passphrase
            passphrase = None
            if self.settings.private_key_passphrase:
                passphrase = self.settings.private_key_passphrase.encode()
                
            private_key_obj = load_pem_private_key(
                private_key_data,
                password=passphrase
            )
            
            # Serialize to DER format for Snowflake
            self._private_key = private_key_obj.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            logger.info("Private key loaded successfully for JWT authentication")
            
        except Exception as e:
            raise ValueError(f"Failed to load private key: {e}")
    
    def _build_connection_params(self) -> Dict[str, Any]:
        """Build connection parameters from settings."""
        params = {
            'account': self.settings.account,
            'user': self.settings.user,
            'warehouse': self.settings.warehouse,
            'database': self.settings.database,
            'schema': self.settings.schema,
            'client_session_keep_alive': True,
            'network_timeout': self.settings.network_timeout,
            'login_timeout': self.settings.login_timeout,
            'ocsp_response_cache_filename': None,  # Disable for better performance
        }
        
        # Add role if specified
        if self.settings.role:
            params['role'] = self.settings.role
            
        # Add authentication method
        if self._is_jwt_auth():
            params['authenticator'] = 'SNOWFLAKE_JWT'
            # Note: private_key will be added in connect() method after loading
        else:
            # Use password authentication
            if self.settings.password:
                params['password'] = self.settings.password
            else:
                logger.warning("No password provided and JWT auth not configured")
        
        return params
    
    def connect(self, retry_attempts: int = 3) -> SnowflakeConnection:
        """
        Establish connection to Snowflake with retry logic.
        
        Args:
            retry_attempts: Number of retry attempts on connection failure
            
        Returns:
            Active Snowflake connection
            
        Raises:
            ConnectionError: If connection fails after all retry attempts
        """
        last_error = None
        
        for attempt in range(retry_attempts):
            try:
                logger.info(f"Attempting Snowflake connection (attempt {attempt + 1}/{retry_attempts})")
                
                # Add private key for JWT authentication
                connection_params = self._connection_params.copy()
                if self._is_jwt_auth() and self._private_key:
                    connection_params['private_key'] = self._private_key
                
                self._connection = snowflake.connector.connect(**connection_params)
                
                logger.info("Successfully connected to Snowflake")
                return self._connection
                
            except (DatabaseError, OperationalError, InterfaceError) as e:
                last_error = e
                if attempt < retry_attempts - 1:
                    wait_time = (2 ** attempt) * self.settings.retry_delay_base
                    logger.warning(
                        f"Connection attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"All connection attempts failed. Last error: {e}")
        
        raise ConnectionError(f"Failed to connect to Snowflake after {retry_attempts} attempts: {last_error}")
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for getting a connection with automatic cleanup.
        
        Yields:
            SnowflakeConnection: Active connection
        """
        connection = None
        try:
            if not self._connection or self._connection.is_closed():
                connection = self.connect()
            else:
                connection = self._connection
                
            yield connection
            
        except Exception as e:
            logger.error(f"Error during connection usage: {e}")
            raise
        finally:
            # Connection is kept alive for reuse, not closed here
            pass
    
    def execute_query(
        self, 
        query: str, 
        params: Optional[Dict[str, Any]] = None,
        fetch_results: bool = True,
        retry_attempts: int = 3
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Execute SQL query with retry logic and error handling.
        
        Args:
            query: SQL query to execute
            params: Query parameters for parameterized queries
            fetch_results: Whether to fetch and return results
            retry_attempts: Number of retry attempts on query failure
            
        Returns:
            Query results as list of dictionaries, or None if fetch_results=False
            
        Raises:
            QueryExecutionError: If query fails after all retry attempts
        """
        last_error = None
        
        for attempt in range(retry_attempts):
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    
                    try:
                        # Execute query with optional parameters
                        if params:
                            cursor.execute(query, params)
                        else:
                            cursor.execute(query)
                        
                        if fetch_results:
                            # Fetch results and convert to list of dictionaries
                            columns = [desc[0] for desc in cursor.description]
                            results = []
                            
                            while True:
                                batch = cursor.fetchmany(size=10000)  # Fetch in batches
                                if not batch:
                                    break
                                    
                                for row in batch:
                                    results.append(dict(zip(columns, row)))
                            
                            logger.debug(f"Query executed successfully, returned {len(results)} rows")
                            return results
                        else:
                            logger.debug("Query executed successfully (no results fetched)")
                            return None
                            
                    finally:
                        cursor.close()
                        
            except (DatabaseError, OperationalError, ProgrammingError) as e:
                last_error = e
                if attempt < retry_attempts - 1:
                    wait_time = (2 ** attempt) * self.settings.retry_delay_base
                    logger.warning(
                        f"Query execution attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
                    # Force reconnection on next attempt
                    self._connection = None
                else:
                    logger.error(f"Query failed after {retry_attempts} attempts: {e}")
        
        raise QueryExecutionError(f"Query execution failed after {retry_attempts} attempts: {last_error}")
    
    def test_connection(self) -> bool:
        """
        Test connection health by executing a simple query.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            result = self.execute_query("SELECT 1 as test", retry_attempts=1)
            return result is not None and len(result) == 1 and result[0]['TEST'] == 1
        except Exception as e:
            logger.warning(f"Connection health check failed: {e}")
            return False
    
    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get information about the current connection.
        
        Returns:
            Dictionary with connection information
        """
        try:
            with self.get_connection() as conn:
                return {
                    'account': self.settings.account,
                    'user': self.settings.user,
                    'warehouse': self.settings.warehouse,
                    'database': self.settings.database,
                    'schema': self.settings.schema,
                    'role': self.settings.role,
                    'session_id': conn.session_id,
                    'is_closed': conn.is_closed(),
                    'autocommit': conn.autocommit
                }
        except Exception as e:
            logger.error(f"Failed to get connection info: {e}")
            return {}
    
    def close(self):
        """Close the connection if open."""
        if self._connection and not self._connection.is_closed():
            try:
                self._connection.close()
                logger.info("Snowflake connection closed")
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
        self._connection = None


class QueryExecutionError(Exception):
    """Raised when query execution fails after all retry attempts."""
    pass
