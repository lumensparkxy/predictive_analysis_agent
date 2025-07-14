"""
Connection Pool - Manages multiple Snowflake connections for concurrent operations.

This module provides connection pooling to optimize performance and resource usage
when executing multiple concurrent Snowflake operations.
"""

import threading
import time
from typing import List, Optional, Dict, Any
from queue import Queue, Empty, Full
from contextlib import contextmanager

from .snowflake_client import SnowflakeClient, QueryExecutionError
from ..config.settings import SnowflakeSettings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ConnectionPool:
    """
    Connection pool for managing multiple Snowflake connections.
    
    Features:
    - Configurable pool size (default: 5 connections)
    - Connection lifecycle management
    - Health checking and automatic replacement of dead connections
    - Thread-safe connection borrowing and returning
    - Connection usage statistics and monitoring
    """
    
    def __init__(
        self, 
        settings: Optional[SnowflakeSettings] = None,
        max_connections: int = 5,
        min_connections: int = 1,
        connection_timeout: int = 30,
        health_check_interval: int = 300  # 5 minutes
    ):
        """
        Initialize connection pool.
        
        Args:
            settings: Snowflake configuration settings
            max_connections: Maximum number of connections in pool
            min_connections: Minimum number of connections to maintain
            connection_timeout: Timeout for borrowing connections (seconds)
            health_check_interval: Interval for health checking connections (seconds)
        """
        self.settings = settings or SnowflakeSettings()
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.connection_timeout = connection_timeout
        self.health_check_interval = health_check_interval
        
        # Connection pool and management
        self._pool: Queue[SnowflakeClient] = Queue(maxsize=max_connections)
        self._all_connections: List[SnowflakeClient] = []
        self._borrowed_connections: Dict[int, SnowflakeClient] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        self._health_check_lock = threading.Lock()
        
        # Statistics
        self._stats = {
            'total_created': 0,
            'total_borrowed': 0,
            'total_returned': 0,
            'total_health_checks': 0,
            'total_replacements': 0,
            'current_active': 0,
            'current_borrowed': 0
        }
        
        # Health checking
        self._last_health_check = 0
        self._health_check_thread: Optional[threading.Thread] = None
        self._shutdown_flag = threading.Event()
        
        # Initialize pool
        self._initialize_pool()
        self._start_health_check_thread()
    
    def _initialize_pool(self):
        """Initialize the connection pool with minimum connections."""
        logger.info(f"Initializing connection pool with {self.min_connections} initial connections")
        
        with self._lock:
            for i in range(self.min_connections):
                try:
                    client = self._create_connection()
                    self._pool.put(client)
                    self._all_connections.append(client)
                    self._stats['total_created'] += 1
                    logger.debug(f"Created initial connection {i + 1}/{self.min_connections}")
                except Exception as e:
                    logger.error(f"Failed to create initial connection {i + 1}: {e}")
            
            self._stats['current_active'] = self._pool.qsize()
            logger.info(f"Connection pool initialized with {self._stats['current_active']} connections")
    
    def _create_connection(self) -> SnowflakeClient:
        """Create a new Snowflake client connection."""
        client = SnowflakeClient(self.settings)
        client.connect()  # Establish connection immediately
        return client
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for borrowing and returning connections.
        
        Yields:
            SnowflakeClient: Connection from the pool
            
        Raises:
            PoolExhaustedException: If no connections available within timeout
        """
        connection = self.borrow_connection()
        try:
            yield connection
        finally:
            self.return_connection(connection)
    
    def borrow_connection(self) -> SnowflakeClient:
        """
        Borrow a connection from the pool.
        
        Returns:
            SnowflakeClient: Connection from the pool
            
        Raises:
            PoolExhaustedException: If no connections available within timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < self.connection_timeout:
            try:
                # Try to get existing connection from pool
                connection = self._pool.get_nowait()
                
                with self._lock:
                    connection_id = id(connection)
                    self._borrowed_connections[connection_id] = connection
                    self._stats['total_borrowed'] += 1
                    self._stats['current_borrowed'] += 1
                
                # Test connection health before returning
                if connection.test_connection():
                    logger.debug(f"Borrowed connection {connection_id}")
                    return connection
                else:
                    # Connection is unhealthy, replace it
                    logger.warning(f"Borrowed connection {connection_id} is unhealthy, replacing")
                    self._replace_connection(connection)
                    continue
                    
            except Empty:
                # Pool is empty, try to create new connection if under limit
                with self._lock:
                    if len(self._all_connections) < self.max_connections:
                        try:
                            new_connection = self._create_connection()
                            self._all_connections.append(new_connection)
                            self._stats['total_created'] += 1
                            
                            connection_id = id(new_connection)
                            self._borrowed_connections[connection_id] = new_connection
                            self._stats['total_borrowed'] += 1
                            self._stats['current_borrowed'] += 1
                            
                            logger.debug(f"Created and borrowed new connection {connection_id}")
                            return new_connection
                            
                        except Exception as e:
                            logger.error(f"Failed to create new connection: {e}")
            
            # Wait a bit before retrying
            time.sleep(0.1)
        
        raise PoolExhaustedException(
            f"No connections available within {self.connection_timeout} seconds. "
            f"Pool stats: {self.get_stats()}"
        )
    
    def return_connection(self, connection: SnowflakeClient):
        """
        Return a borrowed connection to the pool.
        
        Args:
            connection: Connection to return to the pool
        """
        connection_id = id(connection)
        
        with self._lock:
            if connection_id in self._borrowed_connections:
                del self._borrowed_connections[connection_id]
                self._stats['total_returned'] += 1
                self._stats['current_borrowed'] -= 1
                
                # Test connection health before returning to pool
                if connection.test_connection():
                    try:
                        self._pool.put_nowait(connection)
                        logger.debug(f"Returned connection {connection_id} to pool")
                    except Full:
                        # Pool is full, close this connection
                        logger.debug(f"Pool full, closing returned connection {connection_id}")
                        connection.close()
                        self._all_connections.remove(connection)
                else:
                    # Connection is unhealthy, replace it
                    logger.warning(f"Returned connection {connection_id} is unhealthy, replacing")
                    self._replace_connection(connection)
            else:
                logger.warning(f"Attempted to return unknown connection {connection_id}")
    
    def _replace_connection(self, old_connection: SnowflakeClient):
        """Replace an unhealthy connection with a new one."""
        with self._lock:
            try:
                # Close old connection
                old_connection.close()
                if old_connection in self._all_connections:
                    self._all_connections.remove(old_connection)
                
                # Create replacement if under minimum
                if len(self._all_connections) < self.min_connections:
                    new_connection = self._create_connection()
                    self._all_connections.append(new_connection)
                    self._pool.put(new_connection)
                    self._stats['total_replacements'] += 1
                    logger.info("Replaced unhealthy connection with new one")
                
            except Exception as e:
                logger.error(f"Failed to replace unhealthy connection: {e}")
    
    def _start_health_check_thread(self):
        """Start background health checking thread."""
        def health_check_worker():
            while not self._shutdown_flag.is_set():
                try:
                    self._perform_health_checks()
                    time.sleep(self.health_check_interval)
                except Exception as e:
                    logger.error(f"Error in health check thread: {e}")
        
        self._health_check_thread = threading.Thread(
            target=health_check_worker,
            name="ConnectionPoolHealthChecker",
            daemon=True
        )
        self._health_check_thread.start()
        logger.info("Started connection pool health check thread")
    
    def _perform_health_checks(self):
        """Perform health checks on all connections in the pool."""
        if not self._health_check_lock.acquire(blocking=False):
            return  # Health check already in progress
        
        try:
            current_time = time.time()
            if current_time - self._last_health_check < self.health_check_interval:
                return
            
            logger.debug("Performing connection pool health checks")
            unhealthy_connections = []
            
            # Check connections currently in pool
            temp_connections = []
            while not self._pool.empty():
                try:
                    connection = self._pool.get_nowait()
                    if connection.test_connection():
                        temp_connections.append(connection)
                    else:
                        unhealthy_connections.append(connection)
                except Empty:
                    break
            
            # Return healthy connections to pool
            for connection in temp_connections:
                self._pool.put(connection)
            
            # Replace unhealthy connections
            for connection in unhealthy_connections:
                self._replace_connection(connection)
            
            self._stats['total_health_checks'] += 1
            self._last_health_check = current_time
            
            if unhealthy_connections:
                logger.info(f"Health check completed. Replaced {len(unhealthy_connections)} unhealthy connections")
            else:
                logger.debug("Health check completed. All connections healthy")
                
        finally:
            self._health_check_lock.release()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get pool statistics.
        
        Returns:
            Dictionary containing pool statistics
        """
        with self._lock:
            stats = self._stats.copy()
            stats.update({
                'current_active': self._pool.qsize(),
                'current_borrowed': len(self._borrowed_connections),
                'total_connections': len(self._all_connections),
                'max_connections': self.max_connections,
                'min_connections': self.min_connections
            })
            return stats
    
    def shutdown(self):
        """Shutdown the connection pool and close all connections."""
        logger.info("Shutting down connection pool")
        
        # Signal health check thread to stop
        self._shutdown_flag.set()
        
        # Wait for health check thread to finish
        if self._health_check_thread and self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=5)
        
        with self._lock:
            # Close all connections
            for connection in self._all_connections:
                try:
                    connection.close()
                except Exception as e:
                    logger.warning(f"Error closing connection during shutdown: {e}")
            
            # Clear pool and tracking
            while not self._pool.empty():
                try:
                    self._pool.get_nowait()
                except Empty:
                    break
            
            self._all_connections.clear()
            self._borrowed_connections.clear()
        
        logger.info("Connection pool shutdown complete")


class PoolExhaustedException(Exception):
    """Raised when the connection pool is exhausted and no connections are available."""
    pass
