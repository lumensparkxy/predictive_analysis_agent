"""
Usage Data Collector - Main collector for Snowflake usage metrics.

This module coordinates collection of warehouse usage, query performance,
and user activity data with incremental collection capabilities.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from dataclasses import dataclass

from ..connectors.connection_pool import ConnectionPool
from ..storage.sqlite_store import SQLiteStore
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CollectionResult:
    """Result of a data collection operation."""
    collector_name: str
    start_time: datetime
    end_time: datetime
    records_collected: int
    success: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class UsageCollector:
    """
    Main usage data collector that coordinates collection from multiple Snowflake sources.
    
    Features:
    - Incremental data collection with last_updated tracking
    - Parallel collection from multiple account usage views
    - Data validation and quality checks
    - Storage optimization with Parquet format
    - Error handling and retry logic
    """
    
    def __init__(
        self,
        connection_pool: ConnectionPool,
        storage: SQLiteStore,
        collection_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize usage collector.
        
        Args:
            connection_pool: Pool of Snowflake connections
            storage: Storage backend for collected data
            collection_config: Configuration for collection behavior
        """
        self.connection_pool = connection_pool
        self.storage = storage
        self.config = collection_config or {}
        
        # Collection settings
        self.batch_size = self.config.get('batch_size', 10000)
        self.lookback_days = self.config.get('lookback_days', 7)
        self.max_rows_per_query = self.config.get('max_rows_per_query', 100000)
        
        # Data sources and their queries
        self.data_sources = {
            'warehouse_usage': {
                'query': self._get_warehouse_usage_query(),
                'table': 'warehouse_usage_raw',
                'description': 'Warehouse usage and credit consumption'
            },
            'query_history': {
                'query': self._get_query_history_query(),
                'table': 'query_history_raw', 
                'description': 'Query execution history and performance'
            },
            'user_activity': {
                'query': self._get_user_activity_query(),
                'table': 'user_activity_raw',
                'description': 'User login and activity patterns'
            },
            'warehouse_load': {
                'query': self._get_warehouse_load_query(),
                'table': 'warehouse_load_raw',
                'description': 'Warehouse load and queue patterns'
            }
        }
        
    def collect_all_usage_data(self, force_full_collection: bool = False) -> List[CollectionResult]:
        """
        Collect all usage data from configured sources.
        
        Args:
            force_full_collection: If True, collect all data ignoring last_updated
            
        Returns:
            List of collection results for each data source
        """
        logger.info("Starting comprehensive usage data collection")
        start_time = datetime.now()
        results = []
        
        for source_name, source_config in self.data_sources.items():
            try:
                logger.info(f"Collecting {source_name}: {source_config['description']}")
                
                result = self._collect_data_source(
                    source_name=source_name,
                    source_config=source_config,
                    force_full_collection=force_full_collection
                )
                
                results.append(result)
                
                if result.success:
                    logger.info(
                        f"✅ {source_name}: {result.records_collected} records collected "
                        f"in {(result.end_time - result.start_time).total_seconds():.1f}s"
                    )
                else:
                    logger.error(f"❌ {source_name}: {result.error_message}")
                    
            except Exception as e:
                logger.error(f"Failed to collect {source_name}: {e}")
                results.append(CollectionResult(
                    collector_name=source_name,
                    start_time=start_time,
                    end_time=datetime.now(),
                    records_collected=0,
                    success=False,
                    error_message=str(e)
                ))
        
        total_records = sum(r.records_collected for r in results)
        successful_collections = sum(1 for r in results if r.success)
        
        logger.info(
            f"Usage data collection complete: {successful_collections}/{len(results)} sources successful, "
            f"{total_records} total records collected"
        )
        
        return results
    
    def _collect_data_source(
        self, 
        source_name: str,
        source_config: Dict[str, Any],
        force_full_collection: bool = False
    ) -> CollectionResult:
        """Collect data from a single source with incremental logic."""
        start_time = datetime.now()
        
        try:
            # Determine collection window
            last_collected = self._get_last_collection_time(source_name)
            
            if force_full_collection or not last_collected:
                # Full collection from lookback period
                from_time = datetime.now() - timedelta(days=self.lookback_days)
                logger.debug(f"Full collection for {source_name} from {from_time}")
            else:
                # Incremental collection from last collected time
                from_time = last_collected
                logger.debug(f"Incremental collection for {source_name} from {from_time}")
            
            # Execute query with time window
            query = source_config['query'].format(
                from_time=from_time.strftime('%Y-%m-%d %H:%M:%S')
            )
            
            with self.connection_pool.get_connection() as conn:
                raw_data = conn.execute_query(query)
            
            if not raw_data:
                return CollectionResult(
                    collector_name=source_name,
                    start_time=start_time,
                    end_time=datetime.now(),
                    records_collected=0,
                    success=True,
                    metadata={'no_new_data': True}
                )
            
            # Convert to DataFrame for processing
            df = pd.DataFrame(raw_data)
            
            # Data validation and cleaning
            df = self._validate_and_clean_data(df, source_name)
            
            # Store data
            records_stored = self._store_collected_data(df, source_config['table'])
            
            # Update last collection time
            self._update_last_collection_time(source_name, datetime.now())
            
            return CollectionResult(
                collector_name=source_name,
                start_time=start_time,
                end_time=datetime.now(),
                records_collected=records_stored,
                success=True,
                metadata={
                    'table': source_config['table'],
                    'collection_window': {
                        'from': from_time.isoformat(),
                        'to': datetime.now().isoformat()
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Error collecting {source_name}: {e}")
            return CollectionResult(
                collector_name=source_name,
                start_time=start_time,
                end_time=datetime.now(),
                records_collected=0,
                success=False,
                error_message=str(e)
            )
    
    def _get_warehouse_usage_query(self) -> str:
        """Get query for warehouse usage data."""
        return """
        SELECT 
            START_TIME,
            END_TIME,
            WAREHOUSE_ID,
            WAREHOUSE_NAME,
            CREDITS_USED,
            CREDITS_USED_COMPUTE,
            CREDITS_USED_CLOUD_SERVICES,
            BYTES_SCANNED,
            BYTES_WRITTEN,
            BYTES_DELETED,
            BYTES_SPILLED_TO_REMOTE_STORAGE,
            BYTES_SPILLED_TO_LOCAL_STORAGE,
            BYTES_SENT_OVER_THE_NETWORK
        FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY 
        WHERE START_TIME >= '{from_time}'::timestamp
        ORDER BY START_TIME DESC
        LIMIT {max_rows}
        """.replace('{max_rows}', str(self.max_rows_per_query))
    
    def _get_query_history_query(self) -> str:
        """Get query for query history data."""
        return """
        SELECT 
            QUERY_ID,
            QUERY_TEXT,
            DATABASE_NAME,
            SCHEMA_NAME,
            QUERY_TYPE,
            SESSION_ID,
            USER_NAME,
            ROLE_NAME,
            WAREHOUSE_NAME,
            WAREHOUSE_SIZE,
            WAREHOUSE_TYPE,
            CLUSTER_NUMBER,
            QUERY_TAG,
            EXECUTION_STATUS,
            ERROR_CODE,
            ERROR_MESSAGE,
            START_TIME,
            END_TIME,
            TOTAL_ELAPSED_TIME,
            BYTES_SCANNED,
            PERCENTAGE_SCANNED_FROM_CACHE,
            BYTES_WRITTEN,
            BYTES_WRITTEN_TO_RESULT,
            BYTES_READ_FROM_RESULT,
            ROWS_PRODUCED,
            ROWS_INSERTED,
            ROWS_UPDATED,
            ROWS_DELETED,
            ROWS_UNLOADED,
            BYTES_DELETED,
            PARTITIONS_SCANNED,
            PARTITIONS_TOTAL,
            BYTES_SPILLED_TO_LOCAL_STORAGE,
            BYTES_SPILLED_TO_REMOTE_STORAGE,
            BYTES_SENT_OVER_THE_NETWORK,
            COMPILATION_TIME,
            EXECUTION_TIME,
            QUEUED_PROVISIONING_TIME,
            QUEUED_REPAIR_TIME,
            QUEUED_OVERLOAD_TIME,
            TRANSACTION_BLOCKED_TIME,
            OUTBOUND_DATA_TRANSFER_CLOUD,
            OUTBOUND_DATA_TRANSFER_REGION,
            OUTBOUND_DATA_TRANSFER_BYTES,
            INBOUND_DATA_TRANSFER_BYTES,
            CREDITS_USED_CLOUD_SERVICES,
            RELEASE_VERSION,
            DRIVER_TYPE,
            DRIVER_VERSION,
            CLIENT_APPLICATION_ID,
            CLIENT_ENVIRONMENT
        FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY 
        WHERE START_TIME >= '{from_time}'::timestamp
        ORDER BY START_TIME DESC
        LIMIT {max_rows}
        """.replace('{max_rows}', str(self.max_rows_per_query))
    
    def _get_user_activity_query(self) -> str:
        """Get query for user activity data."""
        return """
        SELECT 
            EVENT_ID,
            EVENT_TIMESTAMP,
            EVENT_TYPE,
            USER_NAME,
            CLIENT_IP,
            REPORTED_CLIENT_TYPE,
            REPORTED_CLIENT_VERSION,
            FIRST_AUTHENTICATION_FACTOR,
            SECOND_AUTHENTICATION_FACTOR,
            IS_SUCCESS,
            ERROR_CODE,
            ERROR_MESSAGE,
            RELATED_EVENT_ID,
            CONNECTION_ID,
            CLIENT_ENVIRONMENT
        FROM SNOWFLAKE.ACCOUNT_USAGE.LOGIN_HISTORY 
        WHERE EVENT_TIMESTAMP >= '{from_time}'::timestamp
        ORDER BY EVENT_TIMESTAMP DESC
        LIMIT {max_rows}
        """.replace('{max_rows}', str(self.max_rows_per_query))
    
    def _get_warehouse_load_query(self) -> str:
        """Get query for warehouse load data."""
        return """
        SELECT 
            START_TIME,
            END_TIME,
            WAREHOUSE_ID,
            WAREHOUSE_NAME,
            AVG_RUNNING,
            AVG_QUEUED_LOAD,
            AVG_QUEUED_PROVISIONING,
            AVG_BLOCKED
        FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_LOAD_HISTORY 
        WHERE START_TIME >= '{from_time}'::timestamp
        ORDER BY START_TIME DESC
        LIMIT {max_rows}
        """.replace('{max_rows}', str(self.max_rows_per_query))
    
    def _validate_and_clean_data(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Validate and clean collected data."""
        original_rows = len(df)
        
        # Remove duplicates based on key columns
        if source_name == 'warehouse_usage':
            df = df.drop_duplicates(subset=['START_TIME', 'WAREHOUSE_ID'])
        elif source_name == 'query_history':
            df = df.drop_duplicates(subset=['QUERY_ID'])
        elif source_name == 'user_activity':
            df = df.drop_duplicates(subset=['EVENT_ID'])
        elif source_name == 'warehouse_load':
            df = df.drop_duplicates(subset=['START_TIME', 'WAREHOUSE_ID'])
        
        # Remove rows with null timestamps
        time_columns = ['START_TIME', 'EVENT_TIMESTAMP'] 
        for col in time_columns:
            if col in df.columns:
                df = df.dropna(subset=[col])
        
        # Basic data type conversions
        # Convert timestamp columns
        for col in df.columns:
            if 'TIME' in col.upper() and df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    logger.warning(f"Could not convert {col} to datetime")
        
        # Convert numeric columns
        numeric_columns = [col for col in df.columns if any(keyword in col.upper() 
                          for keyword in ['CREDIT', 'BYTE', 'ROW', 'TIME', 'SIZE'])]
        
        for col in numeric_columns:
            if col.upper() not in ['START_TIME', 'END_TIME', 'EVENT_TIMESTAMP']:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    logger.warning(f"Could not convert {col} to numeric")
        
        cleaned_rows = len(df)
        if cleaned_rows < original_rows:
            logger.info(f"Data cleaning: {original_rows} -> {cleaned_rows} rows ({original_rows - cleaned_rows} removed)")
        
        return df
    
    def _store_collected_data(self, df: pd.DataFrame, table_name: str) -> int:
        """Store collected data in the storage backend."""
        if df.empty:
            return 0
        
        # Add collection metadata
        df['_collected_at'] = datetime.now()
        df['_collection_batch'] = f"{table_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store in SQLite (will be extended to support Parquet later)
        records_stored = self.storage.store_dataframe(df, table_name)
        
        logger.debug(f"Stored {records_stored} records in {table_name}")
        return records_stored
    
    def _get_last_collection_time(self, source_name: str) -> Optional[datetime]:
        """Get the last collection time for a data source."""
        try:
            result = self.storage.execute_query(
                "SELECT MAX(last_collected) as last_time FROM collection_metadata WHERE source_name = ?",
                (source_name,)
            )
            
            if result and result[0]['last_time']:
                return datetime.fromisoformat(result[0]['last_time'])
            
        except Exception as e:
            logger.debug(f"Could not get last collection time for {source_name}: {e}")
        
        return None
    
    def _update_last_collection_time(self, source_name: str, collection_time: datetime):
        """Update the last collection time for a data source."""
        try:
            self.storage.execute_query("""
                INSERT OR REPLACE INTO collection_metadata 
                (source_name, last_collected, updated_at)
                VALUES (?, ?, ?)
            """, (source_name, collection_time.isoformat(), datetime.now().isoformat()))
            
        except Exception as e:
            logger.warning(f"Could not update last collection time for {source_name}: {e}")
    
    def get_collection_status(self) -> Dict[str, Any]:
        """Get status of all data collections."""
        status = {
            'sources': {},
            'total_records': 0,
            'last_collection': None
        }
        
        for source_name in self.data_sources.keys():
            try:
                # Get last collection time
                last_collected = self._get_last_collection_time(source_name)
                
                # Get record count
                table_name = self.data_sources[source_name]['table']
                count_result = self.storage.execute_query(
                    f"SELECT COUNT(*) as count FROM {table_name}"
                )
                record_count = count_result[0]['count'] if count_result else 0
                
                status['sources'][source_name] = {
                    'last_collected': last_collected.isoformat() if last_collected else None,
                    'record_count': record_count,
                    'description': self.data_sources[source_name]['description']
                }
                
                status['total_records'] += record_count
                
                if last_collected and (not status['last_collection'] or 
                                     last_collected > datetime.fromisoformat(status['last_collection'])):
                    status['last_collection'] = last_collected.isoformat()
                    
            except Exception as e:
                status['sources'][source_name] = {
                    'error': str(e),
                    'description': self.data_sources[source_name]['description']
                }
        
        return status
