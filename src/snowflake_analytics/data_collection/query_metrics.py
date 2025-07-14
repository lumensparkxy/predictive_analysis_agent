"""
Query Metrics Collector - Specialized collector for query performance data.

This module provides detailed collection and analysis of Snowflake query execution
metrics including performance patterns, resource usage, and optimization opportunities.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from ..connectors.connection_pool import ConnectionPool
from ..storage.sqlite_store import SQLiteStore
from ..utils.logger import get_logger

logger = get_logger(__name__)


class QueryMetricsCollector:
    """
    Specialized collector for query performance metrics and analysis.
    
    Features:
    - Detailed query execution metrics collection
    - Performance pattern analysis
    - Query optimization recommendations
    - Resource usage tracking
    - Cost per query calculations
    """
    
    def __init__(
        self,
        connection_pool: ConnectionPool,
        storage: SQLiteStore,
        collection_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize query metrics collector."""
        self.connection_pool = connection_pool
        self.storage = storage
        self.config = collection_config or {}
        
        # Collection settings
        self.batch_size = self.config.get('batch_size', 5000)
        self.lookback_hours = self.config.get('lookback_hours', 24)
        self.performance_threshold_ms = self.config.get('performance_threshold_ms', 60000)  # 1 minute
        
        # Query categories for analysis
        self.query_categories = {
            'SELECT': 'read_queries',
            'INSERT': 'write_queries', 
            'UPDATE': 'write_queries',
            'DELETE': 'write_queries',
            'CREATE': 'ddl_queries',
            'ALTER': 'ddl_queries',
            'DROP': 'ddl_queries',
            'COPY': 'data_loading',
            'UNLOAD': 'data_export'
        }
    
    def collect_query_metrics(
        self, 
        hours_back: Optional[int] = None,
        include_query_text: bool = False
    ) -> Dict[str, Any]:
        """
        Collect comprehensive query metrics.
        
        Args:
            hours_back: Hours of history to collect (default: from config)
            include_query_text: Whether to include full query text
            
        Returns:
            Dictionary with collection results and metrics
        """
        logger.info("Starting query metrics collection")
        start_time = datetime.now()
        
        hours_back = hours_back or self.lookback_hours
        from_time = datetime.now() - timedelta(hours=hours_back)
        
        try:
            # Collect raw query data
            raw_queries = self._collect_raw_query_data(from_time, include_query_text)
            
            if not raw_queries:
                return {
                    'success': True,
                    'records_collected': 0,
                    'message': 'No new query data available'
                }
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(raw_queries)
            
            # Clean and validate data
            df = self._clean_query_data(df)
            
            # Calculate derived metrics
            df = self._calculate_derived_metrics(df)
            
            # Categorize queries
            df = self._categorize_queries(df)
            
            # Store processed data
            records_stored = self._store_query_metrics(df)
            
            # Generate performance insights
            insights = self._generate_performance_insights(df)
            
            end_time = datetime.now()
            collection_time = (end_time - start_time).total_seconds()
            
            logger.info(
                f"Query metrics collection complete: {records_stored} records in {collection_time:.1f}s"
            )
            
            return {
                'success': True,
                'records_collected': records_stored,
                'collection_time_seconds': collection_time,
                'time_window': {
                    'from': from_time.isoformat(),
                    'to': end_time.isoformat()
                },
                'insights': insights
            }
            
        except Exception as e:
            logger.error(f"Error in query metrics collection: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'records_collected': 0
            }
    
    def _collect_raw_query_data(self, from_time: datetime, include_query_text: bool) -> List[Dict]:
        """Collect raw query data from Snowflake."""
        # Build query based on requirements
        query_text_field = "QUERY_TEXT," if include_query_text else "LEFT(QUERY_TEXT, 200) as QUERY_TEXT_PREVIEW,"
        
        query = f"""
        SELECT 
            QUERY_ID,
            {query_text_field}
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
            CREDITS_USED_CLOUD_SERVICES
        FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY 
        WHERE START_TIME >= '{from_time.strftime('%Y-%m-%d %H:%M:%S')}'::timestamp
        AND EXECUTION_STATUS IN ('SUCCESS', 'FAIL', 'CANCELLED')
        ORDER BY START_TIME DESC
        LIMIT {self.batch_size * 10}
        """
        
        with self.connection_pool.get_connection() as conn:
            return conn.execute_query(query)
    
    def _clean_query_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate query data."""
        original_rows = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['QUERY_ID'])
        
        # Handle missing values
        numeric_columns = [
            'TOTAL_ELAPSED_TIME', 'BYTES_SCANNED', 'BYTES_WRITTEN', 'ROWS_PRODUCED',
            'COMPILATION_TIME', 'EXECUTION_TIME', 'QUEUED_PROVISIONING_TIME',
            'QUEUED_REPAIR_TIME', 'QUEUED_OVERLOAD_TIME', 'TRANSACTION_BLOCKED_TIME'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Convert timestamps
        for col in ['START_TIME', 'END_TIME']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Remove invalid rows
        df = df.dropna(subset=['QUERY_ID', 'START_TIME'])
        
        cleaned_rows = len(df)
        if cleaned_rows < original_rows:
            logger.debug(f"Query data cleaned: {original_rows} -> {cleaned_rows} rows")
        
        return df
    
    def _calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived performance metrics."""
        # Calculate query duration if not provided
        if 'END_TIME' in df.columns and 'START_TIME' in df.columns:
            df['QUERY_DURATION_MS'] = (
                df['END_TIME'] - df['START_TIME']
            ).dt.total_seconds() * 1000
        else:
            df['QUERY_DURATION_MS'] = df.get('TOTAL_ELAPSED_TIME', 0)
        
        # Calculate efficiency metrics
        df['BYTES_PER_SECOND'] = np.where(
            df['QUERY_DURATION_MS'] > 0,
            (df.get('BYTES_SCANNED', 0) * 1000) / df['QUERY_DURATION_MS'],
            0
        )
        
        df['ROWS_PER_SECOND'] = np.where(
            df['QUERY_DURATION_MS'] > 0,
            (df.get('ROWS_PRODUCED', 0) * 1000) / df['QUERY_DURATION_MS'],
            0
        )
        
        # Calculate cache efficiency
        df['CACHE_HIT_RATIO'] = df.get('PERCENTAGE_SCANNED_FROM_CACHE', 0) / 100
        
        # Calculate spillage ratio
        total_bytes_spilled = (
            df.get('BYTES_SPILLED_TO_LOCAL_STORAGE', 0) + 
            df.get('BYTES_SPILLED_TO_REMOTE_STORAGE', 0)
        )
        df['SPILLAGE_RATIO'] = np.where(
            df.get('BYTES_SCANNED', 0) > 0,
            total_bytes_spilled / df['BYTES_SCANNED'],
            0
        )
        
        # Calculate wait time ratio
        total_wait_time = (
            df.get('QUEUED_PROVISIONING_TIME', 0) +
            df.get('QUEUED_REPAIR_TIME', 0) +
            df.get('QUEUED_OVERLOAD_TIME', 0) +
            df.get('TRANSACTION_BLOCKED_TIME', 0)
        )
        df['WAIT_TIME_RATIO'] = np.where(
            df['QUERY_DURATION_MS'] > 0,
            total_wait_time / df['QUERY_DURATION_MS'],
            0
        )
        
        # Performance classification
        df['PERFORMANCE_CLASS'] = pd.cut(
            df['QUERY_DURATION_MS'],
            bins=[0, 1000, 10000, 60000, 300000, float('inf')],
            labels=['Fast', 'Normal', 'Slow', 'Very Slow', 'Extremely Slow']
        )
        
        return df
    
    def _categorize_queries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize queries by type and purpose."""
        # Extract query type from query text or use provided query_type
        if 'QUERY_TYPE' in df.columns:
            df['QUERY_CATEGORY'] = df['QUERY_TYPE'].map(
                lambda x: self.query_categories.get(x, 'other') if x else 'unknown'
            )
        else:
            df['QUERY_CATEGORY'] = 'unknown'
        
        # Identify automated vs manual queries
        df['IS_AUTOMATED'] = df.get('QUERY_TAG', '').str.contains(
            'automated|scheduled|pipeline|etl|job',
            case=False,
            na=False
        )
        
        # Identify resource-intensive queries
        df['IS_RESOURCE_INTENSIVE'] = (
            (df.get('BYTES_SCANNED', 0) > 1e9) |  # > 1GB scanned
            (df['QUERY_DURATION_MS'] > self.performance_threshold_ms) |
            (df.get('CREDITS_USED_CLOUD_SERVICES', 0) > 1.0)
        )
        
        return df
    
    def _store_query_metrics(self, df: pd.DataFrame) -> int:
        """Store processed query metrics."""
        if df.empty:
            return 0
        
        # Add metadata
        df['_collected_at'] = datetime.now()
        df['_processing_version'] = '1.0'
        
        # Store in query_metrics table
        return self.storage.store_dataframe(df, 'query_metrics')
    
    def _generate_performance_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate performance insights from collected data."""
        if df.empty:
            return {}
        
        insights = {
            'summary': self._get_performance_summary(df),
            'top_issues': self._identify_performance_issues(df),
            'optimization_opportunities': self._find_optimization_opportunities(df),
            'resource_usage_patterns': self._analyze_resource_patterns(df)
        }
        
        return insights
    
    def _get_performance_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get high-level performance summary."""
        return {
            'total_queries': len(df),
            'successful_queries': len(df[df['EXECUTION_STATUS'] == 'SUCCESS']),
            'failed_queries': len(df[df['EXECUTION_STATUS'] == 'FAIL']),
            'cancelled_queries': len(df[df['EXECUTION_STATUS'] == 'CANCELLED']),
            'avg_duration_ms': float(df['QUERY_DURATION_MS'].mean()),
            'median_duration_ms': float(df['QUERY_DURATION_MS'].median()),
            'total_bytes_scanned': int(df.get('BYTES_SCANNED', 0).sum()),
            'total_rows_produced': int(df.get('ROWS_PRODUCED', 0).sum()),
            'avg_cache_hit_ratio': float(df['CACHE_HIT_RATIO'].mean()),
            'slow_queries_count': len(df[df['QUERY_DURATION_MS'] > self.performance_threshold_ms])
        }
    
    def _identify_performance_issues(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify specific performance issues."""
        issues = []
        
        # Slow queries
        slow_queries = df[df['QUERY_DURATION_MS'] > self.performance_threshold_ms]
        if not slow_queries.empty:
            issues.append({
                'type': 'slow_queries',
                'count': len(slow_queries),
                'description': f'{len(slow_queries)} queries took longer than {self.performance_threshold_ms/1000:.0f} seconds',
                'avg_duration_ms': float(slow_queries['QUERY_DURATION_MS'].mean()),
                'sample_query_ids': slow_queries['QUERY_ID'].head(5).tolist()
            })
        
        # High spillage
        high_spillage = df[df['SPILLAGE_RATIO'] > 0.1]  # > 10% spillage
        if not high_spillage.empty:
            issues.append({
                'type': 'high_spillage',
                'count': len(high_spillage),
                'description': f'{len(high_spillage)} queries had high memory spillage',
                'avg_spillage_ratio': float(high_spillage['SPILLAGE_RATIO'].mean()),
                'sample_query_ids': high_spillage['QUERY_ID'].head(5).tolist()
            })
        
        # Low cache utilization
        low_cache = df[df['CACHE_HIT_RATIO'] < 0.1]  # < 10% cache hit
        if not low_cache.empty:
            issues.append({
                'type': 'low_cache_utilization',
                'count': len(low_cache),
                'description': f'{len(low_cache)} queries had low cache utilization',
                'avg_cache_hit_ratio': float(low_cache['CACHE_HIT_RATIO'].mean()),
                'sample_query_ids': low_cache['QUERY_ID'].head(5).tolist()
            })
        
        return issues
    
    def _find_optimization_opportunities(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find query optimization opportunities."""
        opportunities = []
        
        # Repeated similar queries
        if 'QUERY_TEXT_PREVIEW' in df.columns:
            query_patterns = df.groupby('QUERY_TEXT_PREVIEW').size()
            repeated_patterns = query_patterns[query_patterns > 5]  # Repeated > 5 times
            
            if not repeated_patterns.empty:
                opportunities.append({
                    'type': 'repeated_queries',
                    'description': f'{len(repeated_patterns)} query patterns executed multiple times',
                    'patterns': repeated_patterns.head(10).to_dict(),
                    'recommendation': 'Consider query caching or result set caching'
                })
        
        # Large table scans
        large_scans = df[df.get('BYTES_SCANNED', 0) > 1e10]  # > 10GB
        if not large_scans.empty:
            opportunities.append({
                'type': 'large_table_scans',
                'count': len(large_scans),
                'description': f'{len(large_scans)} queries scanned more than 10GB',
                'avg_bytes_scanned': int(large_scans['BYTES_SCANNED'].mean()),
                'recommendation': 'Consider adding filters, partitioning, or clustering'
            })
        
        return opportunities
    
    def _analyze_resource_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze resource usage patterns."""
        patterns = {}
        
        # Usage by warehouse
        if 'WAREHOUSE_NAME' in df.columns:
            warehouse_usage = df.groupby('WAREHOUSE_NAME').agg({
                'QUERY_ID': 'count',
                'QUERY_DURATION_MS': 'mean',
                'BYTES_SCANNED': 'sum'
            }).round(2)
            patterns['by_warehouse'] = warehouse_usage.to_dict('index')
        
        # Usage by user
        if 'USER_NAME' in df.columns:
            user_usage = df.groupby('USER_NAME').agg({
                'QUERY_ID': 'count',
                'QUERY_DURATION_MS': 'sum'
            }).round(2)
            patterns['by_user'] = user_usage.head(20).to_dict('index')
        
        # Usage by hour
        if 'START_TIME' in df.columns:
            df['HOUR'] = df['START_TIME'].dt.hour
            hourly_usage = df.groupby('HOUR').agg({
                'QUERY_ID': 'count',
                'QUERY_DURATION_MS': 'mean'
            }).round(2)
            patterns['by_hour'] = hourly_usage.to_dict('index')
        
        return patterns
    
    def get_query_performance_report(self, hours_back: int = 24) -> Dict[str, Any]:
        """Generate a comprehensive query performance report."""
        logger.info(f"Generating query performance report for last {hours_back} hours")
        
        # Collect recent data
        collection_result = self.collect_query_metrics(hours_back=hours_back)
        
        if not collection_result['success']:
            return collection_result
        
        # Get additional analysis from stored data
        from_time = datetime.now() - timedelta(hours=hours_back)
        
        try:
            stored_queries = self.storage.execute_query("""
                SELECT * FROM query_metrics 
                WHERE START_TIME >= ? 
                ORDER BY START_TIME DESC
            """, (from_time.isoformat(),))
            
            if stored_queries:
                df = pd.DataFrame(stored_queries)
                
                report = {
                    'report_generated_at': datetime.now().isoformat(),
                    'time_window_hours': hours_back,
                    'collection_result': collection_result,
                    'performance_analysis': self._generate_performance_insights(df),
                    'trends': self._analyze_performance_trends(df),
                    'recommendations': self._generate_recommendations(df)
                }
                
                return report
            else:
                return {
                    'report_generated_at': datetime.now().isoformat(),
                    'message': 'No query data available for analysis',
                    'collection_result': collection_result
                }
                
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {
                'error': str(e),
                'collection_result': collection_result
            }
    
    def _analyze_performance_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if df.empty or 'START_TIME' not in df.columns:
            return {}
        
        # Group by hour for trend analysis
        df['hour'] = pd.to_datetime(df['START_TIME']).dt.floor('H')
        hourly_trends = df.groupby('hour').agg({
            'QUERY_ID': 'count',
            'QUERY_DURATION_MS': 'mean',
            'BYTES_SCANNED': 'sum',
            'CACHE_HIT_RATIO': 'mean'
        }).round(2)
        
        return {
            'hourly_query_count': hourly_trends['QUERY_ID'].to_dict(),
            'hourly_avg_duration': hourly_trends['QUERY_DURATION_MS'].to_dict(),
            'hourly_bytes_scanned': hourly_trends['BYTES_SCANNED'].to_dict(),
            'hourly_cache_hit_ratio': hourly_trends['CACHE_HIT_RATIO'].to_dict()
        }
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        if df.empty:
            return recommendations
        
        # Check average performance
        avg_duration = df['QUERY_DURATION_MS'].mean()
        if avg_duration > 30000:  # > 30 seconds
            recommendations.append(
                "Average query duration is high. Consider optimizing frequent queries and adding appropriate indexes."
            )
        
        # Check cache utilization
        avg_cache_hit = df['CACHE_HIT_RATIO'].mean()
        if avg_cache_hit < 0.3:  # < 30%
            recommendations.append(
                "Low cache utilization detected. Consider enabling result caching and optimizing query patterns."
            )
        
        # Check spillage
        avg_spillage = df['SPILLAGE_RATIO'].mean()
        if avg_spillage > 0.05:  # > 5%
            recommendations.append(
                "High memory spillage detected. Consider using larger warehouse sizes for memory-intensive queries."
            )
        
        # Check error rate
        error_rate = len(df[df['EXECUTION_STATUS'] != 'SUCCESS']) / len(df)
        if error_rate > 0.05:  # > 5%
            recommendations.append(
                f"High query error rate ({error_rate:.1%}). Review failed queries and improve error handling."
            )
        
        return recommendations
