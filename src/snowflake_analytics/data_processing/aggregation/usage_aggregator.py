"""
Usage Aggregator

This module provides comprehensive usage pattern aggregation and analysis capabilities for
Snowflake analytics data, focusing on query patterns, warehouse utilization, and user behavior.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
from collections import Counter
import warnings

logger = logging.getLogger(__name__)


class UsageAggregator:
    """
    Handles usage pattern aggregation and analysis of Snowflake analytics data.
    
    This class provides methods for analyzing query patterns, warehouse utilization,
    user behavior, session patterns, and workload characteristics.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the UsageAggregator with optional configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'usage_dimensions': [
                'USER_NAME', 'WAREHOUSE_NAME', 'DATABASE_NAME', 
                'SCHEMA_NAME', 'QUERY_TYPE', 'ROLE_NAME', 'SESSION_ID'
            ],
            'metrics_to_analyze': [
                'query_count', 'execution_time_ms', 'bytes_scanned',
                'rows_produced', 'credits_used', 'partitions_scanned'
            ],
            'time_aggregations': ['hourly', 'daily', 'weekly', 'monthly'],
            'pattern_analysis': {
                'enabled': True,
                'min_pattern_frequency': 5,
                'similarity_threshold': 0.8,
                'analyze_query_text': True,
                'analyze_execution_patterns': True
            },
            'session_analysis': {
                'enabled': True,
                'session_timeout_minutes': 60,
                'min_session_queries': 3,
                'analyze_session_patterns': True
            },
            'workload_analysis': {
                'enabled': True,
                'peak_threshold_percentile': 90,
                'off_peak_threshold_percentile': 25,
                'analyze_concurrency': True,
                'analyze_resource_contention': True
            },
            'user_behavior_analysis': {
                'enabled': True,
                'analyze_query_complexity': True,
                'analyze_data_access_patterns': True,
                'analyze_temporal_patterns': True,
                'segment_users': True
            },
            'warehouse_utilization': {
                'enabled': True,
                'utilization_metrics': ['query_frequency', 'credit_consumption', 'concurrency'],
                'efficiency_analysis': True,
                'capacity_analysis': True
            },
            'percentiles': [25, 50, 75, 90, 95, 99],
            'top_n_analysis': 20,
            'include_anomaly_detection': True,
            'include_trend_analysis': True,
            'include_forecasting_features': True
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
    
    def aggregate_usage_patterns(self, 
                                data: pd.DataFrame,
                                time_column: str = 'START_TIME',
                                analysis_type: str = 'comprehensive') -> Dict[str, pd.DataFrame]:
        """
        Aggregate comprehensive usage patterns from Snowflake data.
        
        Args:
            data: Input DataFrame with Snowflake usage data
            time_column: Name of the timestamp column
            analysis_type: Type of analysis ('comprehensive', 'basic', 'advanced')
        
        Returns:
            Dictionary containing various usage pattern analyses
        """
        logger.info(f"Aggregating {analysis_type} usage patterns...")
        
        if time_column not in data.columns:
            raise ValueError(f"Time column {time_column} not found in data")
        
        # Prepare data
        data = data.copy()
        data[time_column] = pd.to_datetime(data[time_column])
        
        results = {}
        
        # Basic usage aggregations
        results['basic_usage'] = self._create_basic_usage_aggregations(data, time_column)
        
        # Pattern analysis
        if self.config.get('pattern_analysis', {}).get('enabled', True):
            results['query_patterns'] = self._analyze_query_patterns(data)
        
        # Session analysis
        if self.config.get('session_analysis', {}).get('enabled', True):
            results['session_analysis'] = self._analyze_session_patterns(data, time_column)
        
        # Workload analysis
        if self.config.get('workload_analysis', {}).get('enabled', True):
            results['workload_analysis'] = self._analyze_workload_patterns(data, time_column)
        
        # User behavior analysis
        if self.config.get('user_behavior_analysis', {}).get('enabled', True):
            results['user_behavior'] = self._analyze_user_behavior(data, time_column)
        
        # Warehouse utilization
        if self.config.get('warehouse_utilization', {}).get('enabled', True):
            results['warehouse_utilization'] = self._analyze_warehouse_utilization(data, time_column)
        
        if analysis_type == 'advanced':
            # Advanced analyses
            results['concurrency_analysis'] = self._analyze_concurrency_patterns(data, time_column)
            results['resource_contention'] = self._analyze_resource_contention(data, time_column)
            results['usage_forecasting'] = self._create_usage_forecasting_features(data, time_column)
        
        logger.info(f"Generated {len(results)} usage pattern analyses")
        return results
    
    def analyze_query_patterns(self, 
                             data: pd.DataFrame,
                             pattern_type: str = 'execution') -> pd.DataFrame:
        """
        Analyze specific query pattern types.
        
        Args:
            data: Input DataFrame
            pattern_type: Type of pattern analysis ('execution', 'temporal', 'resource')
        
        Returns:
            DataFrame with pattern analysis results
        """
        logger.info(f"Analyzing {pattern_type} query patterns...")
        
        if pattern_type == 'execution':
            return self._analyze_execution_patterns(data)
        elif pattern_type == 'temporal':
            return self._analyze_temporal_patterns(data)
        elif pattern_type == 'resource':
            return self._analyze_resource_patterns(data)
        else:
            raise ValueError(f"Unsupported pattern type: {pattern_type}")
    
    def create_user_segmentation(self, 
                               data: pd.DataFrame,
                               segmentation_method: str = 'behavior') -> pd.DataFrame:
        """
        Create user segmentation based on usage patterns.
        
        Args:
            data: Input DataFrame
            segmentation_method: Method for segmentation ('behavior', 'volume', 'efficiency')
        
        Returns:
            DataFrame with user segments
        """
        logger.info(f"Creating user segmentation by {segmentation_method}...")
        
        if 'USER_NAME' not in data.columns:
            raise ValueError("USER_NAME column required for user segmentation")
        
        if segmentation_method == 'behavior':
            return self._segment_users_by_behavior(data)
        elif segmentation_method == 'volume':
            return self._segment_users_by_volume(data)
        elif segmentation_method == 'efficiency':
            return self._segment_users_by_efficiency(data)
        else:
            raise ValueError(f"Unsupported segmentation method: {segmentation_method}")
    
    def _create_basic_usage_aggregations(self, data: pd.DataFrame, time_column: str) -> pd.DataFrame:
        """
        Create basic usage aggregation metrics.
        
        Args:
            data: Input DataFrame
            time_column: Time column name
        
        Returns:
            DataFrame with basic usage aggregations
        """
        # Overall usage statistics
        total_queries = len(data)
        unique_users = data['USER_NAME'].nunique() if 'USER_NAME' in data.columns else 0
        unique_warehouses = data['WAREHOUSE_NAME'].nunique() if 'WAREHOUSE_NAME' in data.columns else 0
        unique_databases = data['DATABASE_NAME'].nunique() if 'DATABASE_NAME' in data.columns else 0
        
        time_span = (data[time_column].max() - data[time_column].min()).total_seconds() / 3600  # hours
        
        result = pd.DataFrame([{
            'total_queries': total_queries,
            'unique_users': unique_users,
            'unique_warehouses': unique_warehouses,
            'unique_databases': unique_databases,
            'analysis_period_hours': time_span,
            'queries_per_hour': total_queries / max(time_span, 1),
            'analysis_start': data[time_column].min(),
            'analysis_end': data[time_column].max()
        }])
        
        # Add metrics analysis if available
        metrics = self.config.get('metrics_to_analyze', [])
        for metric in metrics:
            if metric == 'query_count':
                result[f'{metric}_total'] = total_queries
            elif metric in data.columns:
                result[f'{metric}_total'] = data[metric].sum()
                result[f'{metric}_mean'] = data[metric].mean()
                result[f'{metric}_median'] = data[metric].median()
                result[f'{metric}_std'] = data[metric].std()
                
                # Add percentiles
                percentiles = self.config.get('percentiles', [50, 75, 90, 95, 99])
                for p in percentiles:
                    result[f'{metric}_p{p}'] = data[metric].quantile(p/100)
        
        # Add temporal distribution
        result = self._add_temporal_distribution_stats(result, data, time_column)
        
        return result
    
    def _analyze_query_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze query execution patterns.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with query pattern analysis
        """
        pattern_config = self.config.get('pattern_analysis', {})
        min_frequency = pattern_config.get('min_pattern_frequency', 5)
        
        patterns = []
        
        # Query type patterns
        if 'QUERY_TYPE' in data.columns:
            query_type_patterns = data.groupby('QUERY_TYPE').agg({
                'QUERY_ID': 'count',
                'CREDITS_USED': ['sum', 'mean', 'std'] if 'CREDITS_USED' in data.columns else [],
                'EXECUTION_TIME_MS': ['mean', 'median', 'std'] if 'EXECUTION_TIME_MS' in data.columns else []
            })
            
            query_type_patterns.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in query_type_patterns.columns]
            query_type_patterns = query_type_patterns.reset_index()
            query_type_patterns['pattern_type'] = 'query_type'
            patterns.append(query_type_patterns)
        
        # Database access patterns
        if 'DATABASE_NAME' in data.columns:
            db_patterns = self._analyze_database_access_patterns(data, min_frequency)
            if not db_patterns.empty:
                patterns.append(db_patterns)
        
        # User query patterns
        if 'USER_NAME' in data.columns:
            user_patterns = self._analyze_user_query_patterns(data, min_frequency)
            if not user_patterns.empty:
                patterns.append(user_patterns)
        
        # Warehouse usage patterns
        if 'WAREHOUSE_NAME' in data.columns:
            warehouse_patterns = self._analyze_warehouse_patterns(data, min_frequency)
            if not warehouse_patterns.empty:
                patterns.append(warehouse_patterns)
        
        if patterns:
            combined_patterns = pd.concat(patterns, ignore_index=True)
            return self._add_pattern_analysis_features(combined_patterns)
        else:
            return pd.DataFrame()
    
    def _analyze_session_patterns(self, data: pd.DataFrame, time_column: str) -> pd.DataFrame:
        """
        Analyze session patterns and user behavior.
        
        Args:
            data: Input DataFrame
            time_column: Time column name
        
        Returns:
            DataFrame with session analysis
        """
        session_config = self.config.get('session_analysis', {})
        if 'USER_NAME' not in data.columns:
            return pd.DataFrame()
        
        timeout_minutes = session_config.get('session_timeout_minutes', 60)
        min_queries = session_config.get('min_session_queries', 3)
        
        sessions = []
        
        # Analyze sessions for each user
        for user in data['USER_NAME'].unique():
            user_data = data[data['USER_NAME'] == user].sort_values(time_column)
            
            if len(user_data) < min_queries:
                continue
            
            # Identify session boundaries
            time_diffs = user_data[time_column].diff().dt.total_seconds() / 60  # minutes
            session_breaks = time_diffs > timeout_minutes
            user_data['session_id'] = session_breaks.cumsum()
            
            # Analyze each session
            for session_id, session_data in user_data.groupby('session_id'):
                if len(session_data) >= min_queries:
                    session_analysis = self._analyze_single_session(session_data, time_column, user)
                    sessions.append(session_analysis)
        
        if sessions:
            return pd.DataFrame(sessions)
        else:
            return pd.DataFrame()
    
    def _analyze_workload_patterns(self, data: pd.DataFrame, time_column: str) -> pd.DataFrame:
        """
        Analyze workload patterns and resource utilization.
        
        Args:
            data: Input DataFrame
            time_column: Time column name
        
        Returns:
            DataFrame with workload analysis
        """
        workload_config = self.config.get('workload_analysis', {})
        
        # Hourly workload analysis
        data['hour'] = data[time_column].dt.hour
        data['day_of_week'] = data[time_column].dt.dayofweek
        data['date'] = data[time_column].dt.date
        
        workload_patterns = []
        
        # Hourly patterns
        hourly_workload = data.groupby('hour').agg({
            'QUERY_ID': 'count',
            'CREDITS_USED': ['sum', 'mean'] if 'CREDITS_USED' in data.columns else [],
            'EXECUTION_TIME_MS': ['sum', 'mean'] if 'EXECUTION_TIME_MS' in data.columns else []
        })
        
        hourly_workload.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in hourly_workload.columns]
        hourly_workload = hourly_workload.reset_index()
        hourly_workload['pattern_type'] = 'hourly'
        workload_patterns.append(hourly_workload)
        
        # Day of week patterns
        dow_workload = data.groupby('day_of_week').agg({
            'QUERY_ID': 'count',
            'CREDITS_USED': ['sum', 'mean'] if 'CREDITS_USED' in data.columns else [],
            'EXECUTION_TIME_MS': ['sum', 'mean'] if 'EXECUTION_TIME_MS' in data.columns else []
        })
        
        dow_workload.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in dow_workload.columns]
        dow_workload = dow_workload.reset_index()
        dow_workload['pattern_type'] = 'day_of_week'
        workload_patterns.append(dow_workload)
        
        # Daily workload trends
        daily_workload = data.groupby('date').agg({
            'QUERY_ID': 'count',
            'CREDITS_USED': ['sum', 'mean'] if 'CREDITS_USED' in data.columns else [],
            'EXECUTION_TIME_MS': ['sum', 'mean'] if 'EXECUTION_TIME_MS' in data.columns else []
        })
        
        daily_workload.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in daily_workload.columns]
        daily_workload = daily_workload.reset_index()
        daily_workload['pattern_type'] = 'daily'
        workload_patterns.append(daily_workload)
        
        if workload_patterns:
            combined_workload = pd.concat(workload_patterns, ignore_index=True)
            return self._add_workload_analysis_features(combined_workload)
        else:
            return pd.DataFrame()
    
    def _analyze_user_behavior(self, data: pd.DataFrame, time_column: str) -> pd.DataFrame:
        """
        Analyze user behavior patterns.
        
        Args:
            data: Input DataFrame
            time_column: Time column name
        
        Returns:
            DataFrame with user behavior analysis
        """
        if 'USER_NAME' not in data.columns:
            return pd.DataFrame()
        
        behavior_config = self.config.get('user_behavior_analysis', {})
        user_behaviors = []
        
        for user in data['USER_NAME'].unique():
            user_data = data[data['USER_NAME'] == user]
            behavior = self._analyze_single_user_behavior(user_data, time_column, user)
            user_behaviors.append(behavior)
        
        if user_behaviors:
            user_behavior_df = pd.DataFrame(user_behaviors)
            
            # Add user segmentation
            if behavior_config.get('segment_users', True):
                user_behavior_df = self._add_user_segments(user_behavior_df)
            
            return user_behavior_df
        else:
            return pd.DataFrame()
    
    def _analyze_warehouse_utilization(self, data: pd.DataFrame, time_column: str) -> pd.DataFrame:
        """
        Analyze warehouse utilization patterns.
        
        Args:
            data: Input DataFrame
            time_column: Time column name
        
        Returns:
            DataFrame with warehouse utilization analysis
        """
        if 'WAREHOUSE_NAME' not in data.columns:
            return pd.DataFrame()
        
        utilization_config = self.config.get('warehouse_utilization', {})
        warehouse_utilizations = []
        
        for warehouse in data['WAREHOUSE_NAME'].unique():
            warehouse_data = data[data['WAREHOUSE_NAME'] == warehouse]
            utilization = self._analyze_single_warehouse_utilization(warehouse_data, time_column, warehouse)
            warehouse_utilizations.append(utilization)
        
        if warehouse_utilizations:
            warehouse_df = pd.DataFrame(warehouse_utilizations)
            
            # Add efficiency analysis
            if utilization_config.get('efficiency_analysis', True):
                warehouse_df = self._add_warehouse_efficiency_analysis(warehouse_df)
            
            # Add capacity analysis
            if utilization_config.get('capacity_analysis', True):
                warehouse_df = self._add_warehouse_capacity_analysis(warehouse_df, data)
            
            return warehouse_df
        else:
            return pd.DataFrame()
    
    def _analyze_database_access_patterns(self, data: pd.DataFrame, min_frequency: int) -> pd.DataFrame:
        """
        Analyze database access patterns.
        
        Args:
            data: Input DataFrame
            min_frequency: Minimum frequency for pattern inclusion
        
        Returns:
            DataFrame with database access patterns
        """
        if 'DATABASE_NAME' not in data.columns:
            return pd.DataFrame()
        
        db_patterns = data.groupby(['DATABASE_NAME', 'SCHEMA_NAME'] if 'SCHEMA_NAME' in data.columns else ['DATABASE_NAME']).agg({
            'QUERY_ID': 'count',
            'CREDITS_USED': ['sum', 'mean'] if 'CREDITS_USED' in data.columns else [],
            'USER_NAME': 'nunique' if 'USER_NAME' in data.columns else []
        })
        
        db_patterns.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in db_patterns.columns]
        db_patterns = db_patterns.reset_index()
        
        # Filter by minimum frequency
        if 'QUERY_ID_count' in db_patterns.columns:
            db_patterns = db_patterns[db_patterns['QUERY_ID_count'] >= min_frequency]
        
        db_patterns['pattern_type'] = 'database_access'
        return db_patterns
    
    def _analyze_user_query_patterns(self, data: pd.DataFrame, min_frequency: int) -> pd.DataFrame:
        """
        Analyze user query patterns.
        
        Args:
            data: Input DataFrame
            min_frequency: Minimum frequency for pattern inclusion
        
        Returns:
            DataFrame with user query patterns
        """
        if 'USER_NAME' not in data.columns:
            return pd.DataFrame()
        
        user_patterns = data.groupby(['USER_NAME', 'QUERY_TYPE'] if 'QUERY_TYPE' in data.columns else ['USER_NAME']).agg({
            'QUERY_ID': 'count',
            'CREDITS_USED': ['sum', 'mean', 'std'] if 'CREDITS_USED' in data.columns else [],
            'EXECUTION_TIME_MS': ['mean', 'std'] if 'EXECUTION_TIME_MS' in data.columns else []
        })
        
        user_patterns.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in user_patterns.columns]
        user_patterns = user_patterns.reset_index()
        
        # Filter by minimum frequency
        if 'QUERY_ID_count' in user_patterns.columns:
            user_patterns = user_patterns[user_patterns['QUERY_ID_count'] >= min_frequency]
        
        user_patterns['pattern_type'] = 'user_query'
        return user_patterns
    
    def _analyze_warehouse_patterns(self, data: pd.DataFrame, min_frequency: int) -> pd.DataFrame:
        """
        Analyze warehouse usage patterns.
        
        Args:
            data: Input DataFrame
            min_frequency: Minimum frequency for pattern inclusion
        
        Returns:
            DataFrame with warehouse patterns
        """
        warehouse_patterns = data.groupby(['WAREHOUSE_NAME', 'QUERY_TYPE'] if 'QUERY_TYPE' in data.columns else ['WAREHOUSE_NAME']).agg({
            'QUERY_ID': 'count',
            'CREDITS_USED': ['sum', 'mean'] if 'CREDITS_USED' in data.columns else [],
            'USER_NAME': 'nunique' if 'USER_NAME' in data.columns else []
        })
        
        warehouse_patterns.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in warehouse_patterns.columns]
        warehouse_patterns = warehouse_patterns.reset_index()
        
        # Filter by minimum frequency
        if 'QUERY_ID_count' in warehouse_patterns.columns:
            warehouse_patterns = warehouse_patterns[warehouse_patterns['QUERY_ID_count'] >= min_frequency]
        
        warehouse_patterns['pattern_type'] = 'warehouse_usage'
        return warehouse_patterns
    
    def _add_pattern_analysis_features(self, patterns: pd.DataFrame) -> pd.DataFrame:
        """
        Add pattern analysis features.
        
        Args:
            patterns: Pattern DataFrame
        
        Returns:
            Enhanced patterns DataFrame
        """
        if 'QUERY_ID_count' in patterns.columns:
            # Frequency-based features
            patterns['frequency_rank'] = patterns.groupby('pattern_type')['QUERY_ID_count'].rank(method='dense', ascending=False)
            patterns['frequency_percentile'] = patterns.groupby('pattern_type')['QUERY_ID_count'].rank(pct=True)
            
            # Pattern significance
            patterns['is_high_frequency'] = (patterns['frequency_percentile'] >= 0.8).astype(int)
            patterns['is_rare_pattern'] = (patterns['frequency_percentile'] <= 0.2).astype(int)
        
        # Cost-based features if available
        if 'CREDITS_USED_sum' in patterns.columns:
            patterns['cost_rank'] = patterns.groupby('pattern_type')['CREDITS_USED_sum'].rank(method='dense', ascending=False)
            patterns['cost_percentile'] = patterns.groupby('pattern_type')['CREDITS_USED_sum'].rank(pct=True)
            patterns['is_high_cost_pattern'] = (patterns['cost_percentile'] >= 0.8).astype(int)
        
        return patterns
    
    def _analyze_single_session(self, session_data: pd.DataFrame, time_column: str, user: str) -> Dict[str, Any]:
        """
        Analyze a single user session.
        
        Args:
            session_data: Session data
            time_column: Time column name
            user: User name
        
        Returns:
            Dictionary with session analysis
        """
        session_start = session_data[time_column].min()
        session_end = session_data[time_column].max()
        session_duration = (session_end - session_start).total_seconds() / 60  # minutes
        
        analysis = {
            'user_name': user,
            'session_start': session_start,
            'session_end': session_end,
            'session_duration_minutes': session_duration,
            'query_count': len(session_data),
            'queries_per_minute': len(session_data) / max(session_duration, 1)
        }
        
        # Add metrics if available
        if 'CREDITS_USED' in session_data.columns:
            analysis['total_credits'] = session_data['CREDITS_USED'].sum()
            analysis['avg_credits_per_query'] = session_data['CREDITS_USED'].mean()
            analysis['credits_per_minute'] = session_data['CREDITS_USED'].sum() / max(session_duration, 1)
        
        if 'EXECUTION_TIME_MS' in session_data.columns:
            analysis['total_execution_time_ms'] = session_data['EXECUTION_TIME_MS'].sum()
            analysis['avg_execution_time_ms'] = session_data['EXECUTION_TIME_MS'].mean()
        
        # Query type diversity
        if 'QUERY_TYPE' in session_data.columns:
            analysis['unique_query_types'] = session_data['QUERY_TYPE'].nunique()
            analysis['query_type_diversity'] = analysis['unique_query_types'] / len(session_data)
        
        # Database access diversity
        if 'DATABASE_NAME' in session_data.columns:
            analysis['unique_databases'] = session_data['DATABASE_NAME'].nunique()
            analysis['database_diversity'] = analysis['unique_databases'] / len(session_data)
        
        return analysis
    
    def _analyze_single_user_behavior(self, user_data: pd.DataFrame, time_column: str, user: str) -> Dict[str, Any]:
        """
        Analyze behavior patterns for a single user.
        
        Args:
            user_data: User's query data
            time_column: Time column name
            user: User name
        
        Returns:
            Dictionary with user behavior analysis
        """
        behavior = {
            'user_name': user,
            'total_queries': len(user_data),
            'first_query': user_data[time_column].min(),
            'last_query': user_data[time_column].max(),
            'activity_period_days': (user_data[time_column].max() - user_data[time_column].min()).days + 1
        }
        
        # Activity frequency
        behavior['queries_per_day'] = len(user_data) / max(behavior['activity_period_days'], 1)
        
        # Temporal patterns
        user_data['hour'] = user_data[time_column].dt.hour
        user_data['day_of_week'] = user_data[time_column].dt.dayofweek
        
        behavior['most_active_hour'] = user_data['hour'].mode().iloc[0] if not user_data['hour'].mode().empty else 0
        behavior['most_active_day'] = user_data['day_of_week'].mode().iloc[0] if not user_data['day_of_week'].mode().empty else 0
        behavior['hour_diversity'] = user_data['hour'].nunique()
        
        # Resource usage patterns
        if 'CREDITS_USED' in user_data.columns:
            behavior['total_credits'] = user_data['CREDITS_USED'].sum()
            behavior['avg_credits_per_query'] = user_data['CREDITS_USED'].mean()
            behavior['credits_std'] = user_data['CREDITS_USED'].std()
            behavior['max_single_query_credits'] = user_data['CREDITS_USED'].max()
        
        # Query complexity indicators
        if 'EXECUTION_TIME_MS' in user_data.columns:
            behavior['avg_execution_time_ms'] = user_data['EXECUTION_TIME_MS'].mean()
            behavior['execution_time_std'] = user_data['EXECUTION_TIME_MS'].std()
            behavior['max_execution_time_ms'] = user_data['EXECUTION_TIME_MS'].max()
        
        if 'BYTES_SCANNED' in user_data.columns:
            behavior['avg_bytes_scanned'] = user_data['BYTES_SCANNED'].mean()
            behavior['max_bytes_scanned'] = user_data['BYTES_SCANNED'].max()
        
        # Query type preferences
        if 'QUERY_TYPE' in user_data.columns:
            query_types = user_data['QUERY_TYPE'].value_counts()
            behavior['primary_query_type'] = query_types.index[0] if len(query_types) > 0 else 'unknown'
            behavior['query_type_diversity'] = user_data['QUERY_TYPE'].nunique()
            behavior['query_type_concentration'] = query_types.iloc[0] / len(user_data) if len(query_types) > 0 else 0
        
        # Database access patterns
        if 'DATABASE_NAME' in user_data.columns:
            behavior['unique_databases_accessed'] = user_data['DATABASE_NAME'].nunique()
            behavior['primary_database'] = user_data['DATABASE_NAME'].mode().iloc[0] if not user_data['DATABASE_NAME'].mode().empty else 'unknown'
        
        # Warehouse usage patterns
        if 'WAREHOUSE_NAME' in user_data.columns:
            behavior['unique_warehouses_used'] = user_data['WAREHOUSE_NAME'].nunique()
            behavior['primary_warehouse'] = user_data['WAREHOUSE_NAME'].mode().iloc[0] if not user_data['WAREHOUSE_NAME'].mode().empty else 'unknown'
        
        return behavior
    
    def _analyze_single_warehouse_utilization(self, warehouse_data: pd.DataFrame, time_column: str, warehouse: str) -> Dict[str, Any]:
        """
        Analyze utilization patterns for a single warehouse.
        
        Args:
            warehouse_data: Warehouse query data
            time_column: Time column name
            warehouse: Warehouse name
        
        Returns:
            Dictionary with warehouse utilization analysis
        """
        utilization = {
            'warehouse_name': warehouse,
            'total_queries': len(warehouse_data),
            'first_query': warehouse_data[time_column].min(),
            'last_query': warehouse_data[time_column].max(),
            'activity_period_hours': (warehouse_data[time_column].max() - warehouse_data[time_column].min()).total_seconds() / 3600
        }
        
        # Query frequency
        utilization['queries_per_hour'] = len(warehouse_data) / max(utilization['activity_period_hours'], 1)
        
        # User diversity
        if 'USER_NAME' in warehouse_data.columns:
            utilization['unique_users'] = warehouse_data['USER_NAME'].nunique()
            utilization['primary_user'] = warehouse_data['USER_NAME'].mode().iloc[0] if not warehouse_data['USER_NAME'].mode().empty else 'unknown'
            
            # User concentration
            user_counts = warehouse_data['USER_NAME'].value_counts()
            utilization['user_concentration'] = user_counts.iloc[0] / len(warehouse_data) if len(user_counts) > 0 else 0
        
        # Resource consumption
        if 'CREDITS_USED' in warehouse_data.columns:
            utilization['total_credits'] = warehouse_data['CREDITS_USED'].sum()
            utilization['avg_credits_per_query'] = warehouse_data['CREDITS_USED'].mean()
            utilization['credits_per_hour'] = warehouse_data['CREDITS_USED'].sum() / max(utilization['activity_period_hours'], 1)
            utilization['max_single_query_credits'] = warehouse_data['CREDITS_USED'].max()
        
        # Performance characteristics
        if 'EXECUTION_TIME_MS' in warehouse_data.columns:
            utilization['avg_execution_time_ms'] = warehouse_data['EXECUTION_TIME_MS'].mean()
            utilization['max_execution_time_ms'] = warehouse_data['EXECUTION_TIME_MS'].max()
            utilization['execution_time_p95'] = warehouse_data['EXECUTION_TIME_MS'].quantile(0.95)
        
        # Workload diversity
        if 'QUERY_TYPE' in warehouse_data.columns:
            utilization['unique_query_types'] = warehouse_data['QUERY_TYPE'].nunique()
            query_types = warehouse_data['QUERY_TYPE'].value_counts()
            utilization['primary_query_type'] = query_types.index[0] if len(query_types) > 0 else 'unknown'
        
        # Data access patterns
        if 'DATABASE_NAME' in warehouse_data.columns:
            utilization['unique_databases'] = warehouse_data['DATABASE_NAME'].nunique()
            utilization['primary_database'] = warehouse_data['DATABASE_NAME'].mode().iloc[0] if not warehouse_data['DATABASE_NAME'].mode().empty else 'unknown'
        
        # Temporal patterns
        warehouse_data['hour'] = warehouse_data[time_column].dt.hour
        utilization['peak_hour'] = warehouse_data['hour'].mode().iloc[0] if not warehouse_data['hour'].mode().empty else 0
        utilization['active_hours'] = warehouse_data['hour'].nunique()
        
        return utilization
    
    def _add_temporal_distribution_stats(self, result: pd.DataFrame, data: pd.DataFrame, time_column: str) -> pd.DataFrame:
        """
        Add temporal distribution statistics.
        
        Args:
            result: Current result DataFrame
            data: Original data
            time_column: Time column name
        
        Returns:
            Enhanced result DataFrame
        """
        # Hour distribution
        hour_dist = data[time_column].dt.hour.value_counts().sort_index()
        result['peak_hour'] = hour_dist.idxmax()
        result['off_peak_hour'] = hour_dist.idxmin()
        result['hour_variance'] = hour_dist.var()
        
        # Day of week distribution
        dow_dist = data[time_column].dt.dayofweek.value_counts().sort_index()
        result['peak_day_of_week'] = dow_dist.idxmax()
        result['weekend_query_ratio'] = dow_dist[[5, 6]].sum() / len(data)  # Saturday, Sunday
        
        # Business hours analysis (assuming 9-17)
        business_hours = data[time_column].dt.hour.between(9, 17)
        result['business_hours_query_ratio'] = business_hours.sum() / len(data)
        
        return result
    
    def _add_workload_analysis_features(self, workload: pd.DataFrame) -> pd.DataFrame:
        """
        Add workload analysis features.
        
        Args:
            workload: Workload DataFrame
        
        Returns:
            Enhanced workload DataFrame
        """
        # Peak detection
        if 'QUERY_ID_count' in workload.columns:
            peak_threshold = workload['QUERY_ID_count'].quantile(0.9)
            workload['is_peak_period'] = (workload['QUERY_ID_count'] >= peak_threshold).astype(int)
            
            off_peak_threshold = workload['QUERY_ID_count'].quantile(0.25)
            workload['is_off_peak_period'] = (workload['QUERY_ID_count'] <= off_peak_threshold).astype(int)
        
        # Load distribution
        if 'CREDITS_USED_sum' in workload.columns:
            workload['credits_rank'] = workload['CREDITS_USED_sum'].rank(method='dense', ascending=False)
            workload['credits_percentile'] = workload['CREDITS_USED_sum'].rank(pct=True)
        
        return workload
    
    def _add_user_segments(self, user_behavior: pd.DataFrame) -> pd.DataFrame:
        """
        Add user segmentation based on behavior patterns.
        
        Args:
            user_behavior: User behavior DataFrame
        
        Returns:
            Enhanced user behavior DataFrame with segments
        """
        # Volume-based segmentation
        if 'total_queries' in user_behavior.columns:
            user_behavior['volume_segment'] = pd.cut(
                user_behavior['total_queries'],
                bins=[0, 10, 100, 1000, float('inf')],
                labels=['low', 'medium', 'high', 'very_high']
            )
        
        # Credit usage segmentation
        if 'total_credits' in user_behavior.columns:
            user_behavior['credit_segment'] = pd.cut(
                user_behavior['total_credits'],
                bins=[0, 1, 10, 100, float('inf')],
                labels=['light', 'moderate', 'heavy', 'very_heavy']
            )
        
        # Activity pattern segmentation
        if 'hour_diversity' in user_behavior.columns:
            user_behavior['activity_pattern'] = pd.cut(
                user_behavior['hour_diversity'],
                bins=[0, 4, 8, 16, 24],
                labels=['narrow', 'focused', 'broad', 'continuous']
            )
        
        # Query complexity segmentation
        if 'avg_execution_time_ms' in user_behavior.columns:
            user_behavior['complexity_segment'] = pd.cut(
                user_behavior['avg_execution_time_ms'],
                bins=[0, 1000, 10000, 60000, float('inf')],
                labels=['simple', 'moderate', 'complex', 'very_complex']
            )
        
        return user_behavior
    
    def _add_warehouse_efficiency_analysis(self, warehouse_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add warehouse efficiency analysis features.
        
        Args:
            warehouse_df: Warehouse DataFrame
        
        Returns:
            Enhanced warehouse DataFrame
        """
        # Query efficiency
        if 'queries_per_hour' in warehouse_df.columns:
            warehouse_df['query_efficiency_rank'] = warehouse_df['queries_per_hour'].rank(method='dense', ascending=False)
            warehouse_df['query_efficiency_percentile'] = warehouse_df['queries_per_hour'].rank(pct=True)
        
        # Credit efficiency
        if 'credits_per_hour' in warehouse_df.columns:
            warehouse_df['credit_efficiency_rank'] = warehouse_df['credits_per_hour'].rank(method='dense', ascending=False)
            warehouse_df['credit_efficiency_percentile'] = warehouse_df['credits_per_hour'].rank(pct=True)
        
        # User utilization efficiency
        if 'unique_users' in warehouse_df.columns and 'total_queries' in warehouse_df.columns:
            warehouse_df['queries_per_user'] = warehouse_df['total_queries'] / warehouse_df['unique_users']
            warehouse_df['user_utilization_rank'] = warehouse_df['queries_per_user'].rank(method='dense', ascending=False)
        
        return warehouse_df
    
    def _add_warehouse_capacity_analysis(self, warehouse_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add warehouse capacity analysis features.
        
        Args:
            warehouse_df: Warehouse DataFrame
            data: Original data
        
        Returns:
            Enhanced warehouse DataFrame
        """
        # Concurrency analysis
        for warehouse in warehouse_df['warehouse_name']:
            warehouse_data = data[data['WAREHOUSE_NAME'] == warehouse]
            
            # Simple concurrency estimate (overlapping queries)
            if 'START_TIME' in warehouse_data.columns and 'END_TIME' in warehouse_data.columns:
                # This is a simplified approach - full concurrency analysis would be more complex
                max_concurrent = self._estimate_max_concurrency(warehouse_data)
                warehouse_df.loc[warehouse_df['warehouse_name'] == warehouse, 'estimated_max_concurrency'] = max_concurrent
            
            # Peak load analysis
            warehouse_data['hour'] = warehouse_data['START_TIME'].dt.hour
            hourly_load = warehouse_data.groupby('hour').size()
            peak_load = hourly_load.max()
            avg_load = hourly_load.mean()
            
            warehouse_df.loc[warehouse_df['warehouse_name'] == warehouse, 'peak_hourly_load'] = peak_load
            warehouse_df.loc[warehouse_df['warehouse_name'] == warehouse, 'avg_hourly_load'] = avg_load
            warehouse_df.loc[warehouse_df['warehouse_name'] == warehouse, 'load_variability'] = hourly_load.std()
        
        return warehouse_df
    
    def _estimate_max_concurrency(self, warehouse_data: pd.DataFrame) -> int:
        """
        Estimate maximum concurrency for a warehouse.
        
        Args:
            warehouse_data: Warehouse query data
        
        Returns:
            Estimated maximum concurrent queries
        """
        if 'END_TIME' not in warehouse_data.columns:
            return 0
        
        # Create events for query start and end
        events = []
        for _, row in warehouse_data.iterrows():
            events.append((row['START_TIME'], 1))  # Query start
            if pd.notna(row['END_TIME']):
                events.append((row['END_TIME'], -1))  # Query end
        
        # Sort events by time
        events.sort(key=lambda x: x[0])
        
        # Track concurrent queries
        current_concurrent = 0
        max_concurrent = 0
        
        for time, change in events:
            current_concurrent += change
            max_concurrent = max(max_concurrent, current_concurrent)
        
        return max_concurrent
    
    def _segment_users_by_behavior(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Segment users by behavior patterns.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with user behavior segments
        """
        user_analysis = self._analyze_user_behavior(data, 'START_TIME')
        return self._add_user_segments(user_analysis)
    
    def _segment_users_by_volume(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Segment users by query volume.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with user volume segments
        """
        user_volumes = data.groupby('USER_NAME').agg({
            'QUERY_ID': 'count',
            'CREDITS_USED': 'sum' if 'CREDITS_USED' in data.columns else lambda x: 0
        })
        
        user_volumes.columns = ['query_count', 'total_credits']
        user_volumes = user_volumes.reset_index()
        
        # Volume-based segmentation
        user_volumes['volume_segment'] = pd.cut(
            user_volumes['query_count'],
            bins=[0, 10, 100, 1000, float('inf')],
            labels=['low', 'medium', 'high', 'very_high']
        )
        
        return user_volumes
    
    def _segment_users_by_efficiency(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Segment users by efficiency metrics.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with user efficiency segments
        """
        if 'CREDITS_USED' not in data.columns:
            return pd.DataFrame()
        
        user_efficiency = data.groupby('USER_NAME').agg({
            'CREDITS_USED': ['sum', 'mean'],
            'EXECUTION_TIME_MS': 'mean' if 'EXECUTION_TIME_MS' in data.columns else lambda x: 0,
            'ROWS_PRODUCED': 'sum' if 'ROWS_PRODUCED' in data.columns else lambda x: 0
        })
        
        user_efficiency.columns = [f"{col[0]}_{col[1]}" for col in user_efficiency.columns]
        user_efficiency = user_efficiency.reset_index()
        
        # Efficiency metrics
        if 'ROWS_PRODUCED_sum' in user_efficiency.columns:
            user_efficiency['credits_per_row'] = user_efficiency['CREDITS_USED_sum'] / (user_efficiency['ROWS_PRODUCED_sum'] + 1)
            
            # Efficiency segmentation
            user_efficiency['efficiency_segment'] = pd.cut(
                user_efficiency['credits_per_row'],
                bins=[0, 0.001, 0.01, 0.1, float('inf')],
                labels=['very_efficient', 'efficient', 'moderate', 'inefficient']
            )
        
        return user_efficiency
    
    def _analyze_concurrency_patterns(self, data: pd.DataFrame, time_column: str) -> pd.DataFrame:
        """
        Analyze concurrency patterns in query execution.
        
        Args:
            data: Input DataFrame
            time_column: Time column name
        
        Returns:
            DataFrame with concurrency analysis
        """
        # This is a simplified concurrency analysis
        # In production, you'd want more sophisticated overlap detection
        
        if 'WAREHOUSE_NAME' not in data.columns:
            return pd.DataFrame()
        
        concurrency_analysis = []
        
        for warehouse in data['WAREHOUSE_NAME'].unique():
            warehouse_data = data[data['WAREHOUSE_NAME'] == warehouse]
            
            # Hourly concurrency approximation
            warehouse_data['hour'] = warehouse_data[time_column].dt.hour
            hourly_concurrency = warehouse_data.groupby('hour').size()
            
            analysis = {
                'warehouse_name': warehouse,
                'avg_hourly_queries': hourly_concurrency.mean(),
                'peak_hourly_queries': hourly_concurrency.max(),
                'concurrency_variance': hourly_concurrency.var(),
                'peak_hour': hourly_concurrency.idxmax()
            }
            
            concurrency_analysis.append(analysis)
        
        return pd.DataFrame(concurrency_analysis)
    
    def _analyze_resource_contention(self, data: pd.DataFrame, time_column: str) -> pd.DataFrame:
        """
        Analyze resource contention patterns.
        
        Args:
            data: Input DataFrame
            time_column: Time column name
        
        Returns:
            DataFrame with resource contention analysis
        """
        contention_analysis = []
        
        # Analyze by warehouse
        if 'WAREHOUSE_NAME' in data.columns:
            for warehouse in data['WAREHOUSE_NAME'].unique():
                warehouse_data = data[data['WAREHOUSE_NAME'] == warehouse]
                
                # Performance degradation indicators
                if 'EXECUTION_TIME_MS' in warehouse_data.columns:
                    # Group by hour to detect performance patterns
                    warehouse_data['hour'] = warehouse_data[time_column].dt.hour
                    hourly_perf = warehouse_data.groupby('hour')['EXECUTION_TIME_MS'].agg(['mean', 'std', 'count'])
                    
                    # Identify hours with high variance (potential contention)
                    high_variance_hours = hourly_perf[hourly_perf['std'] > hourly_perf['std'].quantile(0.8)]
                    
                    analysis = {
                        'warehouse_name': warehouse,
                        'high_contention_hours': len(high_variance_hours),
                        'avg_execution_variance': hourly_perf['std'].mean(),
                        'peak_variance_hour': hourly_perf['std'].idxmax(),
                        'contention_indicator': len(high_variance_hours) / len(hourly_perf)
                    }
                    
                    contention_analysis.append(analysis)
        
        return pd.DataFrame(contention_analysis)
    
    def _create_usage_forecasting_features(self, data: pd.DataFrame, time_column: str) -> pd.DataFrame:
        """
        Create features useful for usage forecasting.
        
        Args:
            data: Input DataFrame
            time_column: Time column name
        
        Returns:
            DataFrame with forecasting features
        """
        # Daily aggregation for forecasting
        data['date'] = data[time_column].dt.date
        daily_usage = data.groupby('date').agg({
            'QUERY_ID': 'count',
            'CREDITS_USED': 'sum' if 'CREDITS_USED' in data.columns else lambda x: 0,
            'USER_NAME': 'nunique' if 'USER_NAME' in data.columns else lambda x: 0
        })
        
        daily_usage.columns = ['daily_queries', 'daily_credits', 'daily_users']
        daily_usage = daily_usage.reset_index()
        daily_usage['date'] = pd.to_datetime(daily_usage['date'])
        
        # Add trend features
        daily_usage = daily_usage.sort_values('date')
        daily_usage['query_trend'] = daily_usage['daily_queries'].pct_change()
        daily_usage['credit_trend'] = daily_usage['daily_credits'].pct_change()
        daily_usage['user_trend'] = daily_usage['daily_users'].pct_change()
        
        # Add moving averages
        daily_usage['query_ma_7'] = daily_usage['daily_queries'].rolling(7).mean()
        daily_usage['credit_ma_7'] = daily_usage['daily_credits'].rolling(7).mean()
        
        # Add seasonality indicators
        daily_usage['day_of_week'] = daily_usage['date'].dt.dayofweek
        daily_usage['is_weekend'] = daily_usage['day_of_week'].isin([5, 6]).astype(int)
        
        return daily_usage
    
    def configure_for_snowflake_data(self):
        """
        Configure the aggregator for typical Snowflake analytics data.
        """
        snowflake_config = {
            'usage_dimensions': [
                'USER_NAME', 'WAREHOUSE_NAME', 'DATABASE_NAME', 'SCHEMA_NAME',
                'QUERY_TYPE', 'ROLE_NAME', 'SESSION_ID', 'QUERY_TAG'
            ],
            'metrics_to_analyze': [
                'query_count', 'EXECUTION_TIME_MS', 'BYTES_SCANNED',
                'ROWS_PRODUCED', 'CREDITS_USED', 'PARTITIONS_SCANNED',
                'COMPILATION_TIME_MS'
            ],
            'session_analysis': {
                'enabled': True,
                'session_timeout_minutes': 30,  # Shorter for analytics workloads
                'min_session_queries': 2,
                'analyze_session_patterns': True
            },
            'pattern_analysis': {
                'min_pattern_frequency': 3,  # Lower threshold for Snowflake
                'analyze_query_text': False,  # Usually too complex for simple analysis
                'analyze_execution_patterns': True
            }
        }
        
        self.config.update(snowflake_config)
        logger.info("Configured UsageAggregator for Snowflake analytics data")
