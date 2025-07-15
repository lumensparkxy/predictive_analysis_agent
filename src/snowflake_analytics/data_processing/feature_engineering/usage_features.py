"""
Usage Pattern Feature Generator

This module provides comprehensive usage pattern feature engineering capabilities
for extracting insights from Snowflake analytics usage data.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)


class UsageFeatureGenerator:
    """
    Generates comprehensive usage pattern features from Snowflake analytics data.
    
    This class extracts various usage patterns including query characteristics,
    user behavior patterns, resource utilization, and workload analysis
    specifically designed for Snowflake analytics data.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the UsageFeatureGenerator with optional configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'usage_columns': {
                'execution_time': 'EXECUTION_TIME_MS',
                'queue_time': 'QUEUE_TIME_MS',
                'compilation_time': 'COMPILATION_TIME_MS',
                'bytes_scanned': 'BYTES_SCANNED',
                'bytes_written': 'BYTES_WRITTEN',
                'rows_produced': 'ROWS_PRODUCED',
                'rows_inserted': 'ROWS_INSERTED',
                'rows_updated': 'ROWS_UPDATED',
                'rows_deleted': 'ROWS_DELETED',
                'partitions_scanned': 'PARTITIONS_SCANNED',
                'warehouse_size': 'WAREHOUSE_SIZE'
            },
            'user_columns': {
                'user_name': 'USER_NAME',
                'role_name': 'ROLE_NAME',
                'session_id': 'SESSION_ID'
            },
            'query_columns': {
                'query_type': 'QUERY_TYPE',
                'query_tag': 'QUERY_TAG',
                'database_name': 'DATABASE_NAME',
                'schema_name': 'SCHEMA_NAME',
                'warehouse_name': 'WAREHOUSE_NAME'
            },
            'time_column': 'START_TIME',
            'percentiles': [25, 50, 75, 90, 95, 99],
            'rolling_windows': [3, 7, 14, 30],  # days
            'min_observations': 10,  # Minimum observations for statistical features
            'outlier_threshold': 3.0,  # Z-score threshold for outliers
            'efficiency_metrics': True,
            'user_behavior_metrics': True,
            'workload_analysis': True
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
    
    def generate_usage_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive usage pattern features from the input DataFrame.
        
        Args:
            data: Input DataFrame containing Snowflake usage data
        
        Returns:
            DataFrame with generated usage features
        """
        logger.info("Generating usage pattern features...")
        
        feature_df = pd.DataFrame(index=data.index)
        
        # Basic usage metrics
        feature_df.update(self._extract_basic_usage_metrics(data))
        
        # Performance efficiency metrics
        if self.config.get('efficiency_metrics', True):
            feature_df.update(self._extract_efficiency_metrics(data))
        
        # Query complexity features
        feature_df.update(self._extract_query_complexity_features(data))
        
        # Resource utilization features
        feature_df.update(self._extract_resource_utilization_features(data))
        
        # User behavior patterns
        if self.config.get('user_behavior_metrics', True):
            feature_df.update(self._extract_user_behavior_features(data))
        
        # Workload analysis features
        if self.config.get('workload_analysis', True):
            feature_df.update(self._extract_workload_analysis_features(data))
        
        # Statistical distribution features
        feature_df.update(self._extract_statistical_features(data))
        
        # Comparative features (vs. historical averages)
        feature_df.update(self._extract_comparative_features(data))
        
        logger.info(f"Generated {len(feature_df.columns)} usage pattern features")
        return feature_df
    
    def _extract_basic_usage_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract basic usage metrics and transformations.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with basic usage metrics
        """
        features = pd.DataFrame(index=data.index)
        usage_cols = self.config['usage_columns']
        
        # Execution time features
        if usage_cols['execution_time'] in data.columns:
            exec_time = data[usage_cols['execution_time']]
            features['execution_time_ms'] = exec_time
            features['execution_time_seconds'] = exec_time / 1000
            features['execution_time_minutes'] = exec_time / 60000
            features['log_execution_time'] = np.log1p(exec_time)
            
            # Execution time categories
            features['execution_time_category'] = pd.cut(
                exec_time,
                bins=[0, 1000, 5000, 30000, 300000, float('inf')],
                labels=['very_fast', 'fast', 'medium', 'slow', 'very_slow'],
                include_lowest=True
            ).astype('category')
        
        # Queue time features
        if usage_cols['queue_time'] in data.columns:
            queue_time = data[usage_cols['queue_time']]
            features['queue_time_ms'] = queue_time
            features['queue_time_seconds'] = queue_time / 1000
            features['log_queue_time'] = np.log1p(queue_time)
            
            # Queue time categories
            features['queue_time_category'] = pd.cut(
                queue_time,
                bins=[0, 100, 1000, 10000, float('inf')],
                labels=['no_queue', 'short_queue', 'medium_queue', 'long_queue'],
                include_lowest=True
            ).astype('category')
        
        # Compilation time features
        if usage_cols['compilation_time'] in data.columns:
            comp_time = data[usage_cols['compilation_time']]
            features['compilation_time_ms'] = comp_time
            features['log_compilation_time'] = np.log1p(comp_time)
        
        # Data volume features
        if usage_cols['bytes_scanned'] in data.columns:
            bytes_scanned = data[usage_cols['bytes_scanned']]
            features['bytes_scanned'] = bytes_scanned
            features['gb_scanned'] = bytes_scanned / (1024**3)
            features['log_bytes_scanned'] = np.log1p(bytes_scanned)
            
            # Data volume categories
            features['data_volume_category'] = pd.cut(
                bytes_scanned,
                bins=[0, 1024**2, 1024**3, 10*1024**3, float('inf')],
                labels=['small', 'medium', 'large', 'very_large'],
                include_lowest=True
            ).astype('category')
        
        # Row count features
        if usage_cols['rows_produced'] in data.columns:
            rows_produced = data[usage_cols['rows_produced']]
            features['rows_produced'] = rows_produced
            features['log_rows_produced'] = np.log1p(rows_produced)
            
            # Row count categories
            features['row_count_category'] = pd.cut(
                rows_produced,
                bins=[0, 100, 10000, 1000000, float('inf')],
                labels=['small', 'medium', 'large', 'very_large'],
                include_lowest=True
            ).astype('category')
        
        return features
    
    def _extract_efficiency_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract performance efficiency metrics.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with efficiency metrics
        """
        features = pd.DataFrame(index=data.index)
        usage_cols = self.config['usage_columns']
        
        # Time efficiency ratios
        exec_time_col = usage_cols.get('execution_time')
        queue_time_col = usage_cols.get('queue_time')
        comp_time_col = usage_cols.get('compilation_time')
        
        if exec_time_col in data.columns and queue_time_col in data.columns:
            exec_time = data[exec_time_col]
            queue_time = data[queue_time_col]
            
            # Queue to execution ratio
            features['queue_to_execution_ratio'] = np.where(
                exec_time > 0, queue_time / exec_time, 0
            )
            
            # Total time
            total_time = exec_time + queue_time
            features['total_time_ms'] = total_time
            
            # Execution efficiency (execution time / total time)
            features['execution_efficiency'] = np.where(
                total_time > 0, exec_time / total_time, 0
            )
        
        if exec_time_col in data.columns and comp_time_col in data.columns:
            exec_time = data[exec_time_col]
            comp_time = data[comp_time_col]
            
            # Compilation overhead ratio
            features['compilation_overhead_ratio'] = np.where(
                exec_time > 0, comp_time / exec_time, 0
            )
        
        # Data efficiency metrics
        bytes_scanned_col = usage_cols.get('bytes_scanned')
        rows_produced_col = usage_cols.get('rows_produced')
        
        if bytes_scanned_col in data.columns and rows_produced_col in data.columns:
            bytes_scanned = data[bytes_scanned_col]
            rows_produced = data[rows_produced_col]
            
            # Bytes per row
            features['bytes_per_row'] = np.where(
                rows_produced > 0, bytes_scanned / rows_produced, 0
            )
            
            # Log transformation for skewed distribution
            features['log_bytes_per_row'] = np.log1p(features['bytes_per_row'])
        
        if exec_time_col in data.columns and bytes_scanned_col in data.columns:
            exec_time = data[exec_time_col]
            bytes_scanned = data[bytes_scanned_col]
            
            # Throughput (bytes per millisecond)
            features['data_throughput'] = np.where(
                exec_time > 0, bytes_scanned / exec_time, 0
            )
            
            # Log transformation
            features['log_data_throughput'] = np.log1p(features['data_throughput'])
        
        if exec_time_col in data.columns and rows_produced_col in data.columns:
            exec_time = data[exec_time_col]
            rows_produced = data[rows_produced_col]
            
            # Row processing rate (rows per millisecond)
            features['row_processing_rate'] = np.where(
                exec_time > 0, rows_produced / exec_time, 0
            )
        
        return features
    
    def _extract_query_complexity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract query complexity and characteristics features.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with query complexity features
        """
        features = pd.DataFrame(index=data.index)
        usage_cols = self.config['usage_columns']
        query_cols = self.config['query_columns']
        
        # Query type features
        if query_cols.get('query_type') in data.columns:
            query_type = data[query_cols['query_type']]
            features['query_type'] = query_type.astype('category')
            
            # Query type indicators
            common_types = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP']
            for qtype in common_types:
                features[f'is_{qtype.lower()}_query'] = (
                    query_type.str.upper().str.contains(qtype, na=False)
                ).astype(int)
            
            # Complex query indicators
            features['is_ddl_query'] = (
                query_type.str.upper().str.contains('CREATE|DROP|ALTER', na=False)
            ).astype(int)
            
            features['is_dml_query'] = (
                query_type.str.upper().str.contains('INSERT|UPDATE|DELETE', na=False)
            ).astype(int)
        
        # Partitions scanned (complexity indicator)
        if usage_cols.get('partitions_scanned') in data.columns:
            partitions = data[usage_cols['partitions_scanned']]
            features['partitions_scanned'] = partitions
            features['log_partitions_scanned'] = np.log1p(partitions)
            
            # Partition complexity categories
            features['partition_complexity'] = pd.cut(
                partitions,
                bins=[0, 1, 10, 100, float('inf')],
                labels=['single', 'few', 'many', 'massive'],
                include_lowest=True
            ).astype('category')
        
        # Warehouse size as complexity indicator
        if usage_cols.get('warehouse_size') in data.columns:
            wh_size = data[usage_cols['warehouse_size']]
            features['warehouse_size'] = wh_size.astype('category')
            
            # Warehouse size encoding
            size_mapping = {
                'X-SMALL': 1, 'SMALL': 2, 'MEDIUM': 3, 'LARGE': 4,
                'X-LARGE': 5, '2X-LARGE': 6, '3X-LARGE': 7, '4X-LARGE': 8
            }
            features['warehouse_size_numeric'] = wh_size.map(size_mapping).fillna(0)
        
        # Create complexity score
        complexity_score = 0
        
        # Add execution time component
        if usage_cols.get('execution_time') in data.columns:
            exec_time = data[usage_cols['execution_time']]
            # Normalize to 0-1 scale using percentile-based scaling
            exec_time_norm = (exec_time - exec_time.quantile(0.1)) / (
                exec_time.quantile(0.9) - exec_time.quantile(0.1)
            )
            complexity_score += exec_time_norm.clip(0, 1) * 0.3
        
        # Add data volume component
        if usage_cols.get('bytes_scanned') in data.columns:
            bytes_scanned = data[usage_cols['bytes_scanned']]
            bytes_norm = (bytes_scanned - bytes_scanned.quantile(0.1)) / (
                bytes_scanned.quantile(0.9) - bytes_scanned.quantile(0.1)
            )
            complexity_score += bytes_norm.clip(0, 1) * 0.3
        
        # Add partition component
        if usage_cols.get('partitions_scanned') in data.columns:
            partitions = data[usage_cols['partitions_scanned']]
            part_norm = (partitions - partitions.quantile(0.1)) / (
                partitions.quantile(0.9) - partitions.quantile(0.1)
            )
            complexity_score += part_norm.clip(0, 1) * 0.2
        
        # Add warehouse size component
        if 'warehouse_size_numeric' in features.columns:
            wh_norm = (features['warehouse_size_numeric'] - 1) / 7  # Normalize to 0-1
            complexity_score += wh_norm * 0.2
        
        features['query_complexity_score'] = complexity_score.clip(0, 1)
        
        return features
    
    def _extract_resource_utilization_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract resource utilization and waste indicators.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with resource utilization features
        """
        features = pd.DataFrame(index=data.index)
        usage_cols = self.config['usage_columns']
        
        # Memory utilization proxies
        bytes_scanned_col = usage_cols.get('bytes_scanned')
        bytes_written_col = usage_cols.get('bytes_written')
        
        if bytes_scanned_col in data.columns and bytes_written_col in data.columns:
            bytes_scanned = data[bytes_scanned_col]
            bytes_written = data[bytes_written_col]
            
            # Write amplification ratio
            features['write_amplification'] = np.where(
                bytes_scanned > 0, bytes_written / bytes_scanned, 0
            )
            
            # Data transformation ratio
            features['data_transformation_ratio'] = np.where(
                bytes_scanned > 0, 
                (bytes_scanned - bytes_written) / bytes_scanned, 
                0
            )
        
        # Row-level efficiency
        rows_produced_col = usage_cols.get('rows_produced')
        rows_inserted_col = usage_cols.get('rows_inserted')
        rows_updated_col = usage_cols.get('rows_updated')
        rows_deleted_col = usage_cols.get('rows_deleted')
        
        total_modified_rows = 0
        row_modification_features = []
        
        if rows_inserted_col in data.columns:
            rows_inserted = data[rows_inserted_col]
            features['rows_inserted'] = rows_inserted
            total_modified_rows += rows_inserted
            row_modification_features.append('insert')
        
        if rows_updated_col in data.columns:
            rows_updated = data[rows_updated_col]
            features['rows_updated'] = rows_updated
            total_modified_rows += rows_updated
            row_modification_features.append('update')
        
        if rows_deleted_col in data.columns:
            rows_deleted = data[rows_deleted_col]
            features['rows_deleted'] = rows_deleted
            total_modified_rows += rows_deleted
            row_modification_features.append('delete')
        
        if len(row_modification_features) > 0:
            features['total_modified_rows'] = total_modified_rows
            
            # Row modification ratios
            if rows_produced_col in data.columns:
                rows_produced = data[rows_produced_col]
                features['modification_to_production_ratio'] = np.where(
                    rows_produced > 0, total_modified_rows / rows_produced, 0
                )
        
        # Compute utilization efficiency score
        utilization_score = 0
        components = 0
        
        # Execution efficiency component
        if 'execution_efficiency' in features.columns:
            utilization_score += features['execution_efficiency']
            components += 1
        
        # Data efficiency component
        if 'data_throughput' in features.columns:
            throughput = features['data_throughput']
            throughput_norm = (throughput - throughput.quantile(0.1)) / (
                throughput.quantile(0.9) - throughput.quantile(0.1)
            )
            utilization_score += throughput_norm.clip(0, 1)
            components += 1
        
        if components > 0:
            features['resource_utilization_score'] = utilization_score / components
        
        return features
    
    def _extract_user_behavior_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract user behavior pattern features.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with user behavior features
        """
        features = pd.DataFrame(index=data.index)
        user_cols = self.config['user_columns']
        
        # User activity indicators
        if user_cols.get('user_name') in data.columns:
            user_name = data[user_cols['user_name']]
            
            # User frequency (how often this user appears)
            user_counts = user_name.value_counts()
            features['user_query_frequency'] = user_name.map(user_counts)
            
            # User rank by activity
            user_ranks = user_counts.rank(method='dense', ascending=False)
            features['user_activity_rank'] = user_name.map(user_ranks)
            
            # Is high-activity user
            high_activity_threshold = user_counts.quantile(0.8)
            features['is_high_activity_user'] = (
                features['user_query_frequency'] >= high_activity_threshold
            ).astype(int)
            
            # User type categories
            features['user_activity_category'] = pd.cut(
                features['user_query_frequency'],
                bins=[0, 1, 10, 50, float('inf')],
                labels=['rare', 'occasional', 'regular', 'power'],
                include_lowest=True
            ).astype('category')
        
        # Role-based features
        if user_cols.get('role_name') in data.columns:
            role_name = data[user_cols['role_name']]
            features['role_name'] = role_name.astype('category')
            
            # Role frequency
            role_counts = role_name.value_counts()
            features['role_usage_frequency'] = role_name.map(role_counts)
            
            # Common role indicators
            common_roles = ['SYSADMIN', 'ACCOUNTADMIN', 'PUBLIC', 'ANALYST']
            for role in common_roles:
                features[f'is_{role.lower()}_role'] = (
                    role_name.str.upper().str.contains(role, na=False)
                ).astype(int)
        
        # Session-based features
        if user_cols.get('session_id') in data.columns:
            session_id = data[user_cols['session_id']]
            
            # Queries per session
            session_counts = session_id.value_counts()
            features['queries_per_session'] = session_id.map(session_counts)
            
            # Session activity level
            features['session_activity_category'] = pd.cut(
                features['queries_per_session'],
                bins=[0, 1, 5, 20, float('inf')],
                labels=['single', 'light', 'moderate', 'heavy'],
                include_lowest=True
            ).astype('category')
        
        return features
    
    def _extract_workload_analysis_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract workload analysis and pattern features.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with workload analysis features
        """
        features = pd.DataFrame(index=data.index)
        usage_cols = self.config['usage_columns']
        query_cols = self.config['query_columns']
        
        # Database and schema usage patterns
        if query_cols.get('database_name') in data.columns:
            database_name = data[query_cols['database_name']]
            
            # Database frequency
            db_counts = database_name.value_counts()
            features['database_usage_frequency'] = database_name.map(db_counts)
            
            # Is primary database
            most_used_db = db_counts.index[0] if len(db_counts) > 0 else None
            if most_used_db:
                features['is_primary_database'] = (
                    database_name == most_used_db
                ).astype(int)
        
        if query_cols.get('schema_name') in data.columns:
            schema_name = data[query_cols['schema_name']]
            
            # Schema frequency
            schema_counts = schema_name.value_counts()
            features['schema_usage_frequency'] = schema_name.map(schema_counts)
        
        # Warehouse usage patterns
        if query_cols.get('warehouse_name') in data.columns:
            warehouse_name = data[query_cols['warehouse_name']]
            
            # Warehouse frequency
            wh_counts = warehouse_name.value_counts()
            features['warehouse_usage_frequency'] = warehouse_name.map(wh_counts)
            
            # Warehouse diversity (number of different warehouses used)
            features['warehouse_diversity'] = len(wh_counts)
        
        # Workload intensity features
        time_col = self.config.get('time_column')
        if time_col in data.columns:
            time_series = pd.to_datetime(data[time_col])
            
            # Hourly query density
            hourly_counts = time_series.dt.floor('H').value_counts()
            features['hourly_query_density'] = time_series.dt.floor('H').map(hourly_counts)
            
            # Daily query density
            daily_counts = time_series.dt.date.value_counts()
            features['daily_query_density'] = time_series.dt.date.map(daily_counts)
        
        return features
    
    def _extract_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract statistical distribution features.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with statistical features
        """
        features = pd.DataFrame(index=data.index)
        usage_cols = self.config['usage_columns']
        percentiles = self.config.get('percentiles', [25, 50, 75, 90, 95])
        
        # Statistical features for key metrics
        key_metrics = ['execution_time', 'bytes_scanned', 'rows_produced']
        
        for metric in key_metrics:
            col_name = usage_cols.get(metric)
            if col_name in data.columns:
                values = data[col_name]
                
                # Percentile-based features
                for p in percentiles:
                    percentile_val = values.quantile(p/100)
                    features[f'{metric}_is_above_p{p}'] = (values > percentile_val).astype(int)
                
                # Z-score (standardized values)
                if len(values) > 1 and values.std() > 0:
                    z_scores = (values - values.mean()) / values.std()
                    features[f'{metric}_z_score'] = z_scores
                    
                    # Outlier indicators
                    outlier_threshold = self.config.get('outlier_threshold', 3.0)
                    features[f'{metric}_is_outlier'] = (
                        np.abs(z_scores) > outlier_threshold
                    ).astype(int)
                
                # Distribution shape indicators
                if len(values) >= self.config.get('min_observations', 10):
                    # Skewness and kurtosis
                    features[f'{metric}_skewness'] = stats.skew(values.dropna())
                    features[f'{metric}_kurtosis'] = stats.kurtosis(values.dropna())
        
        return features
    
    def _extract_comparative_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comparative features against historical averages.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with comparative features
        """
        features = pd.DataFrame(index=data.index)
        usage_cols = self.config['usage_columns']
        user_cols = self.config['user_columns']
        
        # User-specific comparisons
        if user_cols.get('user_name') in data.columns:
            user_name = data[user_cols['user_name']]
            
            # Compare against user's historical average
            for metric in ['execution_time', 'bytes_scanned']:
                col_name = usage_cols.get(metric)
                if col_name in data.columns:
                    values = data[col_name]
                    
                    # User averages
                    user_averages = values.groupby(user_name).mean()
                    features[f'{metric}_vs_user_avg_ratio'] = values / user_name.map(user_averages)
                    
                    # Above user average indicator
                    features[f'{metric}_above_user_avg'] = (
                        features[f'{metric}_vs_user_avg_ratio'] > 1
                    ).astype(int)
        
        # Global comparisons
        for metric in ['execution_time', 'bytes_scanned', 'rows_produced']:
            col_name = usage_cols.get(metric)
            if col_name in data.columns:
                values = data[col_name]
                global_median = values.median()
                global_mean = values.mean()
                
                # Ratio to global statistics
                features[f'{metric}_vs_global_median_ratio'] = values / global_median
                features[f'{metric}_vs_global_mean_ratio'] = values / global_mean
                
                # Above global average indicators
                features[f'{metric}_above_global_median'] = (values > global_median).astype(int)
                features[f'{metric}_above_global_mean'] = (values > global_mean).astype(int)
        
        return features
    
    def configure_for_snowflake_data(self):
        """
        Configure the generator for typical Snowflake analytics data.
        """
        snowflake_config = {
            'usage_columns': {
                'execution_time': 'EXECUTION_TIME_MS',
                'queue_time': 'QUEUE_TIME_MS', 
                'compilation_time': 'COMPILATION_TIME_MS',
                'bytes_scanned': 'BYTES_SCANNED',
                'bytes_written': 'BYTES_WRITTEN',
                'rows_produced': 'ROWS_PRODUCED',
                'rows_inserted': 'ROWS_INSERTED',
                'rows_updated': 'ROWS_UPDATED',
                'rows_deleted': 'ROWS_DELETED',
                'partitions_scanned': 'PARTITIONS_SCANNED',
                'warehouse_size': 'WAREHOUSE_SIZE'
            },
            'user_columns': {
                'user_name': 'USER_NAME',
                'role_name': 'ROLE_NAME',
                'session_id': 'SESSION_ID'
            },
            'query_columns': {
                'query_type': 'QUERY_TYPE',
                'query_tag': 'QUERY_TAG',
                'database_name': 'DATABASE_NAME',
                'schema_name': 'SCHEMA_NAME',
                'warehouse_name': 'WAREHOUSE_NAME'
            },
            'percentiles': [10, 25, 50, 75, 90, 95, 99],
            'outlier_threshold': 2.5,  # More lenient for analytics data
            'min_observations': 5
        }
        
        self.config.update(snowflake_config)
        logger.info("Configured UsageFeatureGenerator for Snowflake analytics data")
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all generated usage features.
        
        Returns:
            Dictionary mapping feature names to descriptions
        """
        descriptions = {
            # Basic metrics
            'execution_time_ms': 'Query execution time in milliseconds',
            'execution_time_seconds': 'Query execution time in seconds',
            'execution_time_minutes': 'Query execution time in minutes',
            'log_execution_time': 'Log-transformed execution time',
            'execution_time_category': 'Execution time category (very_fast to very_slow)',
            
            'queue_time_ms': 'Query queue time in milliseconds',
            'queue_time_seconds': 'Query queue time in seconds',
            'log_queue_time': 'Log-transformed queue time',
            'queue_time_category': 'Queue time category (no_queue to long_queue)',
            
            'bytes_scanned': 'Number of bytes scanned by the query',
            'gb_scanned': 'Gigabytes scanned by the query',
            'log_bytes_scanned': 'Log-transformed bytes scanned',
            'data_volume_category': 'Data volume category (small to very_large)',
            
            # Efficiency metrics
            'queue_to_execution_ratio': 'Ratio of queue time to execution time',
            'execution_efficiency': 'Execution time as fraction of total time',
            'compilation_overhead_ratio': 'Compilation time relative to execution time',
            'bytes_per_row': 'Average bytes per row produced',
            'data_throughput': 'Data processing throughput (bytes/ms)',
            'row_processing_rate': 'Row processing rate (rows/ms)',
            
            # Complexity features
            'query_complexity_score': 'Overall query complexity score (0-1)',
            'partition_complexity': 'Number of partitions scanned category',
            'warehouse_size_numeric': 'Warehouse size as numeric value',
            
            # User behavior
            'user_query_frequency': 'Number of queries by this user',
            'user_activity_rank': 'User rank by query volume',
            'is_high_activity_user': 'Whether user is in top 20% by activity',
            'user_activity_category': 'User activity level (rare to power)',
            
            # Statistical features
            'execution_time_z_score': 'Z-score of execution time',
            'execution_time_is_outlier': 'Whether execution time is an outlier',
            'bytes_scanned_vs_global_median_ratio': 'Bytes scanned vs global median ratio'
        }
        
        return descriptions
