"""
Data Aggregation Pipeline

This module provides the main AggregationPipeline class that orchestrates all data aggregation
operations including temporal, dimensional, cost, and usage aggregations for ML-ready datasets.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import json
from concurrent.futures import ThreadPoolExecutor
import warnings

from .temporal_aggregator import TemporalAggregator
from .dimensional_aggregator import DimensionalAggregator
from .cost_aggregator import CostAggregator
from .usage_aggregator import UsageAggregator

logger = logging.getLogger(__name__)


class AggregationPipeline:
    """
    Main data aggregation pipeline that orchestrates all aggregation operations.
    
    This class provides a unified interface for creating aggregated datasets from
    raw Snowflake analytics data, including temporal summaries, dimensional rollups,
    cost analysis, and usage metrics suitable for ML model training and analytics.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the AggregationPipeline with optional configuration.
        
        Args:
            config: Optional configuration dictionary with aggregation parameters
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'aggregation_types': [
                'temporal', 'dimensional', 'cost', 'usage'
            ],
            'temporal_levels': ['hourly', 'daily', 'weekly', 'monthly'],
            'dimensional_groupings': {
                'user_level': ['user_name'],
                'warehouse_level': ['warehouse_name', 'warehouse_size'],
                'database_level': ['database_name', 'schema_name'],
                'role_level': ['role_name'],
                'combined_level': ['user_name', 'warehouse_name', 'database_name']
            },
            'metrics_to_aggregate': {
                'cost_metrics': ['credits_used', 'credits_used_compute', 'credits_used_cloud_services'],
                'performance_metrics': ['execution_time_ms', 'queue_time_ms', 'compilation_time_ms'],
                'data_metrics': ['bytes_scanned', 'bytes_written', 'rows_produced', 'partitions_scanned'],
                'frequency_metrics': ['query_count', 'session_count', 'error_count']
            },
            'statistics_functions': [
                'sum', 'mean', 'median', 'std', 'min', 'max', 'count',
                'first', 'last', 'nunique'
            ],
            'percentiles': [25, 50, 75, 90, 95, 99],
            'time_column': 'START_TIME',
            'parallel_processing': True,
            'max_workers': 4,
            'output_format': 'pandas',  # 'pandas', 'parquet', 'csv'
            'include_metadata': True,
            'cache_results': True,
            'validate_output': True
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
        
        # Initialize aggregators
        self.temporal_aggregator = TemporalAggregator(self.config.get('temporal_config', {}))
        self.dimensional_aggregator = DimensionalAggregator(self.config.get('dimensional_config', {}))
        self.cost_aggregator = CostAggregator(self.config.get('cost_config', {}))
        self.usage_aggregator = UsageAggregator(self.config.get('usage_config', {}))
        
        # Aggregation statistics
        self.aggregation_stats = {
            'total_aggregations_generated': 0,
            'aggregations_by_type': {},
            'original_rows': 0,
            'aggregated_rows': 0,
            'compression_ratio': 0.0,
            'processing_time': 0.0,
            'memory_usage_mb': 0.0
        }
        
        # Cache for results
        self.cache = {}
    
    def aggregate_data(self, 
                      data: pd.DataFrame, 
                      aggregation_types: Optional[List[str]] = None,
                      output_level: str = 'detailed') -> Dict[str, pd.DataFrame]:
        """
        Perform comprehensive data aggregation across multiple dimensions and time periods.
        
        Args:
            data: Input DataFrame containing raw Snowflake data
            aggregation_types: Optional list of aggregation types to perform.
                             If None, performs all configured aggregations.
            output_level: Level of detail in output ('summary', 'detailed', 'complete')
        
        Returns:
            Dictionary mapping aggregation names to aggregated DataFrames
        """
        start_time = datetime.now()
        
        logger.info(f"Starting data aggregation on {len(data)} rows, {len(data.columns)} columns")
        
        # Initialize statistics
        self.aggregation_stats['original_rows'] = len(data)
        initial_memory = data.memory_usage(deep=True).sum()
        
        if aggregation_types is None:
            aggregation_types = self.config.get('aggregation_types', [])
        
        try:
            aggregated_data = {}
            
            # Validate required columns
            if not self._validate_required_columns(data):
                logger.warning("Some required columns missing, proceeding with available aggregations")
            
            # Prepare data for aggregation
            prepared_data = self._prepare_data_for_aggregation(data)
            
            # Perform aggregations based on configuration
            if self.config.get('parallel_processing', True):
                aggregated_data = self._perform_parallel_aggregations(
                    prepared_data, aggregation_types, output_level
                )
            else:
                aggregated_data = self._perform_sequential_aggregations(
                    prepared_data, aggregation_types, output_level
                )
            
            # Post-process aggregated results
            aggregated_data = self._post_process_aggregations(aggregated_data, output_level)
            
            # Calculate final statistics
            total_aggregated_rows = sum(len(df) for df in aggregated_data.values())
            final_memory = sum(df.memory_usage(deep=True).sum() for df in aggregated_data.values())
            
            self.aggregation_stats.update({
                'total_aggregations_generated': len(aggregated_data),
                'aggregated_rows': total_aggregated_rows,
                'compression_ratio': self.aggregation_stats['original_rows'] / max(total_aggregated_rows, 1),
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'memory_usage_mb': final_memory / 1024 / 1024
            })
            
            logger.info(f"Aggregation completed in {self.aggregation_stats['processing_time']:.2f} seconds")
            logger.info(f"Generated {len(aggregated_data)} aggregated datasets")
            logger.info(f"Compression ratio: {self.aggregation_stats['compression_ratio']:.2f}x")
            
            # Cache results if enabled
            if self.config.get('cache_results', True):
                self.cache.update(aggregated_data)
            
            return aggregated_data
            
        except Exception as e:
            logger.error(f"Error during data aggregation: {str(e)}")
            raise
    
    def _validate_required_columns(self, data: pd.DataFrame) -> bool:
        """
        Validate that required columns are present in the data.
        
        Args:
            data: Input DataFrame
        
        Returns:
            True if all required columns are present
        """
        required_columns = [
            self.config.get('time_column', 'START_TIME')
        ]
        
        # Add metrics columns
        metrics = self.config.get('metrics_to_aggregate', {})
        for metric_group in metrics.values():
            required_columns.extend(metric_group)
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            return False
        
        return True
    
    def _prepare_data_for_aggregation(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for aggregation by cleaning and enriching.
        
        Args:
            data: Input DataFrame
        
        Returns:
            Prepared DataFrame ready for aggregation
        """
        prepared_data = data.copy()
        time_col = self.config.get('time_column', 'START_TIME')
        
        # Ensure datetime format for time column
        if time_col in prepared_data.columns:
            prepared_data[time_col] = pd.to_datetime(prepared_data[time_col])
            
            # Add derived time columns for aggregation
            prepared_data['date'] = prepared_data[time_col].dt.date
            prepared_data['hour'] = prepared_data[time_col].dt.hour
            prepared_data['day_of_week'] = prepared_data[time_col].dt.dayofweek
            prepared_data['week'] = prepared_data[time_col].dt.isocalendar().week
            prepared_data['month'] = prepared_data[time_col].dt.month
            prepared_data['quarter'] = prepared_data[time_col].dt.quarter
            prepared_data['year'] = prepared_data[time_col].dt.year
        
        # Add query count column for frequency metrics
        prepared_data['query_count'] = 1
        
        # Handle missing values in key metrics
        metrics = self.config.get('metrics_to_aggregate', {})
        for metric_group in metrics.values():
            for metric in metric_group:
                if metric in prepared_data.columns:
                    prepared_data[metric] = prepared_data[metric].fillna(0)
        
        # Add session indicators if session_id exists
        if 'session_id' in prepared_data.columns:
            prepared_data['session_count'] = prepared_data.groupby('session_id').cumcount() == 0
            prepared_data['session_count'] = prepared_data['session_count'].astype(int)
        
        logger.info(f"Prepared data with {len(prepared_data.columns)} columns for aggregation")
        return prepared_data
    
    def _perform_parallel_aggregations(self, 
                                     data: pd.DataFrame, 
                                     aggregation_types: List[str],
                                     output_level: str) -> Dict[str, pd.DataFrame]:
        """
        Perform aggregations in parallel using ThreadPoolExecutor.
        
        Args:
            data: Prepared DataFrame
            aggregation_types: List of aggregation types to perform
            output_level: Level of detail in output
        
        Returns:
            Dictionary of aggregated DataFrames
        """
        aggregated_data = {}
        max_workers = self.config.get('max_workers', 4)
        
        # Define aggregation tasks
        aggregation_tasks = []
        
        if 'temporal' in aggregation_types:
            for level in self.config.get('temporal_levels', []):
                aggregation_tasks.append(('temporal', level, data))
        
        if 'dimensional' in aggregation_types:
            groupings = self.config.get('dimensional_groupings', {})
            for grouping_name, grouping_cols in groupings.items():
                if all(col in data.columns for col in grouping_cols):
                    aggregation_tasks.append(('dimensional', grouping_name, data))
        
        if 'cost' in aggregation_types:
            aggregation_tasks.append(('cost', 'cost_analysis', data))
        
        if 'usage' in aggregation_types:
            aggregation_tasks.append(('usage', 'usage_analysis', data))
        
        # Execute tasks in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {}
            
            for task_type, task_name, task_data in aggregation_tasks:
                future = executor.submit(self._execute_single_aggregation, 
                                       task_type, task_name, task_data, output_level)
                future_to_task[future] = (task_type, task_name)
            
            # Collect results
            for future in future_to_task:
                task_type, task_name = future_to_task[future]
                try:
                    result = future.result()
                    if result is not None and not result.empty:
                        key = f"{task_type}_{task_name}"
                        aggregated_data[key] = result
                        
                        # Update statistics
                        if task_type not in self.aggregation_stats['aggregations_by_type']:
                            self.aggregation_stats['aggregations_by_type'][task_type] = 0
                        self.aggregation_stats['aggregations_by_type'][task_type] += 1
                        
                except Exception as e:
                    logger.error(f"Error in {task_type}_{task_name} aggregation: {str(e)}")
        
        return aggregated_data
    
    def _perform_sequential_aggregations(self, 
                                       data: pd.DataFrame, 
                                       aggregation_types: List[str],
                                       output_level: str) -> Dict[str, pd.DataFrame]:
        """
        Perform aggregations sequentially.
        
        Args:
            data: Prepared DataFrame
            aggregation_types: List of aggregation types to perform
            output_level: Level of detail in output
        
        Returns:
            Dictionary of aggregated DataFrames
        """
        aggregated_data = {}
        
        # Temporal aggregations
        if 'temporal' in aggregation_types:
            logger.info("Performing temporal aggregations...")
            for level in self.config.get('temporal_levels', []):
                try:
                    result = self._execute_single_aggregation('temporal', level, data, output_level)
                    if result is not None and not result.empty:
                        aggregated_data[f'temporal_{level}'] = result
                except Exception as e:
                    logger.error(f"Error in temporal {level} aggregation: {str(e)}")
        
        # Dimensional aggregations
        if 'dimensional' in aggregation_types:
            logger.info("Performing dimensional aggregations...")
            groupings = self.config.get('dimensional_groupings', {})
            for grouping_name, grouping_cols in groupings.items():
                if all(col in data.columns for col in grouping_cols):
                    try:
                        result = self._execute_single_aggregation('dimensional', grouping_name, data, output_level)
                        if result is not None and not result.empty:
                            aggregated_data[f'dimensional_{grouping_name}'] = result
                    except Exception as e:
                        logger.error(f"Error in dimensional {grouping_name} aggregation: {str(e)}")
        
        # Cost aggregations
        if 'cost' in aggregation_types:
            logger.info("Performing cost aggregations...")
            try:
                result = self._execute_single_aggregation('cost', 'cost_analysis', data, output_level)
                if result is not None and not result.empty:
                    aggregated_data['cost_analysis'] = result
            except Exception as e:
                logger.error(f"Error in cost aggregation: {str(e)}")
        
        # Usage aggregations
        if 'usage' in aggregation_types:
            logger.info("Performing usage aggregations...")
            try:
                result = self._execute_single_aggregation('usage', 'usage_analysis', data, output_level)
                if result is not None and not result.empty:
                    aggregated_data['usage_analysis'] = result
            except Exception as e:
                logger.error(f"Error in usage aggregation: {str(e)}")
        
        return aggregated_data
    
    def _execute_single_aggregation(self, 
                                   aggregation_type: str, 
                                   aggregation_name: str, 
                                   data: pd.DataFrame,
                                   output_level: str) -> Optional[pd.DataFrame]:
        """
        Execute a single aggregation operation.
        
        Args:
            aggregation_type: Type of aggregation ('temporal', 'dimensional', 'cost', 'usage')
            aggregation_name: Name/level of the specific aggregation
            data: Input DataFrame
            output_level: Level of detail in output
        
        Returns:
            Aggregated DataFrame or None if error
        """
        try:
            if aggregation_type == 'temporal':
                return self.temporal_aggregator.aggregate_by_time_period(data, aggregation_name)
            
            elif aggregation_type == 'dimensional':
                grouping_cols = self.config['dimensional_groupings'].get(aggregation_name, [])
                return self.dimensional_aggregator.aggregate_by_dimensions(data, grouping_cols)
            
            elif aggregation_type == 'cost':
                return self.cost_aggregator.aggregate_cost_metrics(data)
            
            elif aggregation_type == 'usage':
                return self.usage_aggregator.aggregate_usage_metrics(data)
            
            else:
                logger.warning(f"Unknown aggregation type: {aggregation_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error executing {aggregation_type}_{aggregation_name}: {str(e)}")
            return None
    
    def _post_process_aggregations(self, 
                                 aggregated_data: Dict[str, pd.DataFrame],
                                 output_level: str) -> Dict[str, pd.DataFrame]:
        """
        Post-process aggregated results based on output level.
        
        Args:
            aggregated_data: Dictionary of aggregated DataFrames
            output_level: Level of detail in output
        
        Returns:
            Post-processed aggregated data
        """
        processed_data = {}
        
        for name, df in aggregated_data.items():
            if df.empty:
                continue
            
            processed_df = df.copy()
            
            # Add metadata columns if enabled
            if self.config.get('include_metadata', True):
                processed_df['aggregation_type'] = name
                processed_df['aggregation_timestamp'] = datetime.now()
                processed_df['row_count'] = len(df)
            
            # Filter columns based on output level
            if output_level == 'summary':
                # Keep only essential columns
                essential_cols = self._get_essential_columns(processed_df)
                processed_df = processed_df[essential_cols]
            
            elif output_level == 'detailed':
                # Keep most columns but exclude some technical details
                exclude_cols = [col for col in processed_df.columns 
                              if col.endswith('_internal') or col.startswith('_')]
                processed_df = processed_df.drop(columns=exclude_cols, errors='ignore')
            
            # 'complete' level keeps all columns
            
            # Validate output if enabled
            if self.config.get('validate_output', True):
                if self._validate_aggregated_data(processed_df):
                    processed_data[name] = processed_df
                else:
                    logger.warning(f"Validation failed for aggregation: {name}")
            else:
                processed_data[name] = processed_df
        
        return processed_data
    
    def _get_essential_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get essential columns for summary output level.
        
        Args:
            df: Aggregated DataFrame
        
        Returns:
            List of essential column names
        """
        essential_patterns = [
            'date', 'hour', 'user_name', 'warehouse_name', 'database_name',
            'credits_used', 'execution_time', 'bytes_scanned', 'query_count',
            '_sum', '_mean', '_count', '_total'
        ]
        
        essential_cols = []
        for col in df.columns:
            if any(pattern in col.lower() for pattern in essential_patterns):
                essential_cols.append(col)
        
        # Always include index columns
        if hasattr(df.index, 'names') and df.index.names:
            essential_cols.extend([name for name in df.index.names if name])
        
        return list(set(essential_cols))
    
    def _validate_aggregated_data(self, df: pd.DataFrame) -> bool:
        """
        Validate aggregated data for quality and consistency.
        
        Args:
            df: Aggregated DataFrame
        
        Returns:
            True if validation passes
        """
        try:
            # Check for empty dataframe
            if df.empty:
                logger.warning("Aggregated data is empty")
                return False
            
            # Check for all-null columns
            null_columns = df.columns[df.isnull().all()].tolist()
            if null_columns:
                logger.warning(f"Columns with all null values: {null_columns}")
            
            # Check for reasonable value ranges in key metrics
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if df[col].min() < 0 and 'time' not in col.lower():
                    logger.warning(f"Negative values found in {col}")
            
            # Check for reasonable aggregation results
            if 'query_count' in df.columns:
                if df['query_count'].min() < 0:
                    logger.warning("Negative query counts found")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            return False
    
    def get_aggregation_summary(self) -> Dict[str, Any]:
        """
        Get summary of the last aggregation operation.
        
        Returns:
            Dictionary with aggregation statistics and metadata
        """
        return {
            'statistics': self.aggregation_stats.copy(),
            'configuration': self.config.copy(),
            'available_aggregations': list(self.cache.keys()) if self.cache else [],
            'timestamp': datetime.now().isoformat()
        }
    
    def save_aggregated_data(self, 
                           aggregated_data: Dict[str, pd.DataFrame], 
                           output_path: str,
                           format: str = 'parquet') -> Dict[str, str]:
        """
        Save aggregated data to files.
        
        Args:
            aggregated_data: Dictionary of aggregated DataFrames
            output_path: Base output path
            format: Output format ('parquet', 'csv', 'json')
        
        Returns:
            Dictionary mapping aggregation names to file paths
        """
        saved_files = {}
        
        try:
            import os
            os.makedirs(output_path, exist_ok=True)
            
            for name, df in aggregated_data.items():
                if format.lower() == 'parquet':
                    file_path = os.path.join(output_path, f"{name}.parquet")
                    df.to_parquet(file_path)
                elif format.lower() == 'csv':
                    file_path = os.path.join(output_path, f"{name}.csv")
                    df.to_csv(file_path, index=False)
                elif format.lower() == 'json':
                    file_path = os.path.join(output_path, f"{name}.json")
                    df.to_json(file_path, orient='records', date_format='iso')
                
                saved_files[name] = file_path
                logger.info(f"Saved {name} aggregation to {file_path}")
            
            # Save metadata
            metadata_path = os.path.join(output_path, "aggregation_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(self.get_aggregation_summary(), f, indent=2, default=str)
            
            return saved_files
            
        except Exception as e:
            logger.error(f"Error saving aggregated data: {str(e)}")
            raise
    
    def configure_for_snowflake_data(self):
        """
        Configure the pipeline for typical Snowflake analytics data.
        """
        snowflake_config = {
            'temporal_levels': ['hourly', 'daily', 'weekly', 'monthly', 'quarterly'],
            'dimensional_groupings': {
                'user_analysis': ['USER_NAME', 'ROLE_NAME'],
                'warehouse_analysis': ['WAREHOUSE_NAME', 'WAREHOUSE_SIZE'],
                'database_analysis': ['DATABASE_NAME', 'SCHEMA_NAME'],
                'query_analysis': ['QUERY_TYPE', 'USER_NAME'],
                'cost_center_analysis': ['USER_NAME', 'WAREHOUSE_NAME', 'DATABASE_NAME']
            },
            'metrics_to_aggregate': {
                'cost_metrics': ['CREDITS_USED', 'CREDITS_USED_COMPUTE', 'CREDITS_USED_CLOUD_SERVICES'],
                'performance_metrics': ['EXECUTION_TIME_MS', 'QUEUE_TIME_MS', 'COMPILATION_TIME_MS'],
                'data_metrics': ['BYTES_SCANNED', 'BYTES_WRITTEN', 'ROWS_PRODUCED', 'PARTITIONS_SCANNED'],
                'frequency_metrics': ['query_count']
            },
            'time_column': 'START_TIME',
            'percentiles': [50, 75, 90, 95, 99]
        }
        
        self.config.update(snowflake_config)
        
        # Configure sub-aggregators
        self.temporal_aggregator.configure_for_snowflake_data()
        self.dimensional_aggregator.configure_for_snowflake_data()
        self.cost_aggregator.configure_for_snowflake_data()
        self.usage_aggregator.configure_for_snowflake_data()
        
        logger.info("Configured aggregation pipeline for Snowflake analytics data")
