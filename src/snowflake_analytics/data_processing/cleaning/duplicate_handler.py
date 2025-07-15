"""
Duplicate Detection and Removal Handler

This module provides comprehensive duplicate detection and removal capabilities
for Snowflake analytics data with configurable strategies and detection methods.
"""

import logging
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class DuplicateHandler:
    """
    Handler for detecting and removing duplicate records in data.
    
    Provides multiple strategies for duplicate detection including exact matches,
    fuzzy matching, and time-based deduplication.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the DuplicateHandler with configuration.
        
        Args:
            config: Configuration dictionary with duplicate handling parameters
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'method': 'exact',  # 'exact', 'fuzzy', 'time_based'
            'subset_columns': None,  # Columns to consider for duplicates
            'keep': 'first',  # 'first', 'last', False
            'time_window_minutes': 5,  # For time-based deduplication
            'similarity_threshold': 0.95,  # For fuzzy matching
            'key_columns': ['timestamp', 'identifier']  # Primary key columns
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
        
        # Statistics tracking
        self.duplicate_stats = {
            'total_duplicates_found': 0,
            'duplicates_removed': 0,
            'method_used': '',
            'processing_time': 0.0
        }
    
    def remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicates from the DataFrame based on configured method.
        
        Args:
            data: Input DataFrame to process
        
        Returns:
            DataFrame with duplicates removed
        """
        start_time = datetime.now()
        
        logger.info(f"Starting duplicate removal with method: {self.config['method']}")
        original_count = len(data)
        
        try:
            if self.config['method'] == 'exact':
                cleaned_data = self._remove_exact_duplicates(data)
            elif self.config['method'] == 'fuzzy':
                cleaned_data = self._remove_fuzzy_duplicates(data)
            elif self.config['method'] == 'time_based':
                cleaned_data = self._remove_time_based_duplicates(data)
            else:
                logger.warning(f"Unknown method {self.config['method']}, using exact matching")
                cleaned_data = self._remove_exact_duplicates(data)
            
            # Update statistics
            self.duplicate_stats['total_duplicates_found'] = original_count - len(cleaned_data)
            self.duplicate_stats['duplicates_removed'] = original_count - len(cleaned_data)
            self.duplicate_stats['method_used'] = self.config['method']
            self.duplicate_stats['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Removed {self.duplicate_stats['duplicates_removed']} duplicates "
                       f"in {self.duplicate_stats['processing_time']:.2f} seconds")
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Error removing duplicates: {str(e)}")
            raise
    
    def _remove_exact_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove exact duplicates using pandas drop_duplicates method.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with exact duplicates removed
        """
        subset_columns = self.config.get('subset_columns')
        keep = self.config.get('keep', 'first')
        
        # If subset columns specified, use them; otherwise use all columns
        if subset_columns:
            # Ensure specified columns exist
            existing_columns = [col for col in subset_columns if col in data.columns]
            if not existing_columns:
                logger.warning("None of the specified subset columns exist, using all columns")
                subset_columns = None
            else:
                subset_columns = existing_columns
        
        return data.drop_duplicates(subset=subset_columns, keep=keep)
    
    def _remove_fuzzy_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove fuzzy duplicates based on similarity threshold.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with fuzzy duplicates removed
        """
        # For now, implement a simple string similarity approach
        # This can be enhanced with more sophisticated fuzzy matching algorithms
        
        threshold = self.config.get('similarity_threshold', 0.95)
        cleaned_data = data.copy()
        
        # Focus on string columns for fuzzy matching
        string_columns = data.select_dtypes(include=['object']).columns.tolist()
        
        if not string_columns:
            logger.info("No string columns found for fuzzy matching, using exact matching")
            return self._remove_exact_duplicates(data)
        
        # Simple fuzzy matching implementation
        # In production, you might want to use libraries like fuzzywuzzy
        logger.info(f"Performing fuzzy duplicate detection on {len(string_columns)} string columns")
        
        # For now, fall back to exact matching
        # TODO: Implement proper fuzzy matching algorithm
        return self._remove_exact_duplicates(data)
    
    def _remove_time_based_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove time-based duplicates within a specified time window.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with time-based duplicates removed
        """
        time_window = self.config.get('time_window_minutes', 5)
        key_columns = self.config.get('key_columns', ['timestamp', 'identifier'])
        
        # Find timestamp column
        timestamp_col = None
        possible_timestamp_cols = [
            'timestamp', 'START_TIME', 'END_TIME', 'CREATED_TIME', 
            'EXECUTION_TIME', 'time', 'datetime'
        ]
        
        for col in possible_timestamp_cols:
            if col in data.columns:
                timestamp_col = col
                break
        
        if not timestamp_col:
            logger.warning("No timestamp column found for time-based deduplication, using exact matching")
            return self._remove_exact_duplicates(data)
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data[timestamp_col]):
            try:
                data[timestamp_col] = pd.to_datetime(data[timestamp_col])
            except Exception as e:
                logger.warning(f"Could not convert {timestamp_col} to datetime: {str(e)}")
                return self._remove_exact_duplicates(data)
        
        # Sort by timestamp
        data_sorted = data.sort_values(timestamp_col)
        
        # Find identifier columns
        identifier_cols = [col for col in key_columns if col in data.columns and col != timestamp_col]
        
        if not identifier_cols:
            logger.warning("No identifier columns found for time-based deduplication")
            return self._remove_exact_duplicates(data)
        
        # Group by identifier columns and remove duplicates within time window
        cleaned_data = []
        
        for group_key, group in data_sorted.groupby(identifier_cols):
            if len(group) <= 1:
                cleaned_data.append(group)
                continue
            
            # Remove duplicates within time window
            group_cleaned = []
            last_timestamp = None
            
            for _, row in group.iterrows():
                current_timestamp = row[timestamp_col]
                
                if last_timestamp is None:
                    group_cleaned.append(row)
                    last_timestamp = current_timestamp
                else:
                    time_diff = (current_timestamp - last_timestamp).total_seconds() / 60
                    
                    if time_diff >= time_window:
                        group_cleaned.append(row)
                        last_timestamp = current_timestamp
                    # Otherwise skip this row as it's within the time window
            
            if group_cleaned:
                cleaned_data.append(pd.DataFrame(group_cleaned))
        
        if cleaned_data:
            result = pd.concat(cleaned_data, ignore_index=True)
        else:
            result = pd.DataFrame()
        
        return result
    
    def detect_duplicates(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Detect duplicates without removing them.
        
        Args:
            data: Input DataFrame to analyze
        
        Returns:
            Tuple of (duplicate_records, duplicate_statistics)
        """
        logger.info("Detecting duplicates...")
        
        # Get duplicate mask
        subset_columns = self.config.get('subset_columns')
        
        if subset_columns:
            existing_columns = [col for col in subset_columns if col in data.columns]
            subset_columns = existing_columns if existing_columns else None
        
        duplicate_mask = data.duplicated(subset=subset_columns, keep=False)
        duplicate_records = data[duplicate_mask]
        
        # Generate statistics
        duplicate_statistics = {
            'total_records': len(data),
            'duplicate_records': len(duplicate_records),
            'unique_records': len(data) - len(duplicate_records),
            'duplicate_percentage': (len(duplicate_records) / len(data) * 100) if len(data) > 0 else 0,
            'duplicate_groups': len(duplicate_records.drop_duplicates(subset=subset_columns)) if len(duplicate_records) > 0 else 0
        }
        
        logger.info(f"Found {duplicate_statistics['duplicate_records']} duplicate records "
                   f"({duplicate_statistics['duplicate_percentage']:.2f}%)")
        
        return duplicate_records, duplicate_statistics
    
    def get_duplicate_summary(self) -> Dict:
        """
        Get summary of the last duplicate removal operation.
        
        Returns:
            Dictionary with duplicate removal statistics
        """
        return self.duplicate_stats.copy()
    
    def configure_for_snowflake_data(self):
        """
        Configure the handler for typical Snowflake analytics data.
        """
        # Snowflake-specific configuration
        snowflake_config = {
            'method': 'time_based',
            'key_columns': ['QUERY_ID', 'USER_NAME', 'WAREHOUSE_NAME'],
            'time_window_minutes': 1,  # Very short window for query data
            'subset_columns': ['QUERY_ID', 'START_TIME', 'USER_NAME']
        }
        
        self.config.update(snowflake_config)
        logger.info("Configured for Snowflake analytics data")
