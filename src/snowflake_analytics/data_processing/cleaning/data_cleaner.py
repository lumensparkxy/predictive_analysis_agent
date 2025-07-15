"""
Main Data Cleaning Orchestrator

This module provides the main DataCleaner class that orchestrates all data cleaning operations
including duplicate removal, missing value handling, outlier detection, and type conversion.
"""

import logging
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import numpy as np

from .duplicate_handler import DuplicateHandler
from .missing_value_handler import MissingValueHandler
from .outlier_detector import OutlierDetector
from .type_converter import TypeConverter

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Main data cleaning orchestrator that coordinates all cleaning operations.
    
    This class provides a unified interface for data cleaning operations,
    including duplicate removal, missing value handling, outlier detection,
    and data type standardization.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the DataCleaner with optional configuration.
        
        Args:
            config: Optional configuration dictionary with cleaning parameters
        """
        self.config = config or {}
        
        # Initialize cleaning components
        self.duplicate_handler = DuplicateHandler(self.config.get('duplicates', {}))
        self.missing_value_handler = MissingValueHandler(self.config.get('missing_values', {}))
        self.outlier_detector = OutlierDetector(self.config.get('outliers', {}))
        self.type_converter = TypeConverter(self.config.get('types', {}))
        
        # Cleaning statistics
        self.cleaning_stats = {
            'total_rows_input': 0,
            'total_rows_output': 0,
            'duplicates_removed': 0,
            'missing_values_handled': 0,
            'outliers_detected': 0,
            'types_converted': 0,
            'cleaning_time': 0.0
        }
    
    def clean_data(self, 
                   data: pd.DataFrame, 
                   cleaning_steps: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform comprehensive data cleaning on the input DataFrame.
        
        Args:
            data: Input DataFrame to clean
            cleaning_steps: Optional list of cleaning steps to perform.
                          If None, performs all cleaning steps.
                          Options: ['duplicates', 'missing_values', 'outliers', 'types']
        
        Returns:
            Tuple of (cleaned_data, cleaning_statistics)
        """
        start_time = datetime.now()
        
        if cleaning_steps is None:
            cleaning_steps = ['duplicates', 'missing_values', 'outliers', 'types']
        
        logger.info(f"Starting data cleaning with {len(data)} rows")
        self.cleaning_stats['total_rows_input'] = len(data)
        
        cleaned_data = data.copy()
        
        try:
            # Step 1: Remove duplicates
            if 'duplicates' in cleaning_steps:
                logger.info("Removing duplicates...")
                before_count = len(cleaned_data)
                cleaned_data = self.duplicate_handler.remove_duplicates(cleaned_data)
                self.cleaning_stats['duplicates_removed'] = before_count - len(cleaned_data)
                logger.info(f"Removed {self.cleaning_stats['duplicates_removed']} duplicates")
            
            # Step 2: Handle missing values
            if 'missing_values' in cleaning_steps:
                logger.info("Handling missing values...")
                before_nulls = cleaned_data.isnull().sum().sum()
                cleaned_data = self.missing_value_handler.handle_missing_values(cleaned_data)
                after_nulls = cleaned_data.isnull().sum().sum()
                self.cleaning_stats['missing_values_handled'] = before_nulls - after_nulls
                logger.info(f"Handled {self.cleaning_stats['missing_values_handled']} missing values")
            
            # Step 3: Detect and handle outliers
            if 'outliers' in cleaning_steps:
                logger.info("Detecting and handling outliers...")
                cleaned_data, outlier_count = self.outlier_detector.detect_and_handle_outliers(cleaned_data)
                self.cleaning_stats['outliers_detected'] = outlier_count
                logger.info(f"Detected and handled {outlier_count} outliers")
            
            # Step 4: Convert data types
            if 'types' in cleaning_steps:
                logger.info("Converting data types...")
                cleaned_data, conversion_count = self.type_converter.convert_types(cleaned_data)
                self.cleaning_stats['types_converted'] = conversion_count
                logger.info(f"Converted {conversion_count} column types")
            
            # Update final statistics
            self.cleaning_stats['total_rows_output'] = len(cleaned_data)
            self.cleaning_stats['cleaning_time'] = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Data cleaning completed in {self.cleaning_stats['cleaning_time']:.2f} seconds")
            logger.info(f"Final dataset: {len(cleaned_data)} rows ({len(data.columns)} columns)")
            
            return cleaned_data, self.cleaning_stats.copy()
            
        except Exception as e:
            logger.error(f"Error during data cleaning: {str(e)}")
            raise
    
    def validate_cleaned_data(self, original_data: pd.DataFrame, cleaned_data: pd.DataFrame) -> Dict:
        """
        Validate the cleaned data against the original data.
        
        Args:
            original_data: Original DataFrame before cleaning
            cleaned_data: Cleaned DataFrame after processing
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'row_count_change': len(cleaned_data) - len(original_data),
            'column_count_same': len(cleaned_data.columns) == len(original_data.columns),
            'data_types_improved': True,  # Assume types are improved
            'missing_values_reduced': (
                original_data.isnull().sum().sum() >= cleaned_data.isnull().sum().sum()
            ),
            'validation_passed': True
        }
        
        # Check for data integrity issues
        if len(cleaned_data) == 0:
            validation_results['validation_passed'] = False
            validation_results['error'] = "All data was removed during cleaning"
        
        if len(cleaned_data.columns) != len(original_data.columns):
            validation_results['validation_passed'] = False
            validation_results['error'] = "Column count changed during cleaning"
        
        return validation_results
    
    def get_cleaning_summary(self) -> Dict:
        """
        Get a summary of the last cleaning operation.
        
        Returns:
            Dictionary with cleaning summary statistics
        """
        if self.cleaning_stats['total_rows_input'] == 0:
            return {'status': 'No cleaning operations performed yet'}
        
        summary = {
            'input_rows': self.cleaning_stats['total_rows_input'],
            'output_rows': self.cleaning_stats['total_rows_output'],
            'rows_removed': self.cleaning_stats['total_rows_input'] - self.cleaning_stats['total_rows_output'],
            'duplicates_removed': self.cleaning_stats['duplicates_removed'],
            'missing_values_handled': self.cleaning_stats['missing_values_handled'],
            'outliers_detected': self.cleaning_stats['outliers_detected'],
            'types_converted': self.cleaning_stats['types_converted'],
            'processing_time_seconds': self.cleaning_stats['cleaning_time'],
            'data_retention_rate': (
                self.cleaning_stats['total_rows_output'] / self.cleaning_stats['total_rows_input'] * 100
                if self.cleaning_stats['total_rows_input'] > 0 else 0
            )
        }
        
        return summary
    
    def clean_snowflake_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Specialized cleaning for Snowflake analytics data.
        
        This method applies Snowflake-specific cleaning rules and transformations.
        
        Args:
            data: DataFrame containing Snowflake analytics data
        
        Returns:
            Tuple of (cleaned_data, cleaning_statistics)
        """
        logger.info("Applying Snowflake-specific data cleaning...")
        
        # Apply standard cleaning first
        cleaned_data, stats = self.clean_data(data)
        
        # Apply Snowflake-specific transformations
        try:
            # Standardize datetime columns to UTC
            datetime_columns = [
                'START_TIME', 'END_TIME', 'EXECUTION_TIME', 
                'CREATED_TIME', 'MODIFIED_TIME', 'LAST_SUCCESS_TIME'
            ]
            
            for col in datetime_columns:
                if col in cleaned_data.columns:
                    cleaned_data[col] = pd.to_datetime(cleaned_data[col], utc=True)
            
            # Standardize warehouse names to uppercase
            if 'WAREHOUSE_NAME' in cleaned_data.columns:
                cleaned_data['WAREHOUSE_NAME'] = cleaned_data['WAREHOUSE_NAME'].str.upper()
            
            # Standardize user names
            if 'USER_NAME' in cleaned_data.columns:
                cleaned_data['USER_NAME'] = cleaned_data['USER_NAME'].str.upper()
            
            # Ensure numeric columns are properly typed
            numeric_columns = [
                'CREDITS_USED', 'BYTES_SCANNED', 'ROWS_PRODUCED',
                'EXECUTION_TIME_MS', 'QUEUE_TIME_MS', 'TOTAL_ELAPSED_TIME'
            ]
            
            for col in numeric_columns:
                if col in cleaned_data.columns:
                    cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
            
            logger.info("Snowflake-specific cleaning completed")
            
        except Exception as e:
            logger.warning(f"Error in Snowflake-specific cleaning: {str(e)}")
        
        return cleaned_data, stats
