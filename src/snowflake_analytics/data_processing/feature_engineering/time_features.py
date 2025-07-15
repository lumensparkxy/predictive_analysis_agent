"""
Time-based Feature Generator

This module provides comprehensive time-based feature engineering capabilities
for extracting temporal patterns from Snowflake analytics data.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import calendar

logger = logging.getLogger(__name__)


class TimeFeatureGenerator:
    """
    Generates comprehensive time-based features from datetime columns.
    
    This class extracts various temporal patterns including cyclical features,
    business time indicators, seasonality patterns, and time-based aggregations
    specifically designed for analytics workload patterns.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the TimeFeatureGenerator with optional configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'primary_time_column': 'START_TIME',
            'secondary_time_columns': ['END_TIME'],
            'business_hours': (9, 17),  # 9 AM to 5 PM
            'business_days': [0, 1, 2, 3, 4],  # Monday to Friday (0=Monday)
            'timezone': 'UTC',
            'extract_cyclical': True,
            'extract_business_indicators': True,
            'extract_seasonality': True,
            'extract_relative_features': True,
            'custom_time_periods': {
                'peak_hours': [9, 10, 11, 14, 15, 16],
                'off_peak_hours': [0, 1, 2, 3, 4, 5, 22, 23],
                'quarter_end_months': [3, 6, 9, 12],
                'holiday_months': [11, 12, 1]  # Holiday season
            }
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
    
    def generate_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive time-based features from the input DataFrame.
        
        Args:
            data: Input DataFrame containing datetime columns
        
        Returns:
            DataFrame with generated time features
        """
        logger.info("Generating time-based features...")
        
        feature_df = pd.DataFrame(index=data.index)
        primary_time_col = self.config['primary_time_column']
        
        if primary_time_col not in data.columns:
            logger.warning(f"Primary time column {primary_time_col} not found in data")
            return feature_df
        
        # Ensure datetime format
        time_series = pd.to_datetime(data[primary_time_col])
        
        # Basic time components
        feature_df.update(self._extract_basic_time_components(time_series))
        
        # Cyclical features
        if self.config.get('extract_cyclical', True):
            feature_df.update(self._extract_cyclical_features(time_series))
        
        # Business time indicators
        if self.config.get('extract_business_indicators', True):
            feature_df.update(self._extract_business_indicators(time_series))
        
        # Seasonality features
        if self.config.get('extract_seasonality', True):
            feature_df.update(self._extract_seasonality_features(time_series))
        
        # Relative time features
        if self.config.get('extract_relative_features', True):
            feature_df.update(self._extract_relative_time_features(time_series))
        
        # Custom time period indicators
        feature_df.update(self._extract_custom_time_indicators(time_series))
        
        # Duration features if secondary time columns exist
        for sec_col in self.config.get('secondary_time_columns', []):
            if sec_col in data.columns:
                feature_df.update(self._extract_duration_features(
                    data[primary_time_col], data[sec_col], sec_col
                ))
        
        logger.info(f"Generated {len(feature_df.columns)} time-based features")
        return feature_df
    
    def _extract_basic_time_components(self, time_series: pd.Series) -> pd.DataFrame:
        """
        Extract basic time components (hour, day, month, etc.).
        
        Args:
            time_series: Datetime series
        
        Returns:
            DataFrame with basic time components
        """
        features = pd.DataFrame(index=time_series.index)
        
        # Basic temporal components
        features['hour_of_day'] = time_series.dt.hour
        features['day_of_week'] = time_series.dt.dayofweek  # 0=Monday
        features['day_of_month'] = time_series.dt.day
        features['day_of_year'] = time_series.dt.dayofyear
        features['week_of_year'] = time_series.dt.isocalendar().week
        features['month_of_year'] = time_series.dt.month
        features['quarter_of_year'] = time_series.dt.quarter
        features['year'] = time_series.dt.year
        
        # Time of day categories
        features['time_of_day'] = pd.cut(
            time_series.dt.hour,
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        ).astype('category')
        
        return features
    
    def _extract_cyclical_features(self, time_series: pd.Series) -> pd.DataFrame:
        """
        Extract cyclical features using sine and cosine transformations.
        
        Args:
            time_series: Datetime series
        
        Returns:
            DataFrame with cyclical features
        """
        features = pd.DataFrame(index=time_series.index)
        
        # Hour cyclical features (24-hour cycle)
        features['hour_sin'] = np.sin(2 * np.pi * time_series.dt.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * time_series.dt.hour / 24)
        
        # Day of week cyclical features (7-day cycle)
        features['day_of_week_sin'] = np.sin(2 * np.pi * time_series.dt.dayofweek / 7)
        features['day_of_week_cos'] = np.cos(2 * np.pi * time_series.dt.dayofweek / 7)
        
        # Day of month cyclical features (approximately 30-day cycle)
        features['day_of_month_sin'] = np.sin(2 * np.pi * time_series.dt.day / 30.5)
        features['day_of_month_cos'] = np.cos(2 * np.pi * time_series.dt.day / 30.5)
        
        # Month cyclical features (12-month cycle)
        features['month_sin'] = np.sin(2 * np.pi * time_series.dt.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * time_series.dt.month / 12)
        
        # Quarter cyclical features (4-quarter cycle)
        features['quarter_sin'] = np.sin(2 * np.pi * time_series.dt.quarter / 4)
        features['quarter_cos'] = np.cos(2 * np.pi * time_series.dt.quarter / 4)
        
        return features
    
    def _extract_business_indicators(self, time_series: pd.Series) -> pd.DataFrame:
        """
        Extract business time indicators.
        
        Args:
            time_series: Datetime series
        
        Returns:
            DataFrame with business time indicators
        """
        features = pd.DataFrame(index=time_series.index)
        
        business_hours = self.config.get('business_hours', (9, 17))
        business_days = self.config.get('business_days', [0, 1, 2, 3, 4])
        
        # Business day indicator
        features['is_business_day'] = time_series.dt.dayofweek.isin(business_days).astype(int)
        
        # Weekend indicator
        features['is_weekend'] = (~time_series.dt.dayofweek.isin(business_days)).astype(int)
        
        # Business hours indicator
        features['is_business_hours'] = (
            (time_series.dt.hour >= business_hours[0]) & 
            (time_series.dt.hour < business_hours[1])
        ).astype(int)
        
        # Business time (both business day and business hours)
        features['is_business_time'] = (
            features['is_business_day'] & features['is_business_hours']
        ).astype(int)
        
        # After hours indicator
        features['is_after_hours'] = (
            features['is_business_day'] & (~features['is_business_hours'])
        ).astype(int)
        
        # Weekend or after hours
        features['is_non_business_time'] = (
            features['is_weekend'] | features['is_after_hours']
        ).astype(int)
        
        return features
    
    def _extract_seasonality_features(self, time_series: pd.Series) -> pd.DataFrame:
        """
        Extract seasonality and calendar-based features.
        
        Args:
            time_series: Datetime series
        
        Returns:
            DataFrame with seasonality features
        """
        features = pd.DataFrame(index=time_series.index)
        
        # Season indicators
        def get_season(month):
            if month in [12, 1, 2]:
                return 'winter'
            elif month in [3, 4, 5]:
                return 'spring'
            elif month in [6, 7, 8]:
                return 'summer'
            else:
                return 'fall'
        
        features['season'] = time_series.dt.month.apply(get_season).astype('category')
        
        # Month name
        features['month_name'] = time_series.dt.month_name().astype('category')
        
        # Day name
        features['day_name'] = time_series.dt.day_name().astype('category')
        
        # Is month start/end
        features['is_month_start'] = time_series.dt.is_month_start.astype(int)
        features['is_month_end'] = time_series.dt.is_month_end.astype(int)
        
        # Is quarter start/end
        features['is_quarter_start'] = time_series.dt.is_quarter_start.astype(int)
        features['is_quarter_end'] = time_series.dt.is_quarter_end.astype(int)
        
        # Is year start/end
        features['is_year_start'] = time_series.dt.is_year_start.astype(int)
        features['is_year_end'] = time_series.dt.is_year_end.astype(int)
        
        # Days in month
        features['days_in_month'] = time_series.dt.days_in_month
        
        # Is leap year
        features['is_leap_year'] = time_series.dt.is_leap_year.astype(int)
        
        return features
    
    def _extract_relative_time_features(self, time_series: pd.Series) -> pd.DataFrame:
        """
        Extract relative time features (time since/until events).
        
        Args:
            time_series: Datetime series
        
        Returns:
            DataFrame with relative time features
        """
        features = pd.DataFrame(index=time_series.index)
        
        if len(time_series) == 0:
            return features
        
        # Time since start of dataset
        dataset_start = time_series.min()
        features['hours_since_dataset_start'] = (time_series - dataset_start).dt.total_seconds() / 3600
        
        # Time until end of dataset
        dataset_end = time_series.max()
        features['hours_until_dataset_end'] = (dataset_end - time_series).dt.total_seconds() / 3600
        
        # Days since start of current month
        month_start = time_series.dt.to_period('M').dt.start_time
        features['days_since_month_start'] = (time_series - month_start).dt.days
        
        # Days until end of current month
        month_end = time_series.dt.to_period('M').dt.end_time
        features['days_until_month_end'] = (month_end - time_series).dt.days
        
        # Days since start of current quarter
        quarter_start = time_series.dt.to_period('Q').dt.start_time
        features['days_since_quarter_start'] = (time_series - quarter_start).dt.days
        
        # Days until end of current quarter
        quarter_end = time_series.dt.to_period('Q').dt.end_time
        features['days_until_quarter_end'] = (quarter_end - time_series).dt.days
        
        # Days since start of current year
        year_start = time_series.dt.to_period('Y').dt.start_time
        features['days_since_year_start'] = (time_series - year_start).dt.days
        
        return features
    
    def _extract_custom_time_indicators(self, time_series: pd.Series) -> pd.DataFrame:
        """
        Extract custom time period indicators based on configuration.
        
        Args:
            time_series: Datetime series
        
        Returns:
            DataFrame with custom time indicators
        """
        features = pd.DataFrame(index=time_series.index)
        
        custom_periods = self.config.get('custom_time_periods', {})
        
        # Peak hours indicator
        peak_hours = custom_periods.get('peak_hours', [])
        if peak_hours:
            features['is_peak_hours'] = time_series.dt.hour.isin(peak_hours).astype(int)
        
        # Off-peak hours indicator
        off_peak_hours = custom_periods.get('off_peak_hours', [])
        if off_peak_hours:
            features['is_off_peak_hours'] = time_series.dt.hour.isin(off_peak_hours).astype(int)
        
        # Quarter-end months indicator
        quarter_end_months = custom_periods.get('quarter_end_months', [])
        if quarter_end_months:
            features['is_quarter_end_month'] = time_series.dt.month.isin(quarter_end_months).astype(int)
        
        # Holiday months indicator
        holiday_months = custom_periods.get('holiday_months', [])
        if holiday_months:
            features['is_holiday_month'] = time_series.dt.month.isin(holiday_months).astype(int)
        
        # Rush hours (typical commute times)
        features['is_morning_rush'] = (
            (time_series.dt.hour >= 7) & (time_series.dt.hour <= 9)
        ).astype(int)
        
        features['is_evening_rush'] = (
            (time_series.dt.hour >= 17) & (time_series.dt.hour <= 19)
        ).astype(int)
        
        # Lunch hour
        features['is_lunch_hour'] = (
            (time_series.dt.hour >= 12) & (time_series.dt.hour <= 13)
        ).astype(int)
        
        return features
    
    def _extract_duration_features(self, 
                                 start_time: pd.Series, 
                                 end_time: pd.Series, 
                                 suffix: str) -> pd.DataFrame:
        """
        Extract duration-based features from start and end times.
        
        Args:
            start_time: Start time series
            end_time: End time series
            suffix: Suffix for feature names
        
        Returns:
            DataFrame with duration features
        """
        features = pd.DataFrame(index=start_time.index)
        
        # Ensure datetime format
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)
        
        # Duration in various units
        duration = end_time - start_time
        features[f'duration_seconds_{suffix}'] = duration.dt.total_seconds()
        features[f'duration_minutes_{suffix}'] = duration.dt.total_seconds() / 60
        features[f'duration_hours_{suffix}'] = duration.dt.total_seconds() / 3600
        
        # Duration categories
        duration_minutes = duration.dt.total_seconds() / 60
        features[f'duration_category_{suffix}'] = pd.cut(
            duration_minutes,
            bins=[0, 1, 5, 15, 60, 240, float('inf')],
            labels=['very_short', 'short', 'medium', 'long', 'very_long', 'extremely_long'],
            include_lowest=True
        ).astype('category')
        
        # Log duration (for skewed distributions)
        features[f'log_duration_seconds_{suffix}'] = np.log1p(duration.dt.total_seconds())
        
        return features
    
    def configure_for_snowflake_data(self):
        """
        Configure the generator for typical Snowflake analytics data.
        """
        snowflake_config = {
            'primary_time_column': 'START_TIME',
            'secondary_time_columns': ['END_TIME'],
            'business_hours': (8, 18),  # Typical business hours for analytics
            'custom_time_periods': {
                'peak_hours': [9, 10, 11, 14, 15, 16, 17],  # Peak analytics hours
                'off_peak_hours': [0, 1, 2, 3, 4, 5, 22, 23],
                'quarter_end_months': [3, 6, 9, 12],
                'holiday_months': [11, 12, 1],
                'report_generation_hours': [7, 8, 9, 17, 18]  # Common report times
            }
        }
        
        self.config.update(snowflake_config)
        logger.info("Configured TimeFeatureGenerator for Snowflake analytics data")
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all generated time features.
        
        Returns:
            Dictionary mapping feature names to descriptions
        """
        descriptions = {
            # Basic components
            'hour_of_day': 'Hour of the day (0-23)',
            'day_of_week': 'Day of the week (0=Monday, 6=Sunday)',
            'day_of_month': 'Day of the month (1-31)',
            'day_of_year': 'Day of the year (1-366)',
            'week_of_year': 'Week of the year (1-53)',
            'month_of_year': 'Month of the year (1-12)',
            'quarter_of_year': 'Quarter of the year (1-4)',
            'year': 'Year',
            'time_of_day': 'Time period category (night/morning/afternoon/evening)',
            
            # Cyclical features
            'hour_sin': 'Sine transformation of hour (24-hour cycle)',
            'hour_cos': 'Cosine transformation of hour (24-hour cycle)',
            'day_of_week_sin': 'Sine transformation of day of week (7-day cycle)',
            'day_of_week_cos': 'Cosine transformation of day of week (7-day cycle)',
            'day_of_month_sin': 'Sine transformation of day of month (~30-day cycle)',
            'day_of_month_cos': 'Cosine transformation of day of month (~30-day cycle)',
            'month_sin': 'Sine transformation of month (12-month cycle)',
            'month_cos': 'Cosine transformation of month (12-month cycle)',
            'quarter_sin': 'Sine transformation of quarter (4-quarter cycle)',
            'quarter_cos': 'Cosine transformation of quarter (4-quarter cycle)',
            
            # Business indicators
            'is_business_day': 'Whether the day is a business day (1=yes, 0=no)',
            'is_weekend': 'Whether the day is a weekend (1=yes, 0=no)',
            'is_business_hours': 'Whether the time is during business hours (1=yes, 0=no)',
            'is_business_time': 'Whether it is both business day and business hours (1=yes, 0=no)',
            'is_after_hours': 'Whether it is a business day but outside business hours (1=yes, 0=no)',
            'is_non_business_time': 'Whether it is weekend or after hours (1=yes, 0=no)',
            
            # Custom indicators
            'is_peak_hours': 'Whether the hour is in peak usage hours (1=yes, 0=no)',
            'is_off_peak_hours': 'Whether the hour is in off-peak hours (1=yes, 0=no)',
            'is_quarter_end_month': 'Whether the month is a quarter-end month (1=yes, 0=no)',
            'is_holiday_month': 'Whether the month is in holiday season (1=yes, 0=no)',
            'is_morning_rush': 'Whether the time is during morning rush hours (1=yes, 0=no)',
            'is_evening_rush': 'Whether the time is during evening rush hours (1=yes, 0=no)',
            'is_lunch_hour': 'Whether the time is during lunch hour (1=yes, 0=no)'
        }
        
        return descriptions
