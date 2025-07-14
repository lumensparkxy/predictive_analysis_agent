"""
Anomaly Detector - Detects anomalies in data collection patterns.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AnomalyDetector:
    """Detects anomalies in collected data patterns."""
    
    def __init__(self):
        self.baseline_stats = {}
    
    def detect_anomalies(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Detect anomalies in the data."""
        logger.info(f"Detecting anomalies in {table_name}")
        
        anomalies = {
            'volume_anomalies': self._detect_volume_anomalies(df),
            'pattern_anomalies': self._detect_pattern_anomalies(df),
            'value_anomalies': self._detect_value_anomalies(df)
        }
        
        return {
            'table_name': table_name,
            'detection_timestamp': datetime.now().isoformat(),
            'anomalies_found': any(anomalies.values()),
            'anomalies': anomalies
        }
    
    def _detect_volume_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect volume-based anomalies."""
        # Simple implementation - check if record count is significantly different
        current_count = len(df)
        expected_range = (100, 100000)  # Expected range for most tables
        
        if current_count < expected_range[0]:
            return [{'type': 'low_volume', 'count': current_count, 'threshold': expected_range[0]}]
        elif current_count > expected_range[1]:
            return [{'type': 'high_volume', 'count': current_count, 'threshold': expected_range[1]}]
        
        return []
    
    def _detect_pattern_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect pattern-based anomalies."""
        anomalies = []
        
        # Check for unusual time patterns if timestamp columns exist
        time_columns = [col for col in df.columns if 'TIME' in col.upper()]
        for col in time_columns:
            try:
                timestamps = pd.to_datetime(df[col], errors='coerce')
                if timestamps.notna().sum() > 0:
                    # Check for future dates
                    future_dates = (timestamps > datetime.now()).sum()
                    if future_dates > 0:
                        anomalies.append({
                            'type': 'future_timestamps',
                            'column': col,
                            'count': int(future_dates)
                        })
            except:
                pass
        
        return anomalies
    
    def _detect_value_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect value-based anomalies."""
        anomalies = []
        
        # Check numeric columns for extreme values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            try:
                data = df[col].dropna()
                if len(data) > 10:
                    # Simple outlier detection
                    Q1, Q3 = data.quantile([0.25, 0.75])
                    IQR = Q3 - Q1
                    outliers = ((data < Q1 - 3 * IQR) | (data > Q3 + 3 * IQR)).sum()
                    
                    if outliers > len(data) * 0.05:  # > 5% outliers
                        anomalies.append({
                            'type': 'extreme_values',
                            'column': col,
                            'outlier_count': int(outliers),
                            'outlier_percentage': float(outliers / len(data))
                        })
            except:
                pass
        
        return anomalies
