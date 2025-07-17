"""
Anomaly Explainer for Detecting and Explaining Anomalies

Detects anomalies in Snowflake metrics and provides explanations
for unusual patterns or behaviors.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import logging

from ...utils.logger import get_logger

logger = get_logger(__name__)


class AnomalyExplainer:
    """Detects and explains anomalies in Snowflake data."""
    
    def __init__(self, client=None, config: Dict[str, Any] = None):
        """Initialize anomaly explainer."""
        self.client = client
        self.config = config or {}
        
        logger.info("Anomaly explainer initialized")
    
    async def detect_cost_anomalies(self, cost_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect cost anomalies and provide explanations."""
        anomalies = []
        
        try:
            df = self._normalize_data(cost_data)
            if df is None or df.empty:
                return anomalies
            
            # Simple anomaly detection based on standard deviation
            if 'cost_usd' in df.columns:
                mean_cost = df['cost_usd'].mean()
                std_cost = df['cost_usd'].std()
                
                for idx, row in df.iterrows():
                    cost = row['cost_usd']
                    z_score = abs((cost - mean_cost) / std_cost) if std_cost > 0 else 0
                    
                    if z_score > 2:  # More than 2 standard deviations
                        anomalies.append({
                            'title': 'Cost Anomaly Detected',
                            'description': f'Unusual cost of ${cost:.2f} detected (z-score: {z_score:.2f})',
                            'confidence': 0.8,
                            'data_points': {
                                'anomalous_cost': cost,
                                'mean_cost': mean_cost,
                                'z_score': z_score
                            },
                            'recommendations': [
                                'Investigate cause of cost anomaly',
                                'Review activity during anomalous period'
                            ]
                        })
            
        except Exception as e:
            logger.error(f"Cost anomaly detection failed: {e}")
        
        return anomalies
    
    async def detect_usage_anomalies(self, usage_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect usage anomalies and provide explanations."""
        anomalies = []
        
        try:
            df = self._normalize_data(usage_data)
            if df is None or df.empty:
                return anomalies
            
            # Query count anomaly detection
            if 'query_count' in df.columns:
                mean_queries = df['query_count'].mean()
                std_queries = df['query_count'].std()
                
                for idx, row in df.iterrows():
                    queries = row['query_count']
                    z_score = abs((queries - mean_queries) / std_queries) if std_queries > 0 else 0
                    
                    if z_score > 2:
                        anomalies.append({
                            'title': 'Usage Anomaly Detected',
                            'description': f'Unusual query volume of {queries} detected (z-score: {z_score:.2f})',
                            'confidence': 0.8,
                            'data_points': {
                                'anomalous_queries': queries,
                                'mean_queries': mean_queries,
                                'z_score': z_score
                            },
                            'recommendations': [
                                'Investigate cause of usage spike',
                                'Review query patterns during anomaly'
                            ]
                        })
            
        except Exception as e:
            logger.error(f"Usage anomaly detection failed: {e}")
        
        return anomalies
    
    def _normalize_data(self, data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Normalize data into a standard DataFrame format."""
        try:
            if isinstance(data, pd.DataFrame):
                return data
            elif isinstance(data, dict):
                return pd.DataFrame(data)
            elif isinstance(data, list):
                return pd.DataFrame(data)
            else:
                return None
        except Exception as e:
            logger.warning(f"Could not normalize data: {e}")
            return None