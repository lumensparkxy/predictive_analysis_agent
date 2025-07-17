"""
Trend Analyzer for Identifying and Analyzing Trends

Analyzes trends in Snowflake metrics and provides insights
about patterns and forecasts.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import logging

from ...utils.logger import get_logger

logger = get_logger(__name__)


class TrendAnalyzer:
    """Analyzes trends in Snowflake data."""
    
    def __init__(self, client=None, config: Dict[str, Any] = None):
        """Initialize trend analyzer."""
        self.client = client
        self.config = config or {}
        
        logger.info("Trend analyzer initialized")
    
    async def analyze_cost_trends(self, cost_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze cost trends and generate insights."""
        trends = []
        
        try:
            df = self._normalize_data(cost_data)
            if df is None or df.empty:
                return trends
            
            # Simple trend analysis
            if 'cost_usd' in df.columns and len(df) >= 3:
                # Calculate trend direction
                costs = df['cost_usd'].values
                first_half = costs[:len(costs)//2]
                second_half = costs[len(costs)//2:]
                
                first_avg = sum(first_half) / len(first_half)
                second_avg = sum(second_half) / len(second_half)
                
                if second_avg > first_avg * 1.1:
                    trend_direction = "increasing"
                    change_percent = ((second_avg - first_avg) / first_avg) * 100
                elif second_avg < first_avg * 0.9:
                    trend_direction = "decreasing"
                    change_percent = ((first_avg - second_avg) / first_avg) * 100
                else:
                    trend_direction = "stable"
                    change_percent = 0
                
                if trend_direction != "stable":
                    trends.append({
                        'title': f'Cost Trend: {trend_direction.title()}',
                        'description': f'Cost trend is {trend_direction} by {change_percent:.1f}%',
                        'severity': 'medium' if change_percent > 20 else 'low',
                        'confidence': 0.7,
                        'data_points': {
                            'trend_direction': trend_direction,
                            'change_percent': change_percent,
                            'first_period_avg': first_avg,
                            'second_period_avg': second_avg
                        },
                        'recommendations': [
                            f'Monitor {trend_direction} cost trend',
                            'Plan budget adjustments if needed'
                        ]
                    })
            
        except Exception as e:
            logger.error(f"Cost trend analysis failed: {e}")
        
        return trends
    
    async def analyze_usage_trends(self, usage_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze usage trends and generate insights."""
        trends = []
        
        try:
            df = self._normalize_data(usage_data)
            if df is None or df.empty:
                return trends
            
            # Usage trend analysis
            if 'query_count' in df.columns and len(df) >= 3:
                queries = df['query_count'].values
                first_half = queries[:len(queries)//2]
                second_half = queries[len(queries)//2:]
                
                first_avg = sum(first_half) / len(first_half)
                second_avg = sum(second_half) / len(second_half)
                
                if second_avg > first_avg * 1.2:
                    trend_direction = "increasing"
                    change_percent = ((second_avg - first_avg) / first_avg) * 100
                elif second_avg < first_avg * 0.8:
                    trend_direction = "decreasing"
                    change_percent = ((first_avg - second_avg) / first_avg) * 100
                else:
                    trend_direction = "stable"
                    change_percent = 0
                
                if trend_direction != "stable":
                    trends.append({
                        'title': f'Usage Trend: {trend_direction.title()}',
                        'description': f'Query usage trend is {trend_direction} by {change_percent:.1f}%',
                        'severity': 'medium' if change_percent > 30 else 'low',
                        'confidence': 0.7,
                        'data_points': {
                            'trend_direction': trend_direction,
                            'change_percent': change_percent,
                            'first_period_avg': first_avg,
                            'second_period_avg': second_avg
                        },
                        'recommendations': [
                            f'Monitor {trend_direction} usage trend',
                            'Plan capacity adjustments if needed'
                        ]
                    })
            
        except Exception as e:
            logger.error(f"Usage trend analysis failed: {e}")
        
        return trends
    
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