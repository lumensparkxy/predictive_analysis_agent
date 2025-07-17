"""
Usage Analyzer for Intelligent Usage Insights

Analyzes Snowflake usage patterns to generate insights about
query performance, user behavior, and system utilization.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import logging

from ...utils.logger import get_logger

logger = get_logger(__name__)


class UsageAnalyzer:
    """Analyzes usage data and generates intelligent insights."""
    
    def __init__(self, client=None, config: Dict[str, Any] = None):
        """Initialize usage analyzer."""
        self.client = client
        self.config = config or {}
        
        logger.info("Usage analyzer initialized")
    
    async def analyze_usage_patterns(self, usage_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze usage patterns and generate insights."""
        insights = []
        
        try:
            df = self._normalize_usage_data(usage_data)
            if df is None or df.empty:
                return insights
            
            # Peak usage analysis
            if 'query_count' in df.columns:
                total_queries = df['query_count'].sum()
                avg_queries = df['query_count'].mean()
                max_queries = df['query_count'].max()
                
                if max_queries > avg_queries * 2:
                    insights.append({
                        'title': 'Peak Usage Pattern Detected',
                        'description': f'Maximum daily queries ({max_queries}) significantly exceed average ({avg_queries:.1f})',
                        'severity': 'medium',
                        'confidence': 0.8,
                        'data_points': {
                            'total_queries': total_queries,
                            'avg_queries': avg_queries,
                            'max_queries': max_queries
                        },
                        'recommendations': [
                            'Review peak usage periods for optimization',
                            'Consider workload distribution strategies'
                        ]
                    })
            
        except Exception as e:
            logger.error(f"Usage pattern analysis failed: {e}")
        
        return insights
    
    async def analyze_performance_metrics(self, usage_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze performance metrics and generate insights."""
        insights = []
        
        try:
            df = self._normalize_usage_data(usage_data)
            if df is None or df.empty:
                return insights
            
            # Query performance analysis
            if 'execution_time' in df.columns:
                avg_execution_time = df['execution_time'].mean()
                
                if avg_execution_time > 30000:  # 30 seconds
                    insights.append({
                        'title': 'Performance Optimization Opportunity',
                        'description': f'Average query execution time is {avg_execution_time/1000:.1f} seconds',
                        'severity': 'medium',
                        'confidence': 0.7,
                        'data_points': {
                            'avg_execution_time_ms': avg_execution_time
                        },
                        'recommendations': [
                            'Review and optimize slow queries',
                            'Consider query performance tuning'
                        ]
                    })
            
        except Exception as e:
            logger.error(f"Performance metrics analysis failed: {e}")
        
        return insights
    
    def _normalize_usage_data(self, usage_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Normalize usage data into a standard DataFrame format."""
        try:
            if isinstance(usage_data, pd.DataFrame):
                return usage_data
            elif isinstance(usage_data, dict):
                return pd.DataFrame(usage_data)
            elif isinstance(usage_data, list):
                return pd.DataFrame(usage_data)
            else:
                return None
        except Exception as e:
            logger.warning(f"Could not normalize usage data: {e}")
            return None