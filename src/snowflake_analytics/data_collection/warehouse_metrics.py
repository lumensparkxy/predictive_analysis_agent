"""
Warehouse Metrics Collector - Specialized collector for warehouse usage and performance.

This module provides detailed collection and analysis of Snowflake warehouse
metrics including credit consumption, load patterns, and optimization insights.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from ..connectors.connection_pool import ConnectionPool
from ..storage.sqlite_store import SQLiteStore
from ..utils.logger import get_logger

logger = get_logger(__name__)


class WarehouseMetricsCollector:
    """
    Specialized collector for warehouse metrics and analysis.
    
    Features:
    - Warehouse usage and credit consumption tracking
    - Load pattern analysis
    - Auto-scaling recommendations
    - Cost optimization insights
    - Performance bottleneck detection
    """
    
    def __init__(
        self,
        connection_pool: ConnectionPool,
        storage: SQLiteStore,
        collection_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize warehouse metrics collector."""
        self.connection_pool = connection_pool
        self.storage = storage
        self.config = collection_config or {}
        
        # Collection settings
        self.lookback_days = self.config.get('lookback_days', 7)
        self.credit_threshold = self.config.get('credit_threshold', 10.0)
        self.load_threshold = self.config.get('load_threshold', 0.8)
        
        # Warehouse size mapping for cost analysis
        self.warehouse_sizes = {
            'X-Small': 1,
            'Small': 2,
            'Medium': 4,
            'Large': 8,
            'X-Large': 16,
            '2X-Large': 32,
            '3X-Large': 64,
            '4X-Large': 128
        }
    
    def collect_warehouse_metrics(self, days_back: Optional[int] = None) -> Dict[str, Any]:
        """
        Collect comprehensive warehouse metrics.
        
        Args:
            days_back: Days of history to collect (default: from config)
            
        Returns:
            Dictionary with collection results and metrics
        """
        logger.info("Starting warehouse metrics collection")
        start_time = datetime.now()
        
        days_back = days_back or self.lookback_days
        from_time = datetime.now() - timedelta(days=days_back)
        
        try:
            # Collect warehouse usage data
            usage_data = self._collect_warehouse_usage(from_time)
            load_data = self._collect_warehouse_load(from_time)
            
            # Process and analyze data
            usage_df = pd.DataFrame(usage_data) if usage_data else pd.DataFrame()
            load_df = pd.DataFrame(load_data) if load_data else pd.DataFrame()
            
            if usage_df.empty and load_df.empty:
                return {
                    'success': True,
                    'records_collected': 0,
                    'message': 'No warehouse data available'
                }
            
            # Clean and process data
            if not usage_df.empty:
                usage_df = self._clean_usage_data(usage_df)
                usage_df = self._calculate_usage_metrics(usage_df)
            
            if not load_df.empty:
                load_df = self._clean_load_data(load_df)
                load_df = self._calculate_load_metrics(load_df)
            
            # Store processed data
            usage_records = self._store_usage_data(usage_df) if not usage_df.empty else 0
            load_records = self._store_load_data(load_df) if not load_df.empty else 0
            
            # Generate insights
            insights = self._generate_warehouse_insights(usage_df, load_df)
            
            end_time = datetime.now()
            collection_time = (end_time - start_time).total_seconds()
            
            total_records = usage_records + load_records
            logger.info(
                f"Warehouse metrics collection complete: {total_records} records in {collection_time:.1f}s"
            )
            
            return {
                'success': True,
                'records_collected': total_records,
                'usage_records': usage_records,
                'load_records': load_records,
                'collection_time_seconds': collection_time,
                'time_window': {
                    'from': from_time.isoformat(),
                    'to': end_time.isoformat()
                },
                'insights': insights
            }
            
        except Exception as e:
            logger.error(f"Error in warehouse metrics collection: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'records_collected': 0
            }
    
    def _collect_warehouse_usage(self, from_time: datetime) -> List[Dict]:
        """Collect warehouse usage and credit consumption data."""
        query = f"""
        SELECT 
            START_TIME,
            END_TIME,
            WAREHOUSE_ID,
            WAREHOUSE_NAME,
            CREDITS_USED,
            CREDITS_USED_COMPUTE,
            CREDITS_USED_CLOUD_SERVICES,
            BYTES_SCANNED,
            BYTES_WRITTEN,
            BYTES_DELETED,
            BYTES_SPILLED_TO_REMOTE_STORAGE,
            BYTES_SPILLED_TO_LOCAL_STORAGE,
            BYTES_SENT_OVER_THE_NETWORK
        FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY 
        WHERE START_TIME >= '{from_time.strftime('%Y-%m-%d %H:%M:%S')}'::timestamp
        ORDER BY START_TIME DESC
        """
        
        with self.connection_pool.get_connection() as conn:
            return conn.execute_query(query)
    
    def _collect_warehouse_load(self, from_time: datetime) -> List[Dict]:
        """Collect warehouse load and queue patterns."""
        query = f"""
        SELECT 
            START_TIME,
            END_TIME,
            WAREHOUSE_ID,
            WAREHOUSE_NAME,
            AVG_RUNNING,
            AVG_QUEUED_LOAD,
            AVG_QUEUED_PROVISIONING,
            AVG_BLOCKED
        FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_LOAD_HISTORY 
        WHERE START_TIME >= '{from_time.strftime('%Y-%m-%d %H:%M:%S')}'::timestamp
        ORDER BY START_TIME DESC
        """
        
        with self.connection_pool.get_connection() as conn:
            return conn.execute_query(query)
    
    def _clean_usage_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate warehouse usage data."""
        original_rows = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['START_TIME', 'WAREHOUSE_ID'])
        
        # Convert timestamps
        for col in ['START_TIME', 'END_TIME']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Convert numeric columns
        numeric_columns = [
            'CREDITS_USED', 'CREDITS_USED_COMPUTE', 'CREDITS_USED_CLOUD_SERVICES',
            'BYTES_SCANNED', 'BYTES_WRITTEN', 'BYTES_DELETED'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Remove invalid rows
        df = df.dropna(subset=['WAREHOUSE_ID', 'START_TIME'])
        
        cleaned_rows = len(df)
        if cleaned_rows < original_rows:
            logger.debug(f"Usage data cleaned: {original_rows} -> {cleaned_rows} rows")
        
        return df
    
    def _clean_load_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate warehouse load data."""
        original_rows = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['START_TIME', 'WAREHOUSE_ID'])
        
        # Convert timestamps
        for col in ['START_TIME', 'END_TIME']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Convert numeric columns
        numeric_columns = ['AVG_RUNNING', 'AVG_QUEUED_LOAD', 'AVG_QUEUED_PROVISIONING', 'AVG_BLOCKED']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Remove invalid rows
        df = df.dropna(subset=['WAREHOUSE_ID', 'START_TIME'])
        
        cleaned_rows = len(df)
        if cleaned_rows < original_rows:
            logger.debug(f"Load data cleaned: {original_rows} -> {cleaned_rows} rows")
        
        return df
    
    def _calculate_usage_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived warehouse usage metrics."""
        # Calculate time duration for each period
        df['PERIOD_DURATION_HOURS'] = (
            df['END_TIME'] - df['START_TIME']
        ).dt.total_seconds() / 3600
        
        # Calculate credits per hour
        df['CREDITS_PER_HOUR'] = np.where(
            df['PERIOD_DURATION_HOURS'] > 0,
            df['CREDITS_USED'] / df['PERIOD_DURATION_HOURS'],
            0
        )
        
        # Calculate data throughput
        df['DATA_THROUGHPUT_GB_PER_HOUR'] = np.where(
            df['PERIOD_DURATION_HOURS'] > 0,
            (df.get('BYTES_SCANNED', 0) + df.get('BYTES_WRITTEN', 0)) / (1024**3) / df['PERIOD_DURATION_HOURS'],
            0
        )
        
        # Calculate efficiency ratios
        df['COMPUTE_CREDIT_RATIO'] = np.where(
            df['CREDITS_USED'] > 0,
            df.get('CREDITS_USED_COMPUTE', 0) / df['CREDITS_USED'],
            0
        )
        
        df['CLOUD_SERVICES_RATIO'] = np.where(
            df['CREDITS_USED'] > 0,
            df.get('CREDITS_USED_CLOUD_SERVICES', 0) / df['CREDITS_USED'],
            0
        )
        
        # Add cost estimates (assuming $3 per credit)
        df['ESTIMATED_COST_USD'] = df['CREDITS_USED'] * 3.0
        
        # Add time-based groupings
        df['HOUR_OF_DAY'] = df['START_TIME'].dt.hour
        df['DAY_OF_WEEK'] = df['START_TIME'].dt.dayofweek
        df['DATE'] = df['START_TIME'].dt.date
        
        return df
    
    def _calculate_load_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived warehouse load metrics."""
        # Calculate total load
        df['TOTAL_LOAD'] = (
            df.get('AVG_RUNNING', 0) + 
            df.get('AVG_QUEUED_LOAD', 0) + 
            df.get('AVG_QUEUED_PROVISIONING', 0) +
            df.get('AVG_BLOCKED', 0)
        )
        
        # Calculate utilization ratios
        df['UTILIZATION_RATIO'] = np.where(
            df['TOTAL_LOAD'] > 0,
            df.get('AVG_RUNNING', 0) / df['TOTAL_LOAD'],
            0
        )
        
        df['QUEUE_RATIO'] = np.where(
            df['TOTAL_LOAD'] > 0,
            (df.get('AVG_QUEUED_LOAD', 0) + df.get('AVG_QUEUED_PROVISIONING', 0)) / df['TOTAL_LOAD'],
            0
        )
        
        df['BLOCKED_RATIO'] = np.where(
            df['TOTAL_LOAD'] > 0,
            df.get('AVG_BLOCKED', 0) / df['TOTAL_LOAD'],
            0
        )
        
        # Load classification
        df['LOAD_CLASSIFICATION'] = pd.cut(
            df['TOTAL_LOAD'],
            bins=[0, 0.2, 0.5, 0.8, 1.0, float('inf')],
            labels=['Low', 'Medium', 'High', 'Very High', 'Overloaded']
        )
        
        # Add time-based groupings
        df['HOUR_OF_DAY'] = df['START_TIME'].dt.hour
        df['DAY_OF_WEEK'] = df['START_TIME'].dt.dayofweek
        df['DATE'] = df['START_TIME'].dt.date
        
        return df
    
    def _store_usage_data(self, df: pd.DataFrame) -> int:
        """Store warehouse usage data."""
        if df.empty:
            return 0
        
        df['_collected_at'] = datetime.now()
        return self.storage.store_dataframe(df, 'warehouse_usage_metrics')
    
    def _store_load_data(self, df: pd.DataFrame) -> int:
        """Store warehouse load data."""
        if df.empty:
            return 0
        
        df['_collected_at'] = datetime.now()
        return self.storage.store_dataframe(df, 'warehouse_load_metrics')
    
    def _generate_warehouse_insights(self, usage_df: pd.DataFrame, load_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate warehouse insights and recommendations."""
        insights = {
            'usage_analysis': self._analyze_usage_patterns(usage_df),
            'load_analysis': self._analyze_load_patterns(load_df),
            'cost_optimization': self._identify_cost_optimization(usage_df),
            'performance_issues': self._identify_performance_issues(load_df),
            'recommendations': []
        }
        
        # Generate recommendations based on analysis
        insights['recommendations'] = self._generate_recommendations(usage_df, load_df)
        
        return insights
    
    def _analyze_usage_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze warehouse usage patterns."""
        if df.empty:
            return {}
        
        analysis = {
            'summary': {
                'total_credits_used': float(df['CREDITS_USED'].sum()),
                'avg_credits_per_hour': float(df['CREDITS_PER_HOUR'].mean()),
                'total_estimated_cost': float(df['ESTIMATED_COST_USD'].sum()),
                'unique_warehouses': int(df['WAREHOUSE_NAME'].nunique())
            },
            'by_warehouse': {},
            'by_time_patterns': {}
        }
        
        # Analysis by warehouse
        warehouse_summary = df.groupby('WAREHOUSE_NAME').agg({
            'CREDITS_USED': 'sum',
            'ESTIMATED_COST_USD': 'sum',
            'CREDITS_PER_HOUR': 'mean',
            'DATA_THROUGHPUT_GB_PER_HOUR': 'mean',
            'COMPUTE_CREDIT_RATIO': 'mean'
        }).round(2)
        
        analysis['by_warehouse'] = warehouse_summary.to_dict('index')
        
        # Time pattern analysis
        if 'HOUR_OF_DAY' in df.columns:
            hourly_usage = df.groupby('HOUR_OF_DAY')['CREDITS_USED'].sum()
            analysis['by_time_patterns']['hourly'] = hourly_usage.to_dict()
        
        if 'DAY_OF_WEEK' in df.columns:
            daily_usage = df.groupby('DAY_OF_WEEK')['CREDITS_USED'].sum()
            analysis['by_time_patterns']['daily'] = daily_usage.to_dict()
        
        return analysis
    
    def _analyze_load_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze warehouse load patterns."""
        if df.empty:
            return {}
        
        analysis = {
            'summary': {
                'avg_utilization': float(df['UTILIZATION_RATIO'].mean()),
                'avg_queue_ratio': float(df['QUEUE_RATIO'].mean()),
                'avg_blocked_ratio': float(df['BLOCKED_RATIO'].mean()),
                'peak_load': float(df['TOTAL_LOAD'].max()),
                'overloaded_periods': len(df[df['TOTAL_LOAD'] > 1.0])
            },
            'by_warehouse': {},
            'load_distribution': {}
        }
        
        # Analysis by warehouse
        if 'WAREHOUSE_NAME' in df.columns:
            warehouse_load = df.groupby('WAREHOUSE_NAME').agg({
                'UTILIZATION_RATIO': 'mean',
                'QUEUE_RATIO': 'mean',
                'BLOCKED_RATIO': 'mean',
                'TOTAL_LOAD': 'max'
            }).round(3)
            
            analysis['by_warehouse'] = warehouse_load.to_dict('index')
        
        # Load distribution
        if 'LOAD_CLASSIFICATION' in df.columns:
            load_dist = df['LOAD_CLASSIFICATION'].value_counts()
            analysis['load_distribution'] = load_dist.to_dict()
        
        return analysis
    
    def _identify_cost_optimization(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify cost optimization opportunities."""
        opportunities = []
        
        if df.empty:
            return opportunities
        
        # High cost warehouses
        warehouse_costs = df.groupby('WAREHOUSE_NAME')['ESTIMATED_COST_USD'].sum().sort_values(ascending=False)
        high_cost_warehouses = warehouse_costs[warehouse_costs > self.credit_threshold * 3.0].head(5)
        
        if not high_cost_warehouses.empty:
            opportunities.append({
                'type': 'high_cost_warehouses',
                'description': f'{len(high_cost_warehouses)} warehouses with high costs',
                'warehouses': high_cost_warehouses.to_dict(),
                'recommendation': 'Review warehouse sizing and auto-suspend settings'
            })
        
        # Low utilization periods
        low_efficiency = df[df['COMPUTE_CREDIT_RATIO'] < 0.5]  # < 50% compute ratio
        if not low_efficiency.empty:
            cost_impact = low_efficiency['ESTIMATED_COST_USD'].sum()
            opportunities.append({
                'type': 'low_efficiency_periods',
                'count': len(low_efficiency),
                'cost_impact': float(cost_impact),
                'description': f'{len(low_efficiency)} periods with low compute efficiency',
                'recommendation': 'Consider optimizing query patterns or warehouse sizing'
            })
        
        # Weekend usage
        if 'DAY_OF_WEEK' in df.columns:
            weekend_usage = df[df['DAY_OF_WEEK'].isin([5, 6])]  # Saturday, Sunday
            if not weekend_usage.empty:
                weekend_cost = weekend_usage['ESTIMATED_COST_USD'].sum()
                total_cost = df['ESTIMATED_COST_USD'].sum()
                weekend_percentage = (weekend_cost / total_cost) * 100
                
                if weekend_percentage > 20:  # > 20% of costs on weekends
                    opportunities.append({
                        'type': 'weekend_usage',
                        'weekend_cost': float(weekend_cost),
                        'percentage_of_total': float(weekend_percentage),
                        'description': f'{weekend_percentage:.1f}% of costs incurred on weekends',
                        'recommendation': 'Review weekend workload necessity and scheduling'
                    })
        
        return opportunities
    
    def _identify_performance_issues(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify warehouse performance issues."""
        issues = []
        
        if df.empty:
            return issues
        
        # High queue times
        high_queue = df[df['QUEUE_RATIO'] > 0.3]  # > 30% queue time
        if not high_queue.empty:
            issues.append({
                'type': 'high_queue_times',
                'count': len(high_queue),
                'avg_queue_ratio': float(high_queue['QUEUE_RATIO'].mean()),
                'description': f'{len(high_queue)} periods with high queue times',
                'affected_warehouses': high_queue['WAREHOUSE_NAME'].unique().tolist()
            })
        
        # Overloaded warehouses
        overloaded = df[df['TOTAL_LOAD'] > 1.0]
        if not overloaded.empty:
            issues.append({
                'type': 'overloaded_warehouses',
                'count': len(overloaded),
                'max_load': float(overloaded['TOTAL_LOAD'].max()),
                'description': f'{len(overloaded)} periods with warehouse overload',
                'affected_warehouses': overloaded['WAREHOUSE_NAME'].unique().tolist()
            })
        
        # High blocked ratio
        high_blocked = df[df['BLOCKED_RATIO'] > 0.1]  # > 10% blocked
        if not high_blocked.empty:
            issues.append({
                'type': 'high_blocked_ratio',
                'count': len(high_blocked),
                'avg_blocked_ratio': float(high_blocked['BLOCKED_RATIO'].mean()),
                'description': f'{len(high_blocked)} periods with high blocked queries',
                'affected_warehouses': high_blocked['WAREHOUSE_NAME'].unique().tolist()
            })
        
        return issues
    
    def _generate_recommendations(self, usage_df: pd.DataFrame, load_df: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Usage-based recommendations
        if not usage_df.empty:
            avg_cost_per_hour = usage_df['CREDITS_PER_HOUR'].mean() * 3.0  # $3 per credit
            if avg_cost_per_hour > 50:  # > $50/hour
                recommendations.append(
                    "High hourly costs detected. Consider implementing auto-suspend policies and optimizing warehouse sizes."
                )
            
            low_compute_ratio = usage_df['COMPUTE_CREDIT_RATIO'].mean()
            if low_compute_ratio < 0.7:  # < 70% compute
                recommendations.append(
                    "Low compute credit ratio indicates inefficient warehouse usage. Review cloud services usage and query patterns."
                )
        
        # Load-based recommendations
        if not load_df.empty:
            avg_utilization = load_df['UTILIZATION_RATIO'].mean()
            if avg_utilization < 0.3:  # < 30% utilization
                recommendations.append(
                    "Low warehouse utilization detected. Consider consolidating workloads or reducing warehouse sizes."
                )
            
            avg_queue = load_df['QUEUE_RATIO'].mean()
            if avg_queue > 0.2:  # > 20% queue time
                recommendations.append(
                    "High queue times detected. Consider scaling up warehouse sizes or implementing better workload distribution."
                )
        
        # General recommendations
        if not usage_df.empty and not load_df.empty:
            recommendations.append(
                "Monitor warehouse metrics regularly and set up alerts for unusual patterns."
            )
            recommendations.append(
                "Consider implementing auto-scaling policies based on workload patterns."
            )
        
        return recommendations
    
    def get_warehouse_optimization_report(self, days_back: int = 7) -> Dict[str, Any]:
        """Generate a comprehensive warehouse optimization report."""
        logger.info(f"Generating warehouse optimization report for last {days_back} days")
        
        # Collect recent data
        collection_result = self.collect_warehouse_metrics(days_back=days_back)
        
        if not collection_result['success']:
            return collection_result
        
        # Generate comprehensive report
        report = {
            'report_generated_at': datetime.now().isoformat(),
            'time_window_days': days_back,
            'collection_result': collection_result,
            'executive_summary': self._generate_executive_summary(collection_result),
            'detailed_analysis': collection_result.get('insights', {}),
            'action_items': self._prioritize_action_items(collection_result.get('insights', {}))
        }
        
        return report
    
    def _generate_executive_summary(self, collection_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of warehouse performance."""
        insights = collection_result.get('insights', {})
        usage_analysis = insights.get('usage_analysis', {})
        load_analysis = insights.get('load_analysis', {})
        
        summary = {
            'total_cost': usage_analysis.get('summary', {}).get('total_estimated_cost', 0),
            'warehouse_count': usage_analysis.get('summary', {}).get('unique_warehouses', 0),
            'avg_utilization': load_analysis.get('summary', {}).get('avg_utilization', 0),
            'performance_issues_count': len(insights.get('performance_issues', [])),
            'optimization_opportunities_count': len(insights.get('cost_optimization', [])),
            'key_findings': []
        }
        
        # Generate key findings
        if summary['total_cost'] > 1000:
            summary['key_findings'].append(f"High total cost: ${summary['total_cost']:.0f}")
        
        if summary['avg_utilization'] < 0.5:
            summary['key_findings'].append(f"Low utilization: {summary['avg_utilization']:.1%}")
        
        if summary['performance_issues_count'] > 0:
            summary['key_findings'].append(f"{summary['performance_issues_count']} performance issues identified")
        
        return summary
    
    def _prioritize_action_items(self, insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prioritize action items based on impact and urgency."""
        action_items = []
        
        # High priority - performance issues
        for issue in insights.get('performance_issues', []):
            action_items.append({
                'priority': 'High',
                'category': 'Performance',
                'title': issue.get('description', ''),
                'impact': 'User experience and system reliability',
                'urgency': 'Immediate'
            })
        
        # Medium priority - cost optimization
        for opportunity in insights.get('cost_optimization', []):
            if opportunity.get('cost_impact', 0) > 100:  # > $100 impact
                priority = 'High'
            else:
                priority = 'Medium'
                
            action_items.append({
                'priority': priority,
                'category': 'Cost Optimization',
                'title': opportunity.get('description', ''),
                'impact': f"Cost savings: ${opportunity.get('cost_impact', 0):.0f}",
                'urgency': 'Medium'
            })
        
        # Sort by priority
        priority_order = {'High': 1, 'Medium': 2, 'Low': 3}
        action_items.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return action_items
