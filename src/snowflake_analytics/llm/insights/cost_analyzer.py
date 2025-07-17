"""
Cost Analyzer for Intelligent Cost Insights

Analyzes Snowflake cost data to generate intelligent insights,
identify trends, and recommend optimizations.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime, timedelta
import logging

from ...utils.logger import get_logger

logger = get_logger(__name__)


class CostAnalyzer:
    """Analyzes cost data and generates intelligent insights."""
    
    def __init__(self, client=None, config: Dict[str, Any] = None):
        """Initialize cost analyzer.
        
        Args:
            client: LLM client for advanced analysis
            config: Configuration dictionary
        """
        self.client = client
        self.config = config or {}
        
        # Cost analysis thresholds
        self.thresholds = {
            'high_cost_alert': config.get('high_cost_alert', 1000.0),
            'cost_spike_threshold': config.get('cost_spike_threshold', 50.0),  # % increase
            'efficiency_threshold': config.get('efficiency_threshold', 0.7),
            'waste_threshold': config.get('waste_threshold', 100.0)  # USD
        }
        
        logger.info("Cost analyzer initialized")
    
    async def analyze_cost_trends(self, cost_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze cost trends and generate insights.
        
        Args:
            cost_data: Cost metrics data
            
        Returns:
            List of cost trend insights
        """
        insights = []
        
        try:
            # Convert to DataFrame for analysis
            df = self._normalize_cost_data(cost_data)
            if df is None or df.empty:
                return insights
            
            # Daily cost trend analysis
            daily_trends = self._analyze_daily_trends(df)
            insights.extend(daily_trends)
            
            # Weekly and monthly trends
            weekly_trends = self._analyze_weekly_trends(df)
            insights.extend(weekly_trends)
            
            # Warehouse cost analysis
            warehouse_insights = self._analyze_warehouse_costs(df)
            insights.extend(warehouse_insights)
            
            # Cost spike detection
            spike_insights = self._detect_cost_spikes(df)
            insights.extend(spike_insights)
            
        except Exception as e:
            logger.error(f"Cost trend analysis failed: {e}")
        
        return insights
    
    async def analyze_cost_efficiency(self, cost_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze cost efficiency and identify optimization opportunities.
        
        Args:
            cost_data: Cost metrics data
            
        Returns:
            List of cost efficiency insights
        """
        insights = []
        
        try:
            df = self._normalize_cost_data(cost_data)
            if df is None or df.empty:
                return insights
            
            # Compute vs storage cost analysis
            compute_storage_insights = self._analyze_compute_vs_storage(df)
            insights.extend(compute_storage_insights)
            
            # Warehouse utilization efficiency
            utilization_insights = self._analyze_warehouse_utilization(df)
            insights.extend(utilization_insights)
            
            # Credit consumption patterns
            credit_insights = self._analyze_credit_consumption(df)
            insights.extend(credit_insights)
            
            # Idle warehouse detection
            idle_insights = self._detect_idle_warehouses(df)
            insights.extend(idle_insights)
            
        except Exception as e:
            logger.error(f"Cost efficiency analysis failed: {e}")
        
        return insights
    
    def _normalize_cost_data(self, cost_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Normalize cost data into a standard DataFrame format."""
        try:
            if isinstance(cost_data, pd.DataFrame):
                return cost_data
            elif isinstance(cost_data, dict):
                return pd.DataFrame(cost_data)
            elif isinstance(cost_data, list):
                return pd.DataFrame(cost_data)
            else:
                return None
        except Exception as e:
            logger.warning(f"Could not normalize cost data: {e}")
            return None
    
    def _analyze_daily_trends(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze daily cost trends."""
        insights = []
        
        try:
            if 'date' in df.columns and 'cost_usd' in df.columns:
                # Group by date and sum costs
                daily_costs = df.groupby('date')['cost_usd'].sum().sort_index()
                
                if len(daily_costs) >= 7:  # At least a week of data
                    # Calculate week-over-week change
                    recent_week = daily_costs.tail(7).sum()
                    previous_week = daily_costs.tail(14).head(7).sum()
                    
                    if previous_week > 0:
                        change_percent = ((recent_week - previous_week) / previous_week) * 100
                        
                        if abs(change_percent) > 20:  # Significant change
                            severity = "high" if abs(change_percent) > 50 else "medium"
                            direction = "increased" if change_percent > 0 else "decreased"
                            
                            insights.append({
                                'title': f'Weekly Cost Trend: {direction.title()}',
                                'description': f'Daily costs have {direction} by {abs(change_percent):.1f}% this week compared to last week',
                                'severity': severity,
                                'confidence': 0.8,
                                'data_points': {
                                    'recent_week_cost': recent_week,
                                    'previous_week_cost': previous_week,
                                    'change_percent': change_percent
                                },
                                'recommendations': [
                                    f'Investigate reasons for {direction} costs',
                                    'Review warehouse usage patterns',
                                    'Check for any new workloads or queries'
                                ]
                            })
                
        except Exception as e:
            logger.error(f"Daily trend analysis failed: {e}")
        
        return insights
    
    def _analyze_weekly_trends(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze weekly cost trends."""
        insights = []
        
        try:
            if 'date' in df.columns and 'cost_usd' in df.columns:
                # Convert date column to datetime if needed
                df['date'] = pd.to_datetime(df['date'])
                
                # Group by week
                df['week'] = df['date'].dt.isocalendar().week
                weekly_costs = df.groupby('week')['cost_usd'].sum()
                
                if len(weekly_costs) >= 4:  # At least a month of data
                    # Calculate month-over-month trend
                    recent_avg = weekly_costs.tail(2).mean()  # Last 2 weeks
                    previous_avg = weekly_costs.head(-2).tail(2).mean()  # Previous 2 weeks
                    
                    if previous_avg > 0:
                        change_percent = ((recent_avg - previous_avg) / previous_avg) * 100
                        
                        if abs(change_percent) > 15:
                            direction = "upward" if change_percent > 0 else "downward"
                            
                            insights.append({
                                'title': f'Monthly Cost Trend: {direction.title()}',
                                'description': f'Weekly average costs show a {direction} trend of {abs(change_percent):.1f}%',
                                'severity': 'medium',
                                'confidence': 0.7,
                                'data_points': {
                                    'recent_weekly_avg': recent_avg,
                                    'previous_weekly_avg': previous_avg,
                                    'trend_percent': change_percent
                                },
                                'recommendations': [
                                    'Monitor weekly cost patterns',
                                    'Plan capacity based on trends',
                                    'Review quarterly budget projections'
                                ]
                            })
                
        except Exception as e:
            logger.error(f"Weekly trend analysis failed: {e}")
        
        return insights
    
    def _analyze_warehouse_costs(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze costs by warehouse."""
        insights = []
        
        try:
            if 'warehouse' in df.columns and 'cost_usd' in df.columns:
                warehouse_costs = df.groupby('warehouse')['cost_usd'].sum().sort_values(ascending=False)
                
                if len(warehouse_costs) > 1:
                    total_cost = warehouse_costs.sum()
                    top_warehouse = warehouse_costs.index[0]
                    top_cost = warehouse_costs.iloc[0]
                    top_percentage = (top_cost / total_cost) * 100
                    
                    # High cost concentration insight
                    if top_percentage > 60:
                        insights.append({
                            'title': 'High Cost Concentration',
                            'description': f'Warehouse {top_warehouse} accounts for {top_percentage:.1f}% of total costs (${top_cost:.2f})',
                            'severity': 'medium',
                            'confidence': 0.9,
                            'data_points': {
                                'top_warehouse': top_warehouse,
                                'top_cost': top_cost,
                                'concentration_percent': top_percentage,
                                'total_cost': total_cost
                            },
                            'recommendations': [
                                f'Review {top_warehouse} warehouse usage',
                                'Consider workload distribution optimization',
                                'Check for potential rightsizing opportunities'
                            ]
                        })
                    
                    # Cost imbalance insight
                    if len(warehouse_costs) > 2:
                        cost_ratio = warehouse_costs.iloc[0] / warehouse_costs.iloc[-1]
                        if cost_ratio > 10:  # Top warehouse costs 10x more than cheapest
                            insights.append({
                                'title': 'Warehouse Cost Imbalance',
                                'description': f'Significant cost imbalance: {top_warehouse} costs {cost_ratio:.1f}x more than the least expensive warehouse',
                                'severity': 'medium',
                                'confidence': 0.8,
                                'data_points': {
                                    'cost_ratio': cost_ratio,
                                    'highest_cost_warehouse': warehouse_costs.index[0],
                                    'lowest_cost_warehouse': warehouse_costs.index[-1]
                                },
                                'recommendations': [
                                    'Review workload distribution strategy',
                                    'Consider consolidating underutilized warehouses',
                                    'Evaluate warehouse sizing policies'
                                ]
                            })
                
        except Exception as e:
            logger.error(f"Warehouse cost analysis failed: {e}")
        
        return insights
    
    def _detect_cost_spikes(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect unusual cost spikes."""
        insights = []
        
        try:
            if 'date' in df.columns and 'cost_usd' in df.columns:
                daily_costs = df.groupby('date')['cost_usd'].sum()
                
                if len(daily_costs) >= 7:
                    # Calculate rolling average and detect spikes
                    rolling_avg = daily_costs.rolling(window=7).mean()
                    
                    for date, cost in daily_costs.items():
                        avg_cost = rolling_avg.get(date, 0)
                        
                        if avg_cost > 0:
                            spike_percent = ((cost - avg_cost) / avg_cost) * 100
                            
                            if spike_percent > self.thresholds['cost_spike_threshold']:
                                insights.append({
                                    'title': 'Cost Spike Detected',
                                    'description': f'Cost spike of {spike_percent:.1f}% detected on {date} (${cost:.2f} vs ${avg_cost:.2f} average)',
                                    'severity': 'high',
                                    'confidence': 0.9,
                                    'data_points': {
                                        'spike_date': str(date),
                                        'spike_cost': cost,
                                        'average_cost': avg_cost,
                                        'spike_percent': spike_percent
                                    },
                                    'recommendations': [
                                        'Investigate queries run on spike date',
                                        'Check for unusual warehouse activity',
                                        'Review data loading operations',
                                        'Monitor for recurring patterns'
                                    ]
                                })
                
        except Exception as e:
            logger.error(f"Cost spike detection failed: {e}")
        
        return insights
    
    def _analyze_compute_vs_storage(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze compute vs storage cost distribution."""
        insights = []
        
        try:
            if 'compute_cost' in df.columns and 'storage_cost' in df.columns:
                total_compute = df['compute_cost'].sum()
                total_storage = df['storage_cost'].sum()
                total_cost = total_compute + total_storage
                
                if total_cost > 0:
                    compute_percent = (total_compute / total_cost) * 100
                    storage_percent = (total_storage / total_cost) * 100
                    
                    # High compute cost insight
                    if compute_percent > 80:
                        insights.append({
                            'title': 'High Compute Cost Ratio',
                            'description': f'Compute costs represent {compute_percent:.1f}% of total costs (${total_compute:.2f})',
                            'severity': 'medium',
                            'confidence': 0.8,
                            'data_points': {
                                'compute_percent': compute_percent,
                                'compute_cost': total_compute,
                                'storage_cost': total_storage,
                                'total_cost': total_cost
                            },
                            'recommendations': [
                                'Review warehouse auto-suspend settings',
                                'Optimize query performance to reduce compute time',
                                'Consider warehouse rightsizing',
                                'Implement better workload scheduling'
                            ]
                        })
                    
                    # High storage cost insight
                    elif storage_percent > 70:
                        insights.append({
                            'title': 'High Storage Cost Ratio',
                            'description': f'Storage costs represent {storage_percent:.1f}% of total costs (${total_storage:.2f})',
                            'severity': 'medium',
                            'confidence': 0.8,
                            'data_points': {
                                'storage_percent': storage_percent,
                                'storage_cost': total_storage,
                                'compute_cost': total_compute,
                                'total_cost': total_cost
                            },
                            'recommendations': [
                                'Review data retention policies',
                                'Implement data lifecycle management',
                                'Consider data archival strategies',
                                'Optimize table clustering and compression'
                            ]
                        })
                
        except Exception as e:
            logger.error(f"Compute vs storage analysis failed: {e}")
        
        return insights
    
    def _analyze_warehouse_utilization(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze warehouse utilization for efficiency insights."""
        insights = []
        
        try:
            if 'warehouse' in df.columns and 'credits_consumed' in df.columns:
                # Group by warehouse and calculate efficiency metrics
                warehouse_stats = df.groupby('warehouse').agg({
                    'credits_consumed': 'sum',
                    'cost_usd': 'sum'
                })
                
                for warehouse, stats in warehouse_stats.iterrows():
                    credits = stats['credits_consumed']
                    cost = stats['cost_usd']
                    
                    # Check for potential inefficiencies
                    if credits > 0:
                        cost_per_credit = cost / credits
                        
                        # High cost per credit might indicate inefficiency
                        if cost_per_credit > 3.0:  # Typical credit cost is ~$2-3
                            insights.append({
                                'title': f'Potential Inefficiency: {warehouse}',
                                'description': f'Warehouse {warehouse} has high cost per credit ratio: ${cost_per_credit:.2f}',
                                'severity': 'medium',
                                'confidence': 0.7,
                                'data_points': {
                                    'warehouse': warehouse,
                                    'cost_per_credit': cost_per_credit,
                                    'total_credits': credits,
                                    'total_cost': cost
                                },
                                'recommendations': [
                                    f'Review {warehouse} warehouse configuration',
                                    'Check query performance and optimization',
                                    'Consider warehouse sizing adjustments',
                                    'Monitor concurrent query patterns'
                                ]
                            })
                
        except Exception as e:
            logger.error(f"Warehouse utilization analysis failed: {e}")
        
        return insights
    
    def _analyze_credit_consumption(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze credit consumption patterns."""
        insights = []
        
        try:
            if 'credits_consumed' in df.columns and 'date' in df.columns:
                daily_credits = df.groupby('date')['credits_consumed'].sum()
                
                if len(daily_credits) >= 7:
                    avg_daily_credits = daily_credits.mean()
                    max_daily_credits = daily_credits.max()
                    
                    # High daily credit consumption
                    if max_daily_credits > avg_daily_credits * 3:
                        spike_date = daily_credits.idxmax()
                        
                        insights.append({
                            'title': 'High Credit Consumption Spike',
                            'description': f'Peak daily credit consumption of {max_daily_credits:.1f} credits on {spike_date} (avg: {avg_daily_credits:.1f})',
                            'severity': 'medium',
                            'confidence': 0.8,
                            'data_points': {
                                'max_credits': max_daily_credits,
                                'avg_credits': avg_daily_credits,
                                'spike_date': str(spike_date),
                                'spike_ratio': max_daily_credits / avg_daily_credits
                            },
                            'recommendations': [
                                'Investigate high-credit operations on spike date',
                                'Review concurrent query execution',
                                'Consider implementing resource monitors',
                                'Monitor for recurring patterns'
                            ]
                        })
                
        except Exception as e:
            logger.error(f"Credit consumption analysis failed: {e}")
        
        return insights
    
    def _detect_idle_warehouses(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect potentially idle or underutilized warehouses."""
        insights = []
        
        try:
            if 'warehouse' in df.columns and 'credits_consumed' in df.columns:
                # Calculate utilization by warehouse
                warehouse_credits = df.groupby('warehouse')['credits_consumed'].sum()
                
                # Find warehouses with very low credit consumption
                total_credits = warehouse_credits.sum()
                
                for warehouse, credits in warehouse_credits.items():
                    if total_credits > 0:
                        utilization_percent = (credits / total_credits) * 100
                        
                        # Low utilization warning
                        if utilization_percent < 5 and credits < 10:  # Less than 5% and low absolute
                            insights.append({
                                'title': f'Low Utilization: {warehouse}',
                                'description': f'Warehouse {warehouse} has very low utilization: {utilization_percent:.1f}% ({credits:.1f} credits)',
                                'severity': 'low',
                                'confidence': 0.8,
                                'data_points': {
                                    'warehouse': warehouse,
                                    'utilization_percent': utilization_percent,
                                    'credits_consumed': credits,
                                    'total_credits': total_credits
                                },
                                'recommendations': [
                                    f'Consider suspending {warehouse} if not needed',
                                    'Consolidate workloads to more utilized warehouses',
                                    'Review warehouse necessity and configuration',
                                    'Implement auto-suspend if not already enabled'
                                ]
                            })
                
        except Exception as e:
            logger.error(f"Idle warehouse detection failed: {e}")
        
        return insights