"""
Cost Monitoring System

Specialized monitoring for Snowflake cost metrics including budget thresholds,
spend rate anomalies, and cost spike identification.
"""

import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

from .real_time_monitor import RealTimeMonitor, MonitoringMetric, Alert, AlertSeverity


class CostMetricType(Enum):
    """Types of cost metrics"""
    DAILY_SPEND = "daily_spend"
    HOURLY_SPEND = "hourly_spend"
    WAREHOUSE_COST = "warehouse_cost"
    STORAGE_COST = "storage_cost"
    COMPUTE_COST = "compute_cost"
    DATA_TRANSFER_COST = "data_transfer_cost"
    QUERY_COST = "query_cost"


@dataclass
class CostThreshold:
    """Represents a cost threshold configuration"""
    metric_type: CostMetricType
    threshold_value: float
    period: str  # 'daily', 'hourly', 'monthly'
    currency: str = "USD"
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric_type': self.metric_type.value,
            'threshold_value': self.threshold_value,
            'period': self.period,
            'currency': self.currency,
            'enabled': self.enabled
        }


@dataclass
class BudgetConfig:
    """Budget configuration for cost monitoring"""
    name: str
    total_budget: float
    period: str  # 'daily', 'weekly', 'monthly', 'yearly'
    currency: str = "USD"
    alert_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.8, 0.9])  # 50%, 80%, 90%
    auto_actions: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'total_budget': self.total_budget,
            'period': self.period,
            'currency': self.currency,
            'alert_thresholds': self.alert_thresholds,
            'auto_actions': self.auto_actions
        }


class CostMonitor:
    """
    Cost-specific monitoring system for Snowflake
    
    Monitors budget thresholds, spend rate anomalies, and cost spikes
    with intelligent alerting and escalation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize base monitor
        self.monitor = RealTimeMonitor(config)
        
        # Cost-specific configuration
        self.budgets = {}  # budget_name -> BudgetConfig
        self.cost_thresholds = {}  # metric_type -> CostThreshold
        self.spend_history = {}  # date -> spend_amount
        
        # Load configuration
        self._load_cost_config()
        
        # Register alert callback
        self.monitor.add_alert_callback(self._handle_cost_alert)
        
    def _load_cost_config(self):
        """Load cost monitoring configuration"""
        # Load budgets
        budgets_config = self.config.get('budgets', {})
        for budget_name, budget_data in budgets_config.items():
            self.budgets[budget_name] = BudgetConfig(
                name=budget_name,
                total_budget=budget_data.get('total_budget', 1000.0),
                period=budget_data.get('period', 'monthly'),
                currency=budget_data.get('currency', 'USD'),
                alert_thresholds=budget_data.get('alert_thresholds', [0.5, 0.8, 0.9]),
                auto_actions=budget_data.get('auto_actions', {})
            )
        
        # Load cost thresholds
        thresholds_config = self.config.get('thresholds', {})
        for metric_type, threshold_data in thresholds_config.items():
            if hasattr(CostMetricType, metric_type.upper()):
                self.cost_thresholds[metric_type] = CostThreshold(
                    metric_type=CostMetricType(metric_type.lower()),
                    threshold_value=threshold_data.get('threshold_value', 100.0),
                    period=threshold_data.get('period', 'daily'),
                    currency=threshold_data.get('currency', 'USD'),
                    enabled=threshold_data.get('enabled', True)
                )
    
    def start_monitoring(self):
        """Start cost monitoring"""
        self.monitor.start_monitoring()
        self.logger.info("Cost monitoring started")
    
    def stop_monitoring(self):
        """Stop cost monitoring"""
        self.monitor.stop_monitoring()
        self.logger.info("Cost monitoring stopped")
    
    def add_cost_metric(self, metric_type: CostMetricType, value: float, 
                       source: str = "snowflake", tags: Optional[Dict[str, Any]] = None):
        """Add a cost metric for monitoring"""
        metric = MonitoringMetric(
            name=f"cost_{metric_type.value}",
            value=value,
            timestamp=datetime.now(),
            source=source,
            tags=tags or {}
        )
        
        self.monitor.add_metric(metric)
        
        # Check budget thresholds
        self._check_budget_thresholds(metric_type, value)
        
        # Check cost thresholds
        self._check_cost_thresholds(metric_type, value)
        
        # Update spend history
        self._update_spend_history(metric_type, value)
    
    def _check_budget_thresholds(self, metric_type: CostMetricType, value: float):
        """Check if cost exceeds budget thresholds"""
        current_date = datetime.now().date()
        
        for budget_name, budget in self.budgets.items():
            # Calculate current spend for the budget period
            current_spend = self._get_current_spend_for_period(budget.period, current_date)
            spend_percentage = (current_spend / budget.total_budget) * 100
            
            # Check threshold breaches
            for threshold_pct in budget.alert_thresholds:
                if spend_percentage >= (threshold_pct * 100) and not self._alert_already_sent(
                    budget_name, threshold_pct, current_date
                ):
                    self._create_budget_alert(budget, current_spend, spend_percentage, threshold_pct)
    
    def _check_cost_thresholds(self, metric_type: CostMetricType, value: float):
        """Check if cost exceeds defined thresholds"""
        if metric_type.value in self.cost_thresholds:
            threshold = self.cost_thresholds[metric_type.value]
            
            if threshold.enabled and value > threshold.threshold_value:
                self._create_threshold_alert(metric_type, value, threshold)
    
    def _update_spend_history(self, metric_type: CostMetricType, value: float):
        """Update spend history for trend analysis"""
        current_date = datetime.now().date()
        
        if current_date not in self.spend_history:
            self.spend_history[current_date] = {}
        
        if metric_type.value not in self.spend_history[current_date]:
            self.spend_history[current_date][metric_type.value] = 0
        
        self.spend_history[current_date][metric_type.value] += value
    
    def _get_current_spend_for_period(self, period: str, current_date) -> float:
        """Calculate current spend for a given period"""
        if period == 'daily':
            return self.spend_history.get(current_date, {}).get('total', 0.0)
        elif period == 'weekly':
            # Get last 7 days
            start_date = current_date - timedelta(days=7)
            return self._get_spend_for_date_range(start_date, current_date)
        elif period == 'monthly':
            # Get current month
            start_date = current_date.replace(day=1)
            return self._get_spend_for_date_range(start_date, current_date)
        elif period == 'yearly':
            # Get current year
            start_date = current_date.replace(month=1, day=1)
            return self._get_spend_for_date_range(start_date, current_date)
        
        return 0.0
    
    def _get_spend_for_date_range(self, start_date, end_date) -> float:
        """Calculate total spend for a date range"""
        total = 0.0
        current_date = start_date
        
        while current_date <= end_date:
            date_spend = self.spend_history.get(current_date, {})
            total += sum(date_spend.values())
            current_date += timedelta(days=1)
        
        return total
    
    def _alert_already_sent(self, budget_name: str, threshold_pct: float, date) -> bool:
        """Check if alert was already sent for this budget/threshold/date"""
        # Simple implementation - in production would use persistent storage
        # For now, assume alerts are not duplicated within the same day
        return False
    
    def _create_budget_alert(self, budget: BudgetConfig, current_spend: float, 
                           spend_percentage: float, threshold_pct: float):
        """Create budget threshold alert"""
        alert_id = f"budget_{budget.name}_{threshold_pct}_{int(time.time())}"
        
        # Determine severity based on threshold
        if threshold_pct >= 0.9:
            severity = AlertSeverity.CRITICAL
        elif threshold_pct >= 0.8:
            severity = AlertSeverity.HIGH
        elif threshold_pct >= 0.5:
            severity = AlertSeverity.MEDIUM
        else:
            severity = AlertSeverity.LOW
        
        metric = MonitoringMetric(
            name=f"budget_spend_{budget.name}",
            value=current_spend,
            timestamp=datetime.now(),
            source="cost_monitor",
            tags={
                'budget_name': budget.name,
                'threshold_percentage': threshold_pct,
                'spend_percentage': spend_percentage
            }
        )
        
        alert = Alert(
            id=alert_id,
            title=f"Budget Alert: {budget.name}",
            description=f"Budget '{budget.name}' has reached {spend_percentage:.1f}% "
                       f"of allocated budget ({current_spend:.2f} {budget.currency} "
                       f"of {budget.total_budget:.2f} {budget.currency})",
            severity=severity,
            metric=metric,
            threshold=budget.total_budget * threshold_pct,
            timestamp=datetime.now(),
            tags={
                'type': 'budget',
                'budget_name': budget.name,
                'threshold_percentage': threshold_pct,
                'spend_percentage': spend_percentage
            }
        )
        
        self._send_alert(alert)
    
    def _create_threshold_alert(self, metric_type: CostMetricType, value: float, 
                              threshold: CostThreshold):
        """Create cost threshold alert"""
        alert_id = f"threshold_{metric_type.value}_{int(time.time())}"
        
        # Determine severity based on threshold breach magnitude
        breach_ratio = value / threshold.threshold_value
        if breach_ratio >= 2.0:
            severity = AlertSeverity.CRITICAL
        elif breach_ratio >= 1.5:
            severity = AlertSeverity.HIGH
        elif breach_ratio >= 1.2:
            severity = AlertSeverity.MEDIUM
        else:
            severity = AlertSeverity.LOW
        
        metric = MonitoringMetric(
            name=f"cost_{metric_type.value}",
            value=value,
            timestamp=datetime.now(),
            source="cost_monitor",
            tags={
                'metric_type': metric_type.value,
                'threshold_value': threshold.threshold_value,
                'breach_ratio': breach_ratio
            }
        )
        
        alert = Alert(
            id=alert_id,
            title=f"Cost Threshold Alert: {metric_type.value}",
            description=f"Cost metric '{metric_type.value}' exceeded threshold: "
                       f"{value:.2f} {threshold.currency} > {threshold.threshold_value:.2f} {threshold.currency}",
            severity=severity,
            metric=metric,
            threshold=threshold.threshold_value,
            timestamp=datetime.now(),
            tags={
                'type': 'cost_threshold',
                'metric_type': metric_type.value,
                'breach_ratio': breach_ratio
            }
        )
        
        self._send_alert(alert)
    
    def _handle_cost_alert(self, alert: Alert):
        """Handle cost-specific alerts"""
        self.logger.info(f"Cost alert received: {alert.title}")
        
        # Add cost-specific processing
        if alert.tags.get('type') == 'budget':
            self._process_budget_alert(alert)
        elif alert.tags.get('type') == 'cost_threshold':
            self._process_threshold_alert(alert)
    
    def _process_budget_alert(self, alert: Alert):
        """Process budget-specific alerts"""
        budget_name = alert.tags.get('budget_name')
        if budget_name in self.budgets:
            budget = self.budgets[budget_name]
            
            # Execute auto actions if configured
            if budget.auto_actions:
                self._execute_auto_actions(budget.auto_actions, alert)
    
    def _process_threshold_alert(self, alert: Alert):
        """Process threshold-specific alerts"""
        metric_type = alert.tags.get('metric_type')
        self.logger.info(f"Processing threshold alert for metric: {metric_type}")
    
    def _execute_auto_actions(self, auto_actions: Dict[str, Any], alert: Alert):
        """Execute automated actions based on alert"""
        for action_type, action_config in auto_actions.items():
            if action_type == 'suspend_warehouse':
                self._suspend_warehouse(action_config, alert)
            elif action_type == 'scale_down':
                self._scale_down_resources(action_config, alert)
            elif action_type == 'notify_team':
                self._notify_team(action_config, alert)
    
    def _suspend_warehouse(self, config: Dict[str, Any], alert: Alert):
        """Suspend warehouse as auto action"""
        self.logger.warning(f"Auto action: Suspending warehouse due to alert {alert.id}")
        # Implementation would integrate with Snowflake API
    
    def _scale_down_resources(self, config: Dict[str, Any], alert: Alert):
        """Scale down resources as auto action"""
        self.logger.warning(f"Auto action: Scaling down resources due to alert {alert.id}")
        # Implementation would integrate with Snowflake API
    
    def _notify_team(self, config: Dict[str, Any], alert: Alert):
        """Notify team as auto action"""
        self.logger.info(f"Auto action: Notifying team due to alert {alert.id}")
        # Implementation would integrate with notification system
    
    def _send_alert(self, alert: Alert):
        """Send alert through the monitoring system"""
        # Delegate to the base monitor's alert system
        for callback in self.monitor.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost monitoring summary"""
        current_date = datetime.now().date()
        
        summary = {
            'budgets': {},
            'current_spend': {},
            'thresholds': {},
            'recent_alerts': []
        }
        
        # Budget summary
        for budget_name, budget in self.budgets.items():
            current_spend = self._get_current_spend_for_period(budget.period, current_date)
            spend_percentage = (current_spend / budget.total_budget) * 100
            
            summary['budgets'][budget_name] = {
                'total_budget': budget.total_budget,
                'current_spend': current_spend,
                'spend_percentage': spend_percentage,
                'remaining_budget': budget.total_budget - current_spend,
                'period': budget.period,
                'currency': budget.currency
            }
        
        # Current spend by metric type
        summary['current_spend'] = self.spend_history.get(current_date, {})
        
        # Threshold summary
        for metric_type, threshold in self.cost_thresholds.items():
            summary['thresholds'][metric_type] = threshold.to_dict()
        
        return summary
    
    def get_spend_trend(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get spend trend over specified days"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        trend_data = []
        current_date = start_date
        
        while current_date <= end_date:
            date_spend = self.spend_history.get(current_date, {})
            total_spend = sum(date_spend.values())
            
            trend_data.append({
                'date': current_date.isoformat(),
                'total_spend': total_spend,
                'spend_by_type': date_spend
            })
            
            current_date += timedelta(days=1)
        
        return trend_data