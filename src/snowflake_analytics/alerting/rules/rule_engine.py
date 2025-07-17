"""
Alert Rule Engine

Main rule processing engine that evaluates alert rules with complex conditions,
handles rule execution, and manages rule state and lifecycle.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import re


class RuleType(Enum):
    """Types of alert rules"""
    THRESHOLD = "threshold"
    PATTERN = "pattern"
    ANOMALY = "anomaly"
    PREDICTIVE = "predictive"
    COMPOSITE = "composite"


class RuleStatus(Enum):
    """Rule execution status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    ERROR = "error"


class LogicalOperator(Enum):
    """Logical operators for rule conditions"""
    AND = "and"
    OR = "or"
    NOT = "not"


@dataclass
class RuleCondition:
    """Individual rule condition"""
    id: str
    metric_name: str
    operator: str  # >, <, >=, <=, ==, !=
    value: Union[float, str]
    time_window: int = 300  # seconds
    aggregation: str = "latest"  # latest, avg, max, min, sum, count
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'metric_name': self.metric_name,
            'operator': self.operator,
            'value': self.value,
            'time_window': self.time_window,
            'aggregation': self.aggregation
        }


@dataclass
class RuleGroup:
    """Group of conditions with logical operator"""
    logical_operator: LogicalOperator
    conditions: List[RuleCondition]
    groups: List['RuleGroup'] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'logical_operator': self.logical_operator.value,
            'conditions': [c.to_dict() for c in self.conditions],
            'groups': [g.to_dict() for g in self.groups]
        }


@dataclass
class AlertRule:
    """Complete alert rule definition"""
    id: str
    name: str
    description: str
    rule_type: RuleType
    condition_group: RuleGroup
    severity: str
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_executed: Optional[datetime] = None
    last_triggered: Optional[datetime] = None
    execution_count: int = 0
    trigger_count: int = 0
    status: RuleStatus = RuleStatus.ACTIVE
    
    # Rule behavior settings
    cooldown_period: int = 300  # seconds between triggers
    max_triggers_per_hour: int = 10
    auto_resolve: bool = True
    auto_resolve_timeout: int = 3600  # seconds
    
    # Metadata
    tags: Dict[str, Any] = field(default_factory=dict)
    owner: str = ""
    team: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'rule_type': self.rule_type.value,
            'condition_group': self.condition_group.to_dict(),
            'severity': self.severity,
            'enabled': self.enabled,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'last_executed': self.last_executed.isoformat() if self.last_executed else None,
            'last_triggered': self.last_triggered.isoformat() if self.last_triggered else None,
            'execution_count': self.execution_count,
            'trigger_count': self.trigger_count,
            'status': self.status.value,
            'cooldown_period': self.cooldown_period,
            'max_triggers_per_hour': self.max_triggers_per_hour,
            'auto_resolve': self.auto_resolve,
            'auto_resolve_timeout': self.auto_resolve_timeout,
            'tags': self.tags,
            'owner': self.owner,
            'team': self.team
        }


@dataclass
class RuleExecution:
    """Record of rule execution"""
    rule_id: str
    execution_id: str
    timestamp: datetime
    result: bool
    execution_time_ms: float
    conditions_evaluated: int
    triggered: bool
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rule_id': self.rule_id,
            'execution_id': self.execution_id,
            'timestamp': self.timestamp.isoformat(),
            'result': self.result,
            'execution_time_ms': self.execution_time_ms,
            'conditions_evaluated': self.conditions_evaluated,
            'triggered': self.triggered,
            'error': self.error
        }


class RuleEngine:
    """
    Main rule processing engine for alert rules
    
    Evaluates alert rules with complex conditions, manages rule execution,
    and handles rule state and lifecycle.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Rule storage
        self.rules = {}  # rule_id -> AlertRule
        self.rule_executions = []  # Recent executions
        self.active_alerts = {}  # rule_id -> alert_timestamp
        
        # Engine state
        self.is_running = False
        self.engine_thread = None
        
        # Configuration
        self.evaluation_interval = self.config.get('evaluation_interval', 30)  # seconds
        self.max_execution_history = self.config.get('max_execution_history', 1000)
        self.max_concurrent_rules = self.config.get('max_concurrent_rules', 50)
        
        # Metrics and callbacks
        self.metric_provider = None  # Will be set externally
        self.alert_callbacks = []  # List of alert callback functions
        
        # Threading
        self._lock = threading.Lock()
        
        # Import other components
        from .condition_evaluator import ConditionEvaluator
        from .severity_calculator import SeverityCalculator
        
        self.condition_evaluator = ConditionEvaluator()
        self.severity_calculator = SeverityCalculator()
        
    def start(self):
        """Start the rule engine"""
        if self.is_running:
            return
        
        self.is_running = True
        self.engine_thread = threading.Thread(target=self._engine_loop)
        self.engine_thread.daemon = True
        self.engine_thread.start()
        
        self.logger.info("Rule engine started")
    
    def stop(self):
        """Stop the rule engine"""
        self.is_running = False
        if self.engine_thread:
            self.engine_thread.join()
        
        self.logger.info("Rule engine stopped")
    
    def add_rule(self, rule: AlertRule) -> bool:
        """Add a new rule to the engine"""
        with self._lock:
            if rule.id in self.rules:
                return False
            
            self.rules[rule.id] = rule
            self.logger.info(f"Added rule: {rule.name} ({rule.id})")
            return True
    
    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing rule"""
        with self._lock:
            if rule_id not in self.rules:
                return False
            
            rule = self.rules[rule_id]
            for key, value in updates.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            
            rule.updated_at = datetime.now()
            self.logger.info(f"Updated rule: {rule.name} ({rule_id})")
            return True
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule from the engine"""
        with self._lock:
            if rule_id not in self.rules:
                return False
            
            rule = self.rules.pop(rule_id)
            
            # Remove from active alerts
            if rule_id in self.active_alerts:
                del self.active_alerts[rule_id]
            
            self.logger.info(f"Removed rule: {rule.name} ({rule_id})")
            return True
    
    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Get a rule by ID"""
        return self.rules.get(rule_id)
    
    def get_rules(self, filters: Optional[Dict[str, Any]] = None) -> List[AlertRule]:
        """Get rules with optional filters"""
        rules = list(self.rules.values())
        
        if filters:
            if 'enabled' in filters:
                rules = [r for r in rules if r.enabled == filters['enabled']]
            if 'rule_type' in filters:
                rules = [r for r in rules if r.rule_type.value == filters['rule_type']]
            if 'status' in filters:
                rules = [r for r in rules if r.status.value == filters['status']]
            if 'team' in filters:
                rules = [r for r in rules if r.team == filters['team']]
        
        return rules
    
    def set_metric_provider(self, provider: Callable[[str, int], List[Dict[str, Any]]]):
        """Set the metric provider function"""
        self.metric_provider = provider
    
    def add_alert_callback(self, callback: Callable[[AlertRule, Dict[str, Any]], None]):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def evaluate_rule(self, rule: AlertRule) -> bool:
        """Evaluate a single rule"""
        if not self.metric_provider:
            self.logger.warning("No metric provider configured")
            return False
        
        start_time = time.time()
        execution_id = f"{rule.id}_{int(time.time() * 1000)}"
        
        try:
            # Check cooldown period
            if self._is_in_cooldown(rule):
                return False
            
            # Check rate limiting
            if self._is_rate_limited(rule):
                return False
            
            # Evaluate rule conditions
            result = self._evaluate_condition_group(rule.condition_group)
            
            # Update rule statistics
            rule.execution_count += 1
            rule.last_executed = datetime.now()
            
            if result:
                rule.trigger_count += 1
                rule.last_triggered = datetime.now()
                
                # Create alert
                alert_data = self._create_alert_data(rule)
                
                # Store active alert
                self.active_alerts[rule.id] = datetime.now()
                
                # Send alert notifications
                self._send_alert(rule, alert_data)
            
            # Record execution
            execution = RuleExecution(
                rule_id=rule.id,
                execution_id=execution_id,
                timestamp=datetime.now(),
                result=result,
                execution_time_ms=(time.time() - start_time) * 1000,
                conditions_evaluated=self._count_conditions(rule.condition_group),
                triggered=result
            )
            
            self._record_execution(execution)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating rule {rule.id}: {e}")
            
            # Record error execution
            execution = RuleExecution(
                rule_id=rule.id,
                execution_id=execution_id,
                timestamp=datetime.now(),
                result=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                conditions_evaluated=0,
                triggered=False,
                error=str(e)
            )
            
            self._record_execution(execution)
            
            # Update rule status
            rule.status = RuleStatus.ERROR
            
            return False
    
    def _engine_loop(self):
        """Main engine loop for rule evaluation"""
        while self.is_running:
            try:
                self._evaluate_all_rules()
                self._cleanup_old_data()
                time.sleep(self.evaluation_interval)
            except Exception as e:
                self.logger.error(f"Error in engine loop: {e}")
                time.sleep(self.evaluation_interval)
    
    def _evaluate_all_rules(self):
        """Evaluate all enabled rules"""
        with self._lock:
            active_rules = [r for r in self.rules.values() if r.enabled and r.status == RuleStatus.ACTIVE]
        
        # Limit concurrent evaluations
        active_rules = active_rules[:self.max_concurrent_rules]
        
        for rule in active_rules:
            try:
                self.evaluate_rule(rule)
            except Exception as e:
                self.logger.error(f"Error evaluating rule {rule.id}: {e}")
    
    def _evaluate_condition_group(self, group: RuleGroup) -> bool:
        """Evaluate a group of conditions"""
        condition_results = []
        
        # Evaluate individual conditions
        for condition in group.conditions:
            result = self.condition_evaluator.evaluate_condition(condition, self.metric_provider)
            condition_results.append(result)
        
        # Evaluate nested groups
        group_results = []
        for nested_group in group.groups:
            result = self._evaluate_condition_group(nested_group)
            group_results.append(result)
        
        # Combine all results
        all_results = condition_results + group_results
        
        if not all_results:
            return False
        
        # Apply logical operator
        if group.logical_operator == LogicalOperator.AND:
            return all(all_results)
        elif group.logical_operator == LogicalOperator.OR:
            return any(all_results)
        elif group.logical_operator == LogicalOperator.NOT:
            return not all(all_results)
        
        return False
    
    def _is_in_cooldown(self, rule: AlertRule) -> bool:
        """Check if rule is in cooldown period"""
        if rule.id not in self.active_alerts:
            return False
        
        last_alert = self.active_alerts[rule.id]
        cooldown_end = last_alert + timedelta(seconds=rule.cooldown_period)
        
        return datetime.now() < cooldown_end
    
    def _is_rate_limited(self, rule: AlertRule) -> bool:
        """Check if rule is rate limited"""
        if rule.max_triggers_per_hour <= 0:
            return False
        
        # Count triggers in last hour
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_executions = [
            e for e in self.rule_executions 
            if e.rule_id == rule.id and e.triggered and e.timestamp >= one_hour_ago
        ]
        
        return len(recent_executions) >= rule.max_triggers_per_hour
    
    def _create_alert_data(self, rule: AlertRule) -> Dict[str, Any]:
        """Create alert data from rule"""
        # Calculate dynamic severity if needed
        calculated_severity = self.severity_calculator.calculate_severity(rule, self.metric_provider)
        
        return {
            'rule_id': rule.id,
            'rule_name': rule.name,
            'rule_type': rule.rule_type.value,
            'description': rule.description,
            'severity': calculated_severity or rule.severity,
            'timestamp': datetime.now().isoformat(),
            'tags': rule.tags,
            'owner': rule.owner,
            'team': rule.team,
            'conditions': rule.condition_group.to_dict(),
            'auto_resolve': rule.auto_resolve,
            'auto_resolve_timeout': rule.auto_resolve_timeout
        }
    
    def _send_alert(self, rule: AlertRule, alert_data: Dict[str, Any]):
        """Send alert to all registered callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(rule, alert_data)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def _count_conditions(self, group: RuleGroup) -> int:
        """Count total conditions in a group"""
        count = len(group.conditions)
        for nested_group in group.groups:
            count += self._count_conditions(nested_group)
        return count
    
    def _record_execution(self, execution: RuleExecution):
        """Record rule execution"""
        self.rule_executions.append(execution)
        
        # Maintain execution history limit
        if len(self.rule_executions) > self.max_execution_history:
            self.rule_executions = self.rule_executions[-self.max_execution_history:]
    
    def _cleanup_old_data(self):
        """Clean up old execution data and resolved alerts"""
        # Clean up old executions (keep last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.rule_executions = [
            e for e in self.rule_executions 
            if e.timestamp >= cutoff_time
        ]
        
        # Clean up resolved alerts
        resolved_alerts = []
        for rule_id, alert_time in self.active_alerts.items():
            rule = self.rules.get(rule_id)
            if rule and rule.auto_resolve:
                resolve_time = alert_time + timedelta(seconds=rule.auto_resolve_timeout)
                if datetime.now() >= resolve_time:
                    resolved_alerts.append(rule_id)
        
        for rule_id in resolved_alerts:
            del self.active_alerts[rule_id]
            self.logger.info(f"Auto-resolved alert for rule {rule_id}")
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get engine status and statistics"""
        with self._lock:
            total_rules = len(self.rules)
            active_rules = len([r for r in self.rules.values() if r.enabled and r.status == RuleStatus.ACTIVE])
            error_rules = len([r for r in self.rules.values() if r.status == RuleStatus.ERROR])
        
        return {
            'is_running': self.is_running,
            'total_rules': total_rules,
            'active_rules': active_rules,
            'error_rules': error_rules,
            'active_alerts': len(self.active_alerts),
            'recent_executions': len(self.rule_executions),
            'evaluation_interval': self.evaluation_interval,
            'max_concurrent_rules': self.max_concurrent_rules
        }
    
    def get_rule_statistics(self, rule_id: str) -> Dict[str, Any]:
        """Get statistics for a specific rule"""
        rule = self.rules.get(rule_id)
        if not rule:
            return {}
        
        # Get recent executions
        recent_executions = [e for e in self.rule_executions if e.rule_id == rule_id]
        
        # Calculate statistics
        total_executions = len(recent_executions)
        successful_executions = len([e for e in recent_executions if e.error is None])
        triggered_executions = len([e for e in recent_executions if e.triggered])
        
        avg_execution_time = 0
        if recent_executions:
            avg_execution_time = sum(e.execution_time_ms for e in recent_executions) / len(recent_executions)
        
        return {
            'rule_id': rule_id,
            'rule_name': rule.name,
            'status': rule.status.value,
            'total_executions': rule.execution_count,
            'trigger_count': rule.trigger_count,
            'recent_executions': total_executions,
            'successful_executions': successful_executions,
            'triggered_executions': triggered_executions,
            'success_rate': (successful_executions / total_executions * 100) if total_executions > 0 else 0,
            'trigger_rate': (triggered_executions / total_executions * 100) if total_executions > 0 else 0,
            'avg_execution_time_ms': avg_execution_time,
            'last_executed': rule.last_executed.isoformat() if rule.last_executed else None,
            'last_triggered': rule.last_triggered.isoformat() if rule.last_triggered else None,
            'is_active_alert': rule_id in self.active_alerts
        }
    
    def get_execution_history(self, rule_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get execution history"""
        executions = self.rule_executions
        
        if rule_id:
            executions = [e for e in executions if e.rule_id == rule_id]
        
        executions = executions[-limit:]
        return [e.to_dict() for e in executions]