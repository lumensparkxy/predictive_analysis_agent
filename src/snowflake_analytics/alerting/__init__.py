"""
Snowflake Analytics - Intelligent Alert & Notification System

This module provides a comprehensive alerting and notification system for
proactive monitoring of Snowflake costs, usage patterns, and anomalies.

Key Components:
- Real-time monitoring system
- Alert rule engine with configurable thresholds
- Multi-channel notification delivery
- Alert management and escalation workflows
- Intelligent alert filtering and correlation
- Alert dashboard and reporting
"""

from .monitoring import RealTimeMonitor, CostMonitor, UsageMonitor, PerformanceMonitor
from .rules import RuleEngine, RuleBuilder, ConditionEvaluator, SeverityCalculator
from .notifications import EmailNotifier, SlackNotifier, WebhookNotifier, SMSNotifier
from .management import AlertManager, EscalationEngine, GroupingEngine
from .filtering import FatiguePreventer, Correlator, FrequencyLimiter
from .dashboard import AlertDashboard, Analytics, MetricsCalculator

__version__ = "1.0.0"
__all__ = [
    "RealTimeMonitor",
    "CostMonitor", 
    "UsageMonitor",
    "PerformanceMonitor",
    "RuleEngine",
    "RuleBuilder",
    "ConditionEvaluator",
    "SeverityCalculator",
    "EmailNotifier",
    "SlackNotifier",
    "WebhookNotifier",
    "SMSNotifier",
    "AlertManager",
    "EscalationEngine",
    "GroupingEngine",
    "FatiguePreventer",
    "Correlator",
    "FrequencyLimiter",
    "AlertDashboard",
    "Analytics",
    "MetricsCalculator",
]