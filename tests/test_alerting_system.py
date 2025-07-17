"""
Test Suite for Intelligent Alert & Notification System

Basic tests to validate the alert system implementation
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import time

# Import the alerting components
from src.snowflake_analytics.alerting.monitoring.real_time_monitor import (
    RealTimeMonitor, MonitoringMetric, Alert, AlertSeverity
)
from src.snowflake_analytics.alerting.monitoring.cost_monitor import (
    CostMonitor, CostMetricType, BudgetConfig
)
from src.snowflake_analytics.alerting.rules.rule_engine import (
    RuleEngine, AlertRule, RuleCondition, RuleGroup, RuleType, LogicalOperator
)
from src.snowflake_analytics.alerting.rules.rule_builder import (
    RuleBuilder, RuleGroupBuilder, create_simple_threshold_rule
)
from src.snowflake_analytics.alerting.rules.condition_evaluator import ConditionEvaluator
from src.snowflake_analytics.alerting.rules.severity_calculator import SeverityCalculator
from src.snowflake_analytics.alerting.notifications.email_notifier import (
    EmailNotifier, EmailNotification, EmailRecipient
)
from src.snowflake_analytics.alerting.notifications.slack_notifier import (
    SlackNotifier, SlackNotification, SlackMessageType
)


class TestRealTimeMonitor(unittest.TestCase):
    """Test real-time monitoring functionality"""
    
    def setUp(self):
        self.monitor = RealTimeMonitor()
        self.alerts_received = []
        self.monitor.add_alert_callback(self.alert_callback)
    
    def alert_callback(self, alert):
        self.alerts_received.append(alert)
    
    def test_add_metric(self):
        """Test adding metrics to monitor"""
        metric = MonitoringMetric(
            name="test_metric",
            value=100.0,
            timestamp=datetime.now(),
            source="test"
        )
        
        self.monitor.add_metric(metric)
        
        # Check metric was added
        self.assertIn("test_metric", self.monitor.metrics_windows)
        
        # Check metric statistics
        stats = self.monitor.get_metrics_summary()
        self.assertIn("test_metric", stats)
        self.assertEqual(stats["test_metric"]["statistics"]["count"], 1)
    
    def test_anomaly_detection(self):
        """Test anomaly detection"""
        # Add normal metrics
        for i in range(10):
            metric = MonitoringMetric(
                name="anomaly_test",
                value=50.0 + i,  # Gradual increase
                timestamp=datetime.now(),
                source="test"
            )
            self.monitor.add_metric(metric)
        
        # Add anomalous metric
        anomaly_metric = MonitoringMetric(
            name="anomaly_test",
            value=200.0,  # Significant spike
            timestamp=datetime.now(),
            source="test"
        )
        self.monitor.add_metric(anomaly_metric)
        
        # Should trigger anomaly alert
        self.assertTrue(len(self.alerts_received) > 0)
        alert = self.alerts_received[-1]
        self.assertEqual(alert.title, "Anomaly Detected: anomaly_test")


class TestCostMonitor(unittest.TestCase):
    """Test cost monitoring functionality"""
    
    def setUp(self):
        config = {
            'budgets': {
                'monthly_budget': {
                    'total_budget': 1000.0,
                    'period': 'monthly',
                    'alert_thresholds': [0.5, 0.8, 0.9]
                }
            }
        }
        self.cost_monitor = CostMonitor(config)
        self.alerts_received = []
        self.cost_monitor.monitor.add_alert_callback(self.alert_callback)
    
    def alert_callback(self, alert):
        self.alerts_received.append(alert)
    
    def test_cost_metric_addition(self):
        """Test adding cost metrics"""
        self.cost_monitor.add_cost_metric(
            metric_type=CostMetricType.DAILY_SPEND,
            value=100.0,
            source="test"
        )
        
        # Check metric was added
        summary = self.cost_monitor.get_cost_summary()
        self.assertIn('budgets', summary)
        self.assertIn('monthly_budget', summary['budgets'])
    
    def test_budget_threshold_alert(self):
        """Test budget threshold alerts"""
        # Add cost that exceeds 50% of budget
        self.cost_monitor.add_cost_metric(
            metric_type=CostMetricType.DAILY_SPEND,
            value=600.0,  # 60% of 1000 budget
            source="test"
        )
        
        # Should trigger budget alert
        self.assertTrue(len(self.alerts_received) > 0)


class TestRuleEngine(unittest.TestCase):
    """Test alert rule engine"""
    
    def setUp(self):
        self.rule_engine = RuleEngine()
        self.alerts_received = []
        self.rule_engine.add_alert_callback(self.alert_callback)
        
        # Mock metric provider
        self.mock_metric_data = [
            {'value': 100.0, 'timestamp': datetime.now().isoformat()}
        ]
        self.rule_engine.set_metric_provider(self.mock_metric_provider)
    
    def mock_metric_provider(self, metric_name, time_window):
        return self.mock_metric_data
    
    def alert_callback(self, rule, alert_data):
        self.alerts_received.append((rule, alert_data))
    
    def test_rule_creation_and_evaluation(self):
        """Test creating and evaluating rules"""
        # Create a simple threshold rule
        rule = create_simple_threshold_rule(
            rule_id="test_rule",
            name="Test Rule",
            metric_name="test_metric",
            operator=">",
            threshold=50.0,
            severity="medium"
        )
        
        # Add rule to engine
        self.assertTrue(self.rule_engine.add_rule(rule))
        
        # Evaluate rule (should trigger since mock data is 100.0 > 50.0)
        result = self.rule_engine.evaluate_rule(rule)
        self.assertTrue(result)
        
        # Check alert was triggered
        self.assertEqual(len(self.alerts_received), 1)
        triggered_rule, alert_data = self.alerts_received[0]
        self.assertEqual(triggered_rule.id, "test_rule")
        self.assertEqual(alert_data['severity'], "medium")


class TestRuleBuilder(unittest.TestCase):
    """Test rule builder functionality"""
    
    def test_rule_builder_fluent_interface(self):
        """Test fluent interface for rule building"""
        condition_group = (RuleGroupBuilder()
                          .and_operator()
                          .condition("cond1", "metric1", ">", 100.0)
                          .condition("cond2", "metric2", "<", 50.0)
                          .build())
        
        rule = (RuleBuilder()
                .id("test_rule")
                .name("Test Rule")
                .description("Test rule description")
                .rule_type("threshold")
                .severity("high")
                .condition_group(condition_group)
                .cooldown(600)
                .max_triggers_per_hour(5)
                .build())
        
        self.assertEqual(rule.id, "test_rule")
        self.assertEqual(rule.name, "Test Rule")
        self.assertEqual(rule.severity, "high")
        self.assertEqual(rule.cooldown_period, 600)
        self.assertEqual(len(rule.condition_group.conditions), 2)
    
    def test_simple_threshold_rule_creation(self):
        """Test simple threshold rule creation"""
        rule = create_simple_threshold_rule(
            rule_id="simple_test",
            name="Simple Test Rule",
            metric_name="cpu_usage",
            operator=">",
            threshold=80.0,
            severity="high"
        )
        
        self.assertEqual(rule.id, "simple_test")
        self.assertEqual(rule.name, "Simple Test Rule")
        self.assertEqual(rule.severity, "high")
        self.assertEqual(len(rule.condition_group.conditions), 1)
        
        condition = rule.condition_group.conditions[0]
        self.assertEqual(condition.metric_name, "cpu_usage")
        self.assertEqual(condition.operator, ">")
        self.assertEqual(condition.value, 80.0)


class TestConditionEvaluator(unittest.TestCase):
    """Test condition evaluation"""
    
    def setUp(self):
        self.evaluator = ConditionEvaluator()
        self.mock_metric_data = [
            {'value': 80.0, 'timestamp': datetime.now().isoformat()},
            {'value': 85.0, 'timestamp': datetime.now().isoformat()},
            {'value': 90.0, 'timestamp': datetime.now().isoformat()}
        ]
    
    def mock_metric_provider(self, metric_name, time_window):
        return self.mock_metric_data
    
    def test_condition_evaluation(self):
        """Test basic condition evaluation"""
        condition = RuleCondition(
            id="test_condition",
            metric_name="test_metric",
            operator=">",
            value=75.0,
            aggregation="avg"
        )
        
        result = self.evaluator.evaluate_condition(condition, self.mock_metric_provider)
        self.assertTrue(result)  # Average of [80, 85, 90] = 85 > 75
    
    def test_aggregation_functions(self):
        """Test different aggregation functions"""
        condition = RuleCondition(
            id="test_condition",
            metric_name="test_metric",
            operator=">",
            value=89.0,
            aggregation="max"
        )
        
        result = self.evaluator.evaluate_condition(condition, self.mock_metric_provider)
        self.assertTrue(result)  # Max of [80, 85, 90] = 90 > 89
    
    def test_condition_validation(self):
        """Test condition validation"""
        valid_condition = RuleCondition(
            id="valid_condition",
            metric_name="test_metric",
            operator=">",
            value=50.0,
            aggregation="avg"
        )
        
        errors = self.evaluator.validate_condition(valid_condition)
        self.assertEqual(len(errors), 0)
        
        invalid_condition = RuleCondition(
            id="",  # Invalid: empty ID
            metric_name="test_metric",
            operator="invalid_op",  # Invalid operator
            value=50.0,
            aggregation="invalid_agg"  # Invalid aggregation
        )
        
        errors = self.evaluator.validate_condition(invalid_condition)
        self.assertGreater(len(errors), 0)


class TestSeverityCalculator(unittest.TestCase):
    """Test severity calculation"""
    
    def setUp(self):
        self.severity_calculator = SeverityCalculator()
    
    def test_severity_calculation(self):
        """Test basic severity calculation"""
        # Create a test rule
        rule = create_simple_threshold_rule(
            rule_id="severity_test",
            name="Severity Test",
            metric_name="test_metric",
            operator=">",
            threshold=50.0,
            severity="medium"
        )
        
        # Mock metric provider
        def mock_provider(metric_name, time_window):
            return [{'value': 100.0, 'timestamp': datetime.now().isoformat()}]
        
        calculated_severity = self.severity_calculator.calculate_severity(rule, mock_provider)
        self.assertIn(calculated_severity, ['low', 'medium', 'high', 'critical'])


class TestEmailNotifier(unittest.TestCase):
    """Test email notification system"""
    
    def setUp(self):
        config = {
            'smtp_server': 'localhost',
            'smtp_port': 587,
            'sender_email': 'test@example.com',
            'sender_name': 'Test System'
        }
        self.email_notifier = EmailNotifier(config)
    
    def test_email_notification_creation(self):
        """Test email notification creation"""
        notification = self.email_notifier.create_alert_notification(
            alert_data={
                'rule_id': 'test_rule',
                'rule_name': 'Test Alert',
                'severity': 'high',
                'description': 'Test alert description',
                'timestamp': datetime.now().isoformat()
            },
            recipients=['test@example.com'],
            template_data={}
        )
        
        self.assertIsInstance(notification, EmailNotification)
        self.assertEqual(len(notification.recipients), 1)
        self.assertEqual(notification.recipients[0].email, 'test@example.com')
        self.assertIn('Test Alert', notification.subject)
    
    @patch('smtplib.SMTP')
    def test_email_sending(self, mock_smtp):
        """Test email sending (mocked)"""
        mock_server = Mock()
        mock_smtp.return_value = mock_server
        
        notification = EmailNotification(
            notification_id="test_notification",
            subject="Test Subject",
            html_body="<p>Test HTML</p>",
            recipients=[EmailRecipient(email="test@example.com")]
        )
        
        result = self.email_notifier.send_notification(notification)
        
        # Verify SMTP was called (mocked)
        mock_smtp.assert_called_once()
        self.assertEqual(result.notification_id, "test_notification")


class TestSlackNotifier(unittest.TestCase):
    """Test Slack notification system"""
    
    def setUp(self):
        config = {
            'bot_token': 'test_token',
            'bot_name': 'Test Bot',
            'channels': {
                'alerts': {
                    'channel_id': 'C123456',
                    'severity_levels': ['high', 'critical']
                }
            }
        }
        self.slack_notifier = SlackNotifier(config)
    
    def test_slack_notification_creation(self):
        """Test Slack notification creation"""
        notification = self.slack_notifier.create_alert_notification(
            alert_data={
                'rule_id': 'test_rule',
                'rule_name': 'Test Alert',
                'severity': 'high',
                'description': 'Test alert description',
                'rule_type': 'threshold',
                'team': 'test_team',
                'owner': 'test_owner',
                'timestamp': datetime.now().isoformat()
            },
            channel='alerts',
            template_data={}
        )
        
        self.assertIsInstance(notification, SlackNotification)
        self.assertEqual(notification.channel, 'alerts')
        self.assertEqual(notification.message_type, SlackMessageType.RICH)
        self.assertTrue(len(notification.blocks) > 0)
    
    def test_channel_routing(self):
        """Test channel routing based on alert properties"""
        alert_data = {
            'severity': 'high',
            'rule_type': 'threshold',
            'team': 'test_team'
        }
        
        channel = self.slack_notifier.get_appropriate_channel(alert_data)
        self.assertEqual(channel, 'alerts')  # Should match configured channel
    
    @patch('requests.post')
    def test_slack_webhook_sending(self, mock_post):
        """Test Slack webhook sending (mocked)"""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        # Configure for webhook
        self.slack_notifier.bot_token = ''
        self.slack_notifier.default_webhook_url = 'https://hooks.slack.com/test'
        
        notification = SlackNotification(
            notification_id="test_notification",
            channel="alerts",
            message_type=SlackMessageType.SIMPLE,
            text="Test message"
        )
        
        result = self.slack_notifier.send_notification(notification)
        
        # Verify webhook was called (mocked)
        mock_post.assert_called_once()
        self.assertEqual(result.notification_id, "test_notification")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete alert system"""
    
    def setUp(self):
        self.monitor = RealTimeMonitor()
        self.rule_engine = RuleEngine()
        self.alerts_received = []
        
        # Mock metric provider
        self.mock_metric_data = []
        self.rule_engine.set_metric_provider(self.mock_metric_provider)
        self.rule_engine.add_alert_callback(self.alert_callback)
    
    def mock_metric_provider(self, metric_name, time_window):
        return self.mock_metric_data
    
    def alert_callback(self, rule, alert_data):
        self.alerts_received.append((rule, alert_data))
    
    def test_end_to_end_alert_flow(self):
        """Test complete alert flow from metric to notification"""
        # 1. Create and add a rule
        rule = create_simple_threshold_rule(
            rule_id="integration_test",
            name="Integration Test Rule",
            metric_name="cpu_usage",
            operator=">",
            threshold=80.0,
            severity="high"
        )
        
        self.rule_engine.add_rule(rule)
        
        # 2. Set up mock data that will trigger the rule
        self.mock_metric_data = [
            {'value': 90.0, 'timestamp': datetime.now().isoformat()}
        ]
        
        # 3. Evaluate the rule
        result = self.rule_engine.evaluate_rule(rule)
        
        # 4. Verify alert was triggered
        self.assertTrue(result)
        self.assertEqual(len(self.alerts_received), 1)
        
        triggered_rule, alert_data = self.alerts_received[0]
        self.assertEqual(triggered_rule.id, "integration_test")
        self.assertEqual(alert_data['severity'], "high")
        self.assertEqual(alert_data['rule_name'], "Integration Test Rule")
    
    def test_multi_condition_rule(self):
        """Test rule with multiple conditions"""
        # Create rule with AND conditions
        condition_group = (RuleGroupBuilder()
                          .and_operator()
                          .condition("cond1", "cpu_usage", ">", 80.0)
                          .condition("cond2", "memory_usage", ">", 70.0)
                          .build())
        
        rule = (RuleBuilder()
                .id("multi_condition_test")
                .name("Multi Condition Test")
                .condition_group(condition_group)
                .build())
        
        self.rule_engine.add_rule(rule)
        
        # Mock metric provider to return different data for different metrics
        def multi_metric_provider(metric_name, time_window):
            if metric_name == "cpu_usage":
                return [{'value': 85.0, 'timestamp': datetime.now().isoformat()}]
            elif metric_name == "memory_usage":
                return [{'value': 75.0, 'timestamp': datetime.now().isoformat()}]
            return []
        
        self.rule_engine.set_metric_provider(multi_metric_provider)
        
        # Evaluate rule - should trigger since both conditions are met
        result = self.rule_engine.evaluate_rule(rule)
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()