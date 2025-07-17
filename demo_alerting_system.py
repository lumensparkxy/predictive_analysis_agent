#!/usr/bin/env python3
"""
Snowflake Analytics - Intelligent Alert System Demo

This example demonstrates the complete intelligent alert and notification system
for Snowflake analytics with real-time monitoring, rule engine, and notifications.
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from snowflake_analytics.alerting.monitoring.real_time_monitor import RealTimeMonitor, MonitoringMetric
from snowflake_analytics.alerting.monitoring.cost_monitor import CostMonitor, CostMetricType
from snowflake_analytics.alerting.monitoring.usage_monitor import UsageMonitor, UsageMetricType
from snowflake_analytics.alerting.monitoring.performance_monitor import PerformanceMonitor, PerformanceMetricType
from snowflake_analytics.alerting.monitoring.threshold_manager import ThresholdManager, Threshold, ThresholdType, ThresholdCondition

from snowflake_analytics.alerting.rules.rule_engine import RuleEngine
from snowflake_analytics.alerting.rules.rule_builder import RuleBuilder, RuleTemplateBuilder, create_simple_threshold_rule
from snowflake_analytics.alerting.rules.rule_manager import RuleManager
from snowflake_analytics.alerting.rules.condition_evaluator import ConditionEvaluator
from snowflake_analytics.alerting.rules.severity_calculator import SeverityCalculator

from snowflake_analytics.alerting.notifications.email_notifier import EmailNotifier, EmailNotification, EmailRecipient
from snowflake_analytics.alerting.notifications.slack_notifier import SlackNotifier, SlackNotification


class SnowflakeAlertingDemo:
    """
    Comprehensive demo of the Snowflake alerting system
    """
    
    def __init__(self):
        print("🚀 Initializing Snowflake Intelligent Alert System Demo")
        print("=" * 60)
        
        # Initialize monitoring components
        self.real_time_monitor = RealTimeMonitor()
        self.cost_monitor = CostMonitor(self.get_cost_config())
        self.usage_monitor = UsageMonitor(self.get_usage_config())
        self.performance_monitor = PerformanceMonitor(self.get_performance_config())
        self.threshold_manager = ThresholdManager()
        
        # Initialize rule engine
        self.rule_engine = RuleEngine()
        self.rule_manager = RuleManager()
        self.severity_calculator = SeverityCalculator()
        
        # Initialize notification system
        self.email_notifier = EmailNotifier(self.get_email_config())
        self.slack_notifier = SlackNotifier(self.get_slack_config())
        
        # Connect components
        self.setup_system_connections()
        
        # Storage for demo data
        self.alerts_received = []
        self.notifications_sent = []
        
        print("✅ System initialized successfully!")
        print()
    
    def get_cost_config(self) -> Dict[str, Any]:
        """Get cost monitoring configuration"""
        return {
            'budgets': {
                'monthly_production': {
                    'total_budget': 10000.0,
                    'period': 'monthly',
                    'currency': 'USD',
                    'alert_thresholds': [0.5, 0.8, 0.9, 0.95],
                    'auto_actions': {
                        'scale_down': {'enabled': True, 'threshold': 0.9},
                        'notify_team': {'enabled': True, 'threshold': 0.8}
                    }
                },
                'daily_development': {
                    'total_budget': 500.0,
                    'period': 'daily',
                    'currency': 'USD',
                    'alert_thresholds': [0.7, 0.9]
                }
            },
            'thresholds': {
                'daily_spend': {
                    'threshold_value': 1000.0,
                    'period': 'daily',
                    'enabled': True
                },
                'hourly_spend': {
                    'threshold_value': 50.0,
                    'period': 'hourly',
                    'enabled': True
                }
            }
        }
    
    def get_usage_config(self) -> Dict[str, Any]:
        """Get usage monitoring configuration"""
        return {
            'usage_thresholds': {
                'warehouse_utilization': {
                    'threshold_value': 85.0,
                    'comparison_operator': '>',
                    'enabled': True,
                    'severity': 'high'
                },
                'query_volume': {
                    'threshold_value': 1000.0,
                    'comparison_operator': '>',
                    'enabled': True,
                    'severity': 'medium'
                },
                'failed_queries': {
                    'threshold_value': 50.0,
                    'comparison_operator': '>',
                    'enabled': True,
                    'severity': 'high'
                }
            },
            'baseline_metrics': {
                'warehouse_utilization': 60.0,
                'query_volume': 500.0,
                'concurrent_users': 20.0
            }
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance monitoring configuration"""
        return {
            'performance_slas': {
                'query_execution_time': {
                    'target_value': 30.0,  # 30 seconds
                    'tolerance_percentage': 20.0,
                    'measurement_period': 300,
                    'enabled': True
                },
                'warehouse_response_time': {
                    'target_value': 5.0,  # 5 seconds
                    'tolerance_percentage': 50.0,
                    'measurement_period': 300,
                    'enabled': True
                }
            },
            'degradation_threshold': 0.3  # 30% degradation
        }
    
    def get_email_config(self) -> Dict[str, Any]:
        """Get email notification configuration"""
        return {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'sender_email': 'alerts@company.com',
            'sender_name': 'Snowflake Analytics',
            'use_tls': True,
            'max_retries': 3,
            'timeout': 30
        }
    
    def get_slack_config(self) -> Dict[str, Any]:
        """Get Slack notification configuration"""
        return {
            'bot_name': 'Snowflake Analytics Bot',
            'bot_emoji': ':snowflake:',
            'default_webhook_url': 'https://hooks.slack.com/services/demo/webhook',
            'channels': {
                'alerts': {
                    'channel_id': 'C123456',
                    'severity_levels': ['high', 'critical'],
                    'alert_types': ['threshold', 'anomaly']
                },
                'cost-alerts': {
                    'channel_id': 'C789012',
                    'severity_levels': ['medium', 'high', 'critical'],
                    'alert_types': ['cost_threshold', 'budget']
                },
                'performance': {
                    'channel_id': 'C345678',
                    'severity_levels': ['high', 'critical'],
                    'alert_types': ['performance_degradation', 'sla_violation']
                }
            }
        }
    
    def setup_system_connections(self):
        """Set up connections between system components"""
        # Connect monitoring to rule engine
        self.rule_engine.set_metric_provider(self.mock_metric_provider)
        
        # Connect rule engine to notifications
        self.rule_engine.add_alert_callback(self.handle_alert)
        
        # Connect monitors to alert callbacks
        self.real_time_monitor.add_alert_callback(self.handle_monitor_alert)
        self.cost_monitor.monitor.add_alert_callback(self.handle_monitor_alert)
        self.usage_monitor.monitor.add_alert_callback(self.handle_monitor_alert)
        self.performance_monitor.monitor.add_alert_callback(self.handle_monitor_alert)
        
        # Start monitoring
        self.real_time_monitor.start_monitoring()
        self.cost_monitor.start_monitoring()
        self.usage_monitor.start_monitoring()
        self.performance_monitor.start_monitoring()
        
        print("🔗 System connections established")
    
    def mock_metric_provider(self, metric_name: str, time_window: int) -> List[Dict[str, Any]]:
        """Mock metric provider for demonstration"""
        # Generate mock data based on metric name
        base_time = datetime.now()
        data = []
        
        if "cost" in metric_name.lower():
            # Generate cost data
            for i in range(10):
                data.append({
                    'value': 100.0 + (i * 10) + (i * i * 2),  # Increasing cost
                    'timestamp': (base_time - timedelta(minutes=i)).isoformat()
                })
        elif "usage" in metric_name.lower():
            # Generate usage data
            for i in range(10):
                data.append({
                    'value': 60.0 + (i * 5) + (i % 3 * 10),  # Variable usage
                    'timestamp': (base_time - timedelta(minutes=i)).isoformat()
                })
        elif "performance" in metric_name.lower():
            # Generate performance data
            for i in range(10):
                data.append({
                    'value': 20.0 + (i * 2) + (i % 2 * 5),  # Degrading performance
                    'timestamp': (base_time - timedelta(minutes=i)).isoformat()
                })
        else:
            # Generate generic data
            for i in range(10):
                data.append({
                    'value': 50.0 + (i * 5),
                    'timestamp': (base_time - timedelta(minutes=i)).isoformat()
                })
        
        return data
    
    def handle_alert(self, rule, alert_data):
        """Handle alerts from rule engine"""
        self.alerts_received.append((rule, alert_data))
        
        print(f"🚨 ALERT: {alert_data['rule_name']}")
        print(f"   Severity: {alert_data['severity'].upper()}")
        print(f"   Description: {alert_data['description']}")
        print(f"   Rule ID: {alert_data['rule_id']}")
        print(f"   Timestamp: {alert_data['timestamp']}")
        print()
        
        # Send notifications
        self.send_alert_notifications(alert_data)
    
    def handle_monitor_alert(self, alert):
        """Handle alerts from monitoring components"""
        print(f"📊 MONITOR ALERT: {alert.title}")
        print(f"   Severity: {alert.severity.value.upper()}")
        print(f"   Description: {alert.description}")
        print(f"   Metric: {alert.metric.name} = {alert.metric.value}")
        print(f"   Timestamp: {alert.timestamp}")
        print()
    
    def send_alert_notifications(self, alert_data):
        """Send notifications for alerts"""
        try:
            # Send email notification
            email_notification = self.email_notifier.create_alert_notification(
                alert_data=alert_data,
                recipients=['admin@company.com', 'team@company.com'],
                template_data={}
            )
            
            # In a real implementation, this would actually send the email
            print(f"📧 Email notification created: {email_notification.subject}")
            self.notifications_sent.append(('email', email_notification))
            
            # Send Slack notification
            channel = self.slack_notifier.get_appropriate_channel(alert_data)
            slack_notification = self.slack_notifier.create_alert_notification(
                alert_data=alert_data,
                channel=channel,
                template_data={}
            )
            
            # In a real implementation, this would actually send to Slack
            print(f"💬 Slack notification created for channel: {channel}")
            self.notifications_sent.append(('slack', slack_notification))
            
        except Exception as e:
            print(f"❌ Error sending notifications: {e}")
    
    def create_sample_rules(self):
        """Create sample alert rules for demonstration"""
        print("📋 Creating sample alert rules...")
        
        # 1. Cost threshold rule
        cost_rule = create_simple_threshold_rule(
            rule_id="cost_threshold_daily",
            name="Daily Cost Threshold",
            metric_name="cost_daily_spend",
            operator=">",
            threshold=800.0,
            severity="high"
        )
        cost_rule.description = "Alert when daily spend exceeds $800"
        cost_rule.team = "finance"
        cost_rule.owner = "cost-team@company.com"
        
        self.rule_engine.add_rule(cost_rule)
        print(f"   ✅ Created: {cost_rule.name}")
        
        # 2. Usage anomaly rule
        template_builder = RuleTemplateBuilder()
        usage_rule = template_builder.usage_anomaly_rule(
            rule_id="warehouse_utilization_anomaly",
            name="Warehouse Utilization Anomaly",
            usage_metric="usage_warehouse_utilization",
            anomaly_threshold=2.0,
            severity="medium"
        )
        usage_rule.team = "operations"
        usage_rule.owner = "ops-team@company.com"
        
        self.rule_engine.add_rule(usage_rule)
        print(f"   ✅ Created: {usage_rule.name}")
        
        # 3. Performance degradation rule
        perf_rule = template_builder.performance_degradation_rule(
            rule_id="query_performance_degradation",
            name="Query Performance Degradation",
            performance_metric="performance_query_execution_time",
            baseline_multiplier=1.5,
            severity="high"
        )
        perf_rule.team = "database"
        perf_rule.owner = "db-team@company.com"
        
        self.rule_engine.add_rule(perf_rule)
        print(f"   ✅ Created: {perf_rule.name}")
        
        # 4. Composite rule
        composite_rule = template_builder.composite_cost_performance_rule(
            rule_id="cost_performance_composite",
            name="High Cost & Poor Performance",
            cost_metric="cost_hourly_spend",
            cost_threshold=100.0,
            performance_metric="performance_query_execution_time",
            performance_threshold=60.0,
            severity="critical"
        )
        composite_rule.team = "platform"
        composite_rule.owner = "platform-team@company.com"
        
        self.rule_engine.add_rule(composite_rule)
        print(f"   ✅ Created: {composite_rule.name}")
        
        print(f"📋 Created {len(self.rule_engine.rules)} alert rules")
        print()
    
    def simulate_metrics_and_alerts(self):
        """Simulate metrics and trigger alerts"""
        print("🔄 Simulating metrics and alerts...")
        print()
        
        # Simulate cost metrics
        print("💰 Simulating cost metrics...")
        self.cost_monitor.add_cost_metric(CostMetricType.DAILY_SPEND, 850.0)  # Should trigger alert
        self.cost_monitor.add_cost_metric(CostMetricType.HOURLY_SPEND, 120.0)  # Should trigger alert
        
        # Simulate usage metrics
        print("📊 Simulating usage metrics...")
        self.usage_monitor.add_usage_metric(UsageMetricType.WAREHOUSE_UTILIZATION, 95.0)  # Should trigger alert
        self.usage_monitor.add_usage_metric(UsageMetricType.QUERY_VOLUME, 1200.0)  # Should trigger alert
        
        # Simulate performance metrics
        print("⚡ Simulating performance metrics...")
        self.performance_monitor.add_performance_metric(PerformanceMetricType.QUERY_EXECUTION_TIME, 45.0)  # Should trigger alert
        self.performance_monitor.add_performance_metric(PerformanceMetricType.WAREHOUSE_RESPONSE_TIME, 8.0)  # Should trigger alert
        
        # Simulate anomaly detection
        print("🔍 Simulating anomaly detection...")
        for i in range(5):
            # Normal metrics
            metric = MonitoringMetric(
                name="anomaly_test_metric",
                value=50.0 + i,
                timestamp=datetime.now(),
                source="demo"
            )
            self.real_time_monitor.add_metric(metric)
        
        # Anomalous metric
        anomaly_metric = MonitoringMetric(
            name="anomaly_test_metric",
            value=200.0,  # Significant spike
            timestamp=datetime.now(),
            source="demo"
        )
        self.real_time_monitor.add_metric(anomaly_metric)
        
        print("🔄 Metrics simulation complete")
        print()
    
    def evaluate_rules(self):
        """Evaluate all rules against current metrics"""
        print("⚖️  Evaluating alert rules...")
        print()
        
        for rule_id, rule in self.rule_engine.rules.items():
            try:
                result = self.rule_engine.evaluate_rule(rule)
                if result:
                    print(f"   🔴 Rule '{rule.name}' TRIGGERED")
                else:
                    print(f"   🟢 Rule '{rule.name}' OK")
            except Exception as e:
                print(f"   ❌ Rule '{rule.name}' ERROR: {e}")
        
        print()
    
    def display_system_status(self):
        """Display overall system status"""
        print("📊 SYSTEM STATUS")
        print("=" * 40)
        
        # Monitoring status
        print("🔍 Monitoring:")
        print(f"   Real-time Monitor: {'Running' if self.real_time_monitor.is_monitoring else 'Stopped'}")
        print(f"   Cost Monitor: {'Running' if self.cost_monitor.monitor.is_monitoring else 'Stopped'}")
        print(f"   Usage Monitor: {'Running' if self.usage_monitor.monitor.is_monitoring else 'Stopped'}")
        print(f"   Performance Monitor: {'Running' if self.performance_monitor.monitor.is_monitoring else 'Stopped'}")
        
        # Rule engine status
        rule_status = self.rule_engine.get_engine_status()
        print(f"\n⚖️  Rule Engine:")
        print(f"   Status: {'Running' if rule_status['is_running'] else 'Stopped'}")
        print(f"   Total Rules: {rule_status['total_rules']}")
        print(f"   Active Rules: {rule_status['active_rules']}")
        print(f"   Active Alerts: {rule_status['active_alerts']}")
        
        # Alerts summary
        print(f"\n🚨 Alerts:")
        print(f"   Total Alerts Received: {len(self.alerts_received)}")
        print(f"   Notifications Sent: {len(self.notifications_sent)}")
        
        # Cost monitoring summary
        cost_summary = self.cost_monitor.get_cost_summary()
        print(f"\n💰 Cost Summary:")
        for budget_name, budget_info in cost_summary['budgets'].items():
            print(f"   {budget_name}: {budget_info['spend_percentage']:.1f}% of budget")
        
        # Performance monitoring summary
        perf_summary = self.performance_monitor.get_performance_summary()
        print(f"\n⚡ Performance Summary:")
        print(f"   SLA Compliance: {len(perf_summary.get('sla_compliance', {}))}/{len(perf_summary.get('slas', {}))}")
        print(f"   Recent Trends: {len(perf_summary.get('trends', []))}")
        
        print()
    
    def run_demo(self):
        """Run the complete demo"""
        print("🎬 Starting Snowflake Alerting System Demo")
        print("=" * 60)
        
        # Step 1: Create sample rules
        self.create_sample_rules()
        
        # Step 2: Simulate metrics and alerts
        self.simulate_metrics_and_alerts()
        
        # Step 3: Evaluate rules
        self.evaluate_rules()
        
        # Step 4: Display system status
        self.display_system_status()
        
        # Step 5: Show detailed results
        self.show_detailed_results()
        
        print("✅ Demo completed successfully!")
        print()
        print("🔗 The intelligent alert system is now monitoring your Snowflake environment")
        print("   and will proactively notify you of cost overruns, usage anomalies,")
        print("   and performance issues to help optimize your analytics workloads.")
    
    def show_detailed_results(self):
        """Show detailed results from the demo"""
        print("📋 DETAILED RESULTS")
        print("=" * 40)
        
        # Show triggered alerts
        if self.alerts_received:
            print(f"\n🚨 Triggered Alerts ({len(self.alerts_received)}):")
            for i, (rule, alert_data) in enumerate(self.alerts_received, 1):
                print(f"   {i}. {alert_data['rule_name']} ({alert_data['severity'].upper()})")
                print(f"      Description: {alert_data['description']}")
                print(f"      Team: {alert_data.get('team', 'Not specified')}")
                print(f"      Owner: {alert_data.get('owner', 'Not specified')}")
                print()
        
        # Show notifications sent
        if self.notifications_sent:
            print(f"📤 Notifications Sent ({len(self.notifications_sent)}):")
            for i, (channel, notification) in enumerate(self.notifications_sent, 1):
                if channel == 'email':
                    print(f"   {i}. Email: {notification.subject}")
                    print(f"      Recipients: {len(notification.recipients)}")
                elif channel == 'slack':
                    print(f"   {i}. Slack: {notification.channel}")
                    print(f"      Message Type: {notification.message_type.value}")
                print()
        
        # Show rule statistics
        print("📊 Rule Statistics:")
        for rule_id, rule in self.rule_engine.rules.items():
            stats = self.rule_engine.get_rule_statistics(rule_id)
            print(f"   {rule.name}:")
            print(f"      Status: {stats['status']}")
            print(f"      Executions: {stats['total_executions']}")
            print(f"      Triggers: {stats['trigger_count']}")
            print(f"      Success Rate: {stats['success_rate']:.1f}%")
            print()
    
    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up resources...")
        
        # Stop monitoring
        self.real_time_monitor.stop_monitoring()
        self.cost_monitor.stop_monitoring()
        self.usage_monitor.stop_monitoring()
        self.performance_monitor.stop_monitoring()
        
        # Stop rule engine
        self.rule_engine.stop()
        
        print("✅ Cleanup completed")


def main():
    """Main function to run the demo"""
    try:
        demo = SnowflakeAlertingDemo()
        demo.run_demo()
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 Thank you for trying the Snowflake Intelligent Alert System!")


if __name__ == "__main__":
    main()