"""
Alerting Manager for Snowflake Analytics
Comprehensive alerting system with multiple notification channels and escalation.
"""

import os
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import structlog
import smtplib
import requests
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

logger = structlog.get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


class AlertStatus(Enum):
    """Alert status."""
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SILENCED = "silenced"


class AlertingManager:
    """
    Comprehensive alerting system with multiple notification channels.
    """
    
    def __init__(self):
        """Initialize alerting manager."""
        # Configuration
        self.enabled = os.getenv('ALERTING_ENABLED', 'true').lower() == 'true'
        self.email_enabled = os.getenv('EMAIL_ALERTS_ENABLED', 'false').lower() == 'true'
        self.slack_enabled = os.getenv('SLACK_ALERTS_ENABLED', 'false').lower() == 'true'
        self.webhook_enabled = os.getenv('WEBHOOK_ALERTS_ENABLED', 'false').lower() == 'true'
        
        # Alert storage
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self.silenced_alerts: Dict[str, datetime] = {}
        
        # Notification settings
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_username = os.getenv('SMTP_USERNAME')
        self.smtp_password = os.getenv('SMTP_PASSWORD')
        self.email_from = os.getenv('ALERT_EMAIL_FROM', 'alerts@company.com')
        self.email_to = os.getenv('ALERT_EMAIL_TO', '').split(',')
        
        self.slack_webhook_url = os.getenv('SLACK_WEBHOOK_URL')
        self.slack_channel = os.getenv('SLACK_CHANNEL', '#alerts')
        
        self.webhook_urls = os.getenv('ALERT_WEBHOOK_URLS', '').split(',')
        
        # Alert rules
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self._load_default_rules()
        
        # Background processing
        self._processing_active = False
        self._processing_thread = None
        
        logger.info("AlertingManager initialized", 
                   enabled=self.enabled,
                   email_enabled=self.email_enabled,
                   slack_enabled=self.slack_enabled)
    
    def _load_default_rules(self):
        """Load default alerting rules."""
        self.alert_rules = {
            'high_cpu_usage': {
                'condition': lambda metrics: metrics.get('system.cpu.usage_percent', 0) > 90,
                'severity': AlertSeverity.WARNING,
                'message': 'High CPU usage detected: {value}%',
                'cooldown_minutes': 15
            },
            'high_memory_usage': {
                'condition': lambda metrics: metrics.get('system.memory.usage_percent', 0) > 90,
                'severity': AlertSeverity.WARNING,
                'message': 'High memory usage detected: {value}%',
                'cooldown_minutes': 15
            },
            'disk_space_low': {
                'condition': lambda metrics: metrics.get('system.disk.usage_percent', 0) > 95,
                'severity': AlertSeverity.CRITICAL,
                'message': 'Disk space critically low: {value}%',
                'cooldown_minutes': 5
            },
            'api_health_failed': {
                'condition': lambda metrics: not metrics.get('health.api_health', True),
                'severity': AlertSeverity.CRITICAL,
                'message': 'API health check failed',
                'cooldown_minutes': 5
            },
            'database_connection_failed': {
                'condition': lambda metrics: not metrics.get('health.database_connection', True),
                'severity': AlertSeverity.CRITICAL,
                'message': 'Database connection failed',
                'cooldown_minutes': 5
            },
            'data_collection_stale': {
                'condition': lambda metrics: metrics.get('data.collection.hours_since_last', 0) > 6,
                'severity': AlertSeverity.WARNING,
                'message': 'Data collection is stale: {value} hours since last run',
                'cooldown_minutes': 30
            }
        }
    
    def start_processing(self):
        """Start background alert processing."""
        if self._processing_active:
            logger.warning("Alert processing already active")
            return
        
        self._processing_active = True
        self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._processing_thread.start()
        
        logger.info("Alert processing started")
    
    def stop_processing(self):
        """Stop background alert processing."""
        if not self._processing_active:
            return
        
        self._processing_active = False
        if self._processing_thread:
            self._processing_thread.join(timeout=10)
        
        logger.info("Alert processing stopped")
    
    def create_alert(self, alert_id: str, severity: AlertSeverity, title: str, 
                    description: str, source: str = 'system',
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new alert."""
        if not self.enabled:
            return False
        
        # Check if alert is silenced
        if self._is_silenced(alert_id):
            logger.debug("Alert silenced", alert_id=alert_id)
            return False
        
        # Check if alert already exists and is recent
        if alert_id in self.active_alerts:
            existing_alert = self.active_alerts[alert_id]
            last_triggered = datetime.fromisoformat(existing_alert['last_triggered'])
            cooldown_minutes = existing_alert.get('cooldown_minutes', 15)
            
            if datetime.utcnow() - last_triggered < timedelta(minutes=cooldown_minutes):
                logger.debug("Alert in cooldown period", alert_id=alert_id)
                return False
        
        timestamp = datetime.utcnow()
        
        alert_data = {
            'id': alert_id,
            'severity': severity.value,
            'title': title,
            'description': description,
            'source': source,
            'status': AlertStatus.OPEN.value,
            'created_at': timestamp.isoformat(),
            'last_triggered': timestamp.isoformat(),
            'trigger_count': 1,
            'metadata': metadata or {},
            'acknowledgements': []
        }
        
        # Update existing alert or create new one
        if alert_id in self.active_alerts:
            existing_alert = self.active_alerts[alert_id]
            alert_data['trigger_count'] = existing_alert['trigger_count'] + 1
            alert_data['created_at'] = existing_alert['created_at']
        
        self.active_alerts[alert_id] = alert_data
        self.alert_history.append(alert_data.copy())
        
        # Send notifications
        self._send_notifications(alert_data)
        
        logger.info("Alert created", alert_id=alert_id, severity=severity.value, title=title)
        return True
    
    def resolve_alert(self, alert_id: str, resolved_by: str = 'system', 
                     resolution_note: Optional[str] = None) -> bool:
        """Resolve an active alert."""
        if alert_id not in self.active_alerts:
            logger.warning("Alert not found for resolution", alert_id=alert_id)
            return False
        
        alert_data = self.active_alerts[alert_id]
        alert_data['status'] = AlertStatus.RESOLVED.value
        alert_data['resolved_at'] = datetime.utcnow().isoformat()
        alert_data['resolved_by'] = resolved_by
        alert_data['resolution_note'] = resolution_note
        
        # Move to history and remove from active
        self.alert_history.append(alert_data.copy())
        del self.active_alerts[alert_id]
        
        # Send resolution notification
        self._send_resolution_notification(alert_data)
        
        logger.info("Alert resolved", alert_id=alert_id, resolved_by=resolved_by)
        return True
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str,
                         acknowledgement_note: Optional[str] = None) -> bool:
        """Acknowledge an active alert."""
        if alert_id not in self.active_alerts:
            logger.warning("Alert not found for acknowledgement", alert_id=alert_id)
            return False
        
        alert_data = self.active_alerts[alert_id]
        alert_data['status'] = AlertStatus.ACKNOWLEDGED.value
        
        acknowledgement = {
            'acknowledged_by': acknowledged_by,
            'acknowledged_at': datetime.utcnow().isoformat(),
            'note': acknowledgement_note
        }
        
        alert_data['acknowledgements'].append(acknowledgement)
        
        logger.info("Alert acknowledged", alert_id=alert_id, acknowledged_by=acknowledged_by)
        return True
    
    def silence_alert(self, alert_pattern: str, duration_minutes: int = 60) -> bool:
        """Silence alerts matching a pattern."""
        silence_until = datetime.utcnow() + timedelta(minutes=duration_minutes)
        self.silenced_alerts[alert_pattern] = silence_until
        
        logger.info("Alert silenced", pattern=alert_pattern, duration_minutes=duration_minutes)
        return True
    
    def evaluate_metrics(self, metrics: Dict[str, Any]):
        """Evaluate metrics against alert rules."""
        if not self.enabled:
            return
        
        for rule_name, rule in self.alert_rules.items():
            try:
                if rule['condition'](metrics):
                    # Extract value for message formatting
                    value = self._extract_metric_value(rule_name, metrics)
                    message = rule['message'].format(value=value)
                    
                    self.create_alert(
                        alert_id=f"rule_{rule_name}",
                        severity=rule['severity'],
                        title=f"Alert: {rule_name.replace('_', ' ').title()}",
                        description=message,
                        source='metrics_evaluation',
                        metadata={'rule': rule_name, 'value': value}
                    )
                else:
                    # Check if we should resolve the alert
                    alert_id = f"rule_{rule_name}"
                    if alert_id in self.active_alerts:
                        self.resolve_alert(alert_id, 'system', 'Condition no longer met')
                        
            except Exception as e:
                logger.error("Failed to evaluate alert rule", rule=rule_name, error=str(e))
    
    def _send_notifications(self, alert_data: Dict[str, Any]):
        """Send alert notifications through configured channels."""
        try:
            if self.email_enabled and self.email_to:
                self._send_email_notification(alert_data)
            
            if self.slack_enabled and self.slack_webhook_url:
                self._send_slack_notification(alert_data)
            
            if self.webhook_enabled and self.webhook_urls:
                self._send_webhook_notifications(alert_data)
                
        except Exception as e:
            logger.error("Failed to send alert notifications", alert_id=alert_data['id'], error=str(e))
    
    def _send_email_notification(self, alert_data: Dict[str, Any]):
        """Send email notification."""
        try:
            if not all([self.smtp_username, self.smtp_password]):
                logger.warning("Email credentials not configured")
                return
            
            msg = MimeMultipart()
            msg['From'] = self.email_from
            msg['To'] = ', '.join(self.email_to)
            msg['Subject'] = f"[{alert_data['severity'].upper()}] {alert_data['title']}"
            
            body = f"""
Alert Details:
- ID: {alert_data['id']}
- Severity: {alert_data['severity']}
- Source: {alert_data['source']}
- Created: {alert_data['created_at']}
- Description: {alert_data['description']}

Metadata:
{json.dumps(alert_data.get('metadata', {}), indent=2)}

--
Snowflake Analytics Monitoring System
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            server.send_message(msg)
            server.quit()
            
            logger.info("Email notification sent", alert_id=alert_data['id'])
            
        except Exception as e:
            logger.error("Failed to send email notification", error=str(e))
    
    def _send_slack_notification(self, alert_data: Dict[str, Any]):
        """Send Slack notification."""
        try:
            severity_colors = {
                'info': '#36a64f',
                'warning': '#ff9800',
                'critical': '#f44336',
                'fatal': '#9c27b0'
            }
            
            payload = {
                'channel': self.slack_channel,
                'username': 'Analytics Monitor',
                'icon_emoji': ':warning:',
                'attachments': [{
                    'color': severity_colors.get(alert_data['severity'], '#ff9800'),
                    'title': alert_data['title'],
                    'text': alert_data['description'],
                    'fields': [
                        {
                            'title': 'Severity',
                            'value': alert_data['severity'].upper(),
                            'short': True
                        },
                        {
                            'title': 'Source',
                            'value': alert_data['source'],
                            'short': True
                        },
                        {
                            'title': 'Alert ID',
                            'value': alert_data['id'],
                            'short': True
                        },
                        {
                            'title': 'Created',
                            'value': alert_data['created_at'],
                            'short': True
                        }
                    ],
                    'footer': 'Snowflake Analytics',
                    'ts': int(datetime.utcnow().timestamp())
                }]
            }
            
            response = requests.post(self.slack_webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info("Slack notification sent", alert_id=alert_data['id'])
            
        except Exception as e:
            logger.error("Failed to send Slack notification", error=str(e))
    
    def _send_webhook_notifications(self, alert_data: Dict[str, Any]):
        """Send webhook notifications."""
        for webhook_url in self.webhook_urls:
            if not webhook_url.strip():
                continue
                
            try:
                payload = {
                    'event_type': 'alert_created',
                    'alert': alert_data,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                response = requests.post(webhook_url, json=payload, timeout=10)
                response.raise_for_status()
                
                logger.info("Webhook notification sent", webhook_url=webhook_url, alert_id=alert_data['id'])
                
            except Exception as e:
                logger.error("Failed to send webhook notification", webhook_url=webhook_url, error=str(e))
    
    def _send_resolution_notification(self, alert_data: Dict[str, Any]):
        """Send alert resolution notification."""
        # Simplified resolution notification
        if self.slack_enabled and self.slack_webhook_url:
            try:
                payload = {
                    'channel': self.slack_channel,
                    'username': 'Analytics Monitor',
                    'icon_emoji': ':white_check_mark:',
                    'text': f"✅ Alert Resolved: {alert_data['title']} (ID: {alert_data['id']})"
                }
                
                requests.post(self.slack_webhook_url, json=payload, timeout=10)
                
            except Exception as e:
                logger.error("Failed to send resolution notification", error=str(e))
    
    def _is_silenced(self, alert_id: str) -> bool:
        """Check if alert is silenced."""
        current_time = datetime.utcnow()
        
        for pattern, silence_until in list(self.silenced_alerts.items()):
            if current_time > silence_until:
                # Remove expired silences
                del self.silenced_alerts[pattern]
                continue
            
            if pattern in alert_id or alert_id in pattern:
                return True
        
        return False
    
    def _extract_metric_value(self, rule_name: str, metrics: Dict[str, Any]) -> Any:
        """Extract relevant metric value for alert message."""
        metric_mappings = {
            'high_cpu_usage': 'system.cpu.usage_percent',
            'high_memory_usage': 'system.memory.usage_percent',
            'disk_space_low': 'system.disk.usage_percent',
            'data_collection_stale': 'data.collection.hours_since_last'
        }
        
        metric_key = metric_mappings.get(rule_name)
        return metrics.get(metric_key, 'N/A') if metric_key else 'N/A'
    
    def _processing_loop(self):
        """Background processing loop."""
        while self._processing_active:
            try:
                # Clean up resolved alerts from history
                self._cleanup_old_alerts()
                
                # Check for expired silences
                current_time = datetime.utcnow()
                expired_silences = [
                    pattern for pattern, silence_until in self.silenced_alerts.items()
                    if current_time > silence_until
                ]
                
                for pattern in expired_silences:
                    del self.silenced_alerts[pattern]
                    logger.info("Alert silence expired", pattern=pattern)
                
            except Exception as e:
                logger.error("Error in alert processing loop", error=str(e))
            
            time.sleep(60)  # Check every minute
    
    def _cleanup_old_alerts(self):
        """Clean up old alert history."""
        cutoff_time = datetime.utcnow() - timedelta(days=30)
        
        self.alert_history = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['created_at']) > cutoff_time
        ]
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alert history."""
        return sorted(self.alert_history, key=lambda x: x['created_at'], reverse=True)[:limit]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        now = datetime.utcnow()
        last_24h = now - timedelta(hours=24)
        last_week = now - timedelta(days=7)
        
        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['created_at']) > last_24h
        ]
        
        weekly_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['created_at']) > last_week
        ]
        
        return {
            'active_alerts': len(self.active_alerts),
            'alerts_last_24h': len(recent_alerts),
            'alerts_last_week': len(weekly_alerts),
            'silenced_patterns': len(self.silenced_alerts),
            'severity_breakdown': {
                severity.value: len([a for a in self.active_alerts.values() if a['severity'] == severity.value])
                for severity in AlertSeverity
            },
            'notification_channels': {
                'email': self.email_enabled,
                'slack': self.slack_enabled,
                'webhook': self.webhook_enabled
            }
        }