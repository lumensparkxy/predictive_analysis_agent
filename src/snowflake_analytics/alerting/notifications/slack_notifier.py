"""
Slack Notification System

Provides Slack integration for team alerts with channel routing,
interactive responses, and alert acknowledgment workflows.
"""

import json
import time
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging


class SlackMessageType(Enum):
    """Types of Slack messages"""
    SIMPLE = "simple"
    RICH = "rich"
    INTERACTIVE = "interactive"


@dataclass
class SlackChannel:
    """Slack channel configuration"""
    channel_id: str
    channel_name: str
    webhook_url: Optional[str] = None
    alert_types: List[str] = field(default_factory=list)
    severity_levels: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'channel_id': self.channel_id,
            'channel_name': self.channel_name,
            'webhook_url': self.webhook_url,
            'alert_types': self.alert_types,
            'severity_levels': self.severity_levels
        }


@dataclass
class SlackNotification:
    """Slack notification configuration"""
    notification_id: str
    channel: str
    message_type: SlackMessageType
    text: str
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    username: Optional[str] = None
    icon_emoji: Optional[str] = None
    thread_ts: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'notification_id': self.notification_id,
            'channel': self.channel,
            'message_type': self.message_type.value,
            'text': self.text,
            'attachments': self.attachments,
            'blocks': self.blocks,
            'username': self.username,
            'icon_emoji': self.icon_emoji,
            'thread_ts': self.thread_ts,
            'tags': self.tags
        }


@dataclass
class SlackDeliveryResult:
    """Slack delivery result"""
    notification_id: str
    success: bool
    timestamp: datetime
    message: str
    channel: str
    slack_ts: Optional[str] = None
    error_details: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'notification_id': self.notification_id,
            'success': self.success,
            'timestamp': self.timestamp.isoformat(),
            'message': self.message,
            'channel': self.channel,
            'slack_ts': self.slack_ts,
            'error_details': self.error_details
        }


class SlackNotifier:
    """
    Slack notification system with rich formatting and interactive features
    
    Provides Slack integration with channel routing, interactive responses,
    alert acknowledgment, and comprehensive delivery tracking.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Slack configuration
        self.bot_token = self.config.get('bot_token', '')
        self.default_webhook_url = self.config.get('default_webhook_url', '')
        self.bot_name = self.config.get('bot_name', 'Snowflake Analytics')
        self.bot_emoji = self.config.get('bot_emoji', ':snowflake:')
        
        # Channel configuration
        self.channels = {}  # channel_name -> SlackChannel
        self.load_channels()
        
        # Delivery settings
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 5)  # seconds
        self.timeout = self.config.get('timeout', 10)  # seconds
        
        # Delivery tracking
        self.delivery_results = []
        self.max_delivery_history = self.config.get('max_delivery_history', 1000)
        
        # Interactive features
        self.acknowledgment_enabled = self.config.get('acknowledgment_enabled', True)
        self.escalation_enabled = self.config.get('escalation_enabled', True)
        
        # Base API URL
        self.api_base_url = "https://slack.com/api"
        
    def load_channels(self):
        """Load channel configuration from config"""
        channels_config = self.config.get('channels', {})
        
        for channel_name, channel_data in channels_config.items():
            self.channels[channel_name] = SlackChannel(
                channel_id=channel_data.get('channel_id', ''),
                channel_name=channel_name,
                webhook_url=channel_data.get('webhook_url', self.default_webhook_url),
                alert_types=channel_data.get('alert_types', []),
                severity_levels=channel_data.get('severity_levels', [])
            )
    
    def send_notification(self, notification: SlackNotification) -> SlackDeliveryResult:
        """
        Send Slack notification
        
        Args:
            notification: Slack notification to send
            
        Returns:
            SlackDeliveryResult: Delivery result
        """
        start_time = time.time()
        
        try:
            # Send notification with retries
            result = self._send_slack_message_with_retries(notification)
            
            # Record delivery result
            self._record_delivery_result(result)
            
            # Log delivery
            elapsed_time = time.time() - start_time
            self.logger.info(f"Slack notification {notification.notification_id} "
                           f"{'sent' if result.success else 'failed'} in {elapsed_time:.2f}s")
            
            return result
            
        except Exception as e:
            result = SlackDeliveryResult(
                notification_id=notification.notification_id,
                success=False,
                timestamp=datetime.now(),
                message=f"Failed to send Slack notification: {str(e)}",
                channel=notification.channel,
                error_details=str(e)
            )
            
            self._record_delivery_result(result)
            self.logger.error(f"Slack notification {notification.notification_id} failed: {e}")
            
            return result
    
    def _send_slack_message_with_retries(self, notification: SlackNotification) -> SlackDeliveryResult:
        """Send Slack message with retry logic"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Determine send method
                if self.bot_token:
                    response = self._send_via_api(notification)
                else:
                    response = self._send_via_webhook(notification)
                
                if response.get('ok'):
                    return SlackDeliveryResult(
                        notification_id=notification.notification_id,
                        success=True,
                        timestamp=datetime.now(),
                        message=f"Slack message sent successfully on attempt {attempt + 1}",
                        channel=notification.channel,
                        slack_ts=response.get('ts')
                    )
                else:
                    raise Exception(f"Slack API error: {response.get('error', 'Unknown error')}")
                    
            except Exception as e:
                last_error = e
                self.logger.warning(f"Slack send attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        return SlackDeliveryResult(
            notification_id=notification.notification_id,
            success=False,
            timestamp=datetime.now(),
            message=f"Failed to send Slack message after {self.max_retries} attempts",
            channel=notification.channel,
            error_details=str(last_error)
        )
    
    def _send_via_api(self, notification: SlackNotification) -> Dict[str, Any]:
        """Send message via Slack API"""
        headers = {
            'Authorization': f'Bearer {self.bot_token}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'channel': notification.channel,
            'text': notification.text,
            'username': notification.username or self.bot_name,
            'icon_emoji': notification.icon_emoji or self.bot_emoji
        }
        
        if notification.attachments:
            payload['attachments'] = notification.attachments
        
        if notification.blocks:
            payload['blocks'] = notification.blocks
        
        if notification.thread_ts:
            payload['thread_ts'] = notification.thread_ts
        
        response = requests.post(
            f"{self.api_base_url}/chat.postMessage",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        
        response.raise_for_status()
        return response.json()
    
    def _send_via_webhook(self, notification: SlackNotification) -> Dict[str, Any]:
        """Send message via webhook"""
        # Get webhook URL for channel
        webhook_url = self._get_webhook_url_for_channel(notification.channel)
        
        if not webhook_url:
            raise Exception(f"No webhook URL configured for channel: {notification.channel}")
        
        payload = {
            'text': notification.text,
            'username': notification.username or self.bot_name,
            'icon_emoji': notification.icon_emoji or self.bot_emoji
        }
        
        if notification.attachments:
            payload['attachments'] = notification.attachments
        
        if notification.blocks:
            payload['blocks'] = notification.blocks
        
        response = requests.post(
            webhook_url,
            json=payload,
            timeout=self.timeout
        )
        
        response.raise_for_status()
        
        # Webhook returns 'ok' for success
        return {'ok': True}
    
    def _get_webhook_url_for_channel(self, channel: str) -> Optional[str]:
        """Get webhook URL for a channel"""
        if channel in self.channels:
            return self.channels[channel].webhook_url
        
        return self.default_webhook_url
    
    def create_alert_notification(self, alert_data: Dict[str, Any], 
                                channel: str, template_data: Dict[str, Any]) -> SlackNotification:
        """
        Create Slack notification for alert
        
        Args:
            alert_data: Alert information
            channel: Target Slack channel
            template_data: Template variables
            
        Returns:
            SlackNotification: Configured Slack notification
        """
        notification_id = f"alert_{alert_data.get('rule_id', 'unknown')}_{int(time.time())}"
        
        # Create rich message with blocks
        blocks = self._create_alert_blocks(alert_data, template_data)
        
        # Create fallback text
        fallback_text = self._create_alert_fallback_text(alert_data)
        
        return SlackNotification(
            notification_id=notification_id,
            channel=channel,
            message_type=SlackMessageType.RICH,
            text=fallback_text,
            blocks=blocks,
            tags={'alert_id': alert_data.get('rule_id', ''), 'type': 'alert'}
        )
    
    def _create_alert_blocks(self, alert_data: Dict[str, Any], template_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create Slack blocks for alert notification"""
        severity = alert_data.get('severity', 'medium')
        severity_emoji = {
            'low': ':large_blue_circle:',
            'medium': ':large_yellow_circle:',
            'high': ':large_orange_circle:',
            'critical': ':red_circle:'
        }.get(severity, ':white_circle:')
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{severity_emoji} {alert_data.get('rule_name', 'Alert Notification')}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Severity:*\n{severity.upper()}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Rule Type:*\n{alert_data.get('rule_type', 'Unknown')}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Team:*\n{alert_data.get('team', 'Not specified')}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Owner:*\n{alert_data.get('owner', 'Not specified')}"
                    }
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Description:*\n{alert_data.get('description', 'No description available')}"
                }
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Rule ID: {alert_data.get('rule_id', 'Unknown')} | "
                               f"Timestamp: {alert_data.get('timestamp', datetime.now().isoformat())}"
                    }
                ]
            }
        ]
        
        # Add interactive elements if enabled
        if self.acknowledgment_enabled:
            blocks.append({
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "Acknowledge"
                        },
                        "style": "primary",
                        "value": f"acknowledge_{alert_data.get('rule_id', 'unknown')}",
                        "action_id": "acknowledge_alert"
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "Resolve"
                        },
                        "style": "primary",
                        "value": f"resolve_{alert_data.get('rule_id', 'unknown')}",
                        "action_id": "resolve_alert"
                    }
                ]
            })
        
        return blocks
    
    def _create_alert_fallback_text(self, alert_data: Dict[str, Any]) -> str:
        """Create fallback text for alert notification"""
        severity = alert_data.get('severity', 'medium')
        return (f"[{severity.upper()}] {alert_data.get('rule_name', 'Alert')} - "
                f"{alert_data.get('description', 'No description available')}")
    
    def send_alert_acknowledgment(self, alert_id: str, user_id: str, 
                                channel: str, timestamp: str) -> SlackDeliveryResult:
        """
        Send alert acknowledgment notification
        
        Args:
            alert_id: Alert identifier
            user_id: User who acknowledged
            channel: Target channel
            timestamp: Original message timestamp
            
        Returns:
            SlackDeliveryResult: Delivery result
        """
        notification_id = f"ack_{alert_id}_{int(time.time())}"
        
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f":white_check_mark: Alert acknowledged by <@{user_id}>"
                }
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Alert ID: {alert_id} | Acknowledged at: {datetime.now().isoformat()}"
                    }
                ]
            }
        ]
        
        notification = SlackNotification(
            notification_id=notification_id,
            channel=channel,
            message_type=SlackMessageType.RICH,
            text=f"Alert {alert_id} acknowledged by user {user_id}",
            blocks=blocks,
            thread_ts=timestamp,
            tags={'alert_id': alert_id, 'type': 'acknowledgment'}
        )
        
        return self.send_notification(notification)
    
    def send_alert_escalation(self, alert_data: Dict[str, Any], 
                            escalation_channel: str, escalation_reason: str) -> SlackDeliveryResult:
        """
        Send alert escalation notification
        
        Args:
            alert_data: Alert information
            escalation_channel: Escalation channel
            escalation_reason: Reason for escalation
            
        Returns:
            SlackDeliveryResult: Delivery result
        """
        notification_id = f"escalation_{alert_data.get('rule_id', 'unknown')}_{int(time.time())}"
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f":exclamation: Alert Escalation"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Alert:* {alert_data.get('rule_name', 'Unknown')}\n"
                           f"*Severity:* {alert_data.get('severity', 'medium').upper()}\n"
                           f"*Escalation Reason:* {escalation_reason}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Description:*\n{alert_data.get('description', 'No description available')}"
                }
            }
        ]
        
        notification = SlackNotification(
            notification_id=notification_id,
            channel=escalation_channel,
            message_type=SlackMessageType.RICH,
            text=f"Alert escalation: {alert_data.get('rule_name', 'Unknown')}",
            blocks=blocks,
            tags={'alert_id': alert_data.get('rule_id', ''), 'type': 'escalation'}
        )
        
        return self.send_notification(notification)
    
    def get_appropriate_channel(self, alert_data: Dict[str, Any]) -> str:
        """
        Get appropriate channel for alert based on routing rules
        
        Args:
            alert_data: Alert information
            
        Returns:
            str: Channel name
        """
        severity = alert_data.get('severity', 'medium')
        rule_type = alert_data.get('rule_type', 'threshold')
        team = alert_data.get('team', '')
        
        # Check each channel for routing rules
        for channel_name, channel_config in self.channels.items():
            # Check severity levels
            if channel_config.severity_levels and severity not in channel_config.severity_levels:
                continue
            
            # Check alert types
            if channel_config.alert_types and rule_type not in channel_config.alert_types:
                continue
            
            # This channel matches the criteria
            return channel_name
        
        # Default to general channel or first configured channel
        if 'general' in self.channels:
            return 'general'
        elif self.channels:
            return list(self.channels.keys())[0]
        else:
            return '#alerts'  # Fallback
    
    def _record_delivery_result(self, result: SlackDeliveryResult):
        """Record delivery result for tracking"""
        self.delivery_results.append(result)
        
        # Maintain history limit
        if len(self.delivery_results) > self.max_delivery_history:
            self.delivery_results = self.delivery_results[-self.max_delivery_history:]
    
    def get_delivery_statistics(self) -> Dict[str, Any]:
        """Get Slack delivery statistics"""
        if not self.delivery_results:
            return {
                'total_sent': 0,
                'successful': 0,
                'failed': 0,
                'success_rate': 0.0
            }
        
        total_sent = len(self.delivery_results)
        successful = sum(1 for r in self.delivery_results if r.success)
        failed = total_sent - successful
        success_rate = (successful / total_sent) * 100 if total_sent > 0 else 0.0
        
        # Channel statistics
        channel_stats = {}
        for result in self.delivery_results:
            channel = result.channel
            if channel not in channel_stats:
                channel_stats[channel] = {'total': 0, 'successful': 0}
            
            channel_stats[channel]['total'] += 1
            if result.success:
                channel_stats[channel]['successful'] += 1
        
        # Add success rates to channel stats
        for channel, stats in channel_stats.items():
            stats['success_rate'] = (stats['successful'] / stats['total']) * 100 if stats['total'] > 0 else 0.0
        
        return {
            'total_sent': total_sent,
            'successful': successful,
            'failed': failed,
            'success_rate': success_rate,
            'channel_statistics': channel_stats
        }
    
    def get_recent_delivery_results(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent delivery results"""
        recent_results = self.delivery_results[-limit:]
        return [r.to_dict() for r in recent_results]
    
    def test_slack_configuration(self) -> Dict[str, Any]:
        """Test Slack configuration"""
        try:
            # Create test notification
            test_notification = SlackNotification(
                notification_id="test_config",
                channel=list(self.channels.keys())[0] if self.channels else '#general',
                message_type=SlackMessageType.SIMPLE,
                text="Test notification to verify Slack configuration",
                tags={'type': 'test'}
            )
            
            # Try to send test notification
            result = self.send_notification(test_notification)
            
            return {
                'configuration_valid': result.success,
                'test_result': result.to_dict(),
                'bot_token_configured': bool(self.bot_token),
                'webhook_url_configured': bool(self.default_webhook_url),
                'channels_configured': len(self.channels),
                'channel_list': list(self.channels.keys())
            }
            
        except Exception as e:
            return {
                'configuration_valid': False,
                'error': str(e),
                'bot_token_configured': bool(self.bot_token),
                'webhook_url_configured': bool(self.default_webhook_url),
                'channels_configured': len(self.channels)
            }
    
    def add_channel(self, channel_name: str, channel_config: Dict[str, Any]) -> bool:
        """Add a new channel configuration"""
        try:
            self.channels[channel_name] = SlackChannel(
                channel_id=channel_config.get('channel_id', ''),
                channel_name=channel_name,
                webhook_url=channel_config.get('webhook_url', self.default_webhook_url),
                alert_types=channel_config.get('alert_types', []),
                severity_levels=channel_config.get('severity_levels', [])
            )
            
            self.logger.info(f"Added Slack channel: {channel_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding Slack channel {channel_name}: {e}")
            return False
    
    def remove_channel(self, channel_name: str) -> bool:
        """Remove a channel configuration"""
        if channel_name in self.channels:
            del self.channels[channel_name]
            self.logger.info(f"Removed Slack channel: {channel_name}")
            return True
        
        return False
    
    def get_channel_list(self) -> List[Dict[str, Any]]:
        """Get list of configured channels"""
        return [channel.to_dict() for channel in self.channels.values()]