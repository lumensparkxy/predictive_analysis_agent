"""
Email Notification System

Provides email notification delivery with HTML formatting, attachment support,
template customization, and delivery confirmation tracking.
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import logging
import json
import os


@dataclass
class EmailRecipient:
    """Email recipient information"""
    email: str
    name: Optional[str] = None
    recipient_type: str = "to"  # "to", "cc", "bcc"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'email': self.email,
            'name': self.name,
            'recipient_type': self.recipient_type
        }


@dataclass
class EmailAttachment:
    """Email attachment information"""
    filename: str
    filepath: str
    content_type: str = "application/octet-stream"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'filename': self.filename,
            'filepath': self.filepath,
            'content_type': self.content_type
        }


@dataclass
class EmailNotification:
    """Email notification configuration"""
    notification_id: str
    subject: str
    html_body: str
    text_body: Optional[str] = None
    recipients: List[EmailRecipient] = field(default_factory=list)
    attachments: List[EmailAttachment] = field(default_factory=list)
    priority: str = "normal"  # "low", "normal", "high"
    tags: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'notification_id': self.notification_id,
            'subject': self.subject,
            'html_body': self.html_body,
            'text_body': self.text_body,
            'recipients': [r.to_dict() for r in self.recipients],
            'attachments': [a.to_dict() for a in self.attachments],
            'priority': self.priority,
            'tags': self.tags
        }


@dataclass
class EmailDeliveryResult:
    """Email delivery result"""
    notification_id: str
    success: bool
    timestamp: datetime
    message: str
    smtp_response: Optional[str] = None
    error_details: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'notification_id': self.notification_id,
            'success': self.success,
            'timestamp': self.timestamp.isoformat(),
            'message': self.message,
            'smtp_response': self.smtp_response,
            'error_details': self.error_details
        }


class EmailNotifier:
    """
    Email notification system with rich formatting and delivery tracking
    
    Provides email delivery with HTML formatting, attachment support,
    template customization, and comprehensive delivery tracking.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # SMTP configuration
        self.smtp_server = self.config.get('smtp_server', 'localhost')
        self.smtp_port = self.config.get('smtp_port', 587)
        self.smtp_username = self.config.get('smtp_username', '')
        self.smtp_password = self.config.get('smtp_password', '')
        self.use_tls = self.config.get('use_tls', True)
        self.use_ssl = self.config.get('use_ssl', False)
        
        # Email configuration
        self.sender_email = self.config.get('sender_email', '')
        self.sender_name = self.config.get('sender_name', 'Snowflake Analytics')
        self.reply_to = self.config.get('reply_to', '')
        
        # Delivery settings
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 30)  # seconds
        self.timeout = self.config.get('timeout', 30)  # seconds
        
        # Delivery tracking
        self.delivery_results = []
        self.max_delivery_history = self.config.get('max_delivery_history', 1000)
        
        # Validate configuration
        self._validate_configuration()
        
    def _validate_configuration(self):
        """Validate email configuration"""
        if not self.sender_email:
            raise ValueError("sender_email is required")
        
        if not self.smtp_server:
            raise ValueError("smtp_server is required")
    
    def send_notification(self, notification: EmailNotification) -> EmailDeliveryResult:
        """
        Send email notification
        
        Args:
            notification: Email notification to send
            
        Returns:
            EmailDeliveryResult: Delivery result
        """
        start_time = time.time()
        
        try:
            # Create email message
            message = self._create_email_message(notification)
            
            # Send email with retries
            result = self._send_email_with_retries(message, notification)
            
            # Record delivery result
            self._record_delivery_result(result)
            
            # Log delivery
            elapsed_time = time.time() - start_time
            self.logger.info(f"Email notification {notification.notification_id} "
                           f"{'sent' if result.success else 'failed'} in {elapsed_time:.2f}s")
            
            return result
            
        except Exception as e:
            result = EmailDeliveryResult(
                notification_id=notification.notification_id,
                success=False,
                timestamp=datetime.now(),
                message=f"Failed to send email: {str(e)}",
                error_details=str(e)
            )
            
            self._record_delivery_result(result)
            self.logger.error(f"Email notification {notification.notification_id} failed: {e}")
            
            return result
    
    def _create_email_message(self, notification: EmailNotification) -> MIMEMultipart:
        """Create email message from notification"""
        message = MIMEMultipart("alternative")
        
        # Set headers
        message["Subject"] = notification.subject
        message["From"] = f"{self.sender_name} <{self.sender_email}>"
        message["Reply-To"] = self.reply_to or self.sender_email
        
        # Set recipients
        to_recipients = [r.email for r in notification.recipients if r.recipient_type == "to"]
        cc_recipients = [r.email for r in notification.recipients if r.recipient_type == "cc"]
        bcc_recipients = [r.email for r in notification.recipients if r.recipient_type == "bcc"]
        
        if to_recipients:
            message["To"] = ", ".join(to_recipients)
        if cc_recipients:
            message["Cc"] = ", ".join(cc_recipients)
        # BCC recipients are not added to headers
        
        # Set priority
        if notification.priority == "high":
            message["X-Priority"] = "1"
            message["X-MSMail-Priority"] = "High"
        elif notification.priority == "low":
            message["X-Priority"] = "5"
            message["X-MSMail-Priority"] = "Low"
        
        # Add text body if provided
        if notification.text_body:
            text_part = MIMEText(notification.text_body, "plain")
            message.attach(text_part)
        
        # Add HTML body
        html_part = MIMEText(notification.html_body, "html")
        message.attach(html_part)
        
        # Add attachments
        for attachment in notification.attachments:
            self._add_attachment(message, attachment)
        
        return message
    
    def _add_attachment(self, message: MIMEMultipart, attachment: EmailAttachment):
        """Add attachment to email message"""
        try:
            if not os.path.exists(attachment.filepath):
                self.logger.warning(f"Attachment file not found: {attachment.filepath}")
                return
            
            with open(attachment.filepath, "rb") as attachment_file:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment_file.read())
            
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename= {attachment.filename}",
            )
            
            message.attach(part)
            
        except Exception as e:
            self.logger.error(f"Error adding attachment {attachment.filename}: {e}")
    
    def _send_email_with_retries(self, message: MIMEMultipart, notification: EmailNotification) -> EmailDeliveryResult:
        """Send email with retry logic"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Get all recipients
                all_recipients = [r.email for r in notification.recipients]
                
                # Send email
                smtp_response = self._send_via_smtp(message, all_recipients)
                
                return EmailDeliveryResult(
                    notification_id=notification.notification_id,
                    success=True,
                    timestamp=datetime.now(),
                    message=f"Email sent successfully on attempt {attempt + 1}",
                    smtp_response=smtp_response
                )
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"Email send attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        return EmailDeliveryResult(
            notification_id=notification.notification_id,
            success=False,
            timestamp=datetime.now(),
            message=f"Failed to send email after {self.max_retries} attempts",
            error_details=str(last_error)
        )
    
    def _send_via_smtp(self, message: MIMEMultipart, recipients: List[str]) -> str:
        """Send email via SMTP"""
        if self.use_ssl:
            server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, timeout=self.timeout)
        else:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=self.timeout)
        
        try:
            if self.use_tls and not self.use_ssl:
                server.starttls(context=ssl.create_default_context())
            
            if self.smtp_username and self.smtp_password:
                server.login(self.smtp_username, self.smtp_password)
            
            response = server.send_message(message, to_addrs=recipients)
            
            return str(response)
            
        finally:
            server.quit()
    
    def _record_delivery_result(self, result: EmailDeliveryResult):
        """Record delivery result for tracking"""
        self.delivery_results.append(result)
        
        # Maintain history limit
        if len(self.delivery_results) > self.max_delivery_history:
            self.delivery_results = self.delivery_results[-self.max_delivery_history:]
    
    def send_bulk_notifications(self, notifications: List[EmailNotification]) -> List[EmailDeliveryResult]:
        """
        Send multiple email notifications
        
        Args:
            notifications: List of email notifications
            
        Returns:
            List[EmailDeliveryResult]: Delivery results
        """
        results = []
        
        for notification in notifications:
            result = self.send_notification(notification)
            results.append(result)
        
        success_count = sum(1 for r in results if r.success)
        self.logger.info(f"Bulk email send completed: {success_count}/{len(notifications)} successful")
        
        return results
    
    def create_alert_notification(self, alert_data: Dict[str, Any], 
                                recipients: List[str], template_data: Dict[str, Any]) -> EmailNotification:
        """
        Create email notification for alert
        
        Args:
            alert_data: Alert information
            recipients: List of recipient email addresses
            template_data: Template variables
            
        Returns:
            EmailNotification: Configured email notification
        """
        notification_id = f"alert_{alert_data.get('rule_id', 'unknown')}_{int(time.time())}"
        
        # Create subject
        subject = f"[{alert_data.get('severity', 'MEDIUM').upper()}] {alert_data.get('rule_name', 'Alert')}"
        
        # Create HTML body
        html_body = self._create_alert_html_body(alert_data, template_data)
        
        # Create text body
        text_body = self._create_alert_text_body(alert_data, template_data)
        
        # Create recipient objects
        email_recipients = [
            EmailRecipient(email=email, recipient_type="to") 
            for email in recipients
        ]
        
        return EmailNotification(
            notification_id=notification_id,
            subject=subject,
            html_body=html_body,
            text_body=text_body,
            recipients=email_recipients,
            priority=self._get_priority_from_severity(alert_data.get('severity', 'medium')),
            tags={'alert_id': alert_data.get('rule_id', ''), 'type': 'alert'}
        )
    
    def _create_alert_html_body(self, alert_data: Dict[str, Any], template_data: Dict[str, Any]) -> str:
        """Create HTML body for alert email"""
        severity = alert_data.get('severity', 'medium')
        severity_color = {
            'low': '#28a745',
            'medium': '#ffc107',
            'high': '#fd7e14',
            'critical': '#dc3545'
        }.get(severity, '#6c757d')
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .alert-header {{ background-color: {severity_color}; color: white; padding: 15px; border-radius: 5px; }}
                .alert-body {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-top: 10px; }}
                .alert-details {{ background-color: white; padding: 15px; border-radius: 5px; margin-top: 10px; }}
                .metric-info {{ background-color: #e9ecef; padding: 10px; border-radius: 3px; margin: 5px 0; }}
                .footer {{ margin-top: 20px; padding-top: 20px; border-top: 1px solid #dee2e6; color: #6c757d; }}
            </style>
        </head>
        <body>
            <div class="alert-header">
                <h2>{alert_data.get('rule_name', 'Alert Notification')}</h2>
                <p>Severity: {severity.upper()}</p>
            </div>
            
            <div class="alert-body">
                <p><strong>Description:</strong> {alert_data.get('description', 'No description available')}</p>
                <p><strong>Timestamp:</strong> {alert_data.get('timestamp', datetime.now().isoformat())}</p>
                <p><strong>Rule ID:</strong> {alert_data.get('rule_id', 'Unknown')}</p>
                <p><strong>Rule Type:</strong> {alert_data.get('rule_type', 'Unknown')}</p>
            </div>
            
            <div class="alert-details">
                <h3>Alert Details</h3>
                <div class="metric-info">
                    <p><strong>Team:</strong> {alert_data.get('team', 'Not specified')}</p>
                    <p><strong>Owner:</strong> {alert_data.get('owner', 'Not specified')}</p>
                    <p><strong>Tags:</strong> {json.dumps(alert_data.get('tags', {}), indent=2)}</p>
                </div>
            </div>
            
            <div class="footer">
                <p>This is an automated alert from Snowflake Analytics Monitoring System.</p>
                <p>Timestamp: {datetime.now().isoformat()}</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_alert_text_body(self, alert_data: Dict[str, Any], template_data: Dict[str, Any]) -> str:
        """Create text body for alert email"""
        text = f"""
ALERT NOTIFICATION

Rule: {alert_data.get('rule_name', 'Unknown')}
Severity: {alert_data.get('severity', 'medium').upper()}
Description: {alert_data.get('description', 'No description available')}

Details:
- Rule ID: {alert_data.get('rule_id', 'Unknown')}
- Rule Type: {alert_data.get('rule_type', 'Unknown')}
- Team: {alert_data.get('team', 'Not specified')}
- Owner: {alert_data.get('owner', 'Not specified')}
- Timestamp: {alert_data.get('timestamp', datetime.now().isoformat())}

Tags: {json.dumps(alert_data.get('tags', {}), indent=2)}

---
This is an automated alert from Snowflake Analytics Monitoring System.
Generated at: {datetime.now().isoformat()}
        """
        
        return text.strip()
    
    def _get_priority_from_severity(self, severity: str) -> str:
        """Get email priority from alert severity"""
        severity_priority_map = {
            'low': 'low',
            'medium': 'normal',
            'high': 'high',
            'critical': 'high'
        }
        
        return severity_priority_map.get(severity, 'normal')
    
    def get_delivery_statistics(self) -> Dict[str, Any]:
        """Get email delivery statistics"""
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
        
        # Recent statistics (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_results = [r for r in self.delivery_results if r.timestamp >= recent_cutoff]
        
        recent_total = len(recent_results)
        recent_successful = sum(1 for r in recent_results if r.success)
        recent_success_rate = (recent_successful / recent_total) * 100 if recent_total > 0 else 0.0
        
        return {
            'total_sent': total_sent,
            'successful': successful,
            'failed': failed,
            'success_rate': success_rate,
            'recent_24h': {
                'total_sent': recent_total,
                'successful': recent_successful,
                'failed': recent_total - recent_successful,
                'success_rate': recent_success_rate
            }
        }
    
    def get_recent_delivery_results(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent delivery results"""
        recent_results = self.delivery_results[-limit:]
        return [r.to_dict() for r in recent_results]
    
    def test_email_configuration(self) -> Dict[str, Any]:
        """Test email configuration"""
        try:
            # Create test message
            test_notification = EmailNotification(
                notification_id="test_config",
                subject="Test Email Configuration",
                html_body="<p>This is a test email to verify configuration.</p>",
                text_body="This is a test email to verify configuration.",
                recipients=[EmailRecipient(email=self.sender_email, recipient_type="to")],
                priority="normal"
            )
            
            # Try to send test email
            result = self.send_notification(test_notification)
            
            return {
                'configuration_valid': result.success,
                'test_result': result.to_dict(),
                'smtp_server': self.smtp_server,
                'smtp_port': self.smtp_port,
                'sender_email': self.sender_email,
                'use_tls': self.use_tls,
                'use_ssl': self.use_ssl
            }
            
        except Exception as e:
            return {
                'configuration_valid': False,
                'error': str(e),
                'smtp_server': self.smtp_server,
                'smtp_port': self.smtp_port,
                'sender_email': self.sender_email
            }