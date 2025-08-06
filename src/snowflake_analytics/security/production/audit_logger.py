"""
Security Audit Logging System for Snowflake Analytics
Comprehensive audit logging for security events, compliance, and forensics.
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import hashlib
import threading
from queue import Queue
import structlog

logger = structlog.get_logger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""
    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"
    
    # Authorization events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_CHANGE = "permission_change"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REVOKED = "role_revoked"
    
    # Data access events
    DATA_READ = "data_read"
    DATA_WRITE = "data_write"
    DATA_DELETE = "data_delete"
    DATA_EXPORT = "data_export"
    
    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIG_CHANGE = "config_change"
    
    # Security events
    SECURITY_VIOLATION = "security_violation"
    INTRUSION_ATTEMPT = "intrusion_attempt"
    ANOMALY_DETECTED = "anomaly_detected"
    
    # API events
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    API_REQUEST = "api_request"
    
    # Admin events
    USER_CREATED = "user_created"
    USER_DELETED = "user_deleted"
    USER_DISABLED = "user_disabled"
    
    # Compliance events
    DATA_RETENTION = "data_retention"
    DATA_PURGE = "data_purge"
    BACKUP_CREATED = "backup_created"
    BACKUP_RESTORED = "backup_restored"


class AuditSeverity(Enum):
    """Severity levels for audit events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditLogger:
    """
    Comprehensive security audit logging system.
    Provides tamper-evident logging for security events and compliance.
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """Initialize audit logger."""
        self.log_file = log_file or os.getenv('AUDIT_LOG_FILE', '/opt/analytics/logs/audit.log')
        self.log_dir = os.path.dirname(self.log_file)
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Configuration
        self.max_log_size = int(os.getenv('AUDIT_LOG_MAX_SIZE', str(100 * 1024 * 1024)))  # 100MB
        self.max_log_files = int(os.getenv('AUDIT_LOG_MAX_FILES', '10'))
        self.buffer_size = int(os.getenv('AUDIT_LOG_BUFFER_SIZE', '1000'))
        
        # Integrity protection
        self.integrity_enabled = os.getenv('AUDIT_INTEGRITY_ENABLED', 'true').lower() == 'true'
        self.integrity_key = os.getenv('AUDIT_INTEGRITY_KEY', 'default-integrity-key')
        
        # Buffered logging for performance
        self._log_buffer: Queue = Queue(maxsize=self.buffer_size)
        self._buffer_thread = threading.Thread(target=self._buffer_worker, daemon=True)
        self._buffer_thread.start()
        
        # Event sequence number for integrity
        self._sequence_number = 0
        self._sequence_lock = threading.Lock()
        
        logger.info("AuditLogger initialized", log_file=self.log_file)
    
    def log_event(self, event_type: AuditEventType, severity: AuditSeverity,
                  user_id: Optional[str] = None, session_id: Optional[str] = None,
                  ip_address: Optional[str] = None, user_agent: Optional[str] = None,
                  resource: Optional[str] = None, action: Optional[str] = None,
                  details: Optional[Dict[str, Any]] = None,
                  outcome: Optional[str] = None) -> bool:
        """Log a security audit event."""
        try:
            timestamp = datetime.utcnow()
            
            with self._sequence_lock:
                self._sequence_number += 1
                sequence = self._sequence_number
            
            audit_record = {
                'timestamp': timestamp.isoformat() + 'Z',
                'sequence': sequence,
                'event_type': event_type.value,
                'severity': severity.value,
                'user_id': user_id,
                'session_id': session_id,
                'ip_address': ip_address,
                'user_agent': user_agent,
                'resource': resource,
                'action': action,
                'outcome': outcome,
                'details': details or {},
                'system_info': {
                    'hostname': os.uname().nodename,
                    'process_id': os.getpid(),
                    'thread_id': threading.current_thread().ident
                }
            }
            
            # Add integrity hash if enabled
            if self.integrity_enabled:
                audit_record['integrity_hash'] = self._calculate_integrity_hash(audit_record)
            
            # Add to buffer for async processing
            try:
                self._log_buffer.put_nowait(audit_record)
                return True
            except:
                # Buffer full, log synchronously
                self._write_log_record(audit_record)
                return True
            
        except Exception as e:
            logger.error("Audit logging failed", event_type=event_type.value, error=str(e))
            return False
    
    def log_authentication_success(self, user_id: str, session_id: str,
                                 ip_address: Optional[str] = None,
                                 user_agent: Optional[str] = None) -> bool:
        """Log successful authentication event."""
        return self.log_event(
            event_type=AuditEventType.LOGIN_SUCCESS,
            severity=AuditSeverity.LOW,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            action="login",
            outcome="success"
        )
    
    def log_authentication_failure(self, username: str, reason: str,
                                 ip_address: Optional[str] = None,
                                 user_agent: Optional[str] = None) -> bool:
        """Log failed authentication event."""
        return self.log_event(
            event_type=AuditEventType.LOGIN_FAILURE,
            severity=AuditSeverity.MEDIUM,
            user_id=username,
            ip_address=ip_address,
            user_agent=user_agent,
            action="login",
            outcome="failure",
            details={'reason': reason}
        )
    
    def log_access_denied(self, user_id: str, resource: str, action: str,
                         reason: str, session_id: Optional[str] = None) -> bool:
        """Log access denied event."""
        return self.log_event(
            event_type=AuditEventType.ACCESS_DENIED,
            severity=AuditSeverity.MEDIUM,
            user_id=user_id,
            session_id=session_id,
            resource=resource,
            action=action,
            outcome="denied",
            details={'reason': reason}
        )
    
    def log_data_access(self, user_id: str, resource: str, action: str,
                       details: Optional[Dict[str, Any]] = None,
                       session_id: Optional[str] = None) -> bool:
        """Log data access event."""
        event_type_map = {
            'read': AuditEventType.DATA_READ,
            'write': AuditEventType.DATA_WRITE,
            'delete': AuditEventType.DATA_DELETE,
            'export': AuditEventType.DATA_EXPORT
        }
        
        event_type = event_type_map.get(action.lower(), AuditEventType.DATA_READ)
        severity = AuditSeverity.HIGH if action.lower() in ['delete', 'export'] else AuditSeverity.LOW
        
        return self.log_event(
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            session_id=session_id,
            resource=resource,
            action=action,
            outcome="success",
            details=details
        )
    
    def log_security_violation(self, violation_type: str, details: Dict[str, Any],
                             user_id: Optional[str] = None,
                             ip_address: Optional[str] = None) -> bool:
        """Log security violation event."""
        return self.log_event(
            event_type=AuditEventType.SECURITY_VIOLATION,
            severity=AuditSeverity.HIGH,
            user_id=user_id,
            ip_address=ip_address,
            action="security_violation",
            outcome="detected",
            details={'violation_type': violation_type, **details}
        )
    
    def log_system_event(self, event_type: AuditEventType, action: str,
                        details: Optional[Dict[str, Any]] = None) -> bool:
        """Log system-level event."""
        return self.log_event(
            event_type=event_type,
            severity=AuditSeverity.MEDIUM,
            action=action,
            outcome="success",
            details=details
        )
    
    def log_api_request(self, user_id: str, endpoint: str, method: str,
                       status_code: int, ip_address: Optional[str] = None,
                       session_id: Optional[str] = None,
                       api_key: Optional[str] = None) -> bool:
        """Log API request event."""
        severity = AuditSeverity.HIGH if status_code >= 400 else AuditSeverity.LOW
        outcome = "success" if status_code < 400 else "failure"
        
        details = {
            'method': method,
            'status_code': status_code,
        }
        
        if api_key:
            details['api_key_prefix'] = api_key[:8] + '...'
        
        return self.log_event(
            event_type=AuditEventType.API_REQUEST,
            severity=severity,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            resource=endpoint,
            action=method.lower(),
            outcome=outcome,
            details=details
        )
    
    def search_events(self, start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None,
                     event_types: Optional[List[AuditEventType]] = None,
                     user_id: Optional[str] = None,
                     severity: Optional[AuditSeverity] = None,
                     limit: int = 1000) -> List[Dict[str, Any]]:
        """Search audit events with filters."""
        try:
            events = []
            
            # Read log file(s)
            log_files = self._get_log_files()
            
            for log_file in log_files:
                if not os.path.exists(log_file):
                    continue
                
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            event = json.loads(line.strip())
                            
                            # Apply filters
                            if start_time or end_time:
                                event_time = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
                                if start_time and event_time < start_time:
                                    continue
                                if end_time and event_time > end_time:
                                    continue
                            
                            if event_types and AuditEventType(event['event_type']) not in event_types:
                                continue
                            
                            if user_id and event.get('user_id') != user_id:
                                continue
                            
                            if severity and AuditSeverity(event['severity']) != severity:
                                continue
                            
                            events.append(event)
                            
                            if len(events) >= limit:
                                break
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue
                
                if len(events) >= limit:
                    break
            
            # Sort by timestamp (newest first)
            events.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return events[:limit]
            
        except Exception as e:
            logger.error("Audit event search failed", error=str(e))
            return []
    
    def generate_audit_report(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate comprehensive audit report."""
        try:
            events = self.search_events(start_time=start_time, end_time=end_time, limit=10000)
            
            report = {
                'report_period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat()
                },
                'total_events': len(events),
                'event_summary': {},
                'severity_summary': {},
                'user_activity': {},
                'security_events': [],
                'top_resources': {},
                'anomalies': []
            }
            
            # Analyze events
            for event in events:
                # Event type summary
                event_type = event['event_type']
                report['event_summary'][event_type] = report['event_summary'].get(event_type, 0) + 1
                
                # Severity summary
                severity = event['severity']
                report['severity_summary'][severity] = report['severity_summary'].get(severity, 0) + 1
                
                # User activity
                user_id = event.get('user_id')
                if user_id:
                    if user_id not in report['user_activity']:
                        report['user_activity'][user_id] = {'total': 0, 'types': {}}
                    report['user_activity'][user_id]['total'] += 1
                    report['user_activity'][user_id]['types'][event_type] = \
                        report['user_activity'][user_id]['types'].get(event_type, 0) + 1
                
                # Security events
                if event_type in ['security_violation', 'intrusion_attempt', 'access_denied']:
                    report['security_events'].append({
                        'timestamp': event['timestamp'],
                        'type': event_type,
                        'user': user_id,
                        'details': event.get('details', {})
                    })
                
                # Resource access
                resource = event.get('resource')
                if resource:
                    report['top_resources'][resource] = report['top_resources'].get(resource, 0) + 1
            
            # Detect anomalies (simple heuristics)
            for user_id, activity in report['user_activity'].items():
                if activity['total'] > 1000:  # High activity
                    report['anomalies'].append({
                        'type': 'high_activity',
                        'user': user_id,
                        'count': activity['total']
                    })
                
                failed_logins = activity['types'].get('login_failure', 0)
                if failed_logins > 10:  # Many failed logins
                    report['anomalies'].append({
                        'type': 'multiple_failures',
                        'user': user_id,
                        'count': failed_logins
                    })
            
            return report
            
        except Exception as e:
            logger.error("Audit report generation failed", error=str(e))
            return {}
    
    def verify_integrity(self, start_sequence: int = 1, end_sequence: Optional[int] = None) -> Dict[str, Any]:
        """Verify integrity of audit logs."""
        if not self.integrity_enabled:
            return {'status': 'disabled', 'message': 'Integrity checking is disabled'}
        
        try:
            verified_events = 0
            corrupted_events = []
            missing_sequences = []
            
            log_files = self._get_log_files()
            
            for log_file in log_files:
                if not os.path.exists(log_file):
                    continue
                
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            event = json.loads(line.strip())
                            sequence = event.get('sequence', 0)
                            
                            # Check sequence range
                            if sequence < start_sequence:
                                continue
                            if end_sequence and sequence > end_sequence:
                                continue
                            
                            # Verify integrity hash
                            stored_hash = event.pop('integrity_hash', None)
                            if stored_hash:
                                calculated_hash = self._calculate_integrity_hash(event)
                                if stored_hash != calculated_hash:
                                    corrupted_events.append({
                                        'sequence': sequence,
                                        'timestamp': event['timestamp'],
                                        'stored_hash': stored_hash,
                                        'calculated_hash': calculated_hash
                                    })
                            
                            verified_events += 1
                            
                        except (json.JSONDecodeError, KeyError):
                            continue
            
            return {
                'status': 'completed',
                'verified_events': verified_events,
                'corrupted_events': len(corrupted_events),
                'corruption_details': corrupted_events,
                'missing_sequences': missing_sequences
            }
            
        except Exception as e:
            logger.error("Integrity verification failed", error=str(e))
            return {'status': 'error', 'message': str(e)}
    
    def _buffer_worker(self):
        """Background worker to process log buffer."""
        while True:
            try:
                # Wait for log record
                record = self._log_buffer.get(timeout=5)
                self._write_log_record(record)
                self._log_buffer.task_done()
            except:
                # Timeout or other error, continue
                continue
    
    def _write_log_record(self, record: Dict[str, Any]):
        """Write a single log record to file."""
        try:
            # Check if log rotation is needed
            if os.path.exists(self.log_file) and os.path.getsize(self.log_file) > self.max_log_size:
                self._rotate_logs()
            
            # Write record
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(record, default=str) + '\n')
                f.flush()
        
        except Exception as e:
            logger.error("Failed to write audit log record", error=str(e))
    
    def _rotate_logs(self):
        """Rotate audit log files."""
        try:
            # Move current log to numbered file
            for i in range(self.max_log_files - 1, 0, -1):
                old_file = f"{self.log_file}.{i}"
                new_file = f"{self.log_file}.{i + 1}"
                
                if os.path.exists(old_file):
                    if i == self.max_log_files - 1:
                        os.remove(old_file)  # Remove oldest file
                    else:
                        os.rename(old_file, new_file)
            
            # Move current log to .1
            if os.path.exists(self.log_file):
                os.rename(self.log_file, f"{self.log_file}.1")
            
            logger.info("Audit log rotated")
            
        except Exception as e:
            logger.error("Audit log rotation failed", error=str(e))
    
    def _get_log_files(self) -> List[str]:
        """Get list of audit log files in order (newest first)."""
        files = [self.log_file]
        
        for i in range(1, self.max_log_files + 1):
            log_file = f"{self.log_file}.{i}"
            if os.path.exists(log_file):
                files.append(log_file)
        
        return files
    
    def _calculate_integrity_hash(self, record: Dict[str, Any]) -> str:
        """Calculate integrity hash for a log record."""
        # Create deterministic string representation
        record_copy = record.copy()
        record_copy.pop('integrity_hash', None)  # Remove hash field if present
        
        record_str = json.dumps(record_copy, sort_keys=True, default=str)
        
        # Calculate HMAC-SHA256
        return hashlib.sha256(f"{self.integrity_key}{record_str}".encode()).hexdigest()
    
    def flush(self):
        """Flush any buffered log records."""
        try:
            # Wait for buffer to be processed
            self._log_buffer.join()
        except:
            pass
    
    def __del__(self):
        """Cleanup on destruction."""
        self.flush()