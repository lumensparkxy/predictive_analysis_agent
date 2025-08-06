"""
Access Control System for Snowflake Analytics
Role-based access control (RBAC), permissions management, and authorization.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum
import structlog
import json

logger = structlog.get_logger(__name__)


class Permission(Enum):
    """System permissions enumeration."""
    # Data access permissions
    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    DELETE_DATA = "delete_data"
    
    # Analytics permissions
    VIEW_ANALYTICS = "view_analytics"
    CREATE_ANALYTICS = "create_analytics"
    MODIFY_ANALYTICS = "modify_analytics"
    
    # System permissions
    ADMIN_ACCESS = "admin_access"
    USER_MANAGEMENT = "user_management"
    SYSTEM_CONFIG = "system_config"
    
    # API permissions
    API_ACCESS = "api_access"
    API_WRITE = "api_write"
    API_ADMIN = "api_admin"
    
    # Monitoring permissions
    VIEW_METRICS = "view_metrics"
    VIEW_LOGS = "view_logs"
    VIEW_HEALTH = "view_health"


class Role(Enum):
    """System roles enumeration."""
    VIEWER = "viewer"
    ANALYST = "analyst"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"
    API_USER = "api_user"
    SERVICE = "service"


class AccessControl:
    """
    Comprehensive access control system with RBAC, permissions, and session management.
    """
    
    def __init__(self):
        """Initialize access control system."""
        self._role_permissions: Dict[Role, Set[Permission]] = self._initialize_role_permissions()
        self._user_roles: Dict[str, Set[Role]] = {}
        self._user_sessions: Dict[str, Dict[str, Any]] = {}
        self._api_keys: Dict[str, Dict[str, Any]] = {}
        self._failed_attempts: Dict[str, List[datetime]] = {}
        self._blocked_users: Dict[str, datetime] = {}
        
        # Configuration
        self.max_failed_attempts = int(os.getenv('MAX_LOGIN_ATTEMPTS', '5'))
        self.lockout_duration = int(os.getenv('LOCKOUT_DURATION', '900'))  # 15 minutes
        self.session_timeout = int(os.getenv('SESSION_TIMEOUT', '3600'))  # 1 hour
        self.api_key_expiry = int(os.getenv('API_KEY_EXPIRY_HOURS', '24')) * 3600
        
        logger.info("AccessControl initialized")
    
    def _initialize_role_permissions(self) -> Dict[Role, Set[Permission]]:
        """Initialize default role-permission mappings."""
        role_permissions = {
            Role.VIEWER: {
                Permission.READ_DATA,
                Permission.VIEW_ANALYTICS,
                Permission.VIEW_METRICS,
                Permission.VIEW_HEALTH,
                Permission.API_ACCESS
            },
            Role.ANALYST: {
                Permission.READ_DATA,
                Permission.WRITE_DATA,
                Permission.VIEW_ANALYTICS,
                Permission.CREATE_ANALYTICS,
                Permission.MODIFY_ANALYTICS,
                Permission.VIEW_METRICS,
                Permission.VIEW_LOGS,
                Permission.VIEW_HEALTH,
                Permission.API_ACCESS
            },
            Role.ADMIN: {
                Permission.READ_DATA,
                Permission.WRITE_DATA,
                Permission.DELETE_DATA,
                Permission.VIEW_ANALYTICS,
                Permission.CREATE_ANALYTICS,
                Permission.MODIFY_ANALYTICS,
                Permission.USER_MANAGEMENT,
                Permission.SYSTEM_CONFIG,
                Permission.VIEW_METRICS,
                Permission.VIEW_LOGS,
                Permission.VIEW_HEALTH,
                Permission.API_ACCESS,
                Permission.API_WRITE,
                Permission.API_ADMIN
            },
            Role.SUPER_ADMIN: set(Permission),  # All permissions
            Role.API_USER: {
                Permission.READ_DATA,
                Permission.VIEW_ANALYTICS,
                Permission.API_ACCESS
            },
            Role.SERVICE: {
                Permission.READ_DATA,
                Permission.WRITE_DATA,
                Permission.VIEW_ANALYTICS,
                Permission.CREATE_ANALYTICS,
                Permission.API_ACCESS,
                Permission.API_WRITE
            }
        }
        
        return role_permissions
    
    def assign_role(self, user_id: str, role: Role) -> bool:
        """Assign a role to a user."""
        try:
            if user_id not in self._user_roles:
                self._user_roles[user_id] = set()
            
            self._user_roles[user_id].add(role)
            
            logger.info("Role assigned", user=user_id, role=role.value)
            return True
            
        except Exception as e:
            logger.error("Role assignment failed", user=user_id, role=role.value, error=str(e))
            return False
    
    def revoke_role(self, user_id: str, role: Role) -> bool:
        """Revoke a role from a user."""
        try:
            if user_id in self._user_roles and role in self._user_roles[user_id]:
                self._user_roles[user_id].remove(role)
                
                # Clean up empty role sets
                if not self._user_roles[user_id]:
                    del self._user_roles[user_id]
                
                logger.info("Role revoked", user=user_id, role=role.value)
                return True
            
            logger.warning("Role not found for revocation", user=user_id, role=role.value)
            return False
            
        except Exception as e:
            logger.error("Role revocation failed", user=user_id, role=role.value, error=str(e))
            return False
    
    def get_user_roles(self, user_id: str) -> Set[Role]:
        """Get all roles assigned to a user."""
        return self._user_roles.get(user_id, set())
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for a user based on their roles."""
        user_roles = self.get_user_roles(user_id)
        permissions = set()
        
        for role in user_roles:
            if role in self._role_permissions:
                permissions.update(self._role_permissions[role])
        
        return permissions
    
    def has_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if a user has a specific permission."""
        user_permissions = self.get_user_permissions(user_id)
        has_perm = permission in user_permissions
        
        logger.debug("Permission check", user=user_id, permission=permission.value, granted=has_perm)
        return has_perm
    
    def has_role(self, user_id: str, role: Role) -> bool:
        """Check if a user has a specific role."""
        user_roles = self.get_user_roles(user_id)
        has_role_result = role in user_roles
        
        logger.debug("Role check", user=user_id, role=role.value, granted=has_role_result)
        return has_role_result
    
    def create_session(self, user_id: str, user_agent: Optional[str] = None, 
                      ip_address: Optional[str] = None) -> str:
        """Create a new user session."""
        if self.is_user_blocked(user_id):
            logger.warning("Session creation blocked - user locked", user=user_id)
            raise PermissionError("Account temporarily locked")
        
        session_id = self._generate_session_id()
        session_data = {
            'user_id': user_id,
            'created_at': datetime.utcnow(),
            'last_activity': datetime.utcnow(),
            'user_agent': user_agent,
            'ip_address': ip_address,
            'active': True
        }
        
        self._user_sessions[session_id] = session_data
        
        logger.info("Session created", user=user_id, session=session_id[:8])
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[str]:
        """Validate a session and return user_id if valid."""
        if session_id not in self._user_sessions:
            return None
        
        session = self._user_sessions[session_id]
        
        # Check if session is active
        if not session.get('active', False):
            return None
        
        # Check session timeout
        last_activity = session.get('last_activity', datetime.utcnow())
        if datetime.utcnow() - last_activity > timedelta(seconds=self.session_timeout):
            self.revoke_session(session_id)
            logger.warning("Session expired", session=session_id[:8])
            return None
        
        # Update last activity
        session['last_activity'] = datetime.utcnow()
        
        user_id = session.get('user_id')
        logger.debug("Session validated", user=user_id, session=session_id[:8])
        return user_id
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke a user session."""
        if session_id in self._user_sessions:
            user_id = self._user_sessions[session_id].get('user_id')
            self._user_sessions[session_id]['active'] = False
            
            logger.info("Session revoked", user=user_id, session=session_id[:8])
            return True
        
        return False
    
    def revoke_user_sessions(self, user_id: str) -> int:
        """Revoke all sessions for a user."""
        revoked_count = 0
        
        for session_id, session in self._user_sessions.items():
            if session.get('user_id') == user_id and session.get('active', False):
                session['active'] = False
                revoked_count += 1
        
        logger.info("User sessions revoked", user=user_id, count=revoked_count)
        return revoked_count
    
    def generate_api_key(self, user_id: str, name: str, permissions: Optional[Set[Permission]] = None) -> str:
        """Generate an API key for a user."""
        from .encryption_manager import EncryptionManager
        
        encryption_manager = EncryptionManager()
        api_key = encryption_manager.generate_api_key()
        
        # Limit permissions to user's permissions
        user_permissions = self.get_user_permissions(user_id)
        if permissions:
            api_permissions = permissions.intersection(user_permissions)
        else:
            api_permissions = user_permissions
        
        api_key_data = {
            'user_id': user_id,
            'name': name,
            'permissions': [p.value for p in api_permissions],
            'created_at': datetime.utcnow(),
            'expires_at': datetime.utcnow() + timedelta(seconds=self.api_key_expiry),
            'active': True
        }
        
        self._api_keys[api_key] = api_key_data
        
        logger.info("API key generated", user=user_id, name=name, key=api_key[:8])
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[Tuple[str, Set[Permission]]]:
        """Validate an API key and return user_id and permissions."""
        if api_key not in self._api_keys:
            logger.warning("Invalid API key", key=api_key[:8])
            return None
        
        key_data = self._api_keys[api_key]
        
        # Check if key is active
        if not key_data.get('active', False):
            logger.warning("Inactive API key", key=api_key[:8])
            return None
        
        # Check expiration
        expires_at = key_data.get('expires_at')
        if expires_at and datetime.utcnow() > expires_at:
            key_data['active'] = False
            logger.warning("Expired API key", key=api_key[:8])
            return None
        
        user_id = key_data.get('user_id')
        permissions_list = key_data.get('permissions', [])
        permissions = {Permission(p) for p in permissions_list}
        
        logger.debug("API key validated", user=user_id, key=api_key[:8])
        return user_id, permissions
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        if api_key in self._api_keys:
            user_id = self._api_keys[api_key].get('user_id')
            self._api_keys[api_key]['active'] = False
            
            logger.info("API key revoked", user=user_id, key=api_key[:8])
            return True
        
        return False
    
    def record_failed_attempt(self, user_id: str) -> None:
        """Record a failed authentication attempt."""
        now = datetime.utcnow()
        
        if user_id not in self._failed_attempts:
            self._failed_attempts[user_id] = []
        
        # Clean old attempts (older than lockout duration)
        cutoff = now - timedelta(seconds=self.lockout_duration)
        self._failed_attempts[user_id] = [
            attempt for attempt in self._failed_attempts[user_id]
            if attempt > cutoff
        ]
        
        # Add new attempt
        self._failed_attempts[user_id].append(now)
        
        # Check if user should be blocked
        if len(self._failed_attempts[user_id]) >= self.max_failed_attempts:
            self._blocked_users[user_id] = now + timedelta(seconds=self.lockout_duration)
            logger.warning("User blocked due to failed attempts", user=user_id)
        
        logger.info("Failed attempt recorded", user=user_id, 
                   attempts=len(self._failed_attempts[user_id]))
    
    def clear_failed_attempts(self, user_id: str) -> None:
        """Clear failed attempts for a user (on successful auth)."""
        if user_id in self._failed_attempts:
            del self._failed_attempts[user_id]
        
        if user_id in self._blocked_users:
            del self._blocked_users[user_id]
        
        logger.debug("Failed attempts cleared", user=user_id)
    
    def is_user_blocked(self, user_id: str) -> bool:
        """Check if a user is currently blocked."""
        if user_id not in self._blocked_users:
            return False
        
        block_until = self._blocked_users[user_id]
        if datetime.utcnow() > block_until:
            # Block expired, remove it
            del self._blocked_users[user_id]
            if user_id in self._failed_attempts:
                del self._failed_attempts[user_id]
            return False
        
        return True
    
    def get_security_context(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive security context for a user."""
        return {
            'user_id': user_id,
            'roles': [role.value for role in self.get_user_roles(user_id)],
            'permissions': [perm.value for perm in self.get_user_permissions(user_id)],
            'active_sessions': len([
                s for s in self._user_sessions.values()
                if s.get('user_id') == user_id and s.get('active', False)
            ]),
            'api_keys': len([
                k for k in self._api_keys.values()
                if k.get('user_id') == user_id and k.get('active', False)
            ]),
            'is_blocked': self.is_user_blocked(user_id),
            'failed_attempts': len(self._failed_attempts.get(user_id, []))
        }
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions and return count removed."""
        expired_sessions = []
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.session_timeout)
        
        for session_id, session in self._user_sessions.items():
            last_activity = session.get('last_activity', datetime.utcnow())
            if last_activity < cutoff_time:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.revoke_session(session_id)
        
        logger.info("Expired sessions cleaned", count=len(expired_sessions))
        return len(expired_sessions)
    
    def cleanup_expired_api_keys(self) -> int:
        """Clean up expired API keys and return count removed."""
        expired_keys = []
        now = datetime.utcnow()
        
        for api_key, key_data in self._api_keys.items():
            expires_at = key_data.get('expires_at')
            if expires_at and now > expires_at:
                expired_keys.append(api_key)
        
        for api_key in expired_keys:
            self.revoke_api_key(api_key)
        
        logger.info("Expired API keys cleaned", count=len(expired_keys))
        return len(expired_keys)
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        import secrets
        return secrets.token_urlsafe(32)
    
    def require_permission(self, permission: Permission):
        """Decorator to require specific permission."""
        def decorator(func):
            def wrapper(self, user_id: str, *args, **kwargs):
                if not self.has_permission(user_id, permission):
                    logger.warning("Permission denied", user=user_id, permission=permission.value)
                    raise PermissionError(f"Permission required: {permission.value}")
                return func(self, user_id, *args, **kwargs)
            return wrapper
        return decorator
    
    def require_role(self, role: Role):
        """Decorator to require specific role."""
        def decorator(func):
            def wrapper(self, user_id: str, *args, **kwargs):
                if not self.has_role(user_id, role):
                    logger.warning("Role required", user=user_id, role=role.value)
                    raise PermissionError(f"Role required: {role.value}")
                return func(self, user_id, *args, **kwargs)
            return wrapper
        return decorator