"""
Authentication System for Snowflake Analytics
Handles user authentication, JWT tokens, multi-factor authentication, and SSO integration.
"""

import os
import jwt
import time
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
import hashlib
import pyotp
import qrcode
from io import BytesIO
import base64
import structlog

from .encryption_manager import EncryptionManager
from .access_control import AccessControl, Role

logger = structlog.get_logger(__name__)


@dataclass
class AuthenticationResult:
    """Result of authentication attempt."""
    success: bool
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    token: Optional[str] = None
    message: Optional[str] = None
    requires_mfa: bool = False
    mfa_token: Optional[str] = None


class AuthenticationManager:
    """
    Comprehensive authentication system with JWT, MFA, and SSO support.
    """
    
    def __init__(self, encryption_manager: Optional[EncryptionManager] = None,
                 access_control: Optional[AccessControl] = None):
        """Initialize authentication manager."""
        self.encryption_manager = encryption_manager or EncryptionManager()
        self.access_control = access_control or AccessControl()
        
        # JWT configuration
        self.jwt_secret = os.getenv('JWT_SECRET_KEY', 'default-secret-change-in-production')
        self.jwt_algorithm = 'HS256'
        self.jwt_expiry_hours = int(os.getenv('JWT_EXPIRY_HOURS', '24'))
        
        # MFA configuration
        self.mfa_enabled = os.getenv('MFA_ENABLED', 'false').lower() == 'true'
        self.mfa_issuer = os.getenv('MFA_ISSUER', 'Snowflake Analytics')
        
        # User storage (in production, this would be a database)
        self._users: Dict[str, Dict[str, Any]] = {}
        self._mfa_pending: Dict[str, Dict[str, Any]] = {}
        
        # SSO configuration
        self.sso_enabled = os.getenv('SSO_ENABLED', 'false').lower() == 'true'
        self.oauth_client_id = os.getenv('OAUTH_CLIENT_ID')
        self.oauth_client_secret = os.getenv('OAUTH_CLIENT_SECRET')
        
        logger.info("AuthenticationManager initialized", 
                   mfa_enabled=self.mfa_enabled, sso_enabled=self.sso_enabled)
    
    def register_user(self, username: str, password: str, email: str,
                     roles: Optional[List[Role]] = None) -> bool:
        """Register a new user."""
        try:
            if username in self._users:
                logger.warning("User registration failed - already exists", user=username)
                return False
            
            # Hash password
            password_data = self.encryption_manager.hash_password(password)
            
            # Create user record
            user_data = {
                'username': username,
                'email': email,
                'password_hash': password_data['hash'],
                'password_salt': password_data['salt'],
                'created_at': datetime.utcnow(),
                'last_login': None,
                'login_attempts': 0,
                'is_active': True,
                'mfa_enabled': False,
                'mfa_secret': None,
                'backup_codes': []
            }
            
            # Encrypt sensitive data
            user_data = self.encryption_manager.encrypt_sensitive_data(user_data)
            
            self._users[username] = user_data
            
            # Assign default roles
            if roles:
                for role in roles:
                    self.access_control.assign_role(username, role)
            else:
                self.access_control.assign_role(username, Role.VIEWER)
            
            logger.info("User registered", user=username, email=email)
            return True
            
        except Exception as e:
            logger.error("User registration failed", user=username, error=str(e))
            return False
    
    def authenticate(self, username: str, password: str, 
                    ip_address: Optional[str] = None,
                    user_agent: Optional[str] = None) -> AuthenticationResult:
        """Authenticate user with username and password."""
        try:
            # Check if user exists
            if username not in self._users:
                self.access_control.record_failed_attempt(username)
                logger.warning("Authentication failed - user not found", user=username)
                return AuthenticationResult(
                    success=False,
                    message="Invalid username or password"
                )
            
            # Check if user is blocked
            if self.access_control.is_user_blocked(username):
                logger.warning("Authentication blocked - user locked", user=username)
                return AuthenticationResult(
                    success=False,
                    message="Account temporarily locked due to failed attempts"
                )
            
            user_data = self.encryption_manager.decrypt_sensitive_data(self._users[username])
            
            # Check if user is active
            if not user_data.get('is_active', False):
                logger.warning("Authentication failed - user inactive", user=username)
                return AuthenticationResult(
                    success=False,
                    message="Account is deactivated"
                )
            
            # Verify password
            if not self.encryption_manager.verify_password(
                password, user_data['password_hash'], user_data['password_salt']
            ):
                self.access_control.record_failed_attempt(username)
                logger.warning("Authentication failed - invalid password", user=username)
                return AuthenticationResult(
                    success=False,
                    message="Invalid username or password"
                )
            
            # Clear failed attempts on successful password verification
            self.access_control.clear_failed_attempts(username)
            
            # Check if MFA is enabled
            if user_data.get('mfa_enabled', False):
                # Generate MFA token for verification step
                mfa_token = self._generate_mfa_token(username)
                self._mfa_pending[mfa_token] = {
                    'username': username,
                    'ip_address': ip_address,
                    'user_agent': user_agent,
                    'created_at': datetime.utcnow()
                }
                
                logger.info("MFA required", user=username)
                return AuthenticationResult(
                    success=False,
                    requires_mfa=True,
                    mfa_token=mfa_token,
                    message="MFA verification required"
                )
            
            # Create session and generate token
            session_id = self.access_control.create_session(username, user_agent, ip_address)
            jwt_token = self._generate_jwt_token(username, session_id)
            
            # Update last login
            user_data['last_login'] = datetime.utcnow()
            self._users[username] = self.encryption_manager.encrypt_sensitive_data(user_data)
            
            logger.info("Authentication successful", user=username)
            return AuthenticationResult(
                success=True,
                user_id=username,
                session_id=session_id,
                token=jwt_token,
                message="Authentication successful"
            )
            
        except Exception as e:
            logger.error("Authentication error", user=username, error=str(e))
            return AuthenticationResult(
                success=False,
                message="Authentication error"
            )
    
    def verify_mfa(self, mfa_token: str, mfa_code: str) -> AuthenticationResult:
        """Verify MFA code and complete authentication."""
        try:
            if mfa_token not in self._mfa_pending:
                logger.warning("Invalid MFA token", token=mfa_token[:8])
                return AuthenticationResult(
                    success=False,
                    message="Invalid or expired MFA token"
                )
            
            mfa_data = self._mfa_pending[mfa_token]
            username = mfa_data['username']
            
            # Check MFA token expiry (5 minutes)
            if datetime.utcnow() - mfa_data['created_at'] > timedelta(minutes=5):
                del self._mfa_pending[mfa_token]
                logger.warning("MFA token expired", user=username)
                return AuthenticationResult(
                    success=False,
                    message="MFA token expired"
                )
            
            user_data = self.encryption_manager.decrypt_sensitive_data(self._users[username])
            mfa_secret = user_data.get('mfa_secret')
            
            if not mfa_secret:
                logger.error("MFA secret not found", user=username)
                return AuthenticationResult(
                    success=False,
                    message="MFA not properly configured"
                )
            
            # Verify TOTP code
            totp = pyotp.TOTP(mfa_secret)
            if not totp.verify(mfa_code, valid_window=1):
                # Check backup codes
                backup_codes = user_data.get('backup_codes', [])
                if mfa_code not in backup_codes:
                    self.access_control.record_failed_attempt(username)
                    logger.warning("MFA verification failed", user=username)
                    return AuthenticationResult(
                        success=False,
                        message="Invalid MFA code"
                    )
                else:
                    # Remove used backup code
                    backup_codes.remove(mfa_code)
                    user_data['backup_codes'] = backup_codes
                    self._users[username] = self.encryption_manager.encrypt_sensitive_data(user_data)
                    logger.info("Backup code used", user=username)
            
            # Clean up MFA pending
            del self._mfa_pending[mfa_token]
            
            # Create session and generate token
            session_id = self.access_control.create_session(
                username, mfa_data.get('user_agent'), mfa_data.get('ip_address')
            )
            jwt_token = self._generate_jwt_token(username, session_id)
            
            # Update last login
            user_data['last_login'] = datetime.utcnow()
            self._users[username] = self.encryption_manager.encrypt_sensitive_data(user_data)
            
            logger.info("MFA authentication successful", user=username)
            return AuthenticationResult(
                success=True,
                user_id=username,
                session_id=session_id,
                token=jwt_token,
                message="Authentication successful"
            )
            
        except Exception as e:
            logger.error("MFA verification error", error=str(e))
            return AuthenticationResult(
                success=False,
                message="MFA verification error"
            )
    
    def verify_token(self, token: str) -> Optional[str]:
        """Verify JWT token and return user_id if valid."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            user_id = payload.get('user_id')
            session_id = payload.get('session_id')
            
            # Validate session
            if not self.access_control.validate_session(session_id):
                logger.warning("Token validation failed - invalid session", user=user_id)
                return None
            
            logger.debug("Token validated", user=user_id)
            return user_id
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token validation failed - expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Token validation failed - invalid")
            return None
        except Exception as e:
            logger.error("Token validation error", error=str(e))
            return None
    
    def logout(self, token: str) -> bool:
        """Logout user by invalidating token/session."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            session_id = payload.get('session_id')
            user_id = payload.get('user_id')
            
            if self.access_control.revoke_session(session_id):
                logger.info("Logout successful", user=user_id)
                return True
            
            return False
            
        except Exception as e:
            logger.error("Logout error", error=str(e))
            return False
    
    def enable_mfa(self, username: str) -> Optional[Dict[str, Any]]:
        """Enable MFA for a user and return setup information."""
        try:
            if username not in self._users:
                return None
            
            user_data = self.encryption_manager.decrypt_sensitive_data(self._users[username])
            
            # Generate MFA secret
            mfa_secret = pyotp.random_base32()
            user_data['mfa_secret'] = mfa_secret
            user_data['mfa_enabled'] = True
            
            # Generate backup codes
            backup_codes = [secrets.token_hex(4) for _ in range(10)]
            user_data['backup_codes'] = backup_codes
            
            # Save user data
            self._users[username] = self.encryption_manager.encrypt_sensitive_data(user_data)
            
            # Generate QR code
            totp_uri = pyotp.totp.TOTP(mfa_secret).provisioning_uri(
                name=user_data['email'],
                issuer_name=self.mfa_issuer
            )
            
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(totp_uri)
            qr.make(fit=True)
            
            qr_img = qr.make_image(fill_color="black", back_color="white")
            buffer = BytesIO()
            qr_img.save(buffer, format='PNG')
            qr_code_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            logger.info("MFA enabled", user=username)
            
            return {
                'secret': mfa_secret,
                'qr_code': qr_code_b64,
                'backup_codes': backup_codes,
                'uri': totp_uri
            }
            
        except Exception as e:
            logger.error("MFA enable error", user=username, error=str(e))
            return None
    
    def disable_mfa(self, username: str) -> bool:
        """Disable MFA for a user."""
        try:
            if username not in self._users:
                return False
            
            user_data = self.encryption_manager.decrypt_sensitive_data(self._users[username])
            user_data['mfa_enabled'] = False
            user_data['mfa_secret'] = None
            user_data['backup_codes'] = []
            
            self._users[username] = self.encryption_manager.encrypt_sensitive_data(user_data)
            
            logger.info("MFA disabled", user=username)
            return True
            
        except Exception as e:
            logger.error("MFA disable error", user=username, error=str(e))
            return False
    
    def change_password(self, username: str, current_password: str, new_password: str) -> bool:
        """Change user password."""
        try:
            if username not in self._users:
                return False
            
            user_data = self.encryption_manager.decrypt_sensitive_data(self._users[username])
            
            # Verify current password
            if not self.encryption_manager.verify_password(
                current_password, user_data['password_hash'], user_data['password_salt']
            ):
                logger.warning("Password change failed - incorrect current password", user=username)
                return False
            
            # Hash new password
            password_data = self.encryption_manager.hash_password(new_password)
            user_data['password_hash'] = password_data['hash']
            user_data['password_salt'] = password_data['salt']
            
            self._users[username] = self.encryption_manager.encrypt_sensitive_data(user_data)
            
            # Revoke all existing sessions
            self.access_control.revoke_user_sessions(username)
            
            logger.info("Password changed", user=username)
            return True
            
        except Exception as e:
            logger.error("Password change error", user=username, error=str(e))
            return False
    
    def reset_password(self, username: str, new_password: str, 
                      reset_token: Optional[str] = None) -> bool:
        """Reset user password (admin or with reset token)."""
        try:
            if username not in self._users:
                return False
            
            # In production, validate reset token here
            # For now, just proceed with password reset
            
            user_data = self.encryption_manager.decrypt_sensitive_data(self._users[username])
            
            # Hash new password
            password_data = self.encryption_manager.hash_password(new_password)
            user_data['password_hash'] = password_data['hash']
            user_data['password_salt'] = password_data['salt']
            
            self._users[username] = self.encryption_manager.encrypt_sensitive_data(user_data)
            
            # Revoke all existing sessions
            self.access_control.revoke_user_sessions(username)
            
            logger.info("Password reset", user=username)
            return True
            
        except Exception as e:
            logger.error("Password reset error", user=username, error=str(e))
            return False
    
    def deactivate_user(self, username: str) -> bool:
        """Deactivate a user account."""
        try:
            if username not in self._users:
                return False
            
            user_data = self.encryption_manager.decrypt_sensitive_data(self._users[username])
            user_data['is_active'] = False
            
            self._users[username] = self.encryption_manager.encrypt_sensitive_data(user_data)
            
            # Revoke all sessions and API keys
            self.access_control.revoke_user_sessions(username)
            
            logger.info("User deactivated", user=username)
            return True
            
        except Exception as e:
            logger.error("User deactivation error", user=username, error=str(e))
            return False
    
    def get_user_info(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user information (excluding sensitive data)."""
        try:
            if username not in self._users:
                return None
            
            user_data = self.encryption_manager.decrypt_sensitive_data(self._users[username])
            
            # Return safe user info
            return {
                'username': user_data['username'],
                'email': user_data['email'],
                'created_at': user_data['created_at'],
                'last_login': user_data.get('last_login'),
                'is_active': user_data.get('is_active', False),
                'mfa_enabled': user_data.get('mfa_enabled', False),
                'roles': [role.value for role in self.access_control.get_user_roles(username)],
                'permissions': [perm.value for perm in self.access_control.get_user_permissions(username)]
            }
            
        except Exception as e:
            logger.error("Get user info error", user=username, error=str(e))
            return None
    
    def _generate_jwt_token(self, user_id: str, session_id: str) -> str:
        """Generate JWT token for authenticated user."""
        payload = {
            'user_id': user_id,
            'session_id': session_id,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(hours=self.jwt_expiry_hours)
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def _generate_mfa_token(self, username: str) -> str:
        """Generate temporary MFA token."""
        return secrets.token_urlsafe(32)
    
    def cleanup_expired_mfa_tokens(self) -> int:
        """Clean up expired MFA tokens."""
        expired_tokens = []
        cutoff_time = datetime.utcnow() - timedelta(minutes=5)
        
        for token, data in self._mfa_pending.items():
            if data['created_at'] < cutoff_time:
                expired_tokens.append(token)
        
        for token in expired_tokens:
            del self._mfa_pending[token]
        
        logger.info("Expired MFA tokens cleaned", count=len(expired_tokens))
        return len(expired_tokens)