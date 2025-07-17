"""
Simple authentication middleware.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import hashlib
import secrets
import logging

logger = logging.getLogger(__name__)

class SimpleAuthMiddleware:
    """Simple authentication middleware for API endpoints."""
    
    def __init__(self):
        self.logger = logger
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._initialize_default_keys()
    
    def _initialize_default_keys(self):
        """Initialize default API keys for demo purposes."""
        # In production, these would be loaded from secure storage
        self.api_keys = {
            "demo_key_001": {
                "name": "Demo Dashboard Key",
                "permissions": ["read", "write"],
                "created_at": datetime.now().isoformat(),
                "last_used": None,
                "enabled": True
            },
            "admin_key_001": {
                "name": "Admin Key",
                "permissions": ["read", "write", "admin"],
                "created_at": datetime.now().isoformat(),
                "last_used": None,
                "enabled": True
            }
        }
    
    def generate_api_key(self, name: str, permissions: list = None) -> str:
        """
        Generate a new API key.
        
        Args:
            name: Name/description for the API key
            permissions: List of permissions for the key
            
        Returns:
            Generated API key string
        """
        if permissions is None:
            permissions = ["read"]
        
        # Generate random API key
        key = secrets.token_urlsafe(32)
        
        # Store key information
        self.api_keys[key] = {
            "name": name,
            "permissions": permissions,
            "created_at": datetime.now().isoformat(),
            "last_used": None,
            "enabled": True
        }
        
        return key
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Validate an API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            Key information if valid, None otherwise
        """
        if not api_key or api_key not in self.api_keys:
            return None
        
        key_info = self.api_keys[api_key]
        
        if not key_info.get("enabled", False):
            return None
        
        # Update last used timestamp
        key_info["last_used"] = datetime.now().isoformat()
        
        return key_info
    
    def check_permission(self, key_info: Dict[str, Any], required_permission: str) -> bool:
        """
        Check if a key has the required permission.
        
        Args:
            key_info: Key information from validation
            required_permission: Required permission
            
        Returns:
            True if permission is granted, False otherwise
        """
        if not key_info:
            return False
        
        permissions = key_info.get("permissions", [])
        return required_permission in permissions or "admin" in permissions
    
    def create_session(self, user_id: str, permissions: list = None) -> str:
        """
        Create a user session.
        
        Args:
            user_id: User identifier
            permissions: List of permissions for the session
            
        Returns:
            Session token
        """
        if permissions is None:
            permissions = ["read"]
        
        # Generate session token
        session_token = secrets.token_urlsafe(32)
        
        # Store session information
        self.sessions[session_token] = {
            "user_id": user_id,
            "permissions": permissions,
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=24)).isoformat()
        }
        
        return session_token
    
    def validate_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """
        Validate a session token.
        
        Args:
            session_token: Session token to validate
            
        Returns:
            Session information if valid, None otherwise
        """
        if not session_token or session_token not in self.sessions:
            return None
        
        session_info = self.sessions[session_token]
        
        # Check if session has expired
        expires_at = datetime.fromisoformat(session_info["expires_at"])
        if datetime.now() > expires_at:
            del self.sessions[session_token]
            return None
        
        # Update last accessed timestamp
        session_info["last_accessed"] = datetime.now().isoformat()
        
        return session_info
    
    def revoke_session(self, session_token: str) -> bool:
        """
        Revoke a session token.
        
        Args:
            session_token: Session token to revoke
            
        Returns:
            True if revoked, False if not found
        """
        if session_token in self.sessions:
            del self.sessions[session_token]
            return True
        return False
    
    def get_auth_header(self, headers: Dict[str, str]) -> Optional[str]:
        """
        Extract authentication token from headers.
        
        Args:
            headers: HTTP headers dictionary
            
        Returns:
            Auth token if found, None otherwise
        """
        # Check for API key in headers
        api_key = headers.get("X-API-Key") or headers.get("x-api-key")
        if api_key:
            return api_key
        
        # Check for Bearer token
        auth_header = headers.get("Authorization") or headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix
        
        return None
    
    def authenticate_request(self, headers: Dict[str, str], required_permission: str = "read") -> Optional[Dict[str, Any]]:
        """
        Authenticate a request and check permissions.
        
        Args:
            headers: HTTP headers dictionary
            required_permission: Required permission for the request
            
        Returns:
            Authentication info if successful, None otherwise
        """
        auth_token = self.get_auth_header(headers)
        
        if not auth_token:
            return None
        
        # Try API key authentication first
        key_info = self.validate_api_key(auth_token)
        if key_info and self.check_permission(key_info, required_permission):
            return {
                "type": "api_key",
                "info": key_info,
                "permissions": key_info.get("permissions", [])
            }
        
        # Try session authentication
        session_info = self.validate_session(auth_token)
        if session_info and self.check_permission(session_info, required_permission):
            return {
                "type": "session",
                "info": session_info,
                "permissions": session_info.get("permissions", [])
            }
        
        return None

# Global authentication middleware instance
auth_middleware = SimpleAuthMiddleware()