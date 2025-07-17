"""
Rate limiting middleware for API endpoints.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import time
import logging

logger = logging.getLogger(__name__)

class RateLimitingMiddleware:
    """Rate limiting middleware for API endpoints."""
    
    def __init__(self):
        self.logger = logger
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        self.request_counts: Dict[str, Dict[str, Any]] = {}
        self._initialize_default_limits()
    
    def _initialize_default_limits(self):
        """Initialize default rate limits."""
        self.rate_limits = {
            "default": {
                "requests_per_minute": 60,
                "requests_per_hour": 1000,
                "requests_per_day": 10000,
                "burst_limit": 10  # Burst allowance
            },
            "authenticated": {
                "requests_per_minute": 120,
                "requests_per_hour": 2000,
                "requests_per_day": 20000,
                "burst_limit": 20
            },
            "admin": {
                "requests_per_minute": 300,
                "requests_per_hour": 5000,
                "requests_per_day": 50000,
                "burst_limit": 50
            },
            "websocket": {
                "connections_per_minute": 10,
                "connections_per_hour": 100,
                "max_concurrent": 50
            }
        }
    
    def get_client_key(self, client_ip: str, user_id: str = None) -> str:
        """
        Generate a client key for rate limiting.
        
        Args:
            client_ip: Client IP address
            user_id: User identifier (optional)
            
        Returns:
            Client key string
        """
        if user_id:
            return f"user:{user_id}"
        return f"ip:{client_ip}"
    
    def get_rate_limit_tier(self, permissions: list = None) -> str:
        """
        Determine rate limit tier based on permissions.
        
        Args:
            permissions: List of user permissions
            
        Returns:
            Rate limit tier name
        """
        if permissions:
            if "admin" in permissions:
                return "admin"
            elif "read" in permissions or "write" in permissions:
                return "authenticated"
        return "default"
    
    def _get_time_window_key(self, window_type: str) -> str:
        """
        Get time window key for rate limiting.
        
        Args:
            window_type: Type of time window (minute, hour, day)
            
        Returns:
            Time window key
        """
        now = datetime.now()
        
        if window_type == "minute":
            return f"minute:{now.strftime('%Y-%m-%d-%H-%M')}"
        elif window_type == "hour":
            return f"hour:{now.strftime('%Y-%m-%d-%H')}"
        elif window_type == "day":
            return f"day:{now.strftime('%Y-%m-%d')}"
        else:
            raise ValueError(f"Invalid window type: {window_type}")
    
    def _cleanup_old_entries(self, client_key: str):
        """
        Clean up old rate limit entries.
        
        Args:
            client_key: Client key to clean up
        """
        if client_key not in self.request_counts:
            return
        
        now = datetime.now()
        client_data = self.request_counts[client_key]
        
        # Remove entries older than 1 day
        cutoff_time = now - timedelta(days=1)
        
        keys_to_remove = []
        for key, data in client_data.items():
            if "timestamp" in data:
                entry_time = datetime.fromisoformat(data["timestamp"])
                if entry_time < cutoff_time:
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del client_data[key]
    
    def check_rate_limit(self, client_ip: str, user_id: str = None, 
                        permissions: list = None, endpoint: str = None) -> Dict[str, Any]:
        """
        Check if client has exceeded rate limits.
        
        Args:
            client_ip: Client IP address
            user_id: User identifier (optional)
            permissions: List of user permissions (optional)
            endpoint: API endpoint being accessed (optional)
            
        Returns:
            Dictionary with rate limit check results
        """
        client_key = self.get_client_key(client_ip, user_id)
        tier = self.get_rate_limit_tier(permissions)
        limits = self.rate_limits[tier]
        
        # Initialize client data if not exists
        if client_key not in self.request_counts:
            self.request_counts[client_key] = {}
        
        # Clean up old entries
        self._cleanup_old_entries(client_key)
        
        client_data = self.request_counts[client_key]
        now = datetime.now()
        
        # Check different time windows
        windows = [
            ("minute", limits["requests_per_minute"]),
            ("hour", limits["requests_per_hour"]),
            ("day", limits["requests_per_day"])
        ]
        
        for window_type, limit in windows:
            window_key = self._get_time_window_key(window_type)
            
            if window_key not in client_data:
                client_data[window_key] = {
                    "count": 0,
                    "timestamp": now.isoformat(),
                    "reset_time": self._get_reset_time(window_type).isoformat()
                }
            
            current_count = client_data[window_key]["count"]
            
            if current_count >= limit:
                return {
                    "allowed": False,
                    "limit_type": window_type,
                    "limit": limit,
                    "current": current_count,
                    "reset_time": client_data[window_key]["reset_time"],
                    "tier": tier
                }
        
        # Check burst limit
        burst_key = "burst"
        if burst_key not in client_data:
            client_data[burst_key] = {
                "count": 0,
                "timestamp": now.isoformat(),
                "reset_time": (now + timedelta(minutes=1)).isoformat()
            }
        
        # Reset burst count if it's been more than a minute
        burst_reset_time = datetime.fromisoformat(client_data[burst_key]["reset_time"])
        if now > burst_reset_time:
            client_data[burst_key] = {
                "count": 0,
                "timestamp": now.isoformat(),
                "reset_time": (now + timedelta(minutes=1)).isoformat()
            }
        
        if client_data[burst_key]["count"] >= limits["burst_limit"]:
            return {
                "allowed": False,
                "limit_type": "burst",
                "limit": limits["burst_limit"],
                "current": client_data[burst_key]["count"],
                "reset_time": client_data[burst_key]["reset_time"],
                "tier": tier
            }
        
        return {
            "allowed": True,
            "tier": tier,
            "limits": limits,
            "current_usage": {
                "minute": client_data.get(self._get_time_window_key("minute"), {}).get("count", 0),
                "hour": client_data.get(self._get_time_window_key("hour"), {}).get("count", 0),
                "day": client_data.get(self._get_time_window_key("day"), {}).get("count", 0),
                "burst": client_data.get("burst", {}).get("count", 0)
            }
        }
    
    def record_request(self, client_ip: str, user_id: str = None, 
                      endpoint: str = None) -> None:
        """
        Record a request for rate limiting.
        
        Args:
            client_ip: Client IP address
            user_id: User identifier (optional)
            endpoint: API endpoint being accessed (optional)
        """
        client_key = self.get_client_key(client_ip, user_id)
        
        if client_key not in self.request_counts:
            self.request_counts[client_key] = {}
        
        client_data = self.request_counts[client_key]
        now = datetime.now()
        
        # Update counters for all time windows
        windows = ["minute", "hour", "day", "burst"]
        
        for window_type in windows:
            if window_type == "burst":
                window_key = "burst"
            else:
                window_key = self._get_time_window_key(window_type)
            
            if window_key not in client_data:
                client_data[window_key] = {
                    "count": 0,
                    "timestamp": now.isoformat(),
                    "reset_time": self._get_reset_time(window_type).isoformat()
                }
            
            client_data[window_key]["count"] += 1
            client_data[window_key]["timestamp"] = now.isoformat()
    
    def _get_reset_time(self, window_type: str) -> datetime:
        """
        Get reset time for a time window.
        
        Args:
            window_type: Type of time window
            
        Returns:
            Reset time datetime
        """
        now = datetime.now()
        
        if window_type == "minute":
            return now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        elif window_type == "hour":
            return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        elif window_type == "day":
            return now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        elif window_type == "burst":
            return now + timedelta(minutes=1)
        else:
            raise ValueError(f"Invalid window type: {window_type}")
    
    def get_rate_limit_info(self, client_ip: str, user_id: str = None,
                           permissions: list = None) -> Dict[str, Any]:
        """
        Get rate limit information for a client.
        
        Args:
            client_ip: Client IP address
            user_id: User identifier (optional)
            permissions: List of user permissions (optional)
            
        Returns:
            Dictionary with rate limit information
        """
        client_key = self.get_client_key(client_ip, user_id)
        tier = self.get_rate_limit_tier(permissions)
        limits = self.rate_limits[tier]
        
        client_data = self.request_counts.get(client_key, {})
        
        return {
            "tier": tier,
            "limits": limits,
            "current_usage": {
                "minute": client_data.get(self._get_time_window_key("minute"), {}).get("count", 0),
                "hour": client_data.get(self._get_time_window_key("hour"), {}).get("count", 0),
                "day": client_data.get(self._get_time_window_key("day"), {}).get("count", 0),
                "burst": client_data.get("burst", {}).get("count", 0)
            },
            "reset_times": {
                "minute": self._get_reset_time("minute").isoformat(),
                "hour": self._get_reset_time("hour").isoformat(),
                "day": self._get_reset_time("day").isoformat(),
                "burst": self._get_reset_time("burst").isoformat()
            }
        }

# Global rate limiting middleware instance
rate_limiting_middleware = RateLimitingMiddleware()