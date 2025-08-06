"""
Rate limiter for API throttling and request management.
"""

import time
from typing import Dict, Optional
from collections import defaultdict, deque


class RateLimiter:
    """Rate limiter with sliding window algorithm."""
    
    def __init__(self, requests_per_minute: int = 100):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(deque)
        self.blocked_requests = 0
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        client_requests = self.requests[client_id]
        
        # Remove old requests (older than 1 minute)
        while client_requests and client_requests[0] < now - 60:
            client_requests.popleft()
        
        # Check rate limit
        if len(client_requests) >= self.requests_per_minute:
            self.blocked_requests += 1
            return False
        
        # Add current request
        client_requests.append(now)
        return True
    
    def get_stats(self) -> Dict[str, int]:
        """Get rate limiting statistics."""
        return {
            'active_clients': len(self.requests),
            'blocked_requests': self.blocked_requests,
            'requests_per_minute_limit': self.requests_per_minute
        }