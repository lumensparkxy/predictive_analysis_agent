"""
Rate Limiter for LLM API Calls

Implements token bucket algorithm for rate limiting LLM API requests
to prevent hitting API rate limits and manage costs.
"""

import time
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass
from threading import Lock
import logging

from ...utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    tokens_per_minute: int = 10000
    burst_requests: int = 10
    burst_tokens: int = 2000


class TokenBucket:
    """Token bucket for rate limiting."""
    
    def __init__(self, capacity: int, refill_rate: float):
        """Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens per second refill rate
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = Lock()
    
    def consume(self, tokens: int) -> bool:
        """Try to consume tokens from bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if not enough tokens
        """
        with self._lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def time_to_refill(self, tokens: int) -> float:
        """Calculate time needed to refill enough tokens.
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            Time in seconds until enough tokens are available
        """
        with self._lock:
            self._refill()
            
            if self.tokens >= tokens:
                return 0.0
            
            tokens_needed = tokens - self.tokens
            return tokens_needed / self.refill_rate
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bucket status."""
        with self._lock:
            self._refill()
            return {
                "tokens": self.tokens,
                "capacity": self.capacity,
                "refill_rate": self.refill_rate,
                "utilization": (self.capacity - self.tokens) / self.capacity
            }


class RateLimiter:
    """Rate limiter for LLM API calls using token bucket algorithm."""
    
    def __init__(
        self, 
        requests_per_minute: int = 60,
        tokens_per_minute: int = 10000,
        burst_requests: int = 10,
        burst_tokens: int = 2000,
        **kwargs
    ):
        """Initialize rate limiter.
        
        Args:
            requests_per_minute: Max requests per minute
            tokens_per_minute: Max tokens per minute
            burst_requests: Max burst requests
            burst_tokens: Max burst tokens
        """
        self.config = RateLimitConfig(
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
            burst_requests=burst_requests,
            burst_tokens=burst_tokens
        )
        
        # Create token buckets
        self.request_bucket = TokenBucket(
            capacity=self.config.burst_requests,
            refill_rate=self.config.requests_per_minute / 60.0
        )
        
        self.token_bucket = TokenBucket(
            capacity=self.config.burst_tokens,
            refill_rate=self.config.tokens_per_minute / 60.0
        )
        
        # Statistics
        self.requests_made = 0
        self.tokens_consumed = 0
        self.wait_time_total = 0.0
        self.rate_limit_hits = 0
        
        logger.info(f"Rate limiter initialized: {requests_per_minute} req/min, {tokens_per_minute} tokens/min")
    
    async def acquire(self, estimated_tokens: int = 100) -> None:
        """Acquire permission to make an API call.
        
        Args:
            estimated_tokens: Estimated tokens for the request
        """
        start_time = time.time()
        
        # Check both request and token limits
        request_wait = 0.0
        token_wait = 0.0
        
        # Wait for request bucket
        if not self.request_bucket.consume(1):
            request_wait = self.request_bucket.time_to_refill(1)
            
        # Wait for token bucket
        if not self.token_bucket.consume(estimated_tokens):
            token_wait = self.token_bucket.time_to_refill(estimated_tokens)
        
        # Wait for the longer of the two
        wait_time = max(request_wait, token_wait)
        
        if wait_time > 0:
            self.rate_limit_hits += 1
            logger.debug(f"Rate limit hit, waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
            
            # Try again after waiting
            await self.acquire(estimated_tokens)
        else:
            # Update statistics
            total_wait = time.time() - start_time
            self.wait_time_total += total_wait
            self.requests_made += 1
            self.tokens_consumed += estimated_tokens
    
    def acquire_sync(self, estimated_tokens: int = 100) -> None:
        """Synchronous version of acquire."""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.acquire(estimated_tokens))
        finally:
            loop.close()
    
    def get_status(self) -> Dict[str, Any]:
        """Get rate limiter status."""
        return {
            "config": {
                "requests_per_minute": self.config.requests_per_minute,
                "tokens_per_minute": self.config.tokens_per_minute,
                "burst_requests": self.config.burst_requests,
                "burst_tokens": self.config.burst_tokens
            },
            "request_bucket": self.request_bucket.get_status(),
            "token_bucket": self.token_bucket.get_status(),
            "statistics": {
                "requests_made": self.requests_made,
                "tokens_consumed": self.tokens_consumed,
                "total_wait_time": self.wait_time_total,
                "rate_limit_hits": self.rate_limit_hits,
                "average_wait_time": (
                    self.wait_time_total / self.requests_made if self.requests_made > 0 else 0
                )
            }
        }
    
    def reset_statistics(self):
        """Reset rate limiter statistics."""
        self.requests_made = 0
        self.tokens_consumed = 0
        self.wait_time_total = 0.0
        self.rate_limit_hits = 0
        logger.info("Rate limiter statistics reset")
    
    def update_config(self, config: Dict[str, Any]):
        """Update rate limiter configuration."""
        if "requests_per_minute" in config:
            self.config.requests_per_minute = config["requests_per_minute"]
            self.request_bucket.refill_rate = config["requests_per_minute"] / 60.0
            
        if "tokens_per_minute" in config:
            self.config.tokens_per_minute = config["tokens_per_minute"]
            self.token_bucket.refill_rate = config["tokens_per_minute"] / 60.0
            
        if "burst_requests" in config:
            self.config.burst_requests = config["burst_requests"]
            self.request_bucket.capacity = config["burst_requests"]
            
        if "burst_tokens" in config:
            self.config.burst_tokens = config["burst_tokens"]
            self.token_bucket.capacity = config["burst_tokens"]
        
        logger.info(f"Rate limiter config updated: {self.config}")
    
    def estimate_wait_time(self, estimated_tokens: int = 100) -> float:
        """Estimate wait time for a request.
        
        Args:
            estimated_tokens: Estimated tokens for the request
            
        Returns:
            Estimated wait time in seconds
        """
        request_wait = self.request_bucket.time_to_refill(1)
        token_wait = self.token_bucket.time_to_refill(estimated_tokens)
        return max(request_wait, token_wait)
    
    def can_make_request(self, estimated_tokens: int = 100) -> bool:
        """Check if a request can be made immediately.
        
        Args:
            estimated_tokens: Estimated tokens for the request
            
        Returns:
            True if request can be made without waiting
        """
        return self.estimate_wait_time(estimated_tokens) == 0.0


class AdaptiveRateLimiter(RateLimiter):
    """Adaptive rate limiter that adjusts based on API responses."""
    
    def __init__(self, *args, **kwargs):
        """Initialize adaptive rate limiter."""
        super().__init__(*args, **kwargs)
        self.success_count = 0
        self.error_count = 0
        self.last_adjustment = time.time()
        self.adjustment_interval = 60.0  # Adjust every minute
    
    def record_success(self):
        """Record a successful API call."""
        self.success_count += 1
        self._maybe_adjust_limits()
    
    def record_error(self, error_type: str):
        """Record an API error."""
        self.error_count += 1
        
        # Immediately reduce limits on rate limit errors
        if "rate_limit" in error_type.lower():
            self._reduce_limits()
        
        self._maybe_adjust_limits()
    
    def _maybe_adjust_limits(self):
        """Adjust limits based on success/error ratio."""
        now = time.time()
        if now - self.last_adjustment < self.adjustment_interval:
            return
        
        total_requests = self.success_count + self.error_count
        if total_requests < 10:  # Need minimum data
            return
        
        error_rate = self.error_count / total_requests
        
        if error_rate > 0.1:  # >10% error rate, reduce limits
            self._reduce_limits()
        elif error_rate < 0.02:  # <2% error rate, increase limits
            self._increase_limits()
        
        # Reset counters
        self.success_count = 0
        self.error_count = 0
        self.last_adjustment = now
    
    def _reduce_limits(self):
        """Reduce rate limits."""
        new_req_rate = max(10, int(self.config.requests_per_minute * 0.8))
        new_token_rate = max(1000, int(self.config.tokens_per_minute * 0.8))
        
        self.update_config({
            "requests_per_minute": new_req_rate,
            "tokens_per_minute": new_token_rate
        })
        
        logger.info(f"Rate limits reduced: {new_req_rate} req/min, {new_token_rate} tokens/min")
    
    def _increase_limits(self):
        """Increase rate limits."""
        new_req_rate = min(100, int(self.config.requests_per_minute * 1.1))
        new_token_rate = min(40000, int(self.config.tokens_per_minute * 1.1))
        
        self.update_config({
            "requests_per_minute": new_req_rate,
            "tokens_per_minute": new_token_rate
        })
        
        logger.info(f"Rate limits increased: {new_req_rate} req/min, {new_token_rate} tokens/min")