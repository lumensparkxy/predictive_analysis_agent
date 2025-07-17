"""
OpenAI Client for LLM Integration

Provides OpenAI API integration with retry logic, rate limiting,
and error handling for Snowflake analytics queries.
"""

import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None
    OpenAI = None

from ...utils.logger import get_logger
from .rate_limiter import RateLimiter
from .response_validator import ResponseValidator

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM API call."""
    content: str
    model: str
    usage: Dict[str, Any]
    finish_reason: str
    response_time: float
    cached: bool = False


class OpenAIClient:
    """OpenAI API client with rate limiting and error handling."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI client with configuration."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")
        
        self.config = config
        self.api_key = config.get('openai_api_key')
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Configuration
        self.model = config.get('model', 'gpt-4')
        self.max_tokens = config.get('max_tokens', 2000)
        self.temperature = config.get('temperature', 0.1)
        self.timeout = config.get('timeout', 30)
        
        # Rate limiting
        rate_limit_config = config.get('rate_limiting', {})
        self.rate_limiter = RateLimiter(**rate_limit_config)
        
        # Response validation
        self.response_validator = ResponseValidator()
        
        # Usage tracking
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.request_count = 0
        
        logger.info(f"OpenAI client initialized with model: {self.model}")
    
    async def generate_completion(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using OpenAI API."""
        start_time = time.time()
        
        try:
            # Apply rate limiting
            await self.rate_limiter.acquire()
            
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # API parameters
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
                **kwargs
            }
            
            # Make API call with retry logic
            response = await self._make_api_call_with_retry(params)
            
            # Extract response data
            choice = response.choices[0]
            content = choice.message.content
            usage = response.usage
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Update usage tracking
            self._update_usage_stats(usage, response_time)
            
            # Create response object
            llm_response = LLMResponse(
                content=content,
                model=response.model,
                usage=usage.dict() if hasattr(usage, 'dict') else usage,
                finish_reason=choice.finish_reason,
                response_time=response_time
            )
            
            # Validate response
            self.response_validator.validate_response(llm_response)
            
            logger.debug(f"OpenAI completion generated in {response_time:.2f}s")
            return llm_response
            
        except Exception as e:
            logger.error(f"OpenAI completion failed: {e}")
            raise
    
    async def _make_api_call_with_retry(self, params: Dict[str, Any], max_retries: int = 3) -> Any:
        """Make API call with exponential backoff retry logic."""
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                response = await self.client.chat.completions.create(**params)
                return response
                
            except openai.RateLimitError as e:
                if attempt == max_retries:
                    raise e
                
                # Calculate backoff delay (exponential with jitter)
                delay = min(2 ** attempt + (time.time() % 1), 60)
                logger.warning(f"Rate limit hit, retrying in {delay:.2f}s (attempt {attempt + 1})")
                await asyncio.sleep(delay)
                last_exception = e
                
            except openai.APITimeoutError as e:
                if attempt == max_retries:
                    raise e
                
                delay = min(2 ** attempt, 30)
                logger.warning(f"API timeout, retrying in {delay:.2f}s (attempt {attempt + 1})")
                await asyncio.sleep(delay)
                last_exception = e
                
            except openai.APIError as e:
                # Don't retry on general API errors
                logger.error(f"OpenAI API error: {e}")
                raise e
                
            except Exception as e:
                if attempt == max_retries:
                    raise e
                
                delay = min(2 ** attempt, 30)
                logger.warning(f"Unexpected error, retrying in {delay:.2f}s (attempt {attempt + 1}): {e}")
                await asyncio.sleep(delay)
                last_exception = e
        
        # If we get here, all retries failed
        raise last_exception or Exception("All retry attempts failed")
    
    def _update_usage_stats(self, usage: Any, response_time: float):
        """Update usage statistics."""
        try:
            tokens_used = getattr(usage, 'total_tokens', 0)
            self.total_tokens_used += tokens_used
            self.request_count += 1
            
            # Estimate cost (rough approximation for GPT-4)
            cost_per_token = 0.00003  # $0.03 per 1K tokens (average)
            estimated_cost = tokens_used * cost_per_token
            self.total_cost += estimated_cost
            
            logger.debug(f"Usage: {tokens_used} tokens, estimated cost: ${estimated_cost:.4f}")
            
        except Exception as e:
            logger.warning(f"Failed to update usage stats: {e}")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_requests": self.request_count,
            "total_tokens": self.total_tokens_used,
            "estimated_cost": self.total_cost,
            "average_tokens_per_request": (
                self.total_tokens_used / self.request_count if self.request_count > 0 else 0
            )
        }
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.request_count = 0
        logger.info("Usage statistics reset")
    
    def is_available(self) -> bool:
        """Check if the OpenAI client is available and configured."""
        return OPENAI_AVAILABLE and self.api_key is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "provider": "openai",
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "available": self.is_available()
        }


import asyncio
    # Fallback for sync usage
# Backwards compatibility for sync usage
try:
    # Sync wrapper implementation would go here if needed
    pass
except Exception:
    pass