"""
CORS middleware configuration.
"""

from typing import List, Optional

class CORSMiddleware:
    """CORS middleware configuration for the API."""
    
    def __init__(self, 
                 allow_origins: List[str] = None,
                 allow_methods: List[str] = None,
                 allow_headers: List[str] = None,
                 allow_credentials: bool = False,
                 expose_headers: List[str] = None,
                 max_age: int = 600):
        """
        Initialize CORS middleware.
        
        Args:
            allow_origins: List of allowed origins
            allow_methods: List of allowed HTTP methods
            allow_headers: List of allowed headers
            allow_credentials: Whether to allow credentials
            expose_headers: List of headers to expose
            max_age: Max age for preflight requests
        """
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allow_headers = allow_headers or ["*"]
        self.allow_credentials = allow_credentials
        self.expose_headers = expose_headers or []
        self.max_age = max_age
    
    def get_cors_config(self) -> dict:
        """Get CORS configuration dictionary."""
        return {
            "allow_origins": self.allow_origins,
            "allow_methods": self.allow_methods,
            "allow_headers": self.allow_headers,
            "allow_credentials": self.allow_credentials,
            "expose_headers": self.expose_headers,
            "max_age": self.max_age
        }

# Default CORS configuration
default_cors = CORSMiddleware(
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8080",
        "http://localhost:8000",
        "https://localhost:3000",
        "https://localhost:8080",
        "https://localhost:8000"
    ],
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=[
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "X-API-Key"
    ],
    allow_credentials=True,
    expose_headers=["X-Total-Count", "X-Page-Count"],
    max_age=86400  # 24 hours
)

# Production CORS configuration
production_cors = CORSMiddleware(
    allow_origins=[
        "https://yourdomain.com",
        "https://dashboard.yourdomain.com"
    ],
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Accept",
        "Content-Type",
        "Authorization",
        "X-API-Key"
    ],
    allow_credentials=True,
    expose_headers=["X-Total-Count"],
    max_age=3600  # 1 hour
)