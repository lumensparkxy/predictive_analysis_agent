"""
Uptime Monitor for Snowflake Analytics
Service uptime monitoring and availability tracking.
"""

import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import structlog

logger = structlog.get_logger(__name__)


class UptimeMonitor:
    """Service uptime monitoring and SLA tracking."""
    
    def __init__(self):
        self.monitored_services = {
            'api': 'http://localhost:8000/health',
            'dashboard': 'http://localhost:8501',
            'nginx': 'http://localhost:80/health'
        }
        self.uptime_data: Dict[str, List[Dict[str, Any]]] = {}
        
    def check_service_uptime(self, service: str, url: str) -> Dict[str, Any]:
        """Check if a service is up."""
        start_time = time.time()
        try:
            response = requests.get(url, timeout=10)
            duration = time.time() - start_time
            
            is_up = response.status_code == 200
            return {
                'service': service,
                'status': 'up' if is_up else 'down',
                'response_time': duration,
                'status_code': response.status_code,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            duration = time.time() - start_time
            return {
                'service': service,
                'status': 'down',
                'response_time': duration,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def get_uptime_statistics(self) -> Dict[str, Any]:
        """Get uptime statistics for all services."""
        stats = {}
        for service in self.monitored_services:
            if service in self.uptime_data:
                recent_checks = self.uptime_data[service][-100:]  # Last 100 checks
                up_count = sum(1 for check in recent_checks if check['status'] == 'up')
                uptime_percentage = (up_count / len(recent_checks)) * 100 if recent_checks else 0
                
                stats[service] = {
                    'uptime_percentage': uptime_percentage,
                    'total_checks': len(recent_checks),
                    'up_checks': up_count,
                    'down_checks': len(recent_checks) - up_count
                }
        
        return stats