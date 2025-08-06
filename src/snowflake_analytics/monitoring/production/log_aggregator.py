"""
Log Aggregation System for Snowflake Analytics
Centralized log collection, parsing, and analysis.
"""

import os
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import structlog

logger = structlog.get_logger(__name__)


class LogAggregator:
    """Centralized log aggregation and processing."""
    
    def __init__(self):
        self.log_sources = {
            'application': '/opt/analytics/logs/application.log',
            'nginx': '/var/log/nginx/analytics_access.log', 
            'system': '/var/log/syslog'
        }
        self.aggregated_logs: List[Dict[str, Any]] = []
        self.max_logs = 10000
        
    def collect_logs(self) -> Dict[str, Any]:
        """Collect and aggregate logs from all sources."""
        logs = []
        for source, path in self.log_sources.items():
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        lines = f.readlines()[-100:]  # Last 100 lines
                        for line in lines:
                            logs.append({
                                'timestamp': datetime.utcnow().isoformat(),
                                'source': source,
                                'message': line.strip(),
                                'level': self._extract_level(line)
                            })
                except Exception as e:
                    logger.error(f"Failed to read {source} logs", error=str(e))
        
        return {'logs': logs, 'total': len(logs)}
    
    def _extract_level(self, line: str) -> str:
        """Extract log level from line."""
        for level in ['ERROR', 'WARN', 'INFO', 'DEBUG']:
            if level in line.upper():
                return level
        return 'INFO'