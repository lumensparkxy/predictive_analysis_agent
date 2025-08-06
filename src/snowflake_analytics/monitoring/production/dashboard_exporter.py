"""
Dashboard Exporter for Snowflake Analytics
Export monitoring data for external dashboards and visualization.
"""

import json
from datetime import datetime
from typing import Dict, Any
import structlog

logger = structlog.get_logger(__name__)


class DashboardExporter:
    """Export monitoring data for dashboards."""
    
    def __init__(self):
        self.export_formats = ['json', 'prometheus', 'grafana']
        
    def export_health_data(self, health_data: Dict[str, Any]) -> Dict[str, Any]:
        """Export health check data for dashboards."""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': health_data.get('overall_status', 'unknown'),
            'checks': {
                name: {
                    'status': check.get('status', 'unknown'),
                    'value': check.get('value'),
                    'duration': check.get('duration', 0)
                }
                for name, check in health_data.get('checks', {}).items()
            }
        }
    
    def export_metrics_data(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Export metrics data for dashboards."""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'system': metrics_data.get('system', {}),
            'application': metrics_data.get('application', {}),
            'business': metrics_data.get('business', {})
        }
    
    def export_alert_data(self, alert_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Export alert data for dashboards."""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'active_alerts': len([a for a in alert_data if a['status'] == 'open']),
            'critical_alerts': len([a for a in alert_data if a['severity'] == 'critical']),
            'alerts': alert_data
        }