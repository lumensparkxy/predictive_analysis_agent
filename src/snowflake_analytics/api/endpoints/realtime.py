"""
Real-time WebSocket endpoints for the dashboard.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Set
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)

class RealTimeEndpoints:
    """Handles real-time WebSocket connections and data streaming."""
    
    def __init__(self):
        self.logger = logger
        self.active_connections: Set[Any] = set()
        self.data_stream_active = False
        
    async def connect_websocket(self, websocket):
        """
        Handle new WebSocket connection.
        
        Args:
            websocket: WebSocket connection object
        """
        await websocket.accept()
        self.active_connections.add(websocket)
        self.logger.info(f"New WebSocket connection. Total connections: {len(self.active_connections)}")
        
        # Start data streaming if not already active
        if not self.data_stream_active:
            self.data_stream_active = True
            asyncio.create_task(self._stream_real_time_data())
    
    async def disconnect_websocket(self, websocket):
        """
        Handle WebSocket disconnection.
        
        Args:
            websocket: WebSocket connection object
        """
        self.active_connections.discard(websocket)
        self.logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
        
        # Stop data streaming if no active connections
        if not self.active_connections:
            self.data_stream_active = False
    
    async def _stream_real_time_data(self):
        """Stream real-time data to all connected clients."""
        while self.data_stream_active and self.active_connections:
            try:
                # Generate real-time data
                data = await self._generate_real_time_data()
                
                # Send to all connected clients
                disconnected = []
                for websocket in self.active_connections:
                    try:
                        await websocket.send_text(json.dumps(data))
                    except Exception as e:
                        self.logger.error(f"Error sending data to WebSocket: {e}")
                        disconnected.append(websocket)
                
                # Remove disconnected clients
                for websocket in disconnected:
                    self.active_connections.discard(websocket)
                
                # Wait before next update
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in real-time data stream: {e}")
                await asyncio.sleep(5)
    
    async def _generate_real_time_data(self) -> Dict[str, Any]:
        """Generate mock real-time data for streaming."""
        current_time = datetime.now()
        
        # Generate random but realistic data
        data = {
            "timestamp": current_time.isoformat(),
            "type": "real_time_update",
            "data": {
                "metrics": {
                    "active_queries": random.randint(15, 45),
                    "active_users": random.randint(25, 75),
                    "current_cost": round(random.uniform(800, 1200), 2),
                    "queries_per_second": round(random.uniform(5, 25), 2),
                    "data_processed_mb": round(random.uniform(100, 500), 2),
                    "avg_query_time": round(random.uniform(3, 12), 2)
                },
                "warehouses": [
                    {
                        "name": "ANALYTICS_WH",
                        "status": "running",
                        "utilization": round(random.uniform(60, 90), 1),
                        "queue_depth": random.randint(0, 5),
                        "active_queries": random.randint(5, 15)
                    },
                    {
                        "name": "ETL_WH",
                        "status": "running",
                        "utilization": round(random.uniform(40, 80), 1),
                        "queue_depth": random.randint(0, 3),
                        "active_queries": random.randint(3, 12)
                    },
                    {
                        "name": "REPORTING_WH",
                        "status": random.choice(["running", "suspended"]),
                        "utilization": round(random.uniform(0, 60), 1),
                        "queue_depth": random.randint(0, 2),
                        "active_queries": random.randint(0, 8)
                    }
                ],
                "alerts": {
                    "active_count": random.randint(1, 5),
                    "critical_count": random.randint(0, 2),
                    "last_alert": (current_time - timedelta(minutes=random.randint(1, 30))).isoformat()
                },
                "anomalies": {
                    "detected_count": random.randint(0, 3),
                    "investigating_count": random.randint(0, 2),
                    "confidence_avg": round(random.uniform(0.8, 0.95), 2)
                }
            }
        }
        
        return data
    
    async def send_alert_notification(self, alert_data: Dict[str, Any]):
        """
        Send alert notification to all connected clients.
        
        Args:
            alert_data: Alert data to send
        """
        notification = {
            "timestamp": datetime.now().isoformat(),
            "type": "alert_notification",
            "data": alert_data
        }
        
        disconnected = []
        for websocket in self.active_connections:
            try:
                await websocket.send_text(json.dumps(notification))
            except Exception as e:
                self.logger.error(f"Error sending alert notification: {e}")
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            self.active_connections.discard(websocket)
    
    async def send_anomaly_notification(self, anomaly_data: Dict[str, Any]):
        """
        Send anomaly notification to all connected clients.
        
        Args:
            anomaly_data: Anomaly data to send
        """
        notification = {
            "timestamp": datetime.now().isoformat(),
            "type": "anomaly_notification",
            "data": anomaly_data
        }
        
        disconnected = []
        for websocket in self.active_connections:
            try:
                await websocket.send_text(json.dumps(notification))
            except Exception as e:
                self.logger.error(f"Error sending anomaly notification: {e}")
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            self.active_connections.discard(websocket)
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get WebSocket connection statistics.
        
        Returns:
            Dictionary with connection statistics
        """
        return {
            "status": "success",
            "data": {
                "active_connections": len(self.active_connections),
                "data_stream_active": self.data_stream_active,
                "last_update": datetime.now().isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }

# Global instance
realtime_endpoints = RealTimeEndpoints()