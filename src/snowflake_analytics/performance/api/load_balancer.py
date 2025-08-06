"""
Load balancer for distributing API requests.
"""

from typing import List, Dict, Any
from enum import Enum
import random


class LoadBalanceStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_CONNECTIONS = "least_connections"


class LoadBalancer:
    """Simple load balancer for API requests."""
    
    def __init__(self, servers: List[str], strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN):
        self.servers = servers
        self.strategy = strategy
        self.current_index = 0
        self.server_connections = {server: 0 for server in servers}
    
    def get_server(self) -> str:
        """Get next server based on load balancing strategy."""
        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            server = self.servers[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.servers)
            return server
        elif self.strategy == LoadBalanceStrategy.RANDOM:
            return random.choice(self.servers)
        elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return min(self.server_connections.items(), key=lambda x: x[1])[0]
        return self.servers[0]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        return {
            'servers': len(self.servers),
            'strategy': self.strategy.value,
            'server_connections': dict(self.server_connections)
        }