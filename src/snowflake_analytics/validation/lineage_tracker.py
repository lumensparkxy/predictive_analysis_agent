"""
Data Lineage Tracker - Tracks data lineage and processing history.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import json

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataLineageTracker:
    """Tracks data lineage and processing history."""
    
    def __init__(self):
        self.lineage_records = []
    
    def track_collection(self, source: str, target: str, metadata: Dict[str, Any]):
        """Track a data collection operation."""
        lineage_record = {
            'timestamp': datetime.now().isoformat(),
            'operation_type': 'collection',
            'source': source,
            'target': target,
            'metadata': metadata
        }
        
        self.lineage_records.append(lineage_record)
        logger.debug(f"Tracked lineage: {source} -> {target}")
    
    def track_transformation(self, source: str, target: str, transformation: str, metadata: Dict[str, Any]):
        """Track a data transformation operation."""
        lineage_record = {
            'timestamp': datetime.now().isoformat(),
            'operation_type': 'transformation',
            'source': source,
            'target': target,
            'transformation': transformation,
            'metadata': metadata
        }
        
        self.lineage_records.append(lineage_record)
        logger.debug(f"Tracked transformation: {source} -> {target} ({transformation})")
    
    def get_lineage_for_table(self, table_name: str) -> List[Dict[str, Any]]:
        """Get lineage records for a specific table."""
        return [record for record in self.lineage_records 
                if record.get('target') == table_name or record.get('source') == table_name]
    
    def get_full_lineage(self) -> List[Dict[str, Any]]:
        """Get all lineage records."""
        return self.lineage_records.copy()
