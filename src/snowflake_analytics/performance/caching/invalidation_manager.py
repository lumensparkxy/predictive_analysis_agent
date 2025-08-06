"""
Cache invalidation manager for intelligent cache invalidation policies.
"""

import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Callable, Any
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum


class InvalidationTrigger(Enum):
    """Cache invalidation triggers."""
    TIME_BASED = "time_based"       # TTL expiration
    EVENT_BASED = "event_based"     # Data change events
    TAG_BASED = "tag_based"         # Tag-based invalidation
    DEPENDENCY_BASED = "dependency" # Dependency changes
    MANUAL = "manual"               # Manual invalidation


@dataclass
class InvalidationRule:
    """Cache invalidation rule."""
    rule_id: str
    trigger: InvalidationTrigger
    cache_keys: Set[str]
    tags: Set[str]
    dependencies: Set[str]
    condition: Optional[Callable] = None
    cascade: bool = False
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class InvalidationEvent:
    """Cache invalidation event record."""
    event_id: str
    trigger: InvalidationTrigger
    affected_keys: Set[str]
    tags: Set[str]
    timestamp: datetime
    rule_id: Optional[str] = None
    cascade_level: int = 0


class InvalidationManager:
    """
    Cache invalidation manager that handles intelligent cache invalidation
    based on various triggers and maintains data consistency.
    """
    
    def __init__(self):
        """Initialize invalidation manager."""
        self.invalidation_rules: Dict[str, InvalidationRule] = {}
        self.invalidation_history: List[InvalidationEvent] = []
        self.tag_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.key_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.event_listeners: Dict[str, List[Callable]] = defaultdict(list)
        
        self._lock = threading.Lock()
        self._running = False
        self._background_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.total_invalidations = 0
        self.invalidations_by_trigger = defaultdict(int)
        self.cascade_invalidations = 0
    
    def start(self):
        """Start the invalidation manager background thread."""
        if self._running:
            return
        
        self._running = True
        self._background_thread = threading.Thread(
            target=self._background_processor,
            daemon=True
        )
        self._background_thread.start()
    
    def stop(self):
        """Stop the invalidation manager."""
        self._running = False
        if self._background_thread:
            self._background_thread.join(timeout=5)
    
    def add_rule(self, rule: InvalidationRule) -> bool:
        """Add invalidation rule."""
        try:
            with self._lock:
                self.invalidation_rules[rule.rule_id] = rule
                
                # Update dependency mappings
                for tag in rule.tags:
                    for key in rule.cache_keys:
                        self.tag_dependencies[tag].add(key)
                
                for dep in rule.dependencies:
                    for key in rule.cache_keys:
                        self.key_dependencies[dep].add(key)
            
            return True
        except Exception as e:
            print(f"Error adding invalidation rule: {e}")
            return False
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove invalidation rule."""
        with self._lock:
            rule = self.invalidation_rules.pop(rule_id, None)
            if rule:
                # Clean up dependency mappings
                for tag in rule.tags:
                    for key in rule.cache_keys:
                        self.tag_dependencies[tag].discard(key)
                    if not self.tag_dependencies[tag]:
                        del self.tag_dependencies[tag]
                
                for dep in rule.dependencies:
                    for key in rule.cache_keys:
                        self.key_dependencies[dep].discard(key)
                    if not self.key_dependencies[dep]:
                        del self.key_dependencies[dep]
                
                return True
        
        return False
    
    def invalidate_by_key(self, 
                         keys: List[str], 
                         trigger: InvalidationTrigger = InvalidationTrigger.MANUAL,
                         cascade: bool = False) -> InvalidationEvent:
        """Invalidate cache by specific keys."""
        event_id = f"inv_{int(time.time())}_{len(self.invalidation_history)}"
        affected_keys = set(keys)
        
        # Handle cascading invalidation
        if cascade:
            cascade_keys = self._find_cascade_keys(keys)
            affected_keys.update(cascade_keys)
            if cascade_keys:
                self.cascade_invalidations += 1
        
        # Create invalidation event
        event = InvalidationEvent(
            event_id=event_id,
            trigger=trigger,
            affected_keys=affected_keys,
            tags=set(),
            timestamp=datetime.now()
        )
        
        # Execute invalidation
        self._execute_invalidation(event)
        
        # Record event
        with self._lock:
            self.invalidation_history.append(event)
            self.total_invalidations += len(affected_keys)
            self.invalidations_by_trigger[trigger] += 1
        
        return event
    
    def invalidate_by_tags(self, 
                          tags: List[str],
                          cascade: bool = False) -> InvalidationEvent:
        """Invalidate cache by tags."""
        affected_keys = set()
        
        with self._lock:
            for tag in tags:
                if tag in self.tag_dependencies:
                    affected_keys.update(self.tag_dependencies[tag])
        
        # Handle cascading
        if cascade and affected_keys:
            cascade_keys = self._find_cascade_keys(list(affected_keys))
            affected_keys.update(cascade_keys)
            if cascade_keys:
                self.cascade_invalidations += 1
        
        event_id = f"inv_tag_{int(time.time())}_{len(self.invalidation_history)}"
        event = InvalidationEvent(
            event_id=event_id,
            trigger=InvalidationTrigger.TAG_BASED,
            affected_keys=affected_keys,
            tags=set(tags),
            timestamp=datetime.now()
        )
        
        # Execute invalidation
        self._execute_invalidation(event)
        
        # Record event
        with self._lock:
            self.invalidation_history.append(event)
            self.total_invalidations += len(affected_keys)
            self.invalidations_by_trigger[InvalidationTrigger.TAG_BASED] += 1
        
        return event
    
    def invalidate_by_dependency(self, dependencies: List[str]) -> InvalidationEvent:
        """Invalidate cache based on data dependencies."""
        affected_keys = set()
        
        with self._lock:
            for dep in dependencies:
                if dep in self.key_dependencies:
                    affected_keys.update(self.key_dependencies[dep])
        
        event_id = f"inv_dep_{int(time.time())}_{len(self.invalidation_history)}"
        event = InvalidationEvent(
            event_id=event_id,
            trigger=InvalidationTrigger.DEPENDENCY_BASED,
            affected_keys=affected_keys,
            tags=set(),
            timestamp=datetime.now()
        )
        
        # Execute invalidation
        self._execute_invalidation(event)
        
        # Record event
        with self._lock:
            self.invalidation_history.append(event)
            self.total_invalidations += len(affected_keys)
            self.invalidations_by_trigger[InvalidationTrigger.DEPENDENCY_BASED] += 1
        
        return event
    
    def add_event_listener(self, event_type: str, callback: Callable):
        """Add event listener for cache invalidation events."""
        self.event_listeners[event_type].append(callback)
    
    def _find_cascade_keys(self, keys: List[str]) -> Set[str]:
        """Find keys that should be invalidated due to cascading rules."""
        cascade_keys = set()
        
        with self._lock:
            for rule in self.invalidation_rules.values():
                if rule.cascade:
                    # Check if any of the keys match this rule's dependencies
                    if any(key in rule.dependencies for key in keys):
                        cascade_keys.update(rule.cache_keys)
                    
                    # Check if any tags are affected
                    for tag in rule.tags:
                        if tag in self.tag_dependencies:
                            tag_keys = self.tag_dependencies[tag]
                            if any(key in tag_keys for key in keys):
                                cascade_keys.update(rule.cache_keys)
        
        return cascade_keys
    
    def _execute_invalidation(self, event: InvalidationEvent):
        """Execute the actual cache invalidation."""
        # Notify listeners
        for listener in self.event_listeners.get('invalidation', []):
            try:
                listener(event)
            except Exception as e:
                print(f"Error in invalidation listener: {e}")
        
        # In a real implementation, this would call the cache manager
        # to actually remove the keys from cache
        print(f"Invalidating {len(event.affected_keys)} cache keys")
    
    def _background_processor(self):
        """Background thread for processing time-based invalidation."""
        while self._running:
            try:
                self._process_time_based_invalidation()
                time.sleep(60)  # Check every minute
            except Exception as e:
                print(f"Error in invalidation background processor: {e}")
                time.sleep(60)
    
    def _process_time_based_invalidation(self):
        """Process time-based invalidation rules."""
        with self._lock:
            current_time = datetime.now()
            rules_to_process = []
            
            for rule in self.invalidation_rules.values():
                if rule.trigger == InvalidationTrigger.TIME_BASED:
                    if rule.condition and rule.condition(current_time):
                        rules_to_process.append(rule)
        
        # Execute invalidations outside the lock
        for rule in rules_to_process:
            self._execute_rule_invalidation(rule)
    
    def _execute_rule_invalidation(self, rule: InvalidationRule):
        """Execute invalidation for a specific rule."""
        event_id = f"rule_{rule.rule_id}_{int(time.time())}"
        event = InvalidationEvent(
            event_id=event_id,
            trigger=rule.trigger,
            affected_keys=rule.cache_keys.copy(),
            tags=rule.tags.copy(),
            timestamp=datetime.now(),
            rule_id=rule.rule_id
        )
        
        # Handle cascading if enabled
        if rule.cascade:
            cascade_keys = self._find_cascade_keys(list(rule.cache_keys))
            event.affected_keys.update(cascade_keys)
            if cascade_keys:
                self.cascade_invalidations += 1
        
        self._execute_invalidation(event)
        
        # Record event
        with self._lock:
            self.invalidation_history.append(event)
            self.total_invalidations += len(event.affected_keys)
            self.invalidations_by_trigger[rule.trigger] += 1
    
    def get_invalidation_statistics(self) -> Dict[str, Any]:
        """Get invalidation statistics."""
        with self._lock:
            recent_events = [
                event for event in self.invalidation_history
                if event.timestamp >= datetime.now() - timedelta(hours=24)
            ]
            
            return {
                'total_invalidations': self.total_invalidations,
                'cascade_invalidations': self.cascade_invalidations,
                'invalidations_by_trigger': dict(self.invalidations_by_trigger),
                'recent_events_24h': len(recent_events),
                'active_rules': len(self.invalidation_rules),
                'tag_dependencies': len(self.tag_dependencies),
                'key_dependencies': len(self.key_dependencies),
                'average_keys_per_event': (
                    sum(len(event.affected_keys) for event in recent_events) /
                    max(len(recent_events), 1)
                )
            }
    
    def analyze_invalidation_patterns(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze invalidation patterns over time."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            recent_events = [
                event for event in self.invalidation_history
                if event.timestamp >= cutoff_time
            ]
        
        if not recent_events:
            return {'pattern': 'no_data'}
        
        # Analyze patterns
        trigger_distribution = defaultdict(int)
        hourly_distribution = defaultdict(int)
        cascade_events = 0
        
        for event in recent_events:
            trigger_distribution[event.trigger.value] += 1
            hour_key = event.timestamp.strftime('%Y-%m-%d %H:00')
            hourly_distribution[hour_key] += 1
            
            if event.cascade_level > 0:
                cascade_events += 1
        
        # Find peak hours
        peak_hour = max(hourly_distribution.items(), key=lambda x: x[1])[0] if hourly_distribution else None
        
        return {
            'analysis_period_hours': hours,
            'total_events': len(recent_events),
            'trigger_distribution': dict(trigger_distribution),
            'hourly_distribution': dict(hourly_distribution),
            'peak_hour': peak_hour,
            'cascade_rate': cascade_events / len(recent_events) if recent_events else 0,
            'avg_keys_per_event': sum(len(e.affected_keys) for e in recent_events) / len(recent_events)
        }
    
    def optimize_rules(self) -> Dict[str, Any]:
        """Analyze and optimize invalidation rules."""
        with self._lock:
            rules_analysis = {}
            recommendations = []
            
            for rule_id, rule in self.invalidation_rules.items():
                # Find events triggered by this rule
                rule_events = [
                    event for event in self.invalidation_history
                    if event.rule_id == rule_id
                ]
                
                if rule_events:
                    avg_keys_affected = sum(len(e.affected_keys) for e in rule_events) / len(rule_events)
                    last_triggered = max(event.timestamp for event in rule_events)
                    
                    rules_analysis[rule_id] = {
                        'trigger_count': len(rule_events),
                        'avg_keys_affected': avg_keys_affected,
                        'last_triggered': last_triggered.isoformat(),
                        'days_since_last_trigger': (datetime.now() - last_triggered).days
                    }
                    
                    # Generate recommendations
                    if len(rule_events) == 0:
                        recommendations.append(f"Rule {rule_id} never triggered - consider removing")
                    elif avg_keys_affected > 1000:
                        recommendations.append(f"Rule {rule_id} affects many keys - review for over-invalidation")
                    elif (datetime.now() - last_triggered).days > 30:
                        recommendations.append(f"Rule {rule_id} not triggered recently - review necessity")
                else:
                    rules_analysis[rule_id] = {
                        'trigger_count': 0,
                        'status': 'never_triggered'
                    }
                    recommendations.append(f"Rule {rule_id} never triggered - consider removing")
        
        return {
            'rules_analysis': rules_analysis,
            'recommendations': recommendations,
            'optimization_timestamp': datetime.now().isoformat()
        }
    
    def export_invalidation_report(self, filepath: str) -> bool:
        """Export comprehensive invalidation report."""
        try:
            report = {
                'generated_at': datetime.now().isoformat(),
                'statistics': self.get_invalidation_statistics(),
                'pattern_analysis': self.analyze_invalidation_patterns(),
                'rules_optimization': self.optimize_rules(),
                'active_rules': {
                    rule_id: {
                        'trigger': rule.trigger.value,
                        'cache_keys_count': len(rule.cache_keys),
                        'tags_count': len(rule.tags),
                        'dependencies_count': len(rule.dependencies),
                        'cascade': rule.cascade,
                        'created_at': rule.created_at.isoformat()
                    }
                    for rule_id, rule in self.invalidation_rules.items()
                }
            }
            
            import json
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error exporting invalidation report: {e}")
            return False