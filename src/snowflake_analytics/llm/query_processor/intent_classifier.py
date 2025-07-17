"""
Intent Classifier for Natural Language Queries

Classifies user intents for Snowflake analytics queries to route them
to appropriate handlers and generate relevant responses.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

from ...utils.logger import get_logger

logger = get_logger(__name__)


class QueryIntent(Enum):
    """Types of query intents for Snowflake analytics."""
    
    # Cost-related intents
    COST_OVERVIEW = "cost_overview"
    COST_BREAKDOWN = "cost_breakdown"
    COST_TREND = "cost_trend"
    COST_COMPARISON = "cost_comparison"
    COST_OPTIMIZATION = "cost_optimization"
    
    # Usage-related intents
    USAGE_OVERVIEW = "usage_overview"
    USAGE_PATTERN = "usage_pattern"
    USER_ACTIVITY = "user_activity"
    QUERY_PERFORMANCE = "query_performance"
    WAREHOUSE_UTILIZATION = "warehouse_utilization"
    
    # Prediction intents
    COST_PREDICTION = "cost_prediction"
    USAGE_PREDICTION = "usage_prediction"
    CAPACITY_PLANNING = "capacity_planning"
    
    # Anomaly and alert intents
    ANOMALY_DETECTION = "anomaly_detection"
    ALERT_STATUS = "alert_status"
    ISSUE_INVESTIGATION = "issue_investigation"
    
    # General information
    HELP = "help"
    STATUS = "status"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    """Result of intent classification."""
    intent: QueryIntent
    confidence: float
    entities: Dict[str, Any]
    time_range: Optional[str]
    filters: Dict[str, Any]
    suggested_queries: List[str]


class IntentClassifier:
    """Classifies natural language queries into specific intents."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize intent classifier."""
        self.config = config or {}
        
        # Intent patterns - keyword-based classification
        self.intent_patterns = {
            QueryIntent.COST_OVERVIEW: [
                r"\b(cost|spend|spending|expense|bill|charge)\b",
                r"\bhow much.*cost",
                r"\btotal.*cost",
                r"\bcost.*overview"
            ],
            QueryIntent.COST_BREAKDOWN: [
                r"\bcost.*breakdown",
                r"\bcost.*by\b",
                r"\bspending.*by\b",
                r"\bwhere.*money",
                r"\bcost.*per\b"
            ],
            QueryIntent.COST_TREND: [
                r"\bcost.*trend",
                r"\bcost.*over time",
                r"\bcost.*increase|decrease",
                r"\bspending.*pattern",
                r"\bcost.*month|week|day"
            ],
            QueryIntent.COST_COMPARISON: [
                r"\bcompare.*cost",
                r"\bcost.*vs\b",
                r"\bcost.*compared to",
                r"\bbefore.*after.*cost"
            ],
            QueryIntent.COST_OPTIMIZATION: [
                r"\boptimize.*cost",
                r"\breduce.*cost",
                r"\bsave.*money",
                r"\bcost.*efficient",
                r"\bwaste|wasting"
            ],
            QueryIntent.USAGE_OVERVIEW: [
                r"\busage.*overview",
                r"\bhow much.*use",
                r"\busage.*summary",
                r"\bactivity.*overview"
            ],
            QueryIntent.USAGE_PATTERN: [
                r"\busage.*pattern",
                r"\bwhen.*most.*use",
                r"\bpeak.*usage",
                r"\busage.*time"
            ],
            QueryIntent.USER_ACTIVITY: [
                r"\buser.*activity",
                r"\bwho.*query|queries",
                r"\buser.*usage",
                r"\bmost.*active.*user"
            ],
            QueryIntent.QUERY_PERFORMANCE: [
                r"\bquery.*performance",
                r"\bslow.*query|queries",
                r"\bperformance.*issue",
                r"\bexecution.*time"
            ],
            QueryIntent.WAREHOUSE_UTILIZATION: [
                r"\bwarehouse.*util",
                r"\bwarehouse.*usage",
                r"\bcompute.*usage",
                r"\bwarehouse.*performance"
            ],
            QueryIntent.COST_PREDICTION: [
                r"\bpredict.*cost",
                r"\bforecast.*cost",
                r"\bfuture.*cost",
                r"\bnext.*month.*cost"
            ],
            QueryIntent.USAGE_PREDICTION: [
                r"\bpredict.*usage",
                r"\bforecast.*usage",
                r"\bfuture.*usage",
                r"\bexpected.*usage"
            ],
            QueryIntent.CAPACITY_PLANNING: [
                r"\bcapacity.*plan",
                r"\bscale.*up|down",
                r"\bresource.*plan",
                r"\bgrowth.*plan"
            ],
            QueryIntent.ANOMALY_DETECTION: [
                r"\banomaly|anomalies",
                r"\bunusual.*activity",
                r"\bspike.*usage|cost",
                r"\babnormal.*pattern"
            ],
            QueryIntent.ALERT_STATUS: [
                r"\balert.*status",
                r"\bnotification",
                r"\bwarning",
                r"\bthreshold.*breach"
            ],
            QueryIntent.HELP: [
                r"\bhelp\b",
                r"\bhow.*to\b",
                r"\bwhat.*can",
                r"\bcommand|commands"
            ],
            QueryIntent.STATUS: [
                r"\bstatus\b",
                r"\bhealth\b",
                r"\bsystem.*status",
                r"\bservice.*status"
            ]
        }
        
        # Entity extraction patterns
        self.entity_patterns = {
            'time_range': [
                (r'\b(today|yesterday)\b', 'relative'),
                (r'\b(this|last)\s+(week|month|quarter|year)\b', 'relative'),
                (r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', 'absolute'),
                (r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', 'month')
            ],
            'warehouse': [
                (r'\bwarehouse\s+(\w+)', 'name'),
                (r'\b(\w+)\s+warehouse\b', 'name')
            ],
            'user': [
                (r'\buser\s+(\w+)', 'name'),
                (r'\b(\w+@\w+\.\w+)\b', 'email')
            ],
            'metric': [
                (r'\b(cost|credit|storage|compute|query)\b', 'type')
            ]
        }
        
        logger.info("Intent classifier initialized")
    
    def classify_intent(self, query: str) -> IntentResult:
        """Classify the intent of a natural language query.
        
        Args:
            query: Natural language query string
            
        Returns:
            IntentResult with classified intent and extracted entities
        """
        query_lower = query.lower()
        
        # Score each intent based on pattern matches
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, query_lower)
                score += len(matches) * 10  # Weight each match
                
                # Bonus for exact phrase matches
                if re.search(pattern, query_lower):
                    score += 5
            
            if score > 0:
                intent_scores[intent] = score
        
        # Determine best intent
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            confidence = min(intent_scores[best_intent] / 20.0, 1.0)  # Normalize to 0-1
        else:
            best_intent = QueryIntent.UNKNOWN
            confidence = 0.0
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Extract time range
        time_range = self._extract_time_range(query)
        
        # Generate filters
        filters = self._generate_filters(query, entities)
        
        # Generate suggested queries
        suggested_queries = self._generate_suggestions(best_intent, entities)
        
        result = IntentResult(
            intent=best_intent,
            confidence=confidence,
            entities=entities,
            time_range=time_range,
            filters=filters,
            suggested_queries=suggested_queries
        )
        
        logger.debug(f"Classified intent: {best_intent.value} (confidence: {confidence:.2f})")
        return result
    
    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities from the query."""
        entities = {}
        
        for entity_type, patterns in self.entity_patterns.items():
            matches = []
            for pattern, label in patterns:
                found = re.findall(pattern, query, re.IGNORECASE)
                if found:
                    matches.extend([(match, label) for match in found])
            
            if matches:
                entities[entity_type] = matches
        
        return entities
    
    def _extract_time_range(self, query: str) -> Optional[str]:
        """Extract time range from query."""
        time_patterns = [
            (r'\btoday\b', 'today'),
            (r'\byesterday\b', 'yesterday'),
            (r'\bthis week\b', 'this_week'),
            (r'\blast week\b', 'last_week'),
            (r'\bthis month\b', 'this_month'),
            (r'\blast month\b', 'last_month'),
            (r'\bthis year\b', 'this_year'),
            (r'\blast (\d+) days?\b', 'last_n_days'),
            (r'\blast (\d+) weeks?\b', 'last_n_weeks'),
            (r'\blast (\d+) months?\b', 'last_n_months')
        ]
        
        for pattern, time_type in time_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                if 'n_' in time_type:
                    return f"{time_type}_{match.group(1)}"
                return time_type
        
        return None
    
    def _generate_filters(self, query: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Generate query filters based on entities."""
        filters = {}
        
        # Warehouse filter
        if 'warehouse' in entities:
            filters['warehouse'] = [match[0] for match, _ in entities['warehouse']]
        
        # User filter
        if 'user' in entities:
            filters['user'] = [match[0] for match, _ in entities['user']]
        
        # Metric filter
        if 'metric' in entities:
            filters['metric'] = [match[0] for match, _ in entities['metric']]
        
        # Additional context filters
        if 'expensive' in query.lower() or 'highest' in query.lower():
            filters['order_by'] = 'cost_desc'
        elif 'cheap' in query.lower() or 'lowest' in query.lower():
            filters['order_by'] = 'cost_asc'
        
        if 'top' in query.lower():
            limit_match = re.search(r'top\s+(\d+)', query.lower())
            if limit_match:
                filters['limit'] = int(limit_match.group(1))
            else:
                filters['limit'] = 10
        
        return filters
    
    def _generate_suggestions(self, intent: QueryIntent, entities: Dict[str, Any]) -> List[str]:
        """Generate suggested queries based on intent."""
        suggestions = []
        
        base_suggestions = {
            QueryIntent.COST_OVERVIEW: [
                "What were our total Snowflake costs last month?",
                "Show me cost breakdown by warehouse",
                "How much did we spend on storage vs compute?"
            ],
            QueryIntent.USAGE_OVERVIEW: [
                "What's our current warehouse utilization?",
                "Show me query volume trends",
                "Which users are most active?"
            ],
            QueryIntent.COST_PREDICTION: [
                "What will our costs be next month?",
                "Predict costs for the next quarter",
                "Show cost forecast based on current trends"
            ],
            QueryIntent.ANOMALY_DETECTION: [
                "Show me unusual cost spikes",
                "Detect performance anomalies",
                "Find abnormal usage patterns"
            ],
            QueryIntent.HELP: [
                "What queries can I ask?",
                "Show me example questions",
                "How do I ask about costs?"
            ]
        }
        
        if intent in base_suggestions:
            suggestions.extend(base_suggestions[intent])
        
        # Add contextual suggestions based on entities
        if 'warehouse' in entities:
            warehouse_name = entities['warehouse'][0][0]
            suggestions.append(f"Show costs for {warehouse_name} warehouse")
            suggestions.append(f"Compare {warehouse_name} performance")
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def get_supported_intents(self) -> List[Dict[str, Any]]:
        """Get list of supported intents with descriptions."""
        intent_descriptions = {
            QueryIntent.COST_OVERVIEW: "Get overview of Snowflake costs",
            QueryIntent.COST_BREAKDOWN: "Break down costs by different dimensions",
            QueryIntent.COST_TREND: "Analyze cost trends over time",
            QueryIntent.COST_COMPARISON: "Compare costs between periods",
            QueryIntent.COST_OPTIMIZATION: "Get cost optimization recommendations",
            QueryIntent.USAGE_OVERVIEW: "Get overview of Snowflake usage",
            QueryIntent.USAGE_PATTERN: "Analyze usage patterns",
            QueryIntent.USER_ACTIVITY: "Analyze user activity and behavior",
            QueryIntent.QUERY_PERFORMANCE: "Analyze query performance metrics",
            QueryIntent.WAREHOUSE_UTILIZATION: "Analyze warehouse utilization",
            QueryIntent.COST_PREDICTION: "Predict future costs",
            QueryIntent.USAGE_PREDICTION: "Predict future usage",
            QueryIntent.CAPACITY_PLANNING: "Plan capacity and scaling",
            QueryIntent.ANOMALY_DETECTION: "Detect anomalies and unusual patterns",
            QueryIntent.ALERT_STATUS: "Check alert and notification status",
            QueryIntent.HELP: "Get help and guidance",
            QueryIntent.STATUS: "Check system status"
        }
        
        return [
            {
                "intent": intent.value,
                "description": intent_descriptions.get(intent, "No description available"),
                "examples": self._get_intent_examples(intent)
            }
            for intent in QueryIntent if intent != QueryIntent.UNKNOWN
        ]
    
    def _get_intent_examples(self, intent: QueryIntent) -> List[str]:
        """Get example queries for an intent."""
        examples = {
            QueryIntent.COST_OVERVIEW: [
                "What were our Snowflake costs last month?",
                "Show me total spending this year"
            ],
            QueryIntent.COST_BREAKDOWN: [
                "Break down costs by warehouse",
                "Show spending by user"
            ],
            QueryIntent.USAGE_OVERVIEW: [
                "What's our current usage?",
                "Show me activity summary"
            ],
            QueryIntent.QUERY_PERFORMANCE: [
                "Show me slow queries",
                "What queries take the longest?"
            ]
        }
        
        return examples.get(intent, [])