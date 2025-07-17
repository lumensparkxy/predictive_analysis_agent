"""
LLM Integration Module for Snowflake Analytics

This module provides Large Language Model capabilities including:
- Natural language query processing
- Intelligent insights generation
- Automated recommendations
- Conversational chat interface
- Output processing and visualization

Components:
- client: LLM API clients (OpenAI, Azure OpenAI)
- query_processor: Natural language to SQL conversion
- insights: Automated insight generation
- recommendations: Intelligent recommendations
- chat: Conversational interface
- output: Response processing and visualization
"""

from typing import Optional

# Version info
__version__ = "1.0.0"

# Core LLM client components
try:
    from .client import OpenAIClient, AzureOpenAIClient, RateLimiter, ResponseValidator
except ImportError as e:
    print(f"Warning: LLM client components import failed: {e}")
    OpenAIClient = None
    AzureOpenAIClient = None
    RateLimiter = None
    ResponseValidator = None

# Query processing components
try:
    from .query_processor import (
        NaturalLanguageToSQL, IntentClassifier, QueryValidator, 
        ResultInterpreter, QueryInterface
    )
except ImportError as e:
    print(f"Warning: Query processor components import failed: {e}")
    NaturalLanguageToSQL = None
    IntentClassifier = None
    QueryValidator = None
    ResultInterpreter = None
    QueryInterface = None

# Insights generation components
try:
    from .insights import (
        CostAnalyzer, UsageAnalyzer, AnomalyExplainer, 
        TrendAnalyzer, InsightGenerator
    )
except ImportError as e:
    print(f"Warning: Insights components import failed: {e}")
    CostAnalyzer = None
    UsageAnalyzer = None
    AnomalyExplainer = None
    TrendAnalyzer = None
    InsightGenerator = None

# Recommendation engine components
try:
    from .recommendations import (
        CostOptimizer, PerformanceOptimizer, ActionRecommender,
        Prioritizer, RecommendationEngine
    )
except ImportError as e:
    print(f"Warning: Recommendation components import failed: {e}")
    CostOptimizer = None
    PerformanceOptimizer = None
    ActionRecommender = None
    Prioritizer = None
    RecommendationEngine = None

# Chat interface components
try:
    from .chat import (
        ChatInterface, ContextManager, HistoryManager,
        FollowupHandler, ConversationFlow
    )
except ImportError as e:
    print(f"Warning: Chat components import failed: {e}")
    ChatInterface = None
    ContextManager = None
    HistoryManager = None
    FollowupHandler = None
    ConversationFlow = None

# Output processing components
try:
    from .output import (
        ResponseProcessor, ChartGenerator, ReportGenerator,
        Visualizer, OutputFormatter
    )
except ImportError as e:
    print(f"Warning: Output processing components import failed: {e}")
    ResponseProcessor = None
    ChartGenerator = None
    ReportGenerator = None
    Visualizer = None
    OutputFormatter = None

# Global LLM service instance
_llm_service_instance: Optional['LLMService'] = None


class LLMService:
    """Main LLM service that orchestrates all LLM components."""
    
    def __init__(self, config: dict = None):
        """Initialize LLM service with configuration."""
        self.config = config or {}
        self.client = None
        self.query_interface = None
        self.insight_generator = None
        self.recommendation_engine = None
        self.chat_interface = None
        self.output_processor = None
        
        # Initialize components if available
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize LLM components based on configuration."""
        try:
            # Initialize LLM client
            if OpenAIClient and self.config.get('openai_api_key'):
                self.client = OpenAIClient(config=self.config)
            elif AzureOpenAIClient and self.config.get('azure_openai_endpoint'):
                self.client = AzureOpenAIClient(config=self.config)
            
            # Initialize other components if client is available
            if self.client:
                if QueryInterface:
                    self.query_interface = QueryInterface(client=self.client)
                if InsightGenerator:
                    self.insight_generator = InsightGenerator(client=self.client)
                if RecommendationEngine:
                    self.recommendation_engine = RecommendationEngine(client=self.client)
                if ChatInterface:
                    self.chat_interface = ChatInterface(client=self.client)
                if ResponseProcessor:
                    self.output_processor = ResponseProcessor(client=self.client)
                    
        except Exception as e:
            print(f"Warning: LLM service initialization failed: {e}")
    
    def is_available(self) -> bool:
        """Check if LLM service is available and properly configured."""
        return self.client is not None
    
    def process_natural_language_query(self, query: str, context: dict = None) -> dict:
        """Process a natural language query and return results."""
        if not self.query_interface:
            return {"error": "Query interface not available"}
        
        return self.query_interface.process_query(query, context)
    
    def generate_insights(self, data: dict, insight_type: str = "auto") -> dict:
        """Generate intelligent insights from data."""
        if not self.insight_generator:
            return {"error": "Insight generator not available"}
        
        return self.insight_generator.generate_insights(data, insight_type)
    
    def get_recommendations(self, analysis_data: dict, recommendation_type: str = "cost") -> dict:
        """Get intelligent recommendations based on analysis data."""
        if not self.recommendation_engine:
            return {"error": "Recommendation engine not available"}
        
        return self.recommendation_engine.get_recommendations(analysis_data, recommendation_type)
    
    def chat(self, message: str, conversation_id: str = None) -> dict:
        """Process a chat message and return response."""
        if not self.chat_interface:
            return {"error": "Chat interface not available"}
        
        return self.chat_interface.chat(message, conversation_id)


def create_llm_service(config: dict = None) -> LLMService:
    """Create a new LLM service instance."""
    return LLMService(config=config)


def get_llm_service(config: dict = None) -> LLMService:
    """Get the global LLM service instance, creating it if needed."""
    global _llm_service_instance
    if _llm_service_instance is None:
        _llm_service_instance = create_llm_service(config)
    return _llm_service_instance


def reset_llm_service():
    """Reset the global LLM service instance."""
    global _llm_service_instance
    _llm_service_instance = None


__all__ = [
    # Main service
    "LLMService",
    "create_llm_service", 
    "get_llm_service",
    "reset_llm_service",
    
    # Client components
    "OpenAIClient",
    "AzureOpenAIClient", 
    "RateLimiter",
    "ResponseValidator",
    
    # Query processing
    "NaturalLanguageToSQL",
    "IntentClassifier",
    "QueryValidator",
    "ResultInterpreter", 
    "QueryInterface",
    
    # Insights generation
    "CostAnalyzer",
    "UsageAnalyzer",
    "AnomalyExplainer",
    "TrendAnalyzer",
    "InsightGenerator",
    
    # Recommendations
    "CostOptimizer",
    "PerformanceOptimizer", 
    "ActionRecommender",
    "Prioritizer",
    "RecommendationEngine",
    
    # Chat interface
    "ChatInterface",
    "ContextManager",
    "HistoryManager",
    "FollowupHandler",
    "ConversationFlow",
    
    # Output processing
    "ResponseProcessor",
    "ChartGenerator",
    "ReportGenerator",
    "Visualizer",
    "OutputFormatter",
]