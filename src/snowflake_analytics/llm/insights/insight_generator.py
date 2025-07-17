"""
Insight Generator - Main Insights Orchestrator

Orchestrates the generation of intelligent insights from Snowflake analytics data
using specialized analyzers and LLM capabilities.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

from ...utils.logger import get_logger
from .cost_analyzer import CostAnalyzer
from .usage_analyzer import UsageAnalyzer
from .anomaly_explainer import AnomalyExplainer
from .trend_analyzer import TrendAnalyzer

logger = get_logger(__name__)


@dataclass
class GeneratedInsight:
    """Represents a generated insight."""
    type: str
    category: str
    title: str
    description: str
    severity: str  # low, medium, high, critical
    confidence: float
    data_points: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime


class InsightGenerator:
    """Main insight generation orchestrator."""
    
    def __init__(self, client=None, config: Dict[str, Any] = None):
        """Initialize insight generator.
        
        Args:
            client: LLM client for advanced analysis
            config: Configuration dictionary
        """
        self.client = client
        self.config = config or {}
        
        # Initialize specialized analyzers
        self.cost_analyzer = CostAnalyzer(client, config.get('cost_analyzer', {}))
        self.usage_analyzer = UsageAnalyzer(client, config.get('usage_analyzer', {}))
        self.anomaly_explainer = AnomalyExplainer(client, config.get('anomaly_explainer', {}))
        self.trend_analyzer = TrendAnalyzer(client, config.get('trend_analyzer', {}))
        
        # Insight generation settings
        self.max_insights_per_category = config.get('max_insights_per_category', 5)
        self.min_confidence_threshold = config.get('min_confidence_threshold', 0.6)
        
        logger.info("Insight generator initialized")
    
    async def generate_insights(
        self, 
        data: Dict[str, Any], 
        insight_type: str = "auto",
        context: Dict[str, Any] = None
    ) -> List[GeneratedInsight]:
        """Generate intelligent insights from analytics data.
        
        Args:
            data: Analytics data (costs, usage, performance, etc.)
            insight_type: Type of insights to generate ('auto', 'cost', 'usage', etc.)
            context: Additional context for insight generation
            
        Returns:
            List of GeneratedInsight objects
        """
        context = context or {}
        insights = []
        
        try:
            logger.info(f"Generating insights of type: {insight_type}")
            
            if insight_type == "auto" or insight_type == "cost":
                cost_insights = await self._generate_cost_insights(data, context)
                insights.extend(cost_insights)
            
            if insight_type == "auto" or insight_type == "usage":
                usage_insights = await self._generate_usage_insights(data, context)
                insights.extend(usage_insights)
            
            if insight_type == "auto" or insight_type == "anomaly":
                anomaly_insights = await self._generate_anomaly_insights(data, context)
                insights.extend(anomaly_insights)
            
            if insight_type == "auto" or insight_type == "trend":
                trend_insights = await self._generate_trend_insights(data, context)
                insights.extend(trend_insights)
            
            # Filter insights by confidence threshold
            filtered_insights = [
                insight for insight in insights 
                if insight.confidence >= self.min_confidence_threshold
            ]
            
            # Sort by severity and confidence
            filtered_insights.sort(key=lambda x: (
                self._get_severity_score(x.severity), 
                x.confidence
            ), reverse=True)
            
            logger.info(f"Generated {len(filtered_insights)} insights")
            return filtered_insights
            
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            return []
    
    async def _generate_cost_insights(
        self, 
        data: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> List[GeneratedInsight]:
        """Generate cost-related insights."""
        insights = []
        
        try:
            cost_data = data.get('cost_metrics', {})
            if not cost_data:
                return insights
            
            # Analyze cost trends
            cost_analysis = await self.cost_analyzer.analyze_cost_trends(cost_data)
            
            for analysis in cost_analysis:
                insight = GeneratedInsight(
                    type="cost_trend",
                    category="cost",
                    title=analysis.get('title', 'Cost Analysis'),
                    description=analysis.get('description', ''),
                    severity=analysis.get('severity', 'medium'),
                    confidence=analysis.get('confidence', 0.7),
                    data_points=analysis.get('data_points', {}),
                    recommendations=analysis.get('recommendations', []),
                    timestamp=datetime.now()
                )
                insights.append(insight)
            
            # Analyze cost efficiency
            efficiency_analysis = await self.cost_analyzer.analyze_cost_efficiency(cost_data)
            
            for analysis in efficiency_analysis:
                insight = GeneratedInsight(
                    type="cost_efficiency",
                    category="cost",
                    title=analysis.get('title', 'Cost Efficiency Analysis'),
                    description=analysis.get('description', ''),
                    severity=analysis.get('severity', 'medium'),
                    confidence=analysis.get('confidence', 0.7),
                    data_points=analysis.get('data_points', {}),
                    recommendations=analysis.get('recommendations', []),
                    timestamp=datetime.now()
                )
                insights.append(insight)
            
        except Exception as e:
            logger.error(f"Cost insight generation failed: {e}")
        
        return insights[:self.max_insights_per_category]
    
    async def _generate_usage_insights(
        self, 
        data: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> List[GeneratedInsight]:
        """Generate usage-related insights."""
        insights = []
        
        try:
            usage_data = data.get('usage_metrics', {})
            if not usage_data:
                return insights
            
            # Analyze usage patterns
            pattern_analysis = await self.usage_analyzer.analyze_usage_patterns(usage_data)
            
            for analysis in pattern_analysis:
                insight = GeneratedInsight(
                    type="usage_pattern",
                    category="usage",
                    title=analysis.get('title', 'Usage Pattern Analysis'),
                    description=analysis.get('description', ''),
                    severity=analysis.get('severity', 'medium'),
                    confidence=analysis.get('confidence', 0.7),
                    data_points=analysis.get('data_points', {}),
                    recommendations=analysis.get('recommendations', []),
                    timestamp=datetime.now()
                )
                insights.append(insight)
            
            # Analyze performance metrics
            performance_analysis = await self.usage_analyzer.analyze_performance_metrics(usage_data)
            
            for analysis in performance_analysis:
                insight = GeneratedInsight(
                    type="performance",
                    category="usage",
                    title=analysis.get('title', 'Performance Analysis'),
                    description=analysis.get('description', ''),
                    severity=analysis.get('severity', 'medium'),
                    confidence=analysis.get('confidence', 0.7),
                    data_points=analysis.get('data_points', {}),
                    recommendations=analysis.get('recommendations', []),
                    timestamp=datetime.now()
                )
                insights.append(insight)
            
        except Exception as e:
            logger.error(f"Usage insight generation failed: {e}")
        
        return insights[:self.max_insights_per_category]
    
    async def _generate_anomaly_insights(
        self, 
        data: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> List[GeneratedInsight]:
        """Generate anomaly-related insights."""
        insights = []
        
        try:
            # Check for cost anomalies
            cost_data = data.get('cost_metrics', {})
            if cost_data:
                cost_anomalies = await self.anomaly_explainer.detect_cost_anomalies(cost_data)
                
                for anomaly in cost_anomalies:
                    insight = GeneratedInsight(
                        type="anomaly",
                        category="cost_anomaly",
                        title=anomaly.get('title', 'Cost Anomaly Detected'),
                        description=anomaly.get('description', ''),
                        severity="high",  # Anomalies are typically high priority
                        confidence=anomaly.get('confidence', 0.8),
                        data_points=anomaly.get('data_points', {}),
                        recommendations=anomaly.get('recommendations', []),
                        timestamp=datetime.now()
                    )
                    insights.append(insight)
            
            # Check for usage anomalies
            usage_data = data.get('usage_metrics', {})
            if usage_data:
                usage_anomalies = await self.anomaly_explainer.detect_usage_anomalies(usage_data)
                
                for anomaly in usage_anomalies:
                    insight = GeneratedInsight(
                        type="anomaly",
                        category="usage_anomaly",
                        title=anomaly.get('title', 'Usage Anomaly Detected'),
                        description=anomaly.get('description', ''),
                        severity="high",
                        confidence=anomaly.get('confidence', 0.8),
                        data_points=anomaly.get('data_points', {}),
                        recommendations=anomaly.get('recommendations', []),
                        timestamp=datetime.now()
                    )
                    insights.append(insight)
            
        except Exception as e:
            logger.error(f"Anomaly insight generation failed: {e}")
        
        return insights[:self.max_insights_per_category]
    
    async def _generate_trend_insights(
        self, 
        data: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> List[GeneratedInsight]:
        """Generate trend-related insights."""
        insights = []
        
        try:
            # Analyze cost trends
            cost_data = data.get('cost_metrics', {})
            if cost_data:
                cost_trends = await self.trend_analyzer.analyze_cost_trends(cost_data)
                
                for trend in cost_trends:
                    insight = GeneratedInsight(
                        type="trend",
                        category="cost_trend",
                        title=trend.get('title', 'Cost Trend Analysis'),
                        description=trend.get('description', ''),
                        severity=trend.get('severity', 'medium'),
                        confidence=trend.get('confidence', 0.7),
                        data_points=trend.get('data_points', {}),
                        recommendations=trend.get('recommendations', []),
                        timestamp=datetime.now()
                    )
                    insights.append(insight)
            
            # Analyze usage trends
            usage_data = data.get('usage_metrics', {})
            if usage_data:
                usage_trends = await self.trend_analyzer.analyze_usage_trends(usage_data)
                
                for trend in usage_trends:
                    insight = GeneratedInsight(
                        type="trend",
                        category="usage_trend",
                        title=trend.get('title', 'Usage Trend Analysis'),
                        description=trend.get('description', ''),
                        severity=trend.get('severity', 'medium'),
                        confidence=trend.get('confidence', 0.7),
                        data_points=trend.get('data_points', {}),
                        recommendations=trend.get('recommendations', []),
                        timestamp=datetime.now()
                    )
                    insights.append(insight)
            
        except Exception as e:
            logger.error(f"Trend insight generation failed: {e}")
        
        return insights[:self.max_insights_per_category]
    
    def _get_severity_score(self, severity: str) -> int:
        """Get numeric score for severity level."""
        severity_scores = {
            'critical': 4,
            'high': 3,
            'medium': 2,
            'low': 1
        }
        return severity_scores.get(severity.lower(), 1)
    
    async def generate_summary_insight(
        self, 
        insights: List[GeneratedInsight],
        data: Dict[str, Any]
    ) -> GeneratedInsight:
        """Generate a summary insight from multiple insights.
        
        Args:
            insights: List of individual insights
            data: Original analytics data
            
        Returns:
            Summary insight
        """
        try:
            # Count insights by category and severity
            categories = {}
            severities = {}
            
            for insight in insights:
                categories[insight.category] = categories.get(insight.category, 0) + 1
                severities[insight.severity] = severities.get(insight.severity, 0) + 1
            
            # Determine overall health status
            if severities.get('critical', 0) > 0:
                overall_severity = "critical"
                health_status = "critical issues detected"
            elif severities.get('high', 0) > 0:
                overall_severity = "high"
                health_status = "attention required"
            elif severities.get('medium', 0) > 0:
                overall_severity = "medium"
                health_status = "monitoring recommended"
            else:
                overall_severity = "low"
                health_status = "system healthy"
            
            # Generate summary description
            total_insights = len(insights)
            top_category = max(categories, key=categories.get) if categories else "general"
            
            description = f"""
            Analysis of your Snowflake environment has identified {total_insights} insights.
            Primary focus area: {top_category.replace('_', ' ').title()}.
            Overall status: {health_status}.
            """
            
            # Generate summary recommendations
            all_recommendations = []
            for insight in insights[:5]:  # Top 5 insights
                all_recommendations.extend(insight.recommendations)
            
            # Deduplicate and prioritize recommendations
            unique_recommendations = list(dict.fromkeys(all_recommendations))[:5]
            
            return GeneratedInsight(
                type="summary",
                category="overview",
                title="Snowflake Analytics Summary",
                description=description.strip(),
                severity=overall_severity,
                confidence=0.9,
                data_points={
                    "total_insights": total_insights,
                    "categories": categories,
                    "severities": severities,
                    "health_status": health_status
                },
                recommendations=unique_recommendations,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Summary insight generation failed: {e}")
            return GeneratedInsight(
                type="summary",
                category="overview",
                title="Analysis Summary",
                description="Summary generation encountered an error",
                severity="low",
                confidence=0.1,
                data_points={},
                recommendations=["Review individual insights"],
                timestamp=datetime.now()
            )
    
    async def get_insight_trends(
        self, 
        historical_insights: List[GeneratedInsight],
        time_period_days: int = 30
    ) -> Dict[str, Any]:
        """Analyze trends in insight generation over time.
        
        Args:
            historical_insights: Historical insights to analyze
            time_period_days: Time period to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        try:
            # Group insights by date
            from collections import defaultdict
            daily_insights = defaultdict(list)
            
            for insight in historical_insights:
                date_key = insight.timestamp.strftime('%Y-%m-%d')
                daily_insights[date_key].append(insight)
            
            # Calculate trend metrics
            total_insights_trend = []
            severity_trends = defaultdict(list)
            category_trends = defaultdict(list)
            
            for date, insights in daily_insights.items():
                total_insights_trend.append(len(insights))
                
                # Count by severity
                severity_counts = defaultdict(int)
                category_counts = defaultdict(int)
                
                for insight in insights:
                    severity_counts[insight.severity] += 1
                    category_counts[insight.category] += 1
                
                for severity, count in severity_counts.items():
                    severity_trends[severity].append(count)
                
                for category, count in category_counts.items():
                    category_trends[category].append(count)
            
            return {
                "time_period_days": time_period_days,
                "total_insights_trend": total_insights_trend,
                "severity_trends": dict(severity_trends),
                "category_trends": dict(category_trends),
                "average_daily_insights": sum(total_insights_trend) / len(total_insights_trend) if total_insights_trend else 0,
                "trend_direction": self._calculate_trend_direction(total_insights_trend)
            }
            
        except Exception as e:
            logger.error(f"Insight trend analysis failed: {e}")
            return {"error": str(e)}
    
    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values."""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend calculation
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        if second_avg > first_avg * 1.1:
            return "increasing"
        elif second_avg < first_avg * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def get_generator_stats(self) -> Dict[str, Any]:
        """Get insight generator statistics."""
        return {
            "components": {
                "cost_analyzer": self.cost_analyzer is not None,
                "usage_analyzer": self.usage_analyzer is not None,
                "anomaly_explainer": self.anomaly_explainer is not None,
                "trend_analyzer": self.trend_analyzer is not None
            },
            "configuration": {
                "max_insights_per_category": self.max_insights_per_category,
                "min_confidence_threshold": self.min_confidence_threshold,
                "llm_available": self.client is not None
            },
            "capabilities": [
                "cost_analysis",
                "usage_analysis", 
                "anomaly_detection",
                "trend_analysis",
                "summary_generation"
            ]
        }