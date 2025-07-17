import React from 'react';
import { useQuery } from 'react-query';
import { TrendingUp, Zap, Target, Brain } from 'lucide-react';
import { apiService } from '../services/api';
import { OptimizationRecommendation } from '../types';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';

const PriorityBadge: React.FC<{ priority: string }> = ({ priority }) => {
  const getColorClass = (priority: string) => {
    switch (priority) {
      case 'high': return 'bg-red-100 text-red-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'low': return 'bg-green-100 text-green-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getColorClass(priority)}`}>
      {priority}
    </span>
  );
};

const RecommendationCard: React.FC<{ recommendation: OptimizationRecommendation }> = ({ recommendation }) => {
  const getSavingsColor = (savings: number) => {
    if (savings > 0) return 'text-green-600';
    if (savings < 0) return 'text-red-600';
    return 'text-gray-600';
  };

  return (
    <div className="bg-white rounded-lg shadow p-6 border-l-4 border-l-blue-500">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center space-x-2 mb-2">
            <Target className="h-5 w-5 text-blue-500" />
            <h3 className="text-lg font-medium text-gray-900">{recommendation.title}</h3>
            <PriorityBadge priority={recommendation.priority} />
          </div>
          <p className="text-gray-600 mb-4">{recommendation.description}</p>
          
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4">
            <div>
              <div className="text-sm text-gray-500">Potential Savings</div>
              <div className={`font-medium ${getSavingsColor(recommendation.potential_savings)}`}>
                {recommendation.potential_savings > 0 ? '+' : ''}${recommendation.potential_savings.toLocaleString()}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-500">Implementation Effort</div>
              <div className="font-medium">{recommendation.implementation_effort}</div>
            </div>
            <div>
              <div className="text-sm text-gray-500">Expected Impact</div>
              <div className="font-medium">{recommendation.impact}</div>
            </div>
          </div>
          
          <div className="flex items-center space-x-4 text-sm text-gray-500">
            <div className="flex items-center">
              <span className="font-medium">Category: {recommendation.category}</span>
            </div>
            <div className="flex items-center">
              <span className="font-medium">Type: {recommendation.type}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default function Predictions() {
  const { data: forecastData, isLoading: forecastLoading, error: forecastError } = useQuery(
    'cost-forecast',
    () => apiService.getCostForecast(30),
    {
      refetchInterval: 300000, // 5 minutes
    }
  );

  const { data: recommendationsData, isLoading: recommendationsLoading, error: recommendationsError } = useQuery(
    'optimization-recommendations',
    () => apiService.getOptimizationRecommendations(),
    {
      refetchInterval: 300000,
    }
  );

  const { data: modelData, isLoading: modelLoading } = useQuery(
    'model-performance',
    () => apiService.getModelPerformance(),
    {
      refetchInterval: 300000,
    }
  );

  if (forecastLoading || recommendationsLoading || modelLoading) {
    return <LoadingSpinner />;
  }

  if (forecastError || recommendationsError) {
    return <ErrorMessage message="Failed to load predictions data" />;
  }

  const forecast = forecastData?.data?.summary || {};
  const recommendations = recommendationsData?.data?.recommendations || [];
  const recommendationsSummary = recommendationsData?.data?.summary || {};
  const models = modelData?.data?.models || [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Predictions</h1>
        <p className="text-gray-600 mt-2">
          AI-powered forecasting and optimization recommendations for your Snowflake environment
        </p>
      </div>

      {/* Forecast summary */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Cost Forecast (30 days)</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              ${forecast.total_predicted_cost?.toLocaleString() || 0}
            </div>
            <div className="text-sm text-gray-500">Total Predicted Cost</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">
              ${forecast.avg_daily_cost?.toLocaleString() || 0}
            </div>
            <div className="text-sm text-gray-500">Avg Daily Cost</div>
          </div>
          <div className="text-center">
            <div className={`text-2xl font-bold ${
              forecast.cost_trend === 'increasing' ? 'text-red-600' : 
              forecast.cost_trend === 'decreasing' ? 'text-green-600' : 'text-gray-600'
            }`}>
              {forecast.cost_trend === 'increasing' ? '↗' : 
               forecast.cost_trend === 'decreasing' ? '↘' : '→'}
            </div>
            <div className="text-sm text-gray-500">Cost Trend</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">
              {forecast.confidence === 'high' ? '95%' : forecast.confidence === 'medium' ? '85%' : '75%'}
            </div>
            <div className="text-sm text-gray-500">Confidence</div>
          </div>
        </div>
      </div>

      {/* Chart placeholder */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Cost Forecast Chart</h2>
        <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center">
          <div className="text-center">
            <TrendingUp className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500">Interactive forecast chart will be displayed here</p>
          </div>
        </div>
      </div>

      {/* Model performance */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Model Performance</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {models.map((model: any, index: number) => (
            <div key={index} className="border border-gray-200 rounded-lg p-4">
              <div className="flex items-center mb-3">
                <Brain className="h-5 w-5 text-purple-500 mr-2" />
                <h3 className="font-medium text-gray-900">{model.model_name}</h3>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Type</span>
                  <span className="text-sm font-medium">{model.model_type}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Accuracy</span>
                  <span className="text-sm font-medium">{(model.accuracy * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Status</span>
                  <span className={`text-sm font-medium ${
                    model.status === 'active' ? 'text-green-600' : 'text-gray-600'
                  }`}>
                    {model.status}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Last Trained</span>
                  <span className="text-sm font-medium">
                    {new Date(model.last_trained).toLocaleDateString()}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Recommendations summary */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Optimization Summary</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">{recommendationsSummary.total_recommendations || 0}</div>
            <div className="text-sm text-gray-500">Total Recommendations</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-red-600">{recommendationsSummary.high_priority || 0}</div>
            <div className="text-sm text-gray-500">High Priority</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">
              ${recommendationsSummary.total_potential_savings?.toLocaleString() || 0}
            </div>
            <div className="text-sm text-gray-500">Potential Savings</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">{recommendationsSummary.categories?.length || 0}</div>
            <div className="text-sm text-gray-500">Categories</div>
          </div>
        </div>
      </div>

      {/* Recommendations */}
      <div className="space-y-4">
        <h2 className="text-xl font-semibold text-gray-900">Optimization Recommendations</h2>
        {recommendations.length === 0 ? (
          <div className="bg-white rounded-lg shadow p-6 text-center">
            <Zap className="h-12 w-12 text-blue-500 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No Recommendations Available</h3>
            <p className="text-gray-600">System is running optimally.</p>
          </div>
        ) : (
          <div className="space-y-4">
            {recommendations.map((recommendation: OptimizationRecommendation) => (
              <RecommendationCard key={recommendation.id} recommendation={recommendation} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}