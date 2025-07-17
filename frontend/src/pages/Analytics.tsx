import React from 'react';

export default function Analytics() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Analytics</h1>
        <p className="text-gray-600 mt-2">
          Deep dive into your Snowflake cost and usage analytics
        </p>
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Cost Analytics</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="font-medium text-gray-900 mb-2">Cost Trends</h3>
            <div className="h-64 bg-white rounded border flex items-center justify-center">
              <span className="text-gray-500">Cost trend chart will go here</span>
            </div>
          </div>
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="font-medium text-gray-900 mb-2">Cost Breakdown</h3>
            <div className="h-64 bg-white rounded border flex items-center justify-center">
              <span className="text-gray-500">Cost breakdown chart will go here</span>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Usage Analytics</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="font-medium text-gray-900 mb-2">Query Performance</h3>
            <div className="h-64 bg-white rounded border flex items-center justify-center">
              <span className="text-gray-500">Query performance chart will go here</span>
            </div>
          </div>
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="font-medium text-gray-900 mb-2">Warehouse Utilization</h3>
            <div className="h-64 bg-white rounded border flex items-center justify-center">
              <span className="text-gray-500">Warehouse utilization chart will go here</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}