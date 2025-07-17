# Snowflake Analytics Dashboard

## 🚀 Interactive Dashboard & Real-time Analytics Portal

A comprehensive, real-time dashboard for monitoring and optimizing Snowflake data warehouse usage with ML-powered predictions and anomaly detection.

## 📊 Features

### Backend API Features
- **RESTful API Endpoints**: Complete set of endpoints for costs, usage, predictions, anomalies, and alerts
- **Real-time WebSocket Support**: Live data streaming and notifications
- **Authentication & Authorization**: API key-based security with role-based access
- **Rate Limiting**: Configurable rate limits to protect API resources
- **CORS Support**: Cross-origin resource sharing for frontend integration
- **Request/Response Validation**: Comprehensive schema validation
- **Error Handling**: Robust error handling with retry mechanisms

### Frontend Dashboard Features
- **Modern React Application**: Built with React 18, TypeScript, and Vite
- **Real-time Updates**: WebSocket integration for live data streaming
- **Responsive Design**: Mobile-friendly interface with TailwindCSS
- **Interactive Charts**: Cost trends, usage patterns, and performance metrics
- **Alert Management**: Real-time alert monitoring and acknowledgment
- **Anomaly Detection**: Visual anomaly indicators with detailed analysis
- **Predictive Analytics**: ML-powered forecasting and recommendations
- **Settings Management**: Configurable dashboard preferences

### Key Capabilities
- **Cost Analytics**: Track and forecast Snowflake costs
- **Usage Monitoring**: Monitor query performance and warehouse utilization
- **Anomaly Detection**: Identify unusual patterns with ML algorithms
- **Predictive Forecasting**: AI-powered cost and usage predictions
- **Alert System**: Configurable alerts with multiple notification channels
- **Optimization Recommendations**: Actionable insights for cost reduction

## 🏗️ Architecture

### Backend Stack
- **FastAPI**: High-performance Python web framework
- **WebSocket**: Real-time bi-directional communication
- **Pydantic**: Data validation and settings management
- **SQLAlchemy**: Database ORM (when needed)
- **Redis**: Caching and session management
- **JSON**: API data format

### Frontend Stack
- **React 18**: Modern React with hooks and concurrent features
- **TypeScript**: Type-safe JavaScript development
- **Vite**: Fast development build tool
- **TailwindCSS**: Utility-first CSS framework
- **React Query**: Data fetching and state management
- **Chart.js**: Interactive data visualizations
- **Socket.IO**: Real-time communication
- **Axios**: HTTP client for API requests

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Node.js 18+
- npm or yarn

### Backend Setup

1. **Install Python dependencies**:
```bash
cd /home/runner/work/predictive_analysis_agent/predictive_analysis_agent
pip install fastapi uvicorn python-socketio
```

2. **Start the backend server**:
```bash
python -m uvicorn src.snowflake_analytics.api.main:app --reload --host 0.0.0.0 --port 8000
```

3. **Verify backend is running**:
```bash
curl http://localhost:8000/api/health
```

### Frontend Setup

1. **Install Node.js dependencies**:
```bash
cd frontend
npm install
```

2. **Start the development server**:
```bash
npm run dev
```

3. **Open dashboard in browser**:
```
http://localhost:3000
```

### Quick Test

Run the comprehensive test suite:
```bash
python test_dashboard_modules.py
```

## 📖 API Documentation

### Available Endpoints

#### Cost Analytics
- `GET /api/v1/costs/summary` - Cost overview and summary
- `GET /api/v1/costs/trends` - Cost trends over time
- `GET /api/v1/costs/warehouses` - Cost breakdown by warehouse

#### Usage Metrics
- `GET /api/v1/usage/metrics` - Usage statistics and metrics
- `GET /api/v1/usage/performance` - Query performance data
- `GET /api/v1/usage/warehouses` - Warehouse utilization metrics

#### Predictions
- `GET /api/v1/predictions/forecast` - Cost forecasting
- `GET /api/v1/predictions/usage` - Usage forecasting
- `GET /api/v1/predictions/recommendations` - Optimization recommendations
- `GET /api/v1/predictions/models` - Model performance metrics

#### Anomalies
- `GET /api/v1/anomalies/current` - Current anomalies
- `GET /api/v1/anomalies/history` - Historical anomaly data
- `GET /api/v1/anomalies/{id}` - Detailed anomaly information
- `GET /api/v1/anomalies/statistics` - Anomaly detection statistics

#### Alerts
- `GET /api/v1/alerts/active` - Active alerts
- `GET /api/v1/alerts/rules` - Alert rules configuration
- `GET /api/v1/alerts/history` - Alert history
- `POST /api/v1/alerts/{id}/acknowledge` - Acknowledge alert
- `POST /api/v1/alerts/{id}/resolve` - Resolve alert

#### Real-time WebSocket
- `WebSocket /ws/real-time` - Real-time data streaming
- `WebSocket /ws/alerts` - Real-time alert notifications

### Interactive API Documentation
Visit `http://localhost:8000/docs` for Swagger UI documentation.

## 🎯 Usage Guide

### Dashboard Navigation
1. **Dashboard**: Main overview with key metrics and real-time data
2. **Analytics**: Deep dive into cost and usage analytics
3. **Alerts**: Monitor and manage system alerts
4. **Anomalies**: Review detected anomalies and their analysis
5. **Predictions**: View forecasts and optimization recommendations
6. **Settings**: Configure dashboard preferences and system settings

### Real-time Features
- **Live Updates**: Dashboard updates automatically with real-time data
- **Connection Status**: Visual indicators show WebSocket connection status
- **Notifications**: Real-time alerts appear as they occur
- **Auto-refresh**: Configurable refresh intervals for data updates

### Configuration
- **API Key**: Set your API key in Settings > System
- **Refresh Intervals**: Configure update frequency in Settings > Dashboard
- **Alert Thresholds**: Set notification thresholds in Settings > Monitoring
- **Notification Channels**: Configure email, Slack, and push notifications

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the frontend directory:
```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
VITE_API_KEY=your_api_key_here
VITE_DEBUG=false
```

### API Configuration
The backend API can be configured through environment variables or by modifying the configuration files in `src/snowflake_analytics/config/`.

## 📊 Sample Data

The dashboard includes comprehensive mock data for demonstration:
- **Cost Data**: Monthly cost trends, warehouse breakdowns, and forecasts
- **Usage Metrics**: Query performance, user activity, and utilization
- **Anomalies**: Various types of anomalies with different severities
- **Alerts**: Active alerts with different priorities and states
- **Predictions**: ML model forecasts and optimization recommendations

## 🛠️ Development

### Backend Development
- **Add New Endpoints**: Create new endpoint modules in `src/snowflake_analytics/api/endpoints/`
- **Middleware**: Add custom middleware in `src/snowflake_analytics/api/middleware/`
- **Schemas**: Define request/response schemas in `src/snowflake_analytics/api/schemas/`

### Frontend Development
- **Add New Pages**: Create new page components in `frontend/src/pages/`
- **Add New Components**: Create reusable components in `frontend/src/components/`
- **Add New Services**: Create service modules in `frontend/src/services/`
- **Add New Types**: Define TypeScript types in `frontend/src/types/`

### Testing
- **Backend Tests**: Run `python test_dashboard_modules.py`
- **Frontend Tests**: Run `npm test` (when test suite is added)
- **Integration Tests**: Test full stack functionality

## 🚀 Deployment

### Backend Deployment
```bash
# Production server
uvicorn src.snowflake_analytics.api.main:app --host 0.0.0.0 --port 8000

# With Docker
docker build -t snowflake-analytics-api .
docker run -p 8000:8000 snowflake-analytics-api
```

### Frontend Deployment
```bash
# Build for production
npm run build

# Serve static files
npm run preview

# Deploy to CDN or hosting service
```

## 🔒 Security

### Authentication
- **API Keys**: Secure API key-based authentication
- **Rate Limiting**: Configurable rate limits per endpoint
- **CORS**: Secure cross-origin resource sharing
- **Input Validation**: Comprehensive request validation

### Best Practices
- Store API keys securely
- Use HTTPS in production
- Configure proper CORS origins
- Set appropriate rate limits
- Monitor API usage

## 📈 Performance

### Optimization Features
- **Caching**: Intelligent caching of API responses
- **Real-time Updates**: Efficient WebSocket communication
- **Lazy Loading**: Components load as needed
- **Error Boundaries**: Graceful error handling
- **Retry Logic**: Automatic retry for failed requests

### Monitoring
- **System Metrics**: CPU, memory, and disk usage
- **API Performance**: Response times and error rates
- **WebSocket Status**: Connection health and statistics
- **User Activity**: Dashboard usage analytics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the API documentation at `/docs`
- Review the configuration settings
- Run the test suite to verify functionality

## 🎉 Acknowledgments

- Built with modern web technologies
- Designed for scalability and performance
- Focused on user experience and accessibility
- Comprehensive real-time monitoring capabilities