# 🚀 Snowflake Predictive Analytics Agent

A self-contained, file-based predictive analytics system for Snowflake data with zero external dependencies.

## ✨ Features

- **File-based architecture** (SQLite + JSON + Parquet) - no external services required
- **One-command setup** - get running in under 2 minutes
- **Predictive analytics** - ML models for cost optimization and usage forecasting
- **Real-time monitoring** - automated alerts and dashboard
- **Ultra-portable** - runs anywhere Python 3.9+ is available

## 🚀 Quick Start (2-Minute Setup)

```bash
# 1. Clone the repository
git clone https://github.com/lumensparkxy/predictive_analysis_agent.git
cd predictive_analysis_agent

# 2. Run the automated setup
python scripts/setup.py

# 3. Configure your Snowflake connection
cp .env.example .env
# Edit .env with your Snowflake credentials

# 4. Start the application
python main.py
```

## 📋 Prerequisites

- **Python 3.9+** (tested on 3.9, 3.10, 3.11, 3.12)
- **10MB disk space** for installation
- **Internet connection** for initial package installation only

## 🏗️ Architecture

### File-Based Storage System
```
data/
├── raw/           # Raw Snowflake data (Parquet)
├── processed/     # Cleaned and transformed data
├── models/        # Trained ML models (joblib)
└── exports/       # Dashboard and report exports

storage.db         # SQLite metadata database
cache/            # File-based cache (diskcache)
logs/             # Application logs
```

### Core Components
- **Data Collection**: Automated Snowflake data ingestion
- **Processing Pipeline**: ETL with pandas and numpy
- **ML Models**: scikit-learn for predictive analytics
- **Dashboard**: FastAPI + web interface
- **Alerts**: Configurable notification system

## ⚙️ Configuration

### Environment Variables (.env)
```bash
# Snowflake Connection
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema

# Application Settings
LOG_LEVEL=INFO
CACHE_TTL=3600
DATA_RETENTION_DAYS=30
```

### Configuration Files
- `config/settings.json` - Application settings
- `config/snowflake.json` - Snowflake connection parameters

## 🧪 Development Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install development dependencies
pip install -r requirements-dev.txt

# 3. Run tests
pytest

# 4. Format code
black .
isort .

# 5. Run linting
flake8 .
```

## 📊 Usage

### Start the Dashboard
```bash
python main.py
# Open http://localhost:8000 in your browser
```

### CLI Commands
```bash
# Collect data manually
python -m snowflake_analytics.data_collection.collector

# Train models
python -m snowflake_analytics.models.trainer

# Generate reports
python -m snowflake_analytics.reports.generator
```

## 📁 Project Structure

```
predictive_analysis_agent/
├── README.md                    # This file
├── main.py                      # Application entry point
├── requirements.txt             # Production dependencies
├── requirements-dev.txt         # Development dependencies
├── .env.example                 # Environment template
├── .gitignore                   # Git ignore rules
├── config/                      # Configuration files
│   ├── settings.json
│   └── snowflake.json
├── src/snowflake_analytics/     # Main application package
│   ├── __init__.py
│   ├── storage/                 # Data storage layer
│   ├── data_collection/         # Snowflake data ingestion
│   ├── data_processing/         # ETL pipelines
│   ├── models/                  # ML models
│   └── utils/                   # Common utilities
├── data/                        # Data storage
├── cache/                       # File cache
├── logs/                        # Application logs
├── tests/                       # Test suite
├── scripts/                     # Utility scripts
└── notebooks/                   # Jupyter notebooks
```

## 🔧 Troubleshooting

### Common Issues

**Setup fails with permission error:**
```bash
# Use virtual environment
python -m venv venv
source venv/bin/activate
python scripts/setup.py
```

**Snowflake connection fails:**
- Verify credentials in `.env` file
- Check network connectivity
- Ensure Snowflake account is accessible

**Performance issues:**
- Check disk space (need 100MB+ for data)
- Monitor SQLite database size
- Clear cache: `rm -rf cache/`

### Getting Help

1. Check logs in `logs/` directory
2. Run diagnostics: `python scripts/diagnose.py`
3. Create GitHub issue with error details

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and add tests
4. Run test suite: `pytest`
5. Format code: `black . && isort .`
6. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🔗 Links

- [GitHub Repository](https://github.com/lumensparkxy/predictive_analysis_agent)
- [Documentation](docs/)
- [Issue Tracker](https://github.com/lumensparkxy/predictive_analysis_agent/issues)

---

**Built with ❤️ for Snowflake analytics teams**
