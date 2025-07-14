#!/bin/bash

# Activate the virtual environment
source "/Users/admin/learn_python/720_data_analysis_agent/predictive_analysis_agent/venv/bin/activate"

# Load environment variables from .env file
if [ -f .env ]; then
    set -a  # automatically export all variables
    source .env
    set +a  # disable automatic export
    echo "Environment variables loaded from .env"
fi

echo "Snowflake Analytics Agent environment activated"
echo "Run 'python main.py --help' to see available commands"
