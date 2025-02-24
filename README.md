# Traffic Accident Analysis Dashboard

## Overview
This project provides a comprehensive analysis of traffic accidents using statistical, geospatial, and temporal perspectives. It includes interactive visualizations, predictive modeling, and data-driven insights for accident prevention and road safety improvements.

## Features
- Statistical Analysis of Accident Trends
- Geospatial Hotspot Detection
- Temporal Pattern Analysis
- Predictive Modeling for Accident Risk
- Interactive Data Visualization

## Setup Instructions
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Project Structure
- `app.py`: Main Streamlit application
- `src/`: Source code modules
  - `data_processor.py`: Data cleaning and preprocessing
  - `statistical_analysis.py`: Statistical modeling functions
  - `geospatial_analysis.py`: Geospatial analysis functions
  - `temporal_analysis.py`: Time series analysis
  - `ml_models.py`: Machine learning models
- `requirements.txt`: Project dependencies
- `accident_data.csv`: Dataset

## Data Sources
The analysis uses traffic accident data containing information about:
- Accident location (latitude, longitude)
- Timestamp
- Severity
- Weather conditions
- Road conditions
- Vehicle types
- Casualties
