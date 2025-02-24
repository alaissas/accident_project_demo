import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class TemporalAnalyzer:
    def __init__(self, data):
        self.data = data
        
    def plot_accident_trends(self):
        """
        Plot accident trends over time.
        """
        daily_counts = self.data.groupby('date').size().reset_index(name='count')
        daily_counts.set_index('date', inplace=True)
        
        # Perform time series decomposition
        decomposition = seasonal_decompose(daily_counts['count'], period=7, model='additive')
        
        fig = go.Figure()
        
        # Original data
        fig.add_trace(go.Scatter(
            x=daily_counts.index,
            y=daily_counts['count'],
            name='Original',
            line=dict(color='blue')
        ))
        
        # Trend
        fig.add_trace(go.Scatter(
            x=daily_counts.index,
            y=decomposition.trend,
            name='Trend',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title='Daily Accident Counts Over Time with Trend',
            xaxis_title='Date',
            yaxis_title='Number of Accidents',
            width=800,
            height=500,
            showlegend=True
        )
        
        return fig
    
    def plot_seasonal_patterns(self):
        """
        Plot seasonal patterns in accident occurrence.
        """
        # Group by month and calculate average accidents
        monthly_avg = self.data.groupby('month').size().reset_index(name='count')
        month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        monthly_avg['month_name'] = monthly_avg['month'].map(lambda x: month_names[x-1])
        
        fig = go.Figure(data=go.Bar(
            x=monthly_avg['month_name'],
            y=monthly_avg['count'],
            marker_color='darkred'
        ))
        
        fig.update_layout(
            title='Average Accidents by Month',
            xaxis_title='Month',
            yaxis_title='Number of Accidents',
            width=800,
            height=500
        )
        
        return fig
    
    def plot_hourly_distribution(self):
        """
        Plot hourly distribution of accidents.
        """
        hourly_counts = self.data.groupby('hour').size().reset_index(name='count')
        
        fig = go.Figure(data=go.Bar(
            x=hourly_counts['hour'],
            y=hourly_counts['count'],
            marker_color='darkblue'
        ))
        
        fig.update_layout(
            title='Accidents by Hour of Day',
            xaxis_title='Hour',
            yaxis_title='Number of Accidents',
            width=800,
            height=500
        )
        
        return fig
    
    def predict_future_trend(self, days=30):
        """
        Predict future accident trends using Holt-Winters method.
        """
        daily_counts = self.data.groupby('date').size().reset_index(name='count')
        daily_counts.set_index('date', inplace=True)
        
        # Fit Holt-Winters model
        model = ExponentialSmoothing(
            daily_counts['count'],
            seasonal_periods=7,
            trend='add',
            seasonal='add'
        ).fit()
        
        # Make predictions
        forecast = model.forecast(days)
        
        # Create plot
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=daily_counts.index,
            y=daily_counts['count'],
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Predictions
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast.values,
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f'Accident Trend Forecast (Next {days} Days)',
            xaxis_title='Date',
            yaxis_title='Number of Accidents',
            width=800,
            height=500,
            showlegend=True
        )
        
        return fig
