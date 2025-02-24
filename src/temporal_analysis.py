import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np

class TemporalAnalyzer:
    def __init__(self, data):
        self.data = data
        
    def plot_accident_trends(self):
        """
        Plot accident trends over time.
        """
        daily_counts = self.data.groupby('date').size().reset_index(name='count')
        
        fig = px.line(
            daily_counts,
            x='date',
            y='count',
            title='Daily Accident Counts Over Time'
        )
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Number of Accidents',
            width=800,
            height=500
        )
        
        return fig
    
    def plot_seasonal_patterns(self):
        """
        Plot seasonal patterns in accident occurrence.
        """
        # Group by month and calculate average accidents
        monthly_avg = self.data.groupby('month').size().reset_index(name='count')
        monthly_avg['month'] = pd.to_datetime(monthly_avg['month'], format='%m').dt.strftime('%B')
        
        fig = go.Figure(data=go.Bar(
            x=monthly_avg['month'],
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
    
    def forecast_accidents(self, periods=30):
        """
        Forecast future accident counts using Prophet.
        """
        # Prepare data for Prophet
        daily_counts = self.data.groupby('date').size().reset_index(name='y')
        daily_counts = daily_counts.rename(columns={'date': 'ds'})
        
        # Create and fit Prophet model
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        model.fit(daily_counts)
        
        # Make future predictions
        future_dates = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future_dates)
        
        return forecast
    
    def plot_hourly_patterns(self):
        """
        Plot hourly patterns in accident occurrence.
        """
        hourly_avg = self.data.groupby(['hour', 'is_weekend']).size().reset_index(name='count')
        
        fig = px.line(
            hourly_avg,
            x='hour',
            y='count',
            color='is_weekend',
            title='Hourly Accident Patterns (Weekday vs Weekend)'
        )
        
        fig.update_layout(
            xaxis_title='Hour of Day',
            yaxis_title='Number of Accidents',
            width=800,
            height=500
        )
        
        return fig
