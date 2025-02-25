import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px

class TemporalAnalyzer:
    def __init__(self, data):
        self.data = data
        self.theme = {
            'background': '#111111',
            'text': '#FFFFFF',
            'grid': '#333333',
            'primary': '#00B4D8',
            'secondary': '#90E0EF',
            'accent': '#CAF0F8',
            'warning': '#FF9E00'
        }
        
    def _apply_theme(self, fig):
        """
        Apply consistent theme to plotly figures
        """
        fig.update_layout(
            plot_bgcolor=self.theme['background'],
            paper_bgcolor=self.theme['background'],
            font_color=self.theme['text'],
            title_font_color=self.theme['text'],
            legend_font_color=self.theme['text'],
            title_x=0.5,
            title_font_size=20,
            legend=dict(
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(0,0,0,0)'
            ),
            xaxis=dict(
                gridcolor=self.theme['grid'],
                zerolinecolor=self.theme['grid']
            ),
            yaxis=dict(
                gridcolor=self.theme['grid'],
                zerolinecolor=self.theme['grid']
            )
        )
        return fig
        
    def plot_accident_trends(self):
        """
        Plot accident trends over time with statistical insights.
        """
        daily_counts = self.data.groupby('date').size().reset_index(name='count')
        daily_counts.set_index('date', inplace=True)
        
        # Calculate moving averages
        ma_7 = daily_counts['count'].rolling(window=7).mean()
        ma_30 = daily_counts['count'].rolling(window=30).mean()
        
        # Perform time series decomposition
        decomposition = seasonal_decompose(daily_counts['count'], period=7, model='additive')
        
        fig = go.Figure()
        
        # Original data
        fig.add_trace(go.Scatter(
            x=daily_counts.index,
            y=daily_counts['count'],
            name='Daily Count',
            line=dict(color=self.theme['primary'], width=1),
            opacity=0.7
        ))
        
        # 7-day moving average
        fig.add_trace(go.Scatter(
            x=daily_counts.index,
            y=ma_7,
            name='7-day MA',
            line=dict(color=self.theme['secondary'], width=2)
        ))
        
        # 30-day moving average
        fig.add_trace(go.Scatter(
            x=daily_counts.index,
            y=ma_30,
            name='30-day MA',
            line=dict(color=self.theme['warning'], width=2)
        ))
        
        # Calculate statistics
        mean_accidents = daily_counts['count'].mean()
        std_accidents = daily_counts['count'].std()
        
        # Add mean line
        fig.add_hline(y=mean_accidents, 
                     line_dash="dash", 
                     line_color=self.theme['accent'],
                     annotation_text=f"Mean: {mean_accidents:.1f}",
                     annotation_position="bottom right")
        
        fig = self._apply_theme(fig)
        fig.update_layout(
            title='Daily Accident Trends with Moving Averages',
            xaxis_title='Date',
            yaxis_title='Number of Accidents',
            hovermode='x unified',
            width=1000,
            height=600
        )
        
        return fig, {
            'mean': mean_accidents,
            'std': std_accidents,
            'max': daily_counts['count'].max(),
            'min': daily_counts['count'].min()
        }
    
    def plot_seasonal_patterns(self):
        """
        Plot seasonal patterns with enhanced statistics.
        """
        # Monthly analysis
        monthly_avg = self.data.groupby('month').agg({
            'Accident_ID': 'count',
            'Severity': lambda x: (x == 'Severe').mean() * 100,
            'Casualties': 'mean'
        }).reset_index()
        
        month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        monthly_avg['month_name'] = monthly_avg['month'].map(lambda x: month_names[x-1])
        
        fig = go.Figure()
        
        # Accident counts
        fig.add_trace(go.Bar(
            x=monthly_avg['month_name'],
            y=monthly_avg['Accident_ID'],
            name='Accidents',
            marker_color=self.theme['primary']
        ))
        
        # Severity percentage
        fig.add_trace(go.Scatter(
            x=monthly_avg['month_name'],
            y=monthly_avg['Severity'],
            name='Severe Accidents (%)',
            yaxis='y2',
            line=dict(color=self.theme['warning'], width=3)
        ))
        
        fig = self._apply_theme(fig)
        fig.update_layout(
            title='Monthly Accident Patterns and Severity',
            xaxis_title='Month',
            yaxis_title='Number of Accidents',
            yaxis2=dict(
                title='Severe Accidents (%)',
                overlaying='y',
                side='right',
                gridcolor=self.theme['grid']
            ),
            width=1000,
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def plot_hourly_distribution(self):
        """
        Plot hourly distribution with day/night and rush hour analysis.
        """
        hourly_stats = self.data.groupby(['hour', 'is_rush_hour', 'Day_Night']).agg({
            'Accident_ID': 'count',
            'Severity': lambda x: (x == 'Severe').mean() * 100,
            'Casualties': 'mean'
        }).reset_index()
        
        fig = go.Figure()
        
        # Non-rush hour accidents
        fig.add_trace(go.Bar(
            x=hourly_stats[~hourly_stats['is_rush_hour']]['hour'],
            y=hourly_stats[~hourly_stats['is_rush_hour']]['Accident_ID'],
            name='Normal Hours',
            marker_color=self.theme['primary']
        ))
        
        # Rush hour accidents
        fig.add_trace(go.Bar(
            x=hourly_stats[hourly_stats['is_rush_hour']]['hour'],
            y=hourly_stats[hourly_stats['is_rush_hour']]['Accident_ID'],
            name='Rush Hours',
            marker_color=self.theme['warning']
        ))
        
        # Severity line
        fig.add_trace(go.Scatter(
            x=hourly_stats.groupby('hour')['Severity'].mean().index,
            y=hourly_stats.groupby('hour')['Severity'].mean().values,
            name='Severity %',
            yaxis='y2',
            line=dict(color=self.theme['accent'], width=3)
        ))
        
        fig = self._apply_theme(fig)
        fig.update_layout(
            title='Hourly Accident Distribution with Severity Analysis',
            xaxis_title='Hour of Day',
            yaxis_title='Number of Accidents',
            yaxis2=dict(
                title='Severe Accidents (%)',
                overlaying='y',
                side='right',
                gridcolor=self.theme['grid']
            ),
            width=1000,
            height=600,
            barmode='overlay',
            bargap=0.1
        )
        
        return fig
    
    def predict_future_trend(self, days=30, confidence_interval=0.95):
        """
        Predict future trends with confidence intervals.
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
        
        # Calculate confidence intervals using historical residuals
        residuals = model.resid
        std_resid = residuals.std()
        z_score = stats.norm.ppf((1 + confidence_interval) / 2)
        ci_width = z_score * std_resid
        
        lower_ci = forecast - ci_width
        upper_ci = forecast + ci_width
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=daily_counts.index,
            y=daily_counts['count'],
            name='Historical',
            line=dict(color=self.theme['primary'], width=2)
        ))
        
        # Predictions
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast.values,
            name='Forecast',
            line=dict(color=self.theme['warning'], width=3)
        ))
        
        # Confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=lower_ci,
            fill=None,
            line=dict(color=self.theme['accent'], width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=upper_ci,
            fill='tonexty',
            fillcolor=f'rgba{tuple(int(self.theme["accent"].lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}',
            line=dict(color=self.theme['accent'], width=0),
            name=f'{int(confidence_interval*100)}% CI'
        ))
        
        fig = self._apply_theme(fig)
        fig.update_layout(
            title=f'Accident Trend Forecast (Next {days} Days)',
            xaxis_title='Date',
            yaxis_title='Number of Accidents',
            width=1000,
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Calculate forecast statistics
        forecast_stats = {
            'mean_forecast': forecast.mean(),
            'min_forecast': forecast.min(),
            'max_forecast': forecast.max(),
            'trend': 'Increasing' if forecast.iloc[-1] > forecast.iloc[0] else 'Decreasing',
            'confidence_width': f'±{ci_width:.1f} accidents'
        }
        
        return fig, forecast_stats
