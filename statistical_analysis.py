import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats

class StatisticalAnalyzer:
    def __init__(self, data):
        self.data = data
        
    def plot_correlation_matrix(self):
        """
        Create a correlation matrix heatmap for numeric features.
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        corr_matrix = self.data[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title='Feature Correlation Matrix',
            width=800,
            height=800
        )
        
        return fig
    
    def plot_time_of_day_distribution(self):
        """
        Create a distribution plot of accidents by time of day.
        """
        hourly_counts = self.data['hour'].value_counts().sort_index()
        
        fig = go.Figure(data=go.Bar(
            x=hourly_counts.index,
            y=hourly_counts.values,
            marker_color='darkblue'
        ))
        
        fig.update_layout(
            title='Accident Distribution by Hour of Day',
            xaxis_title='Hour of Day',
            yaxis_title='Number of Accidents',
            width=800,
            height=500
        )
        
        return fig
    
    def calculate_summary_statistics(self):
        """
        Calculate summary statistics for numeric features.
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        summary = self.data[numeric_cols].describe()
        
        return summary
    
    def perform_chi_square_test(self, feature1, feature2):
        """
        Perform chi-square test of independence between two categorical features.
        """
        contingency_table = pd.crosstab(self.data[feature1], self.data[feature2])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        return {
            'chi2': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof
        }
