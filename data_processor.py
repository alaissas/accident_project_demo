import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self, data):
        self.data = data
        
    def preprocess_data(self):
        """
        Preprocess the accident data by cleaning and transforming it.
        """
        df = self.data.copy()
        
        # Convert date and time columns to datetime
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d-%m-%Y %H:%M:%S')
        df['date'] = df['datetime'].dt.date
        
        # Extract time-based features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['year'] = df['datetime'].dt.year
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Create additional features
        df = self.create_features(df)
        
        return df
    
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset.
        """
        # Fill numeric columns with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Fill categorical columns with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])
        
        return df
    
    def create_features(self, df):
        """
        Create additional features for analysis.
        """
        # Create time period categories
        df['time_period'] = pd.cut(df['hour'],
                                 bins=[0, 6, 12, 18, 24],
                                 labels=['Night', 'Morning', 'Afternoon', 'Evening'])
        
        # Create rush hour indicator
        rush_hours = [7, 8, 9, 16, 17, 18]
        df['is_rush_hour'] = df['hour'].isin(rush_hours)
        
        # Create weekend indicator
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        return df
