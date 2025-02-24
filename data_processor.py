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
        
        # Convert boolean columns
        boolean_columns = ['Alcohol_Involved', 'Hit_and_Run', 'Event_Occurrence', 'Traffic_Camera_Footage']
        for col in boolean_columns:
            df[col] = df[col].map({'TRUE': True, 'FALSE': False})
        
        # Convert numeric columns
        numeric_columns = ['Vehicles_Involved', 'Casualties', 'Speed_Limits', 'Traffic_Volume', 
                         'Emergency_Response_Time', 'Distance_to_Hospital', 'Vehicle_Age']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
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
        for col in numeric_columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical columns with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].isna().any():
                mode_value = df[col].mode()
                if not mode_value.empty:
                    df[col] = df[col].fillna(mode_value.iloc[0])
                else:
                    df[col] = df[col].fillna('Unknown')
        
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
        df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9)) | \
                            ((df['hour'] >= 16) & (df['hour'] <= 18))
        
        # Create weekend indicator
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Create severity numeric mapping
        severity_map = {'Low': 1, 'Moderate': 2, 'Severe': 3}
        df['severity_value'] = df['Severity'].map(severity_map)
        
        return df
