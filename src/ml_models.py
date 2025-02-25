import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import plotly.express as px
from datetime import datetime, timedelta

class MLModels:
    def __init__(self, data):
        self.data = data
        self.severity_model = None
        self.frequency_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def prepare_features(self, feature_data):
        """
        Prepare features for model training or prediction.
        """
        required_features = [
            'hour', 'day_of_week', 'month', 'Weather', 'Road_Type',
            'Vehicles_Involved', 'Speed_Limits', 'Traffic_Volume',
            'Visibility', 'Distance_Junction'
        ]
        
        # Handle different input types
        if isinstance(feature_data, pd.DataFrame):
            feature_df = feature_data.copy()
        elif isinstance(feature_data, dict):
            feature_df = pd.DataFrame([feature_data])
        else:
            raise ValueError("feature_data must be either a DataFrame or a dictionary")
        
        # Add missing features from the data if training
        if len(feature_df) > 1:  # Training data
            if 'Visibility' not in feature_df.columns:
                feature_df['Visibility'] = 1000  # Default good visibility
            if 'Distance_Junction' not in feature_df.columns:
                feature_df['Distance_Junction'] = feature_df['Speed_Limits'].apply(lambda x: x * 2)  # Rough estimate
        
        # Encode categorical variables
        categorical_cols = ['Weather', 'Road_Type']
        for col in categorical_cols:
            if col in feature_df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    self.label_encoders[col].fit(feature_df[col].astype(str))
                feature_df[col] = self.label_encoders[col].transform(feature_df[col].astype(str))
        
        # Ensure all required columns are present and in the correct order
        for col in required_features:
            if col not in feature_df.columns:
                feature_df[col] = 0  # Default value for missing features
        
        # Feature engineering
        feature_df['is_rush_hour'] = feature_df['hour'].apply(
            lambda x: 1 if (x >= 7 and x <= 9) or (x >= 16 and x <= 18) else 0
        )
        feature_df['is_night'] = feature_df['hour'].apply(
            lambda x: 1 if x >= 20 or x <= 5 else 0
        )
        
        # Add the engineered features to required features
        required_features.extend(['is_rush_hour', 'is_night'])
        
        return feature_df[required_features]
    
    def train_severity_model(self):
        """
        Train a model to predict accident severity.
        """
        # Extract required columns from data
        X = self.prepare_features(self.data)
        
        # Map severity labels
        severity_map = {'Low': 0, 'Moderate': 1, 'Severe': 2}
        y = self.data['Severity'].map(severity_map)
        
        # Drop any rows with NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model with more trees and better parameters
        self.severity_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.severity_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.severity_model.predict(X_test_scaled)
        self.severity_report = classification_report(y_test, y_pred)
        
        return self.severity_report
    
    def train_frequency_model(self):
        """
        Train a model to predict accident frequency.
        """
        # Aggregate data by date
        daily_counts = self.data.groupby('date').size().reset_index(name='count')
        daily_counts['day_of_week'] = pd.to_datetime(daily_counts['date']).dt.dayofweek
        daily_counts['month'] = pd.to_datetime(daily_counts['date']).dt.month
        
        # Add weather and road type features
        daily_weather = self.data.groupby('date')['Weather'].agg(lambda x: x.mode()[0]).reset_index()
        daily_road = self.data.groupby('date')['Road_Type'].agg(lambda x: x.mode()[0]).reset_index()
        
        daily_data = pd.merge(daily_counts, daily_weather, on='date')
        daily_data = pd.merge(daily_data, daily_road, on='date')
        
        # Encode categorical variables
        for col in ['Weather', 'Road_Type']:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                daily_data[col] = self.label_encoders[col].fit_transform(daily_data[col])
            else:
                daily_data[col] = self.label_encoders[col].transform(daily_data[col])
        
        # Prepare features
        X = daily_data[['day_of_week', 'month', 'Weather', 'Road_Type']]
        y = daily_data['count']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.frequency_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.frequency_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.frequency_model.predict(X_test)
        self.frequency_mse = mean_squared_error(y_test, y_pred)
        self.frequency_r2 = r2_score(y_test, y_pred)
        
        return {'mse': self.frequency_mse, 'r2': self.frequency_r2}
    
    def predict_severity(self, features):
        """
        Make severity predictions for new data.
        """
        if self.severity_model is None:
            raise ValueError("Model not trained yet. Call train_severity_model() first.")
        
        # Prepare features
        feature_df = self.prepare_features(features)
        
        # Scale features
        features_scaled = self.scaler.transform(feature_df)
        
        # Make prediction
        prediction = self.severity_model.predict(features_scaled)
        probabilities = self.severity_model.predict_proba(features_scaled)
        
        # Convert numeric prediction back to category
        severity_categories = ['Low', 'Moderate', 'Severe']
        predicted_severity = severity_categories[prediction[0]]
        
        return predicted_severity, {cat: prob for cat, prob in zip(severity_categories, probabilities[0])}
    
    def predict_frequency(self, features):
        """
        Make frequency predictions for new data.
        """
        if self.frequency_model is None:
            raise ValueError("Model not trained yet. Call train_frequency_model() first.")
        
        # Extract date components
        start_date = features['start_date']
        predictions = []
        
        # Make predictions for each day in the range
        for day in range(features['days']):
            current_date = start_date + timedelta(days=day)
            features_dict = {
                'day_of_week': current_date.weekday(),
                'month': current_date.month,
                'Weather': features.get('Weather', self.data['Weather'].mode()[0]),
                'Road_Type': features.get('Road_Type', self.data['Road_Type'].mode()[0])
            }
            
            # Encode categorical variables
            for col in ['Weather', 'Road_Type']:
                features_dict[col] = self.label_encoders[col].transform([features_dict[col]])[0]
            
            # Create feature array
            X = pd.DataFrame([features_dict])
            
            # Make prediction
            pred = self.frequency_model.predict(X)[0]
            predictions.append(pred)
        
        # Return average daily prediction
        return sum(predictions) / len(predictions)

