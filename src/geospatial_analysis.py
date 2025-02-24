import folium
from folium import plugins
import plotly.express as px
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd

class GeospatialAnalyzer:
    def __init__(self, data):
        self.data = data
        
    def create_heatmap(self):
        """
        Create a heatmap of accident locations.
        """
        # Create base map centered on mean coordinates
        center_lat = self.data['latitude'].mean()
        center_lon = self.data['longitude'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Add heatmap layer
        heat_data = [[row['latitude'], row['longitude']] for index, row in self.data.iterrows()]
        plugins.HeatMap(heat_data).add_to(m)
        
        return m
    
    def identify_hotspots(self, eps=0.1, min_samples=5):
        """
        Identify accident hotspots using DBSCAN clustering.
        """
        # Prepare coordinates for clustering
        coords = self.data[['latitude', 'longitude']].values
        
        # Perform DBSCAN clustering
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        
        # Add cluster labels to data
        self.data['cluster'] = db.labels_
        
        return self.data
    
    def plot_clusters(self):
        """
        Create a scatter plot of accident clusters.
        """
        # Identify hotspots first
        data = self.identify_hotspots()
        
        # Create scatter plot
        fig = px.scatter_mapbox(
            data,
            lat='latitude',
            lon='longitude',
            color='cluster',
            mapbox_style='carto-positron',
            zoom=10,
            title='Accident Clusters'
        )
        
        return fig
    
    def calculate_cluster_statistics(self):
        """
        Calculate statistics for each cluster.
        """
        if 'cluster' not in self.data.columns:
            self.identify_hotspots()
            
        cluster_stats = self.data.groupby('cluster').agg({
            'severity': ['mean', 'count'],
            'is_rush_hour': 'mean',
            'is_weekend': 'mean'
        }).round(2)
        
        return cluster_stats
