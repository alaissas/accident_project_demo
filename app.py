import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import folium_static
from src.data_processor import DataProcessor
from src.statistical_analysis import StatisticalAnalyzer
from src.geospatial_analysis import GeospatialAnalyzer
from src.temporal_analysis import TemporalAnalyzer
from src.ml_models import MLModels

st.set_page_config(page_title="Traffic Accident Analysis", layout="wide")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None

def load_data():
    if st.session_state.data is None:
        data = pd.read_csv('accident_data.csv')
        dp = DataProcessor(data)
        st.session_state.data = dp.preprocess_data()
    return st.session_state.data

def main():
    st.title("Traffic Accident Analysis Dashboard")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Statistical Analysis", "Geospatial Analysis", "Temporal Analysis", "Predictive Models"]
    )
    
    # Load data
    data = load_data()
    
    if page == "Overview":
        st.header("Overview of Traffic Accidents")
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Accidents", len(data))
        with col2:
            severity_map = {'Low': 1, 'Moderate': 2, 'Severe': 3}
            data['severity_value'] = data['Severity'].map(severity_map)
            st.metric("Average Severity", f"{data['Severity'].mode().iloc[0]} ({round(data['severity_value'].mean(), 2)})")
        with col3:
            st.metric("Time Period", f"{data['date'].min()} to {data['date'].max()}")
            
        # Basic visualizations
        st.subheader("Accident Severity Distribution")
        fig = px.histogram(data, x='Severity', title='Distribution of Accident Severity')
        st.plotly_chart(fig)
        
        # Weather conditions
        st.subheader("Weather Conditions")
        fig = px.pie(data, names='Weather', title='Weather Conditions during Accidents')
        st.plotly_chart(fig)
        
        # Road types
        st.subheader("Road Types")
        road_type_counts = data['Road_Type'].value_counts().reset_index()
        road_type_counts.columns = ['Road_Type', 'Count']
        fig = px.bar(road_type_counts, 
                    x='Road_Type', 
                    y='Count',
                    title='Accidents by Road Type')
        st.plotly_chart(fig)
        
    elif page == "Statistical Analysis":
        statistical = StatisticalAnalyzer(data)
        st.header("Statistical Analysis")
        
        # Time of day analysis
        st.subheader("Accidents by Time of Day")
        time_fig = statistical.plot_time_of_day_distribution()
        st.plotly_chart(time_fig)
        
        # Correlation analysis for numeric columns
        st.subheader("Correlation Analysis")
        numeric_cols = ['Vehicles_Involved', 'Casualties', 'Traffic_Volume', 'Speed_Limits']
        corr_fig = px.imshow(data[numeric_cols].corr(),
                           labels=dict(color="Correlation"),
                           title="Correlation between Numeric Features")
        st.plotly_chart(corr_fig)
        
    elif page == "Geospatial Analysis":
        geospatial = GeospatialAnalyzer(data)
        st.header("Geospatial Analysis")
        
        # Create map
        st.subheader("Accident Hotspots")
        m = folium.Map(location=[data['Latitude'].mean(), data['Longitude'].mean()],
                      zoom_start=12)
        
        # Add points to map
        for idx, row in data.iterrows():
            color = 'red' if row['Severity'] == 'Severe' else 'orange' if row['Severity'] == 'Moderate' else 'green'
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                color=color,
                fill=True,
                popup=f"Severity: {row['Severity']}<br>Time: {row['Time']}<br>Casualties: {row['Casualties']}"
            ).add_to(m)
            
        folium_static(m)
        
    elif page == "Temporal Analysis":
        temporal = TemporalAnalyzer(data)
        st.header("Temporal Analysis")
        
        # Time series plot
        st.subheader("Accident Trends Over Time")
        trend_fig = temporal.plot_accident_trends()
        st.plotly_chart(trend_fig)
        
        # Hourly patterns
        st.subheader("Hourly Patterns")
        hourly_fig = temporal.plot_hourly_patterns()
        st.plotly_chart(hourly_fig)
        
    elif page == "Predictive Models":
        ml = MLModels(data)
        st.header("Predictive Models")
        
        # Train models with progress indicator
        with st.spinner('Training models... Please wait.'):
            ml.train_severity_model()
            ml.train_frequency_model()
        
        # Model selection
        model_type = st.selectbox(
            "Select Model",
            ["Severity Prediction", "Accident Frequency Prediction"]
        )
        
        if model_type == "Severity Prediction":
            st.subheader("Accident Severity Prediction")
            
            # Input features for prediction
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Time and Date**")
                hour = st.slider("Hour of Day", 0, 23, 12)
                is_weekend = st.checkbox("Is Weekend?")
                traffic_volume = st.slider("Traffic Volume", 0, 1000, 100)

            with col2:
                st.markdown("**Road Conditions**")
                weather = st.selectbox("Weather Condition", data['Weather'].unique())
                road_type = st.selectbox("Road Type", data['Road_Type'].unique())
                speed_limit = st.slider("Speed Limit", 20, 100, 50)

            with col3:
                st.markdown("**Accident Details**")
                vehicles = st.slider("Number of Vehicles Involved", 1, 10, 2)
                visibility = st.slider("Visibility (meters)", 0, 1000, 500)
                distance = st.slider("Distance from Junction (meters)", 0, 500, 100)
                
            if st.button("Predict Severity", use_container_width=True):
                # Get current date/time for time-based features
                current_date = pd.Timestamp.now()
                if is_weekend:
                    day_of_week = 5 if current_date.dayofweek < 5 else current_date.dayofweek
                else:
                    day_of_week = 1 if current_date.dayofweek >= 5 else current_date.dayofweek
                
                features = {
                    'hour': hour,
                    'day_of_week': day_of_week,
                    'month': current_date.month,
                    'Weather': weather,
                    'Road_Type': road_type,
                    'Vehicles_Involved': vehicles,
                    'Speed_Limits': speed_limit,
                    'Traffic_Volume': traffic_volume,
                    'Visibility': visibility,
                    'Distance_Junction': distance
                }
                
                severity, probabilities = ml.predict_severity(features)
                
                # Display results in an organized way
                st.markdown("---")
                st.markdown("### Prediction Results")
                
                # Display the main prediction with custom styling
                severity_colors = {
                    'Low': 'green',
                    'Moderate': 'orange',
                    'Severe': 'red'
                }
                st.markdown(
                    f"<h2 style='color: {severity_colors[severity]}; text-align: center;'>"
                    f"Predicted Severity: {severity}</h2>",
                    unsafe_allow_html=True
                )
                
                # Create a bar chart for probabilities
                prob_df = pd.DataFrame({
                    'Severity': list(probabilities.keys()),
                    'Probability': list(probabilities.values())
                })
                
                fig = px.bar(prob_df, 
                            x='Severity', 
                            y='Probability',
                            color='Severity',
                            color_discrete_map={
                                'Low': 'green',
                                'Moderate': 'orange',
                                'Severe': 'red'
                            },
                            title='Probability Distribution')
                fig.update_layout(
                    yaxis_tickformat = '.0%',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display feature importance if available
                if hasattr(ml, 'severity_model') and hasattr(ml.severity_model, 'feature_importances_'):
                    st.markdown("### Feature Importance")
                    feature_importance = pd.DataFrame({
                        'Feature': ml.prepare_features(features).columns,
                        'Importance': ml.severity_model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig_importance = px.bar(feature_importance,
                                          x='Feature',
                                          y='Importance',
                                          title='Feature Importance in Prediction')
                    st.plotly_chart(fig_importance, use_container_width=True)
        
        else:
            st.subheader("Accident Frequency Prediction")
            st.write("Select date range for prediction:")
            start_date = st.date_input("Start Date")
            days = st.slider("Number of days to predict", 1, 30, 7)
            
            if st.button("Predict Frequency"):
                features = {
                    'start_date': start_date,
                    'days': days
                }
                frequency = ml.predict_frequency(features)
                st.write(f"Predicted number of accidents: {int(frequency)}")

if __name__ == "__main__":
    main()
