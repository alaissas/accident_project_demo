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
from src.theme_utils import get_streamlit_theme, get_color_sequence, THEME, apply_theme_to_plotly
import plotly.io as pio
import io
import base64

st.set_page_config(page_title="Traffic Accident Analysis", layout="wide")

# Apply theme
st.markdown(get_streamlit_theme(), unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None

def load_data():
    if st.session_state.data is None:
        data = pd.read_csv('accident_data.csv')
        dp = DataProcessor(data)
        st.session_state.data = dp.preprocess_data()
    return st.session_state.data

def download_plot(fig, filename):
    """Generate download link for plot."""
    buffer = io.BytesIO()
    fig.write_image(buffer, format='png', engine='kaleido')
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}.png">Download Plot as PNG</a>'
    return href

def main():
    st.markdown("""
    # Analyzing Traffic Accidents with Data Science
    ## A Statistical, Geospatial, and Temporal Perspective
    """)
    st.markdown("### Project By: Alaissa Shaikh & Zahra Merchant")
    
    # Sidebar
    st.sidebar.markdown("""
    <div class='nav-header'>
        <h1>📊 Navigation Menu</h1>
        <h3>Explore Traffic Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("<div class='nav-section-header'>Select Analysis Type</div>", unsafe_allow_html=True)
    
    # Simple page names for radio button
    page_names = [
        "📈 Dashboard Overview",
        "📊 Statistical Analysis",
        "🗺️ Geospatial Analysis",
        "⏰ Temporal Analysis",
        "🤖 Predictive Models"
    ]
    
    # Select page using radio button
    page = st.sidebar.radio("", page_names, label_visibility="collapsed")
    
    # Show description after selection
    if page:
        descriptions = {
            "📈 Dashboard Overview": {
                "desc": "Summary statistics and key insights",
                "detail": "Comprehensive overview of accident patterns and key metrics"
            },
            "📊 Statistical Analysis": {
                "desc": "Detailed statistical tests and correlations",
                "detail": "In-depth analysis of accident factors and relationships"
            },
            "🗺️ Geospatial Analysis": {
                "desc": "Geographic patterns and hotspots",
                "detail": "Location-based insights and risk area identification"
            },
            "⏰ Temporal Analysis": {
                "desc": "Time-based trends and forecasting",
                "detail": "Temporal patterns and future predictions"
            },
            "🤖 Predictive Models": {
                "desc": "Machine learning predictions",
                "detail": "AI-powered accident risk assessment"
            }
        }
        
        st.sidebar.markdown(f"""
        <div class='nav-item'>
            <div class='nav-item-title'>{page}</div>
            <div class='nav-item-desc'>{descriptions[page]['desc']}</div>
            <div class='nav-item-detail'>{descriptions[page]['detail']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='nav-section-header'>📌 Quick Guide</div>", unsafe_allow_html=True)
    
    # First guide item
    st.sidebar.markdown("""
    <div class='guide-item'>
        <div class='guide-title'>📈 Dashboard</div>
        <div class='guide-desc'>Interactive metrics & trend analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Second guide item
    st.sidebar.markdown("""
    <div class='guide-item'>
        <div class='guide-title'>📊 Statistical</div>
        <div class='guide-desc'>Advanced data pattern analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Third guide item
    st.sidebar.markdown("""
    <div class='guide-item'>
        <div class='guide-title'>🗺️ Geospatial</div>
        <div class='guide-desc'>Location-based risk assessment</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Fourth guide item
    st.sidebar.markdown("""
    <div class='guide-item'>
        <div class='guide-title'>⏰ Temporal</div>
        <div class='guide-desc'>Time-based pattern detection</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Fifth guide item
    st.sidebar.markdown("""
    <div class='guide-item'>
        <div class='guide-title'>🤖 Predictive</div>
        <div class='guide-desc'>ML-powered risk prediction</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Remove emoji for page handling
    page_clean = page.split(" ", 1)[1]
    
    # Load data
    data = load_data()
    
    if page_clean == "Dashboard Overview":
        st.header("Overview of Traffic Accidents")
        
        # Filters
        with st.expander("Filters", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_severity = st.multiselect("Severity", data['Severity'].unique(), default=data['Severity'].unique())
            with col2:
                selected_weather = st.multiselect("Weather", data['Weather'].unique(), default=data['Weather'].unique())
            with col3:
                selected_road_type = st.multiselect("Road Type", data['Road_Type'].unique(), default=data['Road_Type'].unique())
        
        # Filter data
        filtered_data = data[
            (data['Severity'].isin(selected_severity)) &
            (data['Weather'].isin(selected_weather)) &
            (data['Road_Type'].isin(selected_road_type))
        ]
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Accidents", len(filtered_data))
        with col2:
            avg_severity = filtered_data['severity_value'].mean()
            st.metric("Average Severity", f"{avg_severity:.2f}")
        with col3:
            total_casualties = filtered_data['Casualties'].sum()
            st.metric("Total Casualties", total_casualties)
        with col4:
            severe_pct = (filtered_data['Severity'] == 'Severe').mean() * 100
            st.metric("Severe Accidents %", f"{severe_pct:.1f}%")
        
        # Overview charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Severity distribution
            severity_counts = filtered_data['Severity'].value_counts()
            fig = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                title="Accident Severity Distribution",
                color_discrete_sequence=get_color_sequence()
            )
            fig = apply_theme_to_plotly(fig)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(download_plot(fig, "severity_distribution"), unsafe_allow_html=True)
        
        with col2:
            # Weather distribution
            weather_counts = filtered_data['Weather'].value_counts()
            fig = px.bar(
                x=weather_counts.index,
                y=weather_counts.values,
                title="Weather Conditions",
                color_discrete_sequence=[THEME['primary']]
            )
            fig = apply_theme_to_plotly(fig)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(download_plot(fig, "weather_distribution"), unsafe_allow_html=True)
        
        # Time patterns
        st.subheader("Time Patterns")
        col1, col2 = st.columns(2)
        
        with col1:
            # Hour of day
            hourly_counts = filtered_data.groupby('hour').size()
            fig = px.line(
                x=hourly_counts.index,
                y=hourly_counts.values,
                title="Accidents by Hour of Day",
                color_discrete_sequence=[THEME['primary']]
            )
            fig = apply_theme_to_plotly(fig)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(download_plot(fig, "hourly_pattern"), unsafe_allow_html=True)
        
        with col2:
            # Day of week
            day_counts = filtered_data.groupby('day_of_week').size()
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            fig = px.bar(
                x=[days[i] for i in day_counts.index],
                y=day_counts.values,
                title="Accidents by Day of Week",
                color_discrete_sequence=[THEME['primary']]
            )
            fig = apply_theme_to_plotly(fig)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(download_plot(fig, "daily_pattern"), unsafe_allow_html=True)
    
    elif page_clean == "Statistical Analysis":
        statistical = StatisticalAnalyzer(data)
        st.header("Statistical Analysis")
        
        # Time of day analysis
        st.subheader("Accidents by Time of Day")
        time_fig = statistical.plot_time_of_day_distribution()
        time_fig = apply_theme_to_plotly(time_fig)
        st.plotly_chart(time_fig, use_container_width=True)
        st.markdown(download_plot(time_fig, "time_of_day_distribution"), unsafe_allow_html=True)
        
        # Correlation analysis for numeric columns
        st.subheader("Correlation Analysis")
        numeric_cols = ['Vehicles_Involved', 'Casualties', 'Traffic_Volume', 'Speed_Limits']
        corr_fig = px.imshow(data[numeric_cols].corr(),
                           labels=dict(color="Correlation"),
                           title="Correlation between Numeric Features")
        corr_fig = apply_theme_to_plotly(corr_fig)
        st.plotly_chart(corr_fig, use_container_width=True)
        st.markdown(download_plot(corr_fig, "correlation_analysis"), unsafe_allow_html=True)
        
    elif page_clean == "Geospatial Analysis":
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
        
    elif page_clean == "Temporal Analysis":
        temporal = TemporalAnalyzer(data)
        st.header("Temporal Analysis")
        
        # Set dark theme
        st.markdown("""
        <style>
        .stApp {
            background-color: #111111;
            color: #FFFFFF;
        }
        .stMarkdown {
            color: #FFFFFF;
        }
        .stSelectbox > div > div {
            background-color: #333333;
            color: #FFFFFF;
        }
        .stSlider > div > div {
            background-color: #333333;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Time series analysis
        st.subheader("Accident Trends Analysis")
        trend_fig, trend_stats = temporal.plot_accident_trends()
        trend_fig = apply_theme_to_plotly(trend_fig)
        st.plotly_chart(trend_fig, use_container_width=True)
        st.markdown(download_plot(trend_fig, "accident_trends"), unsafe_allow_html=True)
        
        # Display trend statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Daily Accidents", f"{trend_stats['mean']:.1f}")
        with col2:
            st.metric("Standard Deviation", f"{trend_stats['std']:.1f}")
        with col3:
            st.metric("Maximum Daily", f"{trend_stats['max']:.0f}")
        with col4:
            st.metric("Minimum Daily", f"{trend_stats['min']:.0f}")
        
        # Seasonal patterns
        st.subheader("Seasonal and Monthly Analysis")
        seasonal_fig = temporal.plot_seasonal_patterns()
        seasonal_fig = apply_theme_to_plotly(seasonal_fig)
        st.plotly_chart(seasonal_fig, use_container_width=True)
        st.markdown(download_plot(seasonal_fig, "seasonal_patterns"), unsafe_allow_html=True)
        
        # Hourly distribution
        st.subheader("Time-of-Day Analysis")
        
        # Add time period selector
        time_period = st.selectbox(
            "Select Time Period",
            ["All Day", "Morning (6-12)", "Afternoon (12-18)", "Evening (18-24)", "Night (0-6)"]
        )
        
        hourly_fig = temporal.plot_hourly_distribution()
        hourly_fig = apply_theme_to_plotly(hourly_fig)
        st.plotly_chart(hourly_fig, use_container_width=True)
        st.markdown(download_plot(hourly_fig, "hourly_distribution"), unsafe_allow_html=True)
        
        # Future predictions
        st.subheader("Accident Trend Forecast")
        
        # Add forecast controls
        col1, col2 = st.columns(2)
        with col1:
            days = st.slider("Forecast Period (Days)", 7, 90, 30)
        with col2:
            confidence = st.slider("Confidence Interval (%)", 80, 99, 95)
            
        forecast_fig, forecast_stats = temporal.predict_future_trend(days=days, confidence_interval=confidence/100)
        forecast_fig = apply_theme_to_plotly(forecast_fig)
        st.plotly_chart(forecast_fig, use_container_width=True)
        st.markdown(download_plot(forecast_fig, "forecast_trend"), unsafe_allow_html=True)
        
        # Display forecast statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Forecast", f"{forecast_stats['mean_forecast']:.1f}")
        with col2:
            st.metric("Minimum Forecast", f"{forecast_stats['min_forecast']:.1f}")
        with col3:
            st.metric("Maximum Forecast", f"{forecast_stats['max_forecast']:.1f}")
        with col4:
            st.metric("Trend", forecast_stats['trend'])
        
    elif page_clean == "Predictive Models":
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
                fig = apply_theme_to_plotly(fig)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(download_plot(fig, "probability_distribution"), unsafe_allow_html=True)
                
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
                    fig_importance = apply_theme_to_plotly(fig_importance)
                    st.plotly_chart(fig_importance, use_container_width=True)
                    st.markdown(download_plot(fig_importance, "feature_importance"), unsafe_allow_html=True)
        
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
