import streamlit as st
from routes.algos.time_series_test import time_series_forecasting
from routes.algos.app2 import kmc

def show_modelling():
    # st.title("Data Modeling and Machine Learning")
    
    activities = ["About", "Linear Regression", "Multiple Regression", "Logistic Regression", 
                 "Time Series Forecasting", "K-Means Clustering", "Decision Trees", "Random Forest"]
    choice = st.sidebar.selectbox("Modelling", activities)

    if choice == 'About':
        display_modelling_about()
    elif choice == 'Time Series Forecasting':
        time_series_forecasting()
    elif choice == 'K-Means Clustering':
        kmc()
    elif choice in ['Linear Regression', 'Multiple Regression', 'Logistic Regression', 'Decision Trees', 'Random Forest']:
        st.info(f"The {choice} module is coming soon. Please check back later!")

def display_modelling_about():
    """Display information about the modeling section"""
    st.subheader("Modeling and Machine Learning")
    
    st.info("""
    ## Welcome to the Modeling Section
    
    This section provides various machine learning and statistical modeling tools to analyze your data,
    discover patterns, and make predictions.
    
    ### How to use this section:
    1. Select a model type from the sidebar
    2. Upload your data when prompted
    3. Configure model parameters
    4. View results and visualizations
    """)
    
    st.write("## Available Models")
    
    # Linear Regression
    st.write("### Linear Regression")
    st.info("""
    **Linear Regression** analyzes the relationship between a dependent variable and one independent variable.
    
    **Ideal for**:
    - Finding the line of best fit through your data points
    - Making predictions for continuous outcomes
    - Establishing if there's a relationship between two variables
    
    **Input**: CSV file with numeric data columns
    """)
    
    # Multiple Regression
    st.write("### Multiple Regression")
    st.info("""
    **Multiple Regression** extends linear regression to multiple independent variables.
    
    **Ideal for**:
    - Analyzing complex relationships with multiple factors
    - Understanding which factors most strongly influence outcomes
    - Making predictions based on multiple inputs
    
    **Input**: CSV file with numeric and/or categorical data
    """)
    
    # Logistic Regression
    st.write("### Logistic Regression")
    st.info("""
    **Logistic Regression** is used for binary classification problems.
    
    **Ideal for**:
    - Predicting yes/no outcomes
    - Calculating probability of an event occurring
    - Classification tasks with discrete outcomes
    
    **Input**: CSV file with target column containing binary values
    """)
    
    # Time Series Forecasting
    st.write("### Time Series Forecasting")
    st.info("""
    **Time Series Forecasting** predicts future values based on time-ordered past values.
    
    **Ideal for**:
    - Sales forecasting
    - Stock market prediction
    - Demand planning
    - Weather forecasting
    
    **Input**: CSV file with date column and value column
    
    **Features**:
    - Stationarity testing
    - ARIMA modeling
    - Confidence intervals
    - Interactive visualizations
    """)
    
    # K-Means Clustering
    st.write("### K-Means Clustering")
    st.info("""
    **K-Means Clustering** is an unsupervised algorithm that groups similar data points together.
    
    **Ideal for**:
    - Customer segmentation
    - Pattern discovery
    - Feature detection
    - Grouping without predefined labels
    
    **Input**: CSV file with numeric features
    """)
    
    # Decision Trees
    st.write("### Decision Trees")
    st.info("""
    **Decision Trees** create a model that predicts the value of a target variable by learning decision rules.
    
    **Ideal for**:
    - Classification and regression
    - Visual decision making
    - Feature importance analysis
    
    **Input**: CSV file with features and target columns
    """)
    
    # Random Forest
    st.write("### Random Forest")
    st.info("""
    **Random Forest** combines multiple decision trees to improve predictive accuracy.
    
    **Ideal for**:
    - High-dimensional data
    - Complex classification and regression tasks
    - Robust predictions
    
    **Input**: CSV file with features and target columns
    """)
    
    st.write("## Data Upload")
    st.info("""
    ### How to prepare your data:
    
    1. **Format**: All models accept CSV files (.csv)
    2. **Structure**: 
       - Each row represents an observation
       - Each column represents a variable
       - Include header row with column names
    3. **Data Cleaning**:
       - Remove or handle missing values
       - Normalize data if needed
       - Encode categorical variables
    
    The app will help guide you through the data preparation process after uploading.
    """)
    
    st.write("## Development Status")
    st.success("✅ Time Series Forecasting - Available")
    st.success("✅ K-Means Clustering - Available")
    st.warning("⏳ Linear Regression - Coming Soon")
    st.warning("⏳ Multiple Regression - Coming Soon")
    st.warning("⏳ Logistic Regression - Coming Soon")
    st.warning("⏳ Decision Trees - Coming Soon")
    st.warning("⏳ Random Forest - Coming Soon")