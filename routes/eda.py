import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from utils.data_loader import (load_beer_data, beer_data_uploader)
from utils.data_processor import clean_beer_data, encode_categorical_variables, get_numeric_columns
from components.correlation import correlation_analysis
from components.visualizations import (
    create_histograms, 
    create_boxplot,
    create_scatterplot,
    create_pairplot
)

def show_testing():
    """Show Exploratory Data Analysis (EDA) page"""
    st.title("Exploratory Data Analysis (EDA)")
    
    activities = ["Home", "Data Exploration","Bivariate Analysis", "Multi-variate Analysis"]
    choice = st.sidebar.selectbox("EDA", activities)
    
    if choice == "Home":
        display_eda_about()
    else:
        perform_eda_analysis(choice)

def display_eda_about():
    """Display information about the EDA section"""
    st.subheader("Exploratory Data Analysis")
    
    st.info("""
    ## Welcome to Exploratory Data Analysis
    
    This section provides tools to explore, understand, and visualize your data through various 
    statistical methods and graphical representations.
    
    ### How to use this section:
    1. Upload your data using the file uploader below
    2. Once data is loaded, choose analysis options from the checkboxes
    3. Visualize relationships and patterns in your data
    """)
    
    # Data upload section
    # st.write("## Data Upload")
    # df = beer_data_uploader()
    
    # if df is None:
    st.info("""
    ### How to prepare your data:
    
    1. **Format**: Upload Excel files (.xlxs,.xls)
    2. **Structure**: 
       - Each row represents an observation
       - Each column represents a variable
       - Include header row with column names
    3. **Data Types**:
       - Numeric columns will be available for statistical analysis
       - Categorical columns will be available for grouping and segmentation
    
    Once your data is uploaded, you'll have access to all analysis tools.
    """)
    # else:
    #     st.success("Data successfully loaded! Select analysis options below.")
    st.write("## Available Analysis Options")
    
    st.info("""
    ### Basic Data Exploration
    
    **Show shape**: Displays the dimensions of your dataset (rows × columns)
    
    **Show info**: Provides information about each column including:
    - Data types
    - Non-null count
    - Memory usage
    
    **Show Unique Elements**: Counts the number of unique values in each column
    
    **Show summary**: Generates descriptive statistics including:
    - Count, mean, standard deviation
    - Minimum and maximum values
    - Quartiles (25%, 50%, 75%)
    
    **Show null**: Identifies missing values in your dataset and offers cleaning options
    """)
    
    st.info("""
    ### Histogram Analysis
    
    This option allows you to generate histograms for two numeric columns to visualize their distributions.
    
    - Select any two numeric columns from your dataset
    - Compare distributions side by side
    - Identify patterns and outliers
    """)
    
    st.info("""
    ### Correlation Analysis
    
    Analyze relationships between numeric variables using:
    
    - Correlation heatmap visualization
    - Correlation coefficient matrix
    - Identify strong positive and negative correlations
    """)
    
    st.info("""
    ### Bivariate Analysis
    
    Examine relationships between pairs of variables:
    
    - Box plots: Visualize distribution of a numeric variable across categories
    - Scatter plots: Visualize relationships between numeric and categorical variables
    - Identify patterns and outliers in grouped data
    """)
    
    st.info("""
    ### Multivariate Analysis
    
    Explore complex relationships between multiple variables:
    
    - Pairplot: Matrix of scatterplots showing relationships between selected variables
    - Add color coding by categorical variable
    - Identify patterns across multiple dimensions
    """)
    
    st.write("## Getting Started")
    st.success("Select any of the analysis options above to begin exploring your data!")

def perform_eda_analysis(choice):
    """Perform EDA analysis based on selected option"""
    # Get dataframe from the uploader (which uses session state)
    df = beer_data_uploader()
    
    if df is None:
        st.warning("Please upload data to perform analysis")
        return
        
    # Only show analysis options if data is loaded
    if choice == "Bivariate Analysis":
        st.subheader("Bivariate Analysis")
        st.write("Analyze relationships between pairs of variables")
        perform_bivariate_analysis(df)
    elif choice == "Multi-variate Analysis":
        st.subheader("Multi-variate Analysis")
        st.write("Analyze relationships between multiple variables")
        perform_multivariate_analysis(df)
    else:
        # Default analysis if specific option not recognized
        perform_default_analysis(df)

def perform_default_analysis(df):
    """Default EDA analysis options"""
    st.subheader("Data Analysis")
    
    # Insights from data
    if st.checkbox("Show shape", key="show_shape"):
        st.write(df.shape)
            
    if st.checkbox("Show info", key="show_info"):
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
            
    if st.checkbox("Show Unique Elements", key="show_unique"):
        st.write(df.nunique())
            
    if st.checkbox("Show summary", key="show_summary"):
        st.write(df.describe())
            
    clean_df = None
    if st.checkbox("Show null", key="show_null"):
        st.write(df.isnull().sum())
            
        # Clean data by handling nulls
        clean_df = clean_beer_data(df)
            
        st.write("After cleaning:")
        st.write(clean_df.isnull().sum())
        st.write(clean_df.shape)
            
    # Use cleaned data if available, otherwise use original
    analysis_df = clean_df if clean_df is not None else df
    
    # Get numeric and categorical columns for selection
    numeric_columns = get_numeric_columns(analysis_df)
    categorical_columns = [col for col in analysis_df.columns if col not in numeric_columns]
    
    # Histogram Analysis
    if st.checkbox("Histogram Analysis", key="data_transform"):
        # Get flexible histogram column selection
        hist_column1, hist_column2 = create_flexible_histogram_section(analysis_df, numeric_columns)

        if hist_column1 and hist_column2:
            # Generate histogram
            fig = create_histograms(analysis_df, hist_column1, hist_column2)

            # Display histogram with column names in the title
            st.pyplot(fig)

            # Additional information about the selected columns
            st.write(f"Histogram comparing {hist_column1} and {hist_column2}")
        else:
            st.info("Please select two numeric columns to generate histograms.")

    # Correlation Analysis
    if st.checkbox("Correlation with Seaborn", key="correlation"):
        correlation_analysis(analysis_df, numeric_columns)
        
    # Bivariate Analysis    
    # if st.checkbox("Bivariate Analysis", key="bivariate"):
    #     perform_bivariate_analysis(analysis_df)
        
    # # Multivariate Analysis    
    # if st.checkbox("Multivariate Analysis", key="multivariate"):
    #     perform_multivariate_analysis(analysis_df)

def create_flexible_histogram_section(analysis_df, numeric_columns):
    """
    Create a flexible histogram selection section for Streamlit

    Parameters:
    -----------
    analysis_df : pandas.DataFrame
        The dataframe for analysis
    numeric_columns : list
        List of numeric column names

    Returns:
    --------
    tuple
        A tuple of selected columns for histogram generation
    """
    # Ensure we have numeric columns
    if not numeric_columns:
        st.warning("No numeric columns available for histogram analysis.")
        return None, None

    # Create columns for selection
    col1, col2 = st.columns(2)

    with col1:
        hist_column1 = st.selectbox(
            "Select first numeric column for histogram:",
            options=numeric_columns,
            key="hist_col1_flexible"
        )

    with col2:
        # Filter out the first selected column to prevent duplicate selection
        remaining_columns = [col for col in numeric_columns if col != hist_column1]

        hist_column2 = st.selectbox(
            "Select second numeric column for histogram:",
            options=remaining_columns,
            key="hist_col2_flexible"
        )

    return hist_column1, hist_column2

def perform_bivariate_analysis(analysis_df):
    """Perform bivariate analysis on the data"""
    st.subheader('Bivariate Analysis')
    
    # Get numeric and categorical columns
    numeric_columns = get_numeric_columns(analysis_df)
    categorical_columns = [col for col in analysis_df.columns]
    
    # Column selection for bivariate analysis
    st.write("Select columns for analysis:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_numeric = st.selectbox(
            "Select a numeric column:",
            options=numeric_columns,
            index=0 if numeric_columns else None,
            key="bivariate_numeric"
        )
        
    with col2:
        selected_categorical = st.selectbox(
            "Select a categorical column:",
            options=categorical_columns,
            index=0 if categorical_columns else None,
            key="bivariate_categorical"
        )
    
    if selected_numeric and selected_categorical:
        # Box plot
        st.write("Box Plot")
        fig = create_boxplot(analysis_df, selected_numeric, selected_categorical)
        st.pyplot(fig)
        
        # Scatter plot
        st.write("Scatter Plot")
        fig = create_scatterplot(
            analysis_df, 
            selected_numeric, 
            selected_categorical, 
            hue_column=selected_categorical
        )
        st.pyplot(fig)
    else:
        st.info("Please select both a numeric and categorical column to generate visualizations.")

def perform_multivariate_analysis(analysis_df):
    """Perform multivariate analysis on the data"""
    st.subheader('Multivariate Analysis')
    
    # Get numeric and categorical columns
    numeric_columns = get_numeric_columns(analysis_df)
    # categorical_columns = [col for col in analysis_df.columns if col not in numeric_columns]
    categorical_columns = [col for col in analysis_df.columns]
    
    # Column selection for multivariate analysis
    st.write("Select columns for pairplot:")
    
    # Let the user select multiple numeric columns
    selected_numeric_cols = st.multiselect(
        "Select numeric columns (max 5 recommended):",
        options=numeric_columns,
        default=numeric_columns[:min(3, len(numeric_columns))],
        key="multivariate_numeric"
    )
    
    # Select a categorical column for hue
    selected_hue = st.selectbox(
        "Select categorical column for grouping (hue):",
        options=categorical_columns,
        index=0 if categorical_columns else None,
        key="multivariate_categorical"
    )
    
    if selected_numeric_cols and selected_hue:
        if len(selected_numeric_cols) > 5:
            st.warning("Using more than 5 columns may slow down the visualization. Consider selecting fewer columns.")
        
        # Create a subset dataframe with selected columns
        subset_df = analysis_df[selected_numeric_cols + [selected_hue]]
        
        # Generate the pairplot
        with st.spinner("Generating pairplot... This may take a moment."):
            pairplot_fig = create_pairplot(subset_df, selected_hue)
            st.pyplot(pairplot_fig)
    else:
        st.info("Please select at least one numeric column and one categorical column for grouping.")