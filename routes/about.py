import streamlit as st

def show_about():
    """Show About page"""
    st.subheader("About")
    
    st.write("""
    ## Enablers of Confidence
    
    This is a modular Streamlit application designed for comprehensive data analysis and visualization.
    
    ### Features:
    
    * **Exploratory Data Analysis**: Basic statistics, distributions, and correlations
    * **Data Visualization**: Various plot types for understanding patterns in your data
    * **Machine Learning Models**: Quick comparisons of common algorithms
    * **Testing Features**: Advanced analysis and visualization options
    
    ### Development:
    
    This application follows a modular architecture pattern, separating functionality into:
    
    * Core application logic
    * Utility functions
    * Reusable components
    * Page-specific modules
    
    ### Contact:
    
    For more information, please contact the development team.
    """)