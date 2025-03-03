import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.data_loader import load_data
from components.correlation import correlation_analysis
from utils.data_processor import get_numeric_columns

def show_eda():
    """Show Exploratory Data Analysis page"""
    st.subheader("Exploratory Data Analysis")
    
    # File uploader
    data = st.file_uploader("Upload Dataset", type=["csv", "xlsx", "xls"])
    
    if data is not None:
        # Load data
        df = load_data(data)
        
        if df is not None:
            # Show data preview
            st.dataframe(df.head())
            
            # Shape
            if st.checkbox("Show Shape"):
                st.write(df.shape)
                
            # Columns
            if st.checkbox("Show Columns"):
                all_columns = df.columns.to_list()
                st.write(all_columns)
            
            # Select columns to show
            if st.checkbox("Select Columns to Show"):
                all_columns = df.columns.to_list()
                selected_columns = st.multiselect("Select Columns", all_columns)
                if selected_columns:
                    new_df = df[selected_columns]
                    st.dataframe(new_df)
            
            # Summary statistics
            if st.checkbox("Show Summary"):
                st.write(df.describe())
            
            # Value counts for target variable
            if st.checkbox("Show Value Counts"):
                st.write(df.iloc[:,-1].value_counts())
            
            # Correlation analysis
            if st.checkbox("Correlation with Seaborn"):
                numeric_columns = get_numeric_columns(df)
                correlation_analysis(df, numeric_columns)
            
            # Pie chart
            if st.checkbox("Pie Chart"):
                all_columns = df.columns.to_list()
                columns_to_plot = st.selectbox("Select 1 Column", all_columns)
                pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
                st.write(pie_plot)
                st.pyplot()