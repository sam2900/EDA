import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def correlation_analysis(df, numeric_columns):
    """
    Interactive correlation analysis component
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    numeric_columns : list
        List of numeric column names
    """
    # Column selector
    st.write("### Select Columns for Correlation Analysis")
    selected_columns = st.multiselect(
        "Choose columns to analyze (select at least 2 numeric columns)",
        options=numeric_columns,
        default=numeric_columns[:min(5, len(numeric_columns))]  # Default to first 5 numeric columns or less
    )
    
    # Check if enough columns are selected
    if len(selected_columns) < 2:
        st.warning("Please select at least 2 numeric columns for correlation analysis.")
        return
        
    st.write("### Correlation Analysis")
    
    # Correlation method selector
    correlation_method = st.selectbox(
        "Select correlation method",
        options=["pearson", "kendall", "spearman"],
        index=0,  # Default to pearson
        help="Pearson is for linear relationships, Kendall and Spearman for ordinal data"
    )
    
    # Color map selector
    cmap_options = ["coolwarm", "viridis", "Blues", "RdBu_r", "YlGnBu", "YlOrRd"]
    color_map = st.selectbox(
        "Select color map for heatmap",
        options=cmap_options,
        index=0  # Default to coolwarm
    )
    
    # Calculate correlation
    correlation_df = df[selected_columns].corr(method=correlation_method)
    
    # Generate heatmap
    st.write(f"Correlation Heatmap ({correlation_method})")
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(correlation_df, dtype=bool))
    
    # Create heatmap
    heatmap = sns.heatmap(
        correlation_df,
        annot=True,
        mask=mask,
        cmap=color_map,
        vmin=-1, vmax=1,
        fmt=".2f",
        linewidths=0.5,
        ax=ax
    )
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Display the heatmap
    st.pyplot(fig)
    
    # Display the correlation table
    st.write("### Correlation Table")
    st.dataframe(correlation_df.style.background_gradient(cmap=color_map, axis=None, vmin=-1, vmax=1))
    
    # Download button for correlation data
    csv = correlation_df.to_csv(index=True)
    st.download_button(
        label="Download Correlation Data as CSV",
        data=csv,
        file_name="correlation_data.csv",
        mime="text/csv",
    )