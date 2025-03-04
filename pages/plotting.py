import streamlit as st
import pandas as pd

from utils.data_loader import beer_data_uploader

def show_plotting():
    """Show Data Visualization page"""
    st.subheader("Data Visualization")
    
    # File uploader
    df = beer_data_uploader()
    
    if df is not None:
        # Load data
        # df = beer_data_uploader()
        
        if df is not None:
            # Show data preview
            st.dataframe(df.head())
            
            # Get all column names
            all_columns_names = df.columns.tolist()
            
            # Select plot type
            type_of_plot = st.selectbox(
                "Select Type of Plot",
                ["area", "bar", "line", "hist", "box", "kde"]
            )
            
            # Select columns to plot
            selected_columns_names = st.multiselect(
                "Select Columns to Plot",
                all_columns_names
            )
            
            # Generate plot button
            if st.button("Generate Plot"):
                if not selected_columns_names:
                    st.warning("Please select at least one column to plot.")
                else:
                    st.success(f"Generating Customizable Plot of {type_of_plot} for {selected_columns_names}")
                    
                    # Create plot based on selection
                    if type_of_plot in ['area', 'bar', 'line']:
                        cust_data = df[selected_columns_names]
                        
                        if type_of_plot == 'area':
                            st.area_chart(cust_data)
                        elif type_of_plot == 'bar':
                            st.bar_chart(cust_data)
                        elif type_of_plot == 'line':
                            st.line_chart(cust_data)
                    else:
                        # For other plot types
                        fig, ax = plt.subplots()
                        if len(selected_columns_names) == 1:
                            # Single column plot
                            df[selected_columns_names].plot(kind=type_of_plot, ax=ax)
                        else:
                            # Multiple columns plot
                            df[selected_columns_names].plot(kind=type_of_plot, ax=ax)
                        
                        st.pyplot(fig)