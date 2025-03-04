
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
    """Show New Feature Testing page"""
    st.subheader("New Feature Testing")
    
    # Get dataframe from the uploader (which uses session state)
    df = beer_data_uploader()
    
    # Only show analysis options if data is loaded
    if df is not None:
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
            
        # if st.checkbox("Histogram Analysis", key="data_transform"):
            # st.subheader("Data Transformation")
            
            # # Column selection for encoding categorical variables
            # st.write("Categorical Encoding:")
            # categorical_to_encode = st.multiselect(
            #     "Select categorical columns to encode:",
            #     options=categorical_columns,
            #     default=['Baseline Sales Units'] if 'Baseline Sales Units' in categorical_columns else [],
            #     key="encode_categorical"
            # )
            
            # if categorical_to_encode:
            #     # Encode selected categorical variables
            #     transformed_df = encode_categorical_variables(analysis_df, categorical_to_encode)
            #     st.success(f"Encoded {len(categorical_to_encode)} categorical variables.")
                
            #     # Use the transformed dataframe for subsequent operations
            #     analysis_df = transformed_df
            
            # # Create histograms with column selection
            # st.write("Histogram Analysis:")
            
            # col1, col2 = st.columns(2)
            
            # with col1:
            #     hist_column1 = st.selectbox(
            #         "Select first numeric column:",
            #         options=numeric_columns,
            #         index=numeric_columns.index('Sales Units') if 'Sales Units' in numeric_columns else 0,
            #         key="hist_col1"
            #     )
            
            # with col2:
            #     hist_column2 = st.selectbox(
            #         "Select second numeric column:",
            #         options=numeric_columns,
            #         index=numeric_columns.index('Weighted Distribution') if 'Weighted Distribution' in numeric_columns else min(1, len(numeric_columns)-1),
            #         key="hist_col2"
            #     )
            
            # if hist_column1 and hist_column2:
            #     fig = create_histograms(analysis_df, hist_column1, hist_column2)
            #     st.pyplot(fig)
            # else:
            #     st.info("Please select two numeric columns to generate histograms.")
                
            # Additional transformation options
            # st.write("Additional Transformations:")
            
            # transform_options = st.multiselect(
            #     "Select transformations to apply:",
            #     options=["Log Transform", "Standard Scaling", "Min-Max Scaling", "Outlier Removal"],
            #     key="transform_options"
            # )
            
            # if "Log Transform" in transform_options:
            #     log_columns = st.multiselect(
            #         "Select columns for log transformation:",
            #         options=numeric_columns,
            #         key="log_columns"
            #     )
                
            #     if log_columns:
            #         for col in log_columns:
            #             # Avoid log(0) errors by adding a small constant
            #             analysis_df[f"{col}_log"] = np.log1p(analysis_df[col])
                    
            #         st.success(f"Created log-transformed versions of {len(log_columns)} columns.")
            
            # if "Standard Scaling" in transform_options:
            #     scale_columns = st.multiselect(
            #         "Select columns for standard scaling:",
            #         options=numeric_columns,
            #         key="scale_columns"
            #     )
                
            #     if scale_columns:
            #         for col in scale_columns:
            #             mean = analysis_df[col].mean()
            #             std = analysis_df[col].std()
            #             analysis_df[f"{col}_scaled"] = (analysis_df[col] - mean) / std
                    
            #         st.success(f"Created standardized versions of {len(scale_columns)} columns.")
            
            # if "Min-Max Scaling" in transform_options:
            #     minmax_columns = st.multiselect(
            #         "Select columns for min-max scaling:",
            #         options=numeric_columns,
            #         key="minmax_columns"
            #     )
                
            #     if minmax_columns:
            #         for col in minmax_columns:
            #             min_val = analysis_df[col].min()
            #             max_val = analysis_df[col].max()
            #             analysis_df[f"{col}_minmax"] = (analysis_df[col] - min_val) / (max_val - min_val)
                    
            #         st.success(f"Created min-max scaled versions of {len(minmax_columns)} columns.")
            
            # if "Outlier Removal" in transform_options:
            #     outlier_columns = st.multiselect(
            #         "Select columns for outlier removal:",
            #         options=numeric_columns,
            #         key="outlier_columns"
            #     )
                
            #     if outlier_columns:
            #         # Create a copy for outlier removal to avoid modifying the original
            #         outlier_removed_df = analysis_df.copy()
            #         rows_before = len(outlier_removed_df)
                    
            #         for col in outlier_columns:
            #             Q1 = outlier_removed_df[col].quantile(0.25)
            #             Q3 = outlier_removed_df[col].quantile(0.75)
            #             IQR = Q3 - Q1
                        
            #             lower_bound = Q1 - 1.5 * IQR
            #             upper_bound = Q3 + 1.5 * IQR
                        
            #             # Create a mask for outliers
            #             outlier_mask = (outlier_removed_df[col] >= lower_bound) & (outlier_removed_df[col] <= upper_bound)
            #             outlier_removed_df = outlier_removed_df[outlier_mask]
                    
            #         rows_after = len(outlier_removed_df)
            #         rows_removed = rows_before - rows_after
                    
            #         st.success(f"Removed {rows_removed} rows containing outliers ({rows_removed/rows_before:.1%} of data).")
                    
            #         # Option to use the filtered dataset for further analysis
            #         if st.checkbox("Use outlier-filtered dataset for subsequent analyses", key="use_filtered"):
            #             analysis_df = outlier_removed_df
            
            # # Show a preview of the transformed data
            # st.write("Preview of transformed data:")
            # st.dataframe(analysis_df.head())
            
            # # Update the numeric columns based on any new columns that were added
            # numeric_columns = get_numeric_columns(analysis_df)
            
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




        if st.checkbox("Correlation with Seaborn", key="correlation"):
            correlation_analysis(analysis_df, numeric_columns)
            
        if st.checkbox("Bivariate Analysis", key="bivariate"):
            st.subheader('Bivariate Analysis')
            
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
            
        if st.checkbox("Multivariate Analysis", key="multivariate"):
            st.subheader('Multivariate Analysis')
            
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