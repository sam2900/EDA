import pandas as pd
import streamlit as st

def load_data(uploaded_file):
    """
    Load data from uploaded file
    
    Parameters:
    -----------
    uploaded_file : UploadedFile
        The uploaded file object from Streamlit
        
    Returns:
    --------
    pandas.DataFrame or None
        Loaded dataframe or None if file couldn't be loaded
    """
    if uploaded_file is None:
        return None
        
    try:
        # Check file extension
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            # Default to first sheet for regular uploads
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# def load_beer_data(uploaded_file):
#     """
#     Load beer-specific data with special handling
    
#     Parameters:
#     -----------
#     uploaded_file : UploadedFile
#         The uploaded file object from Streamlit
        
#     Returns:
#     --------
#     pandas.DataFrame or None
#         Loaded and preprocessed beer dataframe or None if file couldn't be loaded
#     """
#     if uploaded_file is None:
#         return None
        
#     try:
#         # Special handling for beer data
#         if uploaded_file.name.endswith(('.xlsx', '.xls')):
#             df = pd.read_excel(uploaded_file, sheet_name="2-Beer", skiprows=8)
#         else:
#             df = pd.read_csv(uploaded_file)
            
#         return df
#     except Exception as e:
#         st.error(f"Error loading beer data: {e}")
#         return None

# def load_beer_data(uploaded_file, sheet_name="2-Beer", skiprows=8):
#     """
#     Load beer-specific data with user-configurable parameters
    
#     Parameters:
#     -----------
#     uploaded_file : UploadedFile
#         The uploaded file object from Streamlit
#     sheet_name : str
#         The name of the sheet to load (default: "2-Beer")
#     skiprows : int
#         Number of rows to skip from the top (default: 8)
        
#     Returns:
#     --------
#     pandas.DataFrame or None
#         Loaded and preprocessed beer dataframe or None if file couldn't be loaded
#     """
#     if uploaded_file is None:
#         return None
        
#     try:
#         # Handle Excel files with custom parameters
#         if uploaded_file.name.endswith(('.xlsx', '.xls')):
#             df = pd.read_excel(uploaded_file, sheet_name=sheet_name, skiprows=skiprows)
#         else:
#             # For CSV, just load normally
#             df = pd.read_csv(uploaded_file)
            
#         return df
#     except Exception as e:
#         st.error(f"Error loading beer data: {e}")
#         return None
    

# def beer_data_uploader():
#     st.subheader("Upload Beer Data")
    
#     uploaded_file = None
#     df = None
    
#     with st.expander("Upload Settings", expanded=True):
#         # File uploader
#         uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
        
#         # Only show Excel settings if an Excel file is uploaded
#         if uploaded_file is not None and uploaded_file.name.endswith(('.xlsx', '.xls')):
#             # Two columns for the settings
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 sheet_name = st.text_input("Sheet Name", value="2-Beer", 
#                                            help="Name of the sheet in the Excel file")
            
#             with col2:
#                 skiprows = st.number_input("Skip Rows", value=8, min_value=0,
#                                           help="Number of header rows to skip")
                
#             # Upload button
#             if st.button("Load Data"):
#                 df = load_beer_data(uploaded_file, sheet_name, skiprows)
#                 if df is not None:
#                     st.success(f"Data loaded successfully! {len(df)} rows and {len(df.columns)} columns.")
#                     # st.dataframe(df.head())
                
#         elif uploaded_file is not None and uploaded_file.name.endswith('.csv'):
#             # For CSV files, just show the load button
#             if st.button("Load Data"):
#                 df = load_beer_data(uploaded_file)
#                 if df is not None:
#                     st.success(f"Data loaded successfully! {len(df)} rows and {len(df.columns)} columns.")
#                     # st.dataframe(df.head())
    
#     # Return both the file and dataframe
#     return uploaded_file, df



def load_beer_data(uploaded_file, sheet_name="2-Beer", skiprows=8):
    """
    Load beer-specific data with user-configurable parameters
    
    Parameters:
    -----------
    uploaded_file : UploadedFile
        The uploaded file object from Streamlit
    sheet_name : str
        The name of the sheet to load (default: "2-Beer")
    skiprows : int
        Number of rows to skip from the top (default: 8)
        
    Returns:
    --------
    pandas.DataFrame or None
        Loaded and preprocessed beer dataframe or None if file couldn't be loaded
    """
    if uploaded_file is None:
        return None
        
    try:
        # Handle Excel files with custom parameters
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name, skiprows=skiprows)
        else:
            # For CSV, just load normally
            df = pd.read_csv(uploaded_file)
            
        return df
    except Exception as e:
        st.error(f"Error loading beer data: {e}")
        return None
    

def beer_data_uploader():
    st.subheader("Upload Data")
    
    # Initialize session state variables if they don't exist
    if 'beer_df' not in st.session_state:
        st.session_state.beer_df = None
    
    with st.expander("Upload Settings", expanded=True):
        # File uploader
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"], key="beer_file_uploader")
        
        # Only show Excel settings if an Excel file is uploaded
        if uploaded_file is not None and uploaded_file.name.endswith(('.xlsx', '.xls')):
            # Two columns for the settings
            col1, col2 = st.columns(2)
            
            with col1:
                sheet_name = st.text_input("Sheet Name", value="2-Beer", 
                                           help="Name of the sheet in the Excel file")
            
            with col2:
                skiprows = st.number_input("Skip Rows", value=8, min_value=0,
                                          help="Number of header rows to skip")
                
            # Upload button
            if st.button("Load Data", key="load_excel_btn"):
                # Load the data and store in session state
                st.session_state.beer_df = load_beer_data(uploaded_file, sheet_name, skiprows)
                if st.session_state.beer_df is not None:
                    st.success(f"Data loaded successfully! {len(st.session_state.beer_df)} rows and {len(st.session_state.beer_df.columns)} columns.")
                    st.dataframe(st.session_state.beer_df.head())
                
        elif uploaded_file is not None and uploaded_file.name.endswith('.csv'):
            # For CSV files, just show the load button
            if st.button("Load Data", key="load_csv_btn"):
                # Load the data and store in session state
                st.session_state.beer_df = load_beer_data(uploaded_file)
                if st.session_state.beer_df is not None:
                    st.success(f"Data loaded successfully! {len(st.session_state.beer_df)} rows and {len(st.session_state.beer_df.columns)} columns.")
                    st.dataframe(st.session_state.beer_df.head())
    
    # Return the dataframe from session state
    return st.session_state.beer_df