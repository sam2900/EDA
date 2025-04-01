import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import beer_data_uploader, load_beer_data

def show_data_processing():
    st.title("Data Processing")
    st.markdown("""
    This page allows you to clean and preprocess your data before analysis.
    Upload your data to get started.
    """)

    # Load data
    df = beer_data_uploader()
    
    if df is not None and not df.empty:
        st.success("Data loaded successfully! Now you can process and clean your data.")
        
        # Create tabs for different data processing operations
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Overview", "Missing Values", "Column Operations", 
            "Filtering", "Duplicates", "Data Transformation", "Download"
        ])
        
        # Initialize session state to store the processed dataframe
        if 'processed_df' not in st.session_state:
            st.session_state.processed_df = df.copy()
        
        # Store original number of rows and columns for reference
        original_rows = len(df)
        original_cols = len(df.columns)
        
        # Tab 1: Data Overview
        with tab1:
            st.header("Data Overview")
            
            # Display basic info
            st.subheader("Basic Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", f"{len(st.session_state.processed_df):,}")
            with col2:
                st.metric("Columns", f"{len(st.session_state.processed_df.columns):,}")
            with col3:
                st.metric("Data Points", f"{len(st.session_state.processed_df) * len(st.session_state.processed_df.columns):,}")
            
            # Display data types
            st.subheader("Data Types")
            dtype_df = pd.DataFrame({
                'Column': st.session_state.processed_df.columns,
                'Data Type': st.session_state.processed_df.dtypes.values,
                'Non-Null Count': st.session_state.processed_df.count().values,
                'Null Count': st.session_state.processed_df.isnull().sum().values,
                'Null %': (st.session_state.processed_df.isnull().sum().values / len(st.session_state.processed_df) * 100).round(2)
            })
            st.dataframe(dtype_df)
            
            # Data preview
            st.subheader("Data Preview")
            num_rows = st.slider("Number of rows to display", 5, 100, 10)
            st.dataframe(st.session_state.processed_df.head(num_rows))
            
            # Summary statistics
            if st.checkbox("Show summary statistics"):
                st.subheader("Summary Statistics")
                st.dataframe(st.session_state.processed_df.describe().T)
            
            # Changes summary
            if st.checkbox("Show changes from original data"):
                st.subheader("Changes Summary")
                col1, col2 = st.columns(2)
                with col1:
                    rows_diff = len(st.session_state.processed_df) - original_rows
                    st.metric("Rows Change", f"{rows_diff:+,}", f"{rows_diff:+,}")
                with col2:
                    cols_diff = len(st.session_state.processed_df.columns) - original_cols
                    st.metric("Columns Change", f"{cols_diff:+,}", f"{cols_diff:+,}")
        
        # Tab 2: Missing Values
        with tab2:
            st.header("Missing Values Treatment")
            
            # Display missing values summary
            st.subheader("Missing Values Overview")
            
            # Calculate missing values
            missing_values = st.session_state.processed_df.isnull().sum()
            missing_values_df = pd.DataFrame({
                'Column': missing_values.index,
                'Missing Values': missing_values.values,
                'Missing %': (missing_values / len(st.session_state.processed_df) * 100).round(2)
            })
            missing_values_df = missing_values_df.sort_values('Missing %', ascending=False)
            
            # Filter to only show columns with missing values
            missing_values_df = missing_values_df[missing_values_df['Missing Values'] > 0]
            
            if len(missing_values_df) > 0:
                st.dataframe(missing_values_df)
                
                # Missing values visualization
                st.subheader("Missing Values Visualization")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(st.session_state.processed_df.isnull(), 
                            yticklabels=False, 
                            cmap='viridis', 
                            ax=ax)
                ax.set_title('Missing Values Heatmap')
                st.pyplot(fig)
                
                # Treatment options
                st.subheader("Treatment Options")
                
                # Select columns to treat
                columns_with_missing = missing_values_df['Column'].tolist()
                selected_columns = st.multiselect(
                    "Select columns to treat missing values", 
                    options=columns_with_missing,
                    default=columns_with_missing
                )
                
                if selected_columns:
                    treatment_method = st.selectbox(
                        "Select treatment method",
                        ["Drop rows", "Replace with mean", "Replace with median", 
                         "Replace with mode", "Replace with 0", "Replace with custom value",
                         "Forward fill", "Backward fill"]
                    )
                    
                    custom_value = None
                    if treatment_method == "Replace with custom value":
                        custom_value = st.text_input("Enter custom value")
                    
                    if st.button("Apply Treatment"):
                        df_copy = st.session_state.processed_df.copy()
                        
                        for column in selected_columns:
                            if treatment_method == "Drop rows":
                                df_copy = df_copy.dropna(subset=[column])
                            elif treatment_method == "Replace with mean":
                                if pd.api.types.is_numeric_dtype(df_copy[column]):
                                    df_copy[column] = df_copy[column].fillna(df_copy[column].mean())
                                else:
                                    st.warning(f"Column '{column}' is not numeric. Mean can only be applied to numeric columns.")
                            elif treatment_method == "Replace with median":
                                if pd.api.types.is_numeric_dtype(df_copy[column]):
                                    df_copy[column] = df_copy[column].fillna(df_copy[column].median())
                                else:
                                    st.warning(f"Column '{column}' is not numeric. Median can only be applied to numeric columns.")
                            elif treatment_method == "Replace with mode":
                                df_copy[column] = df_copy[column].fillna(df_copy[column].mode()[0])
                            elif treatment_method == "Replace with 0":
                                df_copy[column] = df_copy[column].fillna(0)
                            elif treatment_method == "Replace with custom value":
                                df_copy[column] = df_copy[column].fillna(custom_value)
                            elif treatment_method == "Forward fill":
                                df_copy[column] = df_copy[column].ffill()
                            elif treatment_method == "Backward fill":
                                df_copy[column] = df_copy[column].bfill()
                        
                        # Update the processed dataframe
                        st.session_state.processed_df = df_copy
                        st.success(f"Missing values treated using {treatment_method}!")
                
                # Option to drop all missing values in one go
                if st.checkbox("Quick action: Drop all rows with any missing values"):
                    if st.button("Drop rows with missing values"):
                        before_count = len(st.session_state.processed_df)
                        st.session_state.processed_df = st.session_state.processed_df.dropna()
                        after_count = len(st.session_state.processed_df)
                        st.success(f"Dropped {before_count - after_count} rows with missing values!")
            else:
                st.success("No missing values found in your data!")
        
        # Tab 3: Column Operations
        with tab3:
            st.header("Column Operations")
            
            # Separate into subtabs
            col_tab1, col_tab2, col_tab3, col_tab4 = st.tabs([
                "Rename Columns", "Add Columns", "Delete Columns", "Reorder Columns"
            ])
            
            # Rename columns
            with col_tab1:
                st.subheader("Rename Columns")
                
                # Two ways to rename: one by one or batch
                rename_method = st.radio(
                    "How would you like to rename columns?",
                    ["Individual column", "Batch rename"]
                )
                
                if rename_method == "Individual column":
                    col1, col2 = st.columns(2)
                    with col1:
                        column_to_rename = st.selectbox("Select column to rename", st.session_state.processed_df.columns)
                    with col2:
                        new_name = st.text_input("New column name", column_to_rename)
                    
                    if st.button("Rename Column"):
                        if new_name and new_name != column_to_rename:
                            # Create a copy to avoid modifying the original
                            rename_dict = {column_to_rename: new_name}
                            st.session_state.processed_df = st.session_state.processed_df.rename(columns=rename_dict)
                            st.success(f"Column '{column_to_rename}' renamed to '{new_name}'!")
                
                elif rename_method == "Batch rename":
                    st.write("Enter old and new column names (one pair per line, separated by comma):")
                    rename_text = st.text_area(
                        "Format: old_name,new_name", 
                        value="\n".join([f"{col},{col}" for col in st.session_state.processed_df.columns[:3]])
                    )
                    
                    if st.button("Batch Rename"):
                        rename_dict = {}
                        for line in rename_text.strip().split("\n"):
                            if "," in line:
                                old_name, new_name = line.split(",", 1)
                                old_name = old_name.strip()
                                new_name = new_name.strip()
                                if old_name in st.session_state.processed_df.columns:
                                    rename_dict[old_name] = new_name
                        
                        if rename_dict:
                            st.session_state.processed_df = st.session_state.processed_df.rename(columns=rename_dict)
                            st.success(f"Renamed {len(rename_dict)} columns!")
                
                # Display current column names
                st.subheader("Current Columns")
                col_df = pd.DataFrame({"Column Name": st.session_state.processed_df.columns})
                st.dataframe(col_df)
            
            # Add columns
            with col_tab2:
                st.subheader("Add New Column")
                
                add_method = st.radio(
                    "How would you like to add a column?",
                    ["Constant value", "Formula/Expression", "Copy existing column"]
                )
                
                new_col_name = st.text_input("New column name", "new_column")
                
                if add_method == "Constant value":
                    const_value = st.text_input("Enter constant value", "0")
                    
                    if st.button("Add Column with Constant"):
                        try:
                            # Try to convert to numeric if possible
                            try:
                                value = float(const_value)
                                if value.is_integer():
                                    value = int(value)
                            except:
                                value = const_value
                                
                            st.session_state.processed_df[new_col_name] = value
                            st.success(f"Added new column '{new_col_name}' with constant value '{value}'!")
                        except Exception as e:
                            st.error(f"Error adding column: {e}")
                
                elif add_method == "Formula/Expression":
                    st.write("Enter a Python expression using column names in curly braces {}")
                    st.write("Example: {Price} * {Quantity} + 10")
                    
                    formula = st.text_input("Enter formula")
                    
                    if st.button("Add Column with Formula"):
                        try:
                            # Replace column names with df references
                            eval_formula = formula
                            for col in st.session_state.processed_df.columns:
                                eval_formula = eval_formula.replace(f"{{{col}}}", f"st.session_state.processed_df['{col}']")
                            
                            # Evaluate the expression
                            result = eval(eval_formula)
                            st.session_state.processed_df[new_col_name] = result
                            st.success(f"Added new column '{new_col_name}' with formula!")
                        except Exception as e:
                            st.error(f"Error evaluating formula: {e}")
                
                elif add_method == "Copy existing column":
                    source_col = st.selectbox("Select column to copy", st.session_state.processed_df.columns)
                    
                    if st.button("Add Copy of Column"):
                        st.session_state.processed_df[new_col_name] = st.session_state.processed_df[source_col]
                        st.success(f"Added new column '{new_col_name}' as a copy of '{source_col}'!")
            
            # Delete columns
            with col_tab3:
                st.subheader("Delete Columns")
                
                cols_to_delete = st.multiselect(
                    "Select columns to delete",
                    options=st.session_state.processed_df.columns
                )
                
                if cols_to_delete:
                    if st.button("Delete Selected Columns"):
                        st.session_state.processed_df = st.session_state.processed_df.drop(columns=cols_to_delete)
                        st.success(f"Deleted {len(cols_to_delete)} columns!")
                
                # Button to restore original columns (in case user deleted too much)
                if st.button("Restore Original Columns"):
                    st.session_state.processed_df = df[df.columns].copy()
                    st.success("Restored original columns!")
            
            # Reorder columns
            with col_tab4:
                st.subheader("Reorder Columns")
                
                # Option to drag and reorder columns
                st.write("Drag and drop columns to reorder them:")
                reordered_cols = st.multiselect(
                    "Column order (drag to reorder)",
                    options=st.session_state.processed_df.columns,
                    default=list(st.session_state.processed_df.columns)
                )
                
                if len(reordered_cols) == len(st.session_state.processed_df.columns):
                    if st.button("Apply New Column Order"):
                        st.session_state.processed_df = st.session_state.processed_df[reordered_cols]
                        st.success("Column order updated!")
                else:
                    st.warning("Please include all columns in the reordering.")
        
        # Tab 4: Filtering
        with tab4:
            st.header("Data Filtering")
            
            # Option to filter by column values
            st.subheader("Filter by Column Values")
            
            # Select column to filter
            filter_col = st.selectbox("Select column to filter", st.session_state.processed_df.columns)
            
            # Different filter options based on column data type
            col_type = st.session_state.processed_df[filter_col].dtype
            
            if pd.api.types.is_numeric_dtype(col_type):
                # For numeric columns, use range filter
                min_val = float(st.session_state.processed_df[filter_col].min())
                max_val = float(st.session_state.processed_df[filter_col].max())
                
                filter_type = st.radio(
                    "Filter type",
                    ["Range", "Equal to", "Greater than", "Less than"]
                )
                
                if filter_type == "Range":
                    filter_range = st.slider(
                        f"Select range for {filter_col}",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val)
                    )
                    if st.button("Apply Range Filter"):
                        st.session_state.processed_df = st.session_state.processed_df[
                            (st.session_state.processed_df[filter_col] >= filter_range[0]) & 
                            (st.session_state.processed_df[filter_col] <= filter_range[1])
                        ]
                        st.success(f"Filtered {filter_col} to range {filter_range}!")
                
                elif filter_type == "Equal to":
                    filter_value = st.number_input(f"Enter value for {filter_col}", value=min_val)
                    if st.button("Apply Equal To Filter"):
                        st.session_state.processed_df = st.session_state.processed_df[
                            st.session_state.processed_df[filter_col] == filter_value
                        ]
                        st.success(f"Filtered {filter_col} equal to {filter_value}!")
                
                elif filter_type == "Greater than":
                    filter_value = st.number_input(f"Enter minimum value for {filter_col}", value=min_val)
                    if st.button("Apply Greater Than Filter"):
                        st.session_state.processed_df = st.session_state.processed_df[
                            st.session_state.processed_df[filter_col] > filter_value
                        ]
                        st.success(f"Filtered {filter_col} greater than {filter_value}!")
                
                elif filter_type == "Less than":
                    filter_value = st.number_input(f"Enter maximum value for {filter_col}", value=max_val)
                    if st.button("Apply Less Than Filter"):
                        st.session_state.processed_df = st.session_state.processed_df[
                            st.session_state.processed_df[filter_col] < filter_value
                        ]
                        st.success(f"Filtered {filter_col} less than {filter_value}!")
            
            else:
                # For categorical columns, use multiselect
                unique_values = st.session_state.processed_df[filter_col].dropna().unique().tolist()
                
                selected_values = st.multiselect(
                    f"Select values to keep from {filter_col}",
                    options=unique_values,
                    default=unique_values
                )
                
                if selected_values and len(selected_values) < len(unique_values):
                    if st.button("Apply Category Filter"):
                        st.session_state.processed_df = st.session_state.processed_df[
                            st.session_state.processed_df[filter_col].isin(selected_values)
                        ]
                        st.success(f"Filtered {filter_col} to values: {', '.join(map(str, selected_values))}!")
            
            # Complex filtering with expressions
            st.subheader("Advanced Filtering with Expressions")
            st.write("Enter a Python expression using column names in square brackets")
            st.write("Example: [Price] > 100 and [Category] == 'Electronics'")
            
            filter_expr = st.text_input("Enter filter expression")
            
            if filter_expr:
                if st.button("Apply Expression Filter"):
                    try:
                        # Replace column names with df references
                        eval_expr = filter_expr
                        for col in st.session_state.processed_df.columns:
                            eval_expr = eval_expr.replace(f"[{col}]", f"st.session_state.processed_df['{col}']")
                        
                        # Evaluate the expression
                        mask = eval(eval_expr)
                        filtered_df = st.session_state.processed_df[mask]
                        
                        st.session_state.processed_df = filtered_df
                        st.success(f"Applied filter: {filter_expr}. Rows remaining: {len(filtered_df)}")
                    except Exception as e:
                        st.error(f"Error in filter expression: {e}")
            
            # Reset filters button
            if st.button("Reset All Filters (Restore All Data)"):
                st.session_state.processed_df = df.copy()
                st.success("Restored original data!")
            
            # Display filtered data preview
            st.subheader("Filtered Data Preview")
            st.dataframe(st.session_state.processed_df.head(10))
            st.info(f"Current number of rows: {len(st.session_state.processed_df)}")
        
        # Tab 5: Duplicates
        with tab5:
            st.header("Duplicate Handling")
            
            # Check for duplicates
            st.subheader("Duplicate Rows Analysis")
            
            # Select columns to consider for duplicates
            dup_cols = st.multiselect(
                "Select columns to check for duplicates (leave empty to check all columns)",
                options=st.session_state.processed_df.columns,
                default=[]
            )
            
            if st.button("Find Duplicates"):
                if not dup_cols:
                    dup_mask = st.session_state.processed_df.duplicated(keep='first')
                    dups = st.session_state.processed_df[dup_mask]
                else:
                    dup_mask = st.session_state.processed_df.duplicated(subset=dup_cols, keep='first')
                    dups = st.session_state.processed_df[dup_mask]
                
                # Store duplicates in session state for later use
                st.session_state.duplicate_rows = dups
                st.session_state.duplicate_columns = dup_cols
                
                if len(dups) > 0:
                    st.warning(f"Found {len(dups)} duplicate rows!")
                    st.dataframe(dups)
                    
                    # Actions for duplicates
                    st.subheader("Handle Duplicates")
                    
                    dup_action = st.radio(
                        "What would you like to do with duplicates?",
                        ["Remove duplicates (keep first)", "Remove duplicates (keep last)", 
                         "Remove all instances of duplicates", "View duplicates only"]
                    )
                    
                    if st.button("Apply Duplicate Action"):
                        if dup_action == "Remove duplicates (keep first)":
                            if not dup_cols:
                                st.session_state.processed_df = st.session_state.processed_df.drop_duplicates(keep='first')
                            else:
                                st.session_state.processed_df = st.session_state.processed_df.drop_duplicates(subset=dup_cols, keep='first')
                            st.success(f"Removed {len(dups)} duplicate rows, keeping first occurrences!")
                            
                        elif dup_action == "Remove duplicates (keep last)":
                            if not dup_cols:
                                st.session_state.processed_df = st.session_state.processed_df.drop_duplicates(keep='last')
                            else:
                                st.session_state.processed_df = st.session_state.processed_df.drop_duplicates(subset=dup_cols, keep='last')
                            st.success(f"Removed {len(dups)} duplicate rows, keeping last occurrences!")
                            
                        elif dup_action == "Remove all instances of duplicates":
                            if not dup_cols:
                                # Get all duplicated values
                                dup_mask = st.session_state.processed_df.duplicated(keep=False)
                                st.session_state.processed_df = st.session_state.processed_df[~dup_mask]
                            else:
                                # For specific columns
                                dup_rows = st.session_state.processed_df[dup_cols].duplicated(keep=False)
                                st.session_state.processed_df = st.session_state.processed_df[~dup_rows]
                            st.success("Removed all instances of duplicate rows!")
                            
                        elif dup_action == "View duplicates only":
                            st.session_state.processed_df = dups
                            st.success("Showing only duplicate rows!")
                else:
                    st.success("No duplicate rows found!")
        
        # Tab 6: Data Transformation
        with tab6:
            st.header("Data Transformation")
            
            # Separate into subtabs
            transf_tab1, transf_tab2, transf_tab3 = st.tabs([
                "Type Conversion", "Text Operations", "Numerical Operations"
            ])
            
            # Type conversion
            with transf_tab1:
                st.subheader("Convert Column Data Types")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    convert_col = st.selectbox("Select column to convert", st.session_state.processed_df.columns)
                
                with col2:
                    current_type = st.session_state.processed_df[convert_col].dtype
                    target_type = st.selectbox(
                        f"Convert from {current_type} to",
                        ["string", "int", "float", "datetime", "category", "boolean"]
                    )
                
                if st.button("Convert Data Type"):
                    try:
                        if target_type == "string":
                            st.session_state.processed_df[convert_col] = st.session_state.processed_df[convert_col].astype(str)
                        elif target_type == "int":
                            st.session_state.processed_df[convert_col] = st.session_state.processed_df[convert_col].astype(int)
                        elif target_type == "float":
                            st.session_state.processed_df[convert_col] = st.session_state.processed_df[convert_col].astype(float)
                        elif target_type == "datetime":
                            st.session_state.processed_df[convert_col] = pd.to_datetime(st.session_state.processed_df[convert_col])
                        elif target_type == "category":
                            st.session_state.processed_df[convert_col] = st.session_state.processed_df[convert_col].astype('category')
                        elif target_type == "boolean":
                            st.session_state.processed_df[convert_col] = st.session_state.processed_df[convert_col].astype(bool)
                        
                        st.success(f"Converted {convert_col} to {target_type}!")
                    except Exception as e:
                        st.error(f"Error converting data type: {e}")
            
            # Text operations
            with transf_tab2:
                st.subheader("Text Operations")
                
                # Only show string columns
                string_cols = st.session_state.processed_df.select_dtypes(include=['object', 'string']).columns.tolist()
                
                if string_cols:
                    text_col = st.selectbox("Select text column", string_cols)
                    
                    text_operation = st.selectbox(
                        "Select operation",
                        ["Convert to uppercase", "Convert to lowercase", "Remove whitespace",
                         "Extract text", "Replace text", "Concatenate with another column"]
                    )
                    
                    if text_operation == "Convert to uppercase":
                        if st.button("Apply Uppercase"):
                            st.session_state.processed_df[text_col] = st.session_state.processed_df[text_col].str.upper()
                            st.success(f"Converted {text_col} to uppercase!")
                    
                    elif text_operation == "Convert to lowercase":
                        if st.button("Apply Lowercase"):
                            st.session_state.processed_df[text_col] = st.session_state.processed_df[text_col].str.lower()
                            st.success(f"Converted {text_col} to lowercase!")
                    
                    elif text_operation == "Remove whitespace":
                        trim_option = st.radio("Trim option", ["Both sides", "Left side", "Right side"])
                        
                        if st.button("Remove Whitespace"):
                            if trim_option == "Both sides":
                                st.session_state.processed_df[text_col] = st.session_state.processed_df[text_col].str.strip()
                            elif trim_option == "Left side":
                                st.session_state.processed_df[text_col] = st.session_state.processed_df[text_col].str.lstrip()
                            elif trim_option == "Right side":
                                st.session_state.processed_df[text_col] = st.session_state.processed_df[text_col].str.rstrip()
                            
                            st.success(f"Removed whitespace from {text_col}!")
                    
                    elif text_operation == "Extract text":
                        extraction_pattern = st.text_input("Enter regex pattern to extract")
                        
                        if extraction_pattern and st.button("Extract Text"):
                            try:
                                st.session_state.processed_df[f"{text_col}_extracted"] = st.session_state.processed_df[text_col].str.extract(f"({extraction_pattern})", expand=False)
                                st.success(f"Extracted text from {text_col} to new column {text_col}_extracted!")
                            except Exception as e:
                                st.error(f"Error extracting text: {e}")
                    
                    elif text_operation == "Replace text":
                        col1, col2 = st.columns(2)
                        with col1:
                            text_to_replace = st.text_input("Text to find")
                        with col2:
                            replacement_text = st.text_input("Replace with")
                        
                        if st.button("Replace Text"):
                            st.session_state.processed_df[text_col] = st.session_state.processed_df[text_col].str.replace(text_to_replace, replacement_text)
                            st.success(f"Replaced '{text_to_replace}' with '{replacement_text}' in {text_col}!")
                    
                    elif text_operation == "Concatenate with another column":
                        other_col = st.selectbox("Select column to concatenate with", string_cols)
                        separator = st.text_input("Separator", " ")
                        new_col_name = st.text_input("New column name", f"{text_col}_{other_col}_concat")
                        
                        if st.button("Concatenate Columns"):
                            st.session_state.processed_df[new_col_name] = st.session_state.processed_df[text_col].astype(str) + separator + st.session_state.processed_df[other_col].astype(str)
                            st.success(f"Created new column {new_col_name} with concatenated values!")
                else:
                    st.warning("No text columns found in your data!")
            
            # Numerical operations
            with transf_tab3:
                st.subheader("Numerical Operations")
                
                # Only show numeric columns
                num_cols = st.session_state.processed_df.select_dtypes(include=['int', 'float']).columns.tolist()
                
                if num_cols:
                    num_col = st.selectbox("Select numeric column", num_cols)
                    
                    num_operation = st.selectbox(
                        "Select operation",
                        ["Normalize (0-1)", "Standardize (z-score)", "Log transform", 
                         "Scale by factor", "Round values", "Bin into categories"]
                    )
                    
                    if num_operation == "Normalize (0-1)":
                        if st.button("Apply Normalization"):
                            if st.session_state.processed_df[num_col].max() == st.session_state.processed_df[num_col].min():
                                st.error("Cannot normalize - all values are the same!")
                            else:
                                st.session_state.processed_df[f"{num_col}_normalized"] = (
                                    (st.session_state.processed_df[num_col] - st.session_state.processed_df[num_col].min()) / 
                                    (st.session_state.processed_df[num_col].max() - st.session_state.processed_df[num_col].min())
                                )
                                st.success(f"Created normalized version of {num_col} in new column {num_col}_normalized!")
                    
                    elif num_operation == "Standardize (z-score)":
                        if st.button("Apply Standardization"):
                            if st.session_state.processed_df[num_col].std() == 0:
                                st.error("Cannot standardize - no variance in the data!")
                            else:
                                st.session_state.processed_df[f"{num_col}_standardized"] = (
                                    (st.session_state.processed_df[num_col] - st.session_state.processed_df[num_col].mean()) / 
                                    st.session_state.processed_df[num_col].std()
                                )
                                st.success(f"Created standardized version of {num_col} in new column {num_col}_standardized!")
                    
                    elif num_operation == "Log transform":
                        log_base = st.selectbox("Select log base", ["Natural log (ln)", "Log base 10", "Log base 2"])
                        handle_zeros = st.checkbox("Add small constant to handle zeros/negatives")
                        
                        if st.button("Apply Log Transform"):
                            # Check for negative values
                            if (st.session_state.processed_df[num_col] <= 0).any():
                                if handle_zeros:
                                    # Add small constant to make all values positive
                                    const = abs(st.session_state.processed_df[num_col].min()) + 1 if st.session_state.processed_df[num_col].min() < 0 else 1
                                    temp_col = st.session_state.processed_df[num_col] + const
                                    st.info(f"Added {const} to all values to handle zeros/negatives")
                                else:
                                    st.error("Cannot log transform - data contains zeros or negative values!")
                                    temp_col = None
                            else:
                                temp_col = st.session_state.processed_df[num_col]
                            
                            if temp_col is not None:
                                if log_base == "Natural log (ln)":
                                    st.session_state.processed_df[f"{num_col}_log"] = np.log(temp_col)
                                elif log_base == "Log base 10":
                                    st.session_state.processed_df[f"{num_col}_log10"] = np.log10(temp_col)
                                elif log_base == "Log base 2":
                                    st.session_state.processed_df[f"{num_col}_log2"] = np.log2(temp_col)
                                
                                st.success(f"Applied log transform to {num_col}!")
                    
                    elif num_operation == "Scale by factor":
                        scale_factor = st.number_input("Enter scaling factor", value=1.0)
                        new_col_name = st.text_input("New column name (leave blank to overwrite)", f"{num_col}_scaled")
                        
                        if st.button("Apply Scaling"):
                            if new_col_name:
                                st.session_state.processed_df[new_col_name] = st.session_state.processed_df[num_col] * scale_factor
                                st.success(f"Scaled {num_col} by {scale_factor} into {new_col_name}!")
                            else:
                                st.session_state.processed_df[num_col] = st.session_state.processed_df[num_col] * scale_factor
                                st.success(f"Scaled {num_col} by {scale_factor}!")
                    
                    elif num_operation == "Round values":
                        decimals = st.number_input("Decimal places", min_value=0, max_value=10, value=2)
                        
                        if st.button("Round Values"):
                            st.session_state.processed_df[num_col] = st.session_state.processed_df[num_col].round(decimals)
                            st.success(f"Rounded {num_col} to {decimals} decimal places!")
                    
                    elif num_operation == "Bin into categories":
                        num_bins = st.number_input("Number of bins", min_value=2, max_value=20, value=5)
                        bin_labels = st.text_input("Bin labels (comma-separated, leave blank for default)", "")
                        
                        if st.button("Create Bins"):
                            try:
                                # Create bin edges
                                bin_edges = np.linspace(
                                    st.session_state.processed_df[num_col].min(),
                                    st.session_state.processed_df[num_col].max(),
                                    num_bins + 1
                                )
                                
                                # Create labels if provided
                                if bin_labels:
                                    labels = [label.strip() for label in bin_labels.split(",")]
                                    if len(labels) != num_bins:
                                        st.warning(f"Number of labels ({len(labels)}) doesn't match number of bins ({num_bins}). Using default labels.")
                                        labels = None
                                else:
                                    labels = None
                                
                                # Create bins
                                st.session_state.processed_df[f"{num_col}_binned"] = pd.cut(
                                    st.session_state.processed_df[num_col],
                                    bins=bin_edges,
                                    labels=labels,
                                    include_lowest=True
                                )
                                
                                st.success(f"Created binned version of {num_col} in new column {num_col}_binned!")
                            except Exception as e:
                                st.error(f"Error creating bins: {e}")
                else:
                    st.warning("No numeric columns found in your data!")
        
        # Tab 7: Download
        with tab7:
            st.header("Download Processed Data")
            
            # Display final data preview
            st.subheader("Final Data Preview")
            st.dataframe(st.session_state.processed_df.head(10))
            
            # Data summary
            st.subheader("Data Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Rows", f"{len(df):,}")
                st.metric("Original Columns", f"{len(df.columns):,}")
            with col2:
                st.metric("Final Rows", f"{len(st.session_state.processed_df):,}")
                st.metric("Final Columns", f"{len(st.session_state.processed_df.columns):,}")
            with col3:
                st.metric("Rows Change", f"{len(st.session_state.processed_df) - len(df):+,}")
                st.metric("Columns Change", f"{len(st.session_state.processed_df.columns) - len(df.columns):+,}")
            
            # Export options
            st.subheader("Export Options")
            
            # Choose file format
            export_format = st.radio(
                "Select export format",
                ["CSV", "Excel", "JSON", "Pickle"]
            )
            
            # Filename for download
            filename = st.text_input("Enter filename (without extension)", "processed_data")
            
            # Download button
            if st.button("Generate Download Link"):
                if export_format == "CSV":
                    csv = st.session_state.processed_df.to_csv(index=False)
                    b64 = pd.io.common.base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download CSV File</a>'
                    st.markdown(href, unsafe_allow_html=True)
                
                elif export_format == "Excel":
                    output = pd.io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        st.session_state.processed_df.to_excel(writer, index=False, sheet_name='Processed_Data')
                    b64 = pd.io.common.base64.b64encode(output.getvalue()).decode()
                    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">Download Excel File</a>'
                    st.markdown(href, unsafe_allow_html=True)
                
                elif export_format == "JSON":
                    json_str = st.session_state.processed_df.to_json(orient="records")
                    b64 = pd.io.common.base64.b64encode(json_str.encode()).decode()
                    href = f'<a href="data:file/json;base64,{b64}" download="{filename}.json">Download JSON File</a>'
                    st.markdown(href, unsafe_allow_html=True)
                
                elif export_format == "Pickle":
                    pickle_byte_obj = pd.io.pickle.dumps(st.session_state.processed_df)
                    b64 = pd.io.common.base64.b64encode(pickle_byte_obj).decode()
                    href = f'<a href="data:file/pickle;base64,{b64}" download="{filename}.pkl">Download Pickle File</a>'
                    st.markdown(href, unsafe_allow_html=True)
                
                st.success("Download link generated successfully!")
            
            # Summary report option
            if st.checkbox("Generate summary report"):
                report_format = st.selectbox(
                    "Report format",
                    ["Text", "HTML", "Markdown"]
                )
                
                if st.button("Generate Report"):
                    # Create summary report
                    if report_format == "Text":
                        report = f"""
                        DATA PROCESSING SUMMARY REPORT
                        =============================
                        
                        Original Data:
                        - Rows: {len(df)}
                        - Columns: {len(df.columns)}
                        
                        Processed Data:
                        - Rows: {len(st.session_state.processed_df)}
                        - Columns: {len(st.session_state.processed_df.columns)}
                        
                        Changes:
                        - Rows Change: {len(st.session_state.processed_df) - len(df):+}
                        - Columns Change: {len(st.session_state.processed_df.columns) - len(df.columns):+}
                        
                        Column Data Types:
                        {st.session_state.processed_df.dtypes.to_string()}
                        
                        Summary Statistics:
                        {st.session_state.processed_df.describe().to_string()}
                        """
                        
                        # Download report
                        b64 = pd.io.common.base64.b64encode(report.encode()).decode()
                        href = f'<a href="data:text/plain;base64,{b64}" download="{filename}_report.txt">Download Report</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    
                    elif report_format == "HTML":
                        # Create HTML report
                        report = f"""
                        <html>
                        <head>
                            <title>Data Processing Summary Report</title>
                            <style>
                                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                                table {{ border-collapse: collapse; width: 100%; }}
                                th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
                                th {{ background-color: #f2f2f2; }}
                                .header {{ font-size: 24px; font-weight: bold; margin-bottom: 20px; }}
                                .section {{ font-size: 18px; font-weight: bold; margin-top: 20px; margin-bottom: 10px; }}
                            </style>
                        </head>
                        <body>
                            <div class="header">Data Processing Summary Report</div>
                            
                            <div class="section">Original vs. Processed Data</div>
                            <table>
                                <tr>
                                    <th>Metric</th>
                                    <th>Original</th>
                                    <th>Processed</th>
                                    <th>Change</th>
                                </tr>
                                <tr>
                                    <td>Rows</td>
                                    <td>{len(df):,}</td>
                                    <td>{len(st.session_state.processed_df):,}</td>
                                    <td>{len(st.session_state.processed_df) - len(df):+,}</td>
                                </tr>
                                <tr>
                                    <td>Columns</td>
                                    <td>{len(df.columns):,}</td>
                                    <td>{len(st.session_state.processed_df.columns):,}</td>
                                    <td>{len(st.session_state.processed_df.columns) - len(df.columns):+,}</td>
                                </tr>
                            </table>
                            
                            <div class="section">Column Data Types</div>
                            <table>
                                <tr>
                                    <th>Column</th>
                                    <th>Data Type</th>
                                </tr>
                                {"".join([f"<tr><td>{col}</td><td>{dtype}</td></tr>" for col, dtype in zip(st.session_state.processed_df.dtypes.index, st.session_state.processed_df.dtypes.values)])}
                            </table>
                            
                            <div class="section">Summary Statistics</div>
                            {st.session_state.processed_df.describe().to_html()}
                        </body>
                        </html>
                        """
                        
                        # Download report
                        b64 = pd.io.common.base64.b64encode(report.encode()).decode()
                        href = f'<a href="data:text/html;base64,{b64}" download="{filename}_report.html">Download HTML Report</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    
                    elif report_format == "Markdown":
                        # Create Markdown report
                        report = f"""
                        # Data Processing Summary Report
                        
                        ## Original vs. Processed Data
                        
                        | Metric | Original | Processed | Change |
                        |--------|----------|-----------|--------|
                        | Rows | {len(df):,} | {len(st.session_state.processed_df):,} | {len(st.session_state.processed_df) - len(df):+,} |
                        | Columns | {len(df.columns):,} | {len(st.session_state.processed_df.columns):,} | {len(st.session_state.processed_df.columns) - len(df.columns):+,} |
                        
                        ## Column Data Types
                        
                        | Column | Data Type |
                        |--------|-----------|
                        {"".join([f"| {col} | {dtype} |\n" for col, dtype in zip(st.session_state.processed_df.dtypes.index, st.session_state.processed_df.dtypes.values)])}
                        
                        ## Summary Statistics
                        
                        {st.session_state.processed_df.describe().to_markdown()}
                        """
                        
                        # Download report
                        b64 = pd.io.common.base64.b64encode(report.encode()).decode()
                        href = f'<a href="data:text/markdown;base64,{b64}" download="{filename}_report.md">Download Markdown Report</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    
                    st.success("Report generated successfully!")
    else:
        st.info("Please upload data to start processing.")



