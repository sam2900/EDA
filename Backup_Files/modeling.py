import streamlit as st
import pandas as pd
import numpy as np

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from utils.data_loader import beer_data_uploader
from utils.data_processor import clean_beer_data

def show_modeling():
    """Show Model Building page"""
    st.subheader("Building ML Model")
    
    # File uploader
    df = beer_data_uploader()
    # df=clean_beer_data(df)

    # In the show_modeling() function, add data validation and preprocessing:

    if df is not None:
        # Show data preview
        st.dataframe(df.head())
    
        # Add data inspection
        st.write("Data types:")
        st.write(df.dtypes)
    
        # Check for non-numeric columns
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        if non_numeric_cols:
            st.warning(f"Non-numeric columns found: {non_numeric_cols}")
        
            # Convert non-numeric columns if needed
            if st.checkbox("Convert categorical columns to numeric"):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                for col in non_numeric_cols:
                    if col != df.columns[-1]:  # Don't encode target yet
                        df[col] = le.fit_transform(df[col].astype(str))
    
        # Handle target column separately
        target_col = df.columns[-1]
        if target_col in non_numeric_cols:
            st.info(f"Converting target column '{target_col}' to numeric")
            df[target_col] = le.fit_transform(df[target_col].astype(str))
    
        # Check for nulls
        if df.isnull().any().any():
            st.warning("Dataset contains missing values. These will be dropped.")
            df = df.dropna()
    
        # Model building section
        try:
            # Split features and target
            X = df.iloc[:, 0:-1]
            Y = df.iloc[:, -1]
        
            # Now check if X and Y are appropriate
            st.write(f"Features shape: {X.shape}, Target shape: {Y.shape}")
            seed = 7
                
                # Define models
            models = []
            models.append(("LR", LogisticRegression()))
            models.append(("LDA", LinearDiscriminantAnalysis()))
            models.append(("KNN", KNeighborsClassifier()))
            models.append(('CART', DecisionTreeClassifier()))
            models.append(('NB', GaussianNB()))
            models.append(('SVM', SVC()))
                
            # Evaluate each model
            model_names = []
            model_mean = []
            model_std = []
            all_models = []
            scoring = 'accuracy'
                
            for name, model in models:
                kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
                cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
                model_names.append(name)
                model_mean.append(cv_results.mean())
                model_std.append(cv_results.std())
                    
                accuracy_results = {
                    "model_name": name,
                    "model_accuracy": cv_results.mean(),
                    "standard_deviation": cv_results.std()
                }
                all_models.append(accuracy_results)
                
            # Display results
            if st.checkbox("Metrics as Table"):
                results_df = pd.DataFrame(
                    zip(model_names, model_mean, model_std),
                    columns=["Model Name", "Model Accuracy", "Standard Deviation"]
                )
                st.dataframe(results_df)
                
            if st.checkbox("Metrics as JSON"):
                st.json(all_models)
                    
        except Exception as e:
            st.error(f"Error in model building: {e}")
            st.info("Make sure your dataset has numeric features and a target variable in the last column.")
        
        
        # Proceed with model building...
    
    # if df is not None:
    #     # Load data
    #     # df = beer_data_uploader()
        
    #     if df is not None:
    #         # Show data preview
    #         st.dataframe(df.head())
            
    #         # Model building section
    #         try:
    #             # Split features and target
    #             X = df.iloc[:, 0:-1]
    #             Y = df.iloc[:, -1]
    #             seed = 7
                
    #             # Define models
    #             models = []
    #             models.append(("LR", LogisticRegression()))
    #             models.append(("LDA", LinearDiscriminantAnalysis()))
    #             models.append(("KNN", KNeighborsClassifier()))
    #             models.append(('CART', DecisionTreeClassifier()))
    #             models.append(('NB', GaussianNB()))
    #             models.append(('SVM', SVC()))
                
    #             # Evaluate each model
    #             model_names = []
    #             model_mean = []
    #             model_std = []
    #             all_models = []
    #             scoring = 'accuracy'
                
    #             for name, model in models:
    #                 kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    #                 cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    #                 model_names.append(name)
    #                 model_mean.append(cv_results.mean())
    #                 model_std.append(cv_results.std())
                    
    #                 accuracy_results = {
    #                     "model_name": name,
    #                     "model_accuracy": cv_results.mean(),
    #                     "standard_deviation": cv_results.std()
    #                 }
    #                 all_models.append(accuracy_results)
                
    #             # Display results
    #             if st.checkbox("Metrics as Table"):
    #                 results_df = pd.DataFrame(
    #                     zip(model_names, model_mean, model_std),
    #                     columns=["Model Name", "Model Accuracy", "Standard Deviation"]
    #                 )
    #                 st.dataframe(results_df)
                
    #             if st.checkbox("Metrics as JSON"):
    #                 st.json(all_models)
                    
    #         except Exception as e:
    #             st.error(f"Error in model building: {e}")
    #             st.info("Make sure your dataset has numeric features and a target variable in the last column.")