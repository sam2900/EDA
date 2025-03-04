import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import beer_data_uploader
from utils.data_processor import detect_column_types



def kmc():

    # Set page config
    # st.set_page_config(page_title="Beer Sales Clustering Analysis", layout="wide")

    # App title and description
    st.title("Clustering Analysis")
    st.markdown("""
    This application performs unsupervised K-means clustering on data.
    Upload your Excel file to get started.
    """)

    # File uploader
    # uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])
    df = beer_data_uploader()

    if df is not None:
        # Load the data
        try:
            # df = pd.read_excel(uploaded_file)
            st.success("File successfully loaded!")

            # Display raw data
            st.subheader("Raw Data Preview")
            st.dataframe(df.head())

            # Data info
            st.subheader("Data Information")

            # Display data types
            buffer = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                buffer.append(f"- **{col}**: {dtype}")
            st.markdown("\n".join(buffer))

            # Data preprocessing
            st.subheader("Data Preprocessing")

            # # Identify categorical and numerical columns old way
            # string_columns = df.iloc[:, :5].columns.tolist()
            # numeric_columns = df.iloc[:, 5:].columns.tolist()
            
            string_columns,numeric_columns = detect_column_types(df)


            # Display column types
            st.write("**Categorical Columns:**", string_columns)
            st.write("**Numeric Columns:**", numeric_columns)

            # Handling missing values
            if df.isnull().sum().sum() > 0:
                st.warning(f"Found {df.isnull().sum().sum()} missing values in the dataset.")
                missing_option = st.radio(
                    "How would you like to handle missing values?",
                    ("Drop rows with missing values", "Fill with mean", "Fill with median", "Fill with 0")
                )

                if missing_option == "Drop rows with missing values":
                    df = df.dropna()
                    st.info(f"Dropped rows with missing values. New shape: {df.shape}")
                elif missing_option == "Fill with mean":
                    for col in numeric_columns:
                        df[col] = df[col].fillna(df[col].mean())
                    st.info("Filled missing values with column means.")
                elif missing_option == "Fill with median":
                    for col in numeric_columns:
                        df[col] = df[col].fillna(df[col].median())
                    st.info("Filled missing values with column medians.")
                else:
                    for col in numeric_columns:
                        df[col] = df[col].fillna(0)
                    st.info("Filled missing values with 0.")
            else:
                st.info("No missing values found in the dataset.")

            # One-Hot Encoding of Categorical Columns
            st.subheader("One-Hot Encoding (Optional)")

            encoding_check = st.checkbox("Apply one-hot encoding to categorical columns")

            if encoding_check:
                # Let user select which categorical columns to encode
                encode_cols = st.multiselect(
                    "Select categorical columns to one-hot encode",
                    string_columns,
                    default=[]
                )

                if encode_cols:
                    # Perform one-hot encoding
                    df_encoded = pd.get_dummies(df, columns=encode_cols, prefix_sep='_')

                    # Show preview of encoded data
                    st.write("Encoded Data Preview:")
                    st.dataframe(df_encoded.head())

                    # Check how many new columns were created
                    new_cols = [col for col in df_encoded.columns if col not in df.columns]
                    st.info(f"Created {len(new_cols)} new columns from one-hot encoding.")

                    # List of new encoded columns
                    with st.expander("View encoded columns"):
                        st.write(new_cols)

                    # Update original dataframe and column lists
                    df = df_encoded

                    # Update column lists - encoded columns become numeric
                    string_columns = [col for col in string_columns if col not in encode_cols]
                    numeric_columns = [col for col in df.columns if col not in string_columns]

                    st.success(f"One-hot encoding completed. New dataframe shape: {df.shape}")

            # Grouping option
            st.subheader("Data Grouping (Optional)")
            group_check = st.checkbox("Group data by categorical columns before clustering")

            if group_check:
                group_cols = st.multiselect(
                    "Select columns to group by",
                    string_columns,
                    default=string_columns[:2] if len(string_columns) >= 2 else string_columns
                )

                if group_cols:
                    agg_dict = {col: 'sum' for col in numeric_columns}
                    df_grouped = df.groupby(group_cols, as_index=False).agg(agg_dict)
                    st.write("Grouped Data Preview:")
                    st.dataframe(df_grouped.head())

                    # Update dataframe and columns
                    df = df_grouped
                    string_columns = group_cols
                    numeric_columns = [col for col in df.columns if col not in string_columns]

                    st.info(f"Data grouped successfully. New shape: {df.shape}")

            # Feature selection for clustering
            st.subheader("Feature Selection for Clustering")
            st.info("Select the numeric columns to use for clustering analysis")

            selected_features = st.multiselect(
                "Select features for clustering",
                numeric_columns,
                default=numeric_columns[:5] if len(numeric_columns) >= 5 else numeric_columns
            )

            if not selected_features:
                st.error("Please select at least one feature for clustering.")
            else:
                # Prepare data for clustering
                X = df[selected_features].copy()

                # Feature scaling
                # st.subheader("Feature Scaling")
                # scale_option = st.radio(
                #     "Scale features?",
                #     ("Yes", "No"),
                #     index=0
                # )
                ### scale option hack
                scale_option="No"

                if scale_option == "Yes":
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    st.info("Features standardized to mean=0 and std=1")
                else:
                    X_scaled = X.values
                    st.info("Using original feature values without scaling")

                # Determining optimal k
                st.subheader("Determining Optimal Number of Clusters")
                max_clusters = min(10, len(df) - 1)  # Limit max clusters

                # Option to find optimal k
                find_k_option = st.radio(
                    "How would you like to determine the number of clusters?",
                    ("Use elbow method", "Specify number directly")
                )

                if find_k_option == "Use elbow method":
                    with st.spinner("Calculating inertia for different values of k..."):
                        inertia = []
                        silhouette_avg = []
                        k_values = range(2, max_clusters + 1)

                        for k in k_values:
                            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                            kmeans.fit(X_scaled)
                            inertia.append(kmeans.inertia_)

                            # Calculate silhouette score
                            if len(set(kmeans.labels_)) > 1:  # Ensure we have valid clusters
                                silhouette_avg.append(silhouette_score(X_scaled, kmeans.labels_))
                            else:
                                silhouette_avg.append(0)

                    # Create elbow method plot
                    fig, ax1 = plt.subplots(figsize=(10, 6))

                    # Plot inertia
                    ax1.set_xlabel('Number of Clusters (k)')
                    ax1.set_ylabel('Inertia', color='tab:blue')
                    ax1.plot(k_values, inertia, 'o-', color='tab:blue')
                    ax1.tick_params(axis='y', labelcolor='tab:blue')

                    # Create second y-axis for silhouette score
                    ax2 = ax1.twinx()
                    ax2.set_ylabel('Silhouette Score', color='tab:red')
                    ax2.plot(k_values, silhouette_avg, 'o-', color='tab:red')
                    ax2.tick_params(axis='y', labelcolor='tab:red')

                    fig.tight_layout()
                    st.pyplot(fig)

                    suggested_k = 2 + np.argmax(silhouette_avg)
                    st.info(f"Suggested optimal number of clusters based on silhouette score: {suggested_k}")

                    # Let user choose k
                    k = st.slider(
                        "Select number of clusters (k)",
                        min_value=2,
                        max_value=max_clusters,
                        value=suggested_k
                    )
                else:
                    # Direct specification
                    k = st.slider(
                        "Select number of clusters (k)",
                        min_value=2,
                        max_value=max_clusters,
                        value=3
                    )

                # Perform clustering
                st.subheader(f"K-means Clustering with k={k}")
                with st.spinner("Performing K-means clustering..."):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(X_scaled)

                    # Add cluster labels to the original dataframe
                    df['Cluster'] = clusters

                    # Get cluster centers
                    if scale_option == "Yes":
                        centers = scaler.inverse_transform(kmeans.cluster_centers_)
                    else:
                        centers = kmeans.cluster_centers_

                    # Create a DataFrame for cluster centers
                    centers_df = pd.DataFrame(centers, columns=selected_features)
                    centers_df.index.name = 'Cluster'
                    centers_df.index = [f"Cluster {i}" for i in range(k)]

                # Display clustering results
                st.subheader("Clustering Results")

                # Show cluster distribution
                cluster_counts = df['Cluster'].value_counts().sort_index()
                st.write("Cluster Distribution:")

                # Create horizontal bar chart
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = ax.barh([f"Cluster {i}" for i in range(k)], cluster_counts.values)
                ax.set_xlabel('Count')
                ax.set_ylabel('Cluster')
                ax.set_title('Number of Samples in Each Cluster')

                # Add count labels
                for i, bar in enumerate(bars):
                    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                            f"{cluster_counts.values[i]}", 
                            va='center')

                st.pyplot(fig)

                # Display cluster centers
                st.subheader("Cluster Centers")
                st.dataframe(centers_df)

                # Cluster profiles
                st.subheader("Cluster Profiles")

                # Mean values by cluster
                cluster_profiles = df.groupby('Cluster')[selected_features].mean()
                cluster_profiles.index = [f"Cluster {i}" for i in range(k)]

                # Create a heatmap of the cluster profiles
                fig, ax = plt.subplots(figsize=(12, max(6, len(selected_features) // 2)))
                sns.heatmap(cluster_profiles, annot=True, cmap="YlGnBu", fmt=".1f", ax=ax)
                ax.set_title('Feature Values by Cluster')
                st.pyplot(fig)

                # Radar chart for cluster comparison
                st.subheader("Cluster Comparison (Radar Chart)")

                # Select subset of features for radar chart if too many
                if len(selected_features) > 10:
                    st.info("Too many features for radar chart. Selecting top 10 most variable features.")
                    feature_variance = X.var()
                    radar_features = feature_variance.nlargest(10).index.tolist()
                else:
                    radar_features = selected_features

                # Normalize the cluster profiles for radar chart
                radar_df = cluster_profiles[radar_features].copy()
                for col in radar_df.columns:
                    max_val = radar_df[col].max()
                    min_val = radar_df[col].min()
                    if max_val > min_val:
                        radar_df[col] = (radar_df[col] - min_val) / (max_val - min_val)
                    else:
                        radar_df[col] = 0  # Avoid division by zero

                # Create radar chart using Plotly
                fig = go.Figure()

                for i in range(k):
                    fig.add_trace(go.Scatterpolar(
                        r=radar_df.iloc[i].values,
                        theta=radar_df.columns,
                        fill='toself',
                        name=f'Cluster {i}'
                    ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title="Normalized Feature Values by Cluster"
                )

                st.plotly_chart(fig)

                # Dimensionality reduction for visualization if more than 2 features
                if len(selected_features) > 2:
                    if st.checkbox("PCA Visualization"):
                        st.subheader("PCA Visualization")

                        # Apply PCA
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_scaled)

                        # Variance explained
                        explained_variance = pca.explained_variance_ratio_
                        st.write(f"Variance explained by 2 principal components: {sum(explained_variance):.2%}")

                        # Create scatter plot with Plotly
                        pca_df = pd.DataFrame({
                            'PCA1': X_pca[:, 0],
                            'PCA2': X_pca[:, 1],
                            'Cluster': [f"Cluster {c}" for c in clusters]
                        })

                        fig = px.scatter(
                            pca_df, x='PCA1', y='PCA2', color='Cluster',
                            title='PCA Visualization of Clusters',
                            labels={'PCA1': f'Principal Component 1 ({explained_variance[0]:.1%})',
                                    'PCA2': f'Principal Component 2 ({explained_variance[1]:.1%})'},
                            color_discrete_sequence=px.colors.qualitative.Set1
                        )

                        # Add cluster centers to the plot
                        centers_pca = pca.transform(kmeans.cluster_centers_)
                        for i, (x, y) in enumerate(centers_pca):
                            fig.add_trace(
                                go.Scatter(
                                    x=[x], y=[y],
                                    mode='markers',
                                    marker=dict(symbol='star', size=15, color=f'rgba(0,0,0,1)'),
                                    name=f'Center {i}',
                                    showlegend=True
                                )
                            )

                        st.plotly_chart(fig)

                        # Feature contributions to principal components
                        loadings = pca.components_

                        # Create a DataFrame of loadings
                        loadings_df = pd.DataFrame(
                            loadings.T,
                            index=selected_features,
                            columns=['PC1', 'PC2']
                        )

                        st.write("Feature Contributions to Principal Components:")
                        st.dataframe(loadings_df)

                        # Visualize loadings
                        fig, ax = plt.subplots(figsize=(10, 8))

                        # Create a horizontal bar chart for PC1
                        ax.barh(loadings_df.index, loadings_df['PC1'], color='blue', alpha=0.6, label='PC1')
                        ax.barh(loadings_df.index, loadings_df['PC2'], color='red', alpha=0.6, label='PC2')

                        ax.set_xlabel('Loading value')
                        ax.set_title('Feature Importance in Principal Components')
                        ax.legend()
                        ax.grid(alpha=0.3)

                        st.pyplot(fig)

                # Direct 2D visualization if exactly 2 features
                elif len(selected_features) == 2:
                    st.subheader("2D Visualization")

                    # Create scatter plot
                    fig = px.scatter(
                        df, x=selected_features[0], y=selected_features[1], 
                        color='Cluster', color_continuous_scale='viridis',
                        title=f'{selected_features[0]} vs {selected_features[1]} by Cluster'
                    )

                    # Add cluster centers
                    for i, center in enumerate(centers):
                        fig.add_trace(
                            go.Scatter(
                                x=[center[0]], y=[center[1]],
                                mode='markers',
                                marker=dict(symbol='star', size=15, color='black'),
                                name=f'Center {i}'
                            )
                        )

                    st.plotly_chart(fig)

                

                if st.checkbox("Cluster Insights"):
                    # Cluster insights
                    st.subheader("Cluster Insights")

                    for i in range(k):
                        cluster_data = df[df['Cluster'] == i]
                        st.write(f"### Cluster {i} ({len(cluster_data)} items)")

                        # Get categorical distribution
                        if string_columns:
                            for col in string_columns:
                                st.write(f"**{col} Distribution:**")

                                value_counts = cluster_data[col].value_counts().head(10)  # Top 10 values

                                fig, ax = plt.subplots(figsize=(10, min(6, max(3, len(value_counts) // 2))))
                                value_counts.plot.barh(ax=ax)
                                ax.set_title(f'{col} Distribution in Cluster {i}')
                                ax.set_xlabel('Count')
                                st.pyplot(fig)

                        # Statistical summary of numeric features
                        st.write("**Numeric Features Summary:**")
                        st.dataframe(cluster_data[selected_features].describe().T)

                        # Comparison to overall average
                        st.write("**Comparison to Overall Average:**")

                        overall_avg = df[selected_features].mean()
                        cluster_avg = cluster_data[selected_features].mean()

                        # Calculate percentage difference
                        pct_diff = ((cluster_avg - overall_avg) / overall_avg * 100).round(1)

                        comparison_df = pd.DataFrame({
                            'Cluster Average': cluster_avg,
                            'Overall Average': overall_avg,
                            'Difference %': pct_diff
                        })

                        st.dataframe(comparison_df)

                        # Highlight key characteristics
                        st.write("**Key Characteristics:**")

                        # Find features where this cluster differs significantly from the overall average
                        significant_features = pct_diff[abs(pct_diff) > 10].sort_values(ascending=False)

                        if not significant_features.empty:
                            for feature, diff in significant_features.items():
                                direction = "higher" if diff > 0 else "lower"
                                st.write(f"- {feature}: {abs(diff):.1f}% {direction} than average")
                        else:
                            st.write("No significant differences from the overall average.")

                
                # Display clustered data
                st.subheader("Data with Cluster Assignments")

                # Make cluster the first column for better visualization
                cols = df.columns.tolist()
                cols.insert(0, cols.pop(cols.index('Cluster')))
                df = df[cols]

                st.dataframe(df)

                # Option to download results
                st.subheader("Download Results")

                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download clustered data as CSV",
                    data=csv,
                    file_name="beer_sales_clustered.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Error processing the file: {e}")
    else:
        # Sample data information
        st.info("""
        Please upload an Excel file with data.

        Your data should have:
        - First 5 columns as categorical/string values (MarketsPeriods, TOTAL BEER, SEGMENT_IB, SUBSEGMENT_IB)
        - Remaining columns as numeric values (sales metrics, distribution metrics, etc.)
        """)

        st.markdown("""
        ### What This App Does

        1. **Data Preprocessing**:
           - Handles missing values
           - Option to apply one-hot encoding to categorical variables
           - Option to group data by categorical columns

        2. **Clustering Analysis**:
           - Helps determine the optimal number of clusters using the elbow method
           - Performs K-means clustering on selected features

        3. **Visualization**:
           - Cluster distribution
           - Cluster profiles
           - PCA visualization
           - 2D feature plots

        4. **Results**:
           - Download clustered data
           - Detailed insights for each cluster
        """)