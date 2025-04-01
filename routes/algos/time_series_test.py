import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from utils.data_loader import beer_data_uploader
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy import stats

def time_series_forecasting():
    """
    Streamlit page for time series forecasting with ARIMA models
    """
    # Page title and description
    st.title("Time Series Forecasting")
    st.markdown("""
    This tool performs time series analysis and forecasting using ARIMA models.
    Upload your data to get started.
    """)
    
    # File uploader
    df = beer_data_uploader()
    
    if df is not None:
        st.success("File successfully loaded!")
        
        # Display raw data preview
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())
        
        # Select date column
        st.subheader("Select Date and Target Columns")
        
        # Get all column names for selection
        all_columns = df.columns.tolist()
        
        # User selects date column
        date_col = st.selectbox(
            "Select the date column",
            all_columns,
            index=0 if all_columns else None
        )
        
        # Check if date column is properly formatted
        if date_col:
            # Try to convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                    st.success(f"Converted '{date_col}' to datetime format.")
                except Exception as e:
                    st.error(f"Error converting '{date_col}' to date format: {e}")
                    st.info("Please select a column that contains valid date information.")
                    date_col = None
        
        # User selects target column for forecasting
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        target_col = st.selectbox(
            "Select the target column to forecast",
            numeric_cols,
            index=0 if numeric_cols else None
        )
        
        if date_col and target_col:
            # Preprocessing
            st.subheader("Data Preprocessing")
            
            # Sort by date
            df = df.sort_values(by=date_col)
            
            # Check for duplicate dates
            duplicate_dates = df.duplicated(subset=[date_col]).sum()
            if duplicate_dates > 0:
                st.warning(f"Found {duplicate_dates} duplicate dates in your data.")
                agg_method = st.selectbox(
                    "How would you like to handle duplicate dates?",
                    ["mean", "sum", "max", "min", "first", "last"],
                    index=0
                )
                # Aggregate by date
                df = df.groupby(date_col).agg({target_col: agg_method}).reset_index()
                st.success(f"Aggregated duplicate dates using {agg_method}.")
            
            # Check for missing dates
            df = df.set_index(date_col)
            date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
            missing_dates = date_range.difference(df.index)
            
            if len(missing_dates) > 0:
                st.warning(f"Found {len(missing_dates)} missing dates in your data.")
                fill_method = st.selectbox(
                    "How would you like to handle missing dates?",
                    ["linear", "ffill", "bfill", "zero", "none"],
                    index=0
                )
                
                if fill_method != "none":
                    # Create a complete time series with the specified frequency
                    if fill_method == "zero":
                        df_reindexed = df.reindex(date_range, fill_value=0)
                    else:
                        df_reindexed = df.reindex(date_range)
                        if fill_method in ["ffill", "bfill"]:
                            df_reindexed = df_reindexed.fillna(method=fill_method)
                        elif fill_method == "linear":
                            df_reindexed = df_reindexed.interpolate(method='linear')
                    
                    df = df_reindexed
                    st.success(f"Filled missing dates using {fill_method} method.")
            
            # Extract time series
            time_series = df[target_col]
            
            # Original time series visualization
            st.subheader("Time Series Visualization")
            
            fig = px.line(
                x=time_series.index, 
                y=time_series.values,
                labels={"x": "Date", "y": target_col},
                title=f"{target_col} Over Time"
            )
            st.plotly_chart(fig)
            
            # Time Series Analysis
            st.subheader("Time Series Analysis")
            diff_order = 0
            
            # Stationarity Check
            st.write("#### Stationarity Test (Augmented Dickey-Fuller)")
            
            result = adfuller(time_series.dropna())
            
            adf_output = {
                'ADF Statistic': result[0],
                'p-value': result[1],
                '1% Critical Value': result[4]['1%'],
                '5% Critical Value': result[4]['5%'],
                '10% Critical Value': result[4]['10%']
            }
            
            adf_df = pd.DataFrame({
                'Metric': list(adf_output.keys()),
                'Value': list(adf_output.values())
            })
            
            st.table(adf_df.set_index('Metric'))
            
            if result[1] <= 0.05:
                st.success("The time series is stationary (p-value <= 0.05)")
            else:
                st.warning("The time series is not stationary (p-value > 0.05)")
                
                st.write("#### Differencing to achieve stationarity")
                
                diff_order = st.slider("Select differencing order", 0, 2, 1)
                
                if diff_order > 0:
                    differenced = time_series.diff(diff_order).dropna()
                    
                    # Plot differenced series
                    fig = px.line(
                        x=differenced.index,
                        y=differenced.values,
                        labels={"x": "Date", "y": f"{diff_order}-Order Differenced {target_col}"},
                        title=f"{diff_order}-Order Differenced Time Series"
                    )
                    st.plotly_chart(fig)
                    
                    # Stationarity test on differenced series
                    result_diff = adfuller(differenced.dropna())
                    
                    diff_adf_output = {
                        'ADF Statistic': result_diff[0],
                        'p-value': result_diff[1],
                        '1% Critical Value': result_diff[4]['1%'],
                        '5% Critical Value': result_diff[4]['5%'],
                        '10% Critical Value': result_diff[4]['10%']
                    }
                    
                    diff_adf_df = pd.DataFrame({
                        'Metric': list(diff_adf_output.keys()),
                        'Value': list(diff_adf_output.values())
                    })
                    
                    st.table(diff_adf_df.set_index('Metric'))
                    
                    if result_diff[1] <= 0.05:
                        st.success(f"The {diff_order}-order differenced series is stationary")
                    else:
                        st.warning(f"The {diff_order}-order differenced series is still not stationary")
            
            # ACF and PACF plots
            st.write("#### Autocorrelation Analysis")
            
            # Use differenced series if available, otherwise use original
            analysis_series = time_series.diff(diff_order).dropna() if diff_order > 0 else time_series
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Autocorrelation Function (ACF)")
                fig, ax = plt.subplots(figsize=(10, 4))
                plot_acf(analysis_series, ax=ax, lags=40)
                st.pyplot(fig)
            
            with col2:
                st.write("Partial Autocorrelation Function (PACF)")
                fig, ax = plt.subplots(figsize=(10, 4))
                plot_pacf(analysis_series, ax=ax, lags=40)
                st.pyplot(fig)
            
            st.write("""
            **How to interpret ACF/PACF plots:**
            - **ACF** shows correlation between a time series and its lagged values
            - **PACF** shows correlation between a time series and its lagged values, with the linear dependence of all shorter lags removed
            - Use these plots to determine appropriate values for p (AR order) and q (MA order) in your ARIMA model
            """)
            
            # ARIMA Modeling
            st.subheader("ARIMA Modeling")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                p = st.slider("p (AR order)", 0, 5, 1)
            with col2:
                d = st.slider("d (Differencing)", 0, 2, diff_order)
            with col3:
                q = st.slider("q (MA order)", 0, 5, 1)
            
            # Fit ARIMA model
            if st.button("Fit ARIMA Model"):
                with st.spinner(f"Fitting ARIMA({p},{d},{q}) model..."):
                    try:
                        model = ARIMA(time_series, order=(p, d, q))
                        
                        model_fit = model.fit()
                        
                        # Store the model fit in session state to use it later
                        st.session_state.model_fit = model_fit
                        st.session_state.time_series = time_series
                        st.session_state.p = p
                        st.session_state.d = d
                        st.session_state.q = q
                        st.session_state.target_col = target_col
            
                        # Display model summary
                        st.write("#### Model Summary")
                        model_summary = model_fit.summary().tables[1].as_html()
                        st.write(model_summary, unsafe_allow_html=True)
                        
                        # Plot residuals
                        st.write("#### Residual Analysis")
                        
                        residuals = model_fit.resid
                        
                        fig = px.line(
                            x=residuals.index,
                            y=residuals.values,
                            labels={"x": "Date", "y": "Residuals"},
                            title="Residuals Over Time"
                        )
                        st.plotly_chart(fig)
                        
                        # Residual distribution
                        fig = px.histogram(
                            residuals,
                            nbins=20,
                            labels={"value": "Residual Value", "count": "Frequency"},
                            title="Residual Distribution"
                        )
                        fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="red")
                        st.plotly_chart(fig)
                        
                        # Residual ACF
                        st.write("#### Residual Autocorrelation")
                        fig, ax = plt.subplots(figsize=(10, 4))
                        plot_acf(residuals.dropna(), ax=ax)
                        st.pyplot(fig)
                        
                        # Forecast
                        st.subheader("Forecasting")
                        
                        # User selects forecast horizon
                        forecast_periods = st.slider(
                            "Select forecast horizon (periods)",
                            1, 365, 30
                        )
                        
                        # Generate forecast and get prediction intervals
                        forecast_result = model_fit.get_forecast(steps=forecast_periods)
                        forecast = forecast_result.predicted_mean
                        pred_interval = forecast_result.conf_int()
                        lower_bound = pred_interval.iloc[:, 0]  # Lower bound
                        upper_bound = pred_interval.iloc[:, 1]  # Upper bound

                    
                        # # Get prediction intervals manually
                        
                        # alpha = 0.05  # 95% confidence interval
                        # forecast_var = model_fit.forecast_variance(steps=forecast_periods)
                        # forecast_std = np.sqrt(forecast_var)
                        # critical_value = stats.norm.ppf(1 - alpha/2)  # For 95% CI
                        # lower_bound = forecast - critical_value * forecast_std
                        # upper_bound = forecast + critical_value * forecast_std
                        
                        # Create forecast index
                        last_date = time_series.index[-1]
                        
                        # Determine frequency of the original data
                        if isinstance(last_date, pd.Timestamp):
                            # Calculate average gap between dates
                            date_diffs = pd.Series(time_series.index[1:]) - pd.Series(time_series.index[:-1])
                            avg_days = date_diffs.mean().days if hasattr(date_diffs.mean(), 'days') else 1
                            
                            forecast_index = pd.date_range(
                                start=last_date + pd.Timedelta(days=avg_days),
                                periods=forecast_periods,
                                freq=f'{avg_days}D'
                            )
                        else:
                            forecast_index = pd.RangeIndex(
                                start=time_series.index[-1] + 1,
                                stop=time_series.index[-1] + forecast_periods + 1
                            )
                        
                        forecast = pd.Series(forecast, index=forecast_index)
                        
                        # Generate forecast and get prediction intervals
                        forecast_result = model_fit.get_forecast(steps=forecast_periods)
                        forecast = forecast_result.predicted_mean
                        pred_interval = forecast_result.conf_int()
                        lower_bound = pred_interval.iloc[:, 0]  # Lower bound
                        upper_bound = pred_interval.iloc[:, 1]  # Upper bound
                        
                        # Combine actual and forecast
                        fig = go.Figure()
                        
                        # Add actual values
                        fig.add_trace(go.Scatter(
                            x=time_series.index,
                            y=time_series.values,
                            mode='lines',
                            name='Actual',
                            line=dict(color='blue')
                        ))
                        
                        # Add forecast
                        fig.add_trace(go.Scatter(
                            x=forecast.index,
                            y=forecast.values,
                            mode='lines',
                            name='Forecast',
                            line=dict(color='red')
                        ))
                        
                        # Add prediction intervals
                        fig.add_trace(go.Scatter(
                            x=forecast.index.tolist() + forecast.index.tolist()[::-1],
                            y=upper_bound.tolist() + lower_bound.tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(231,107,243,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='95% Confidence Interval'
                        ))
                        
                        fig.update_layout(
                            title=f"ARIMA({p},{d},{q}) Forecast for {target_col}",
                            xaxis_title="Date",
                            yaxis_title=target_col,
                            hovermode="x unified"
                        )
                        
                        st.plotly_chart(fig)
                        
                        # Display forecast values
                        st.write("#### Forecast Values")
                        forecast_df = pd.DataFrame({
                            'Date': forecast.index,
                            'Forecast': forecast.values,
                            'Lower Bound (95%)': lower_bound.values,
                            'Upper Bound (95%)': upper_bound.values
                        })
                        st.dataframe(forecast_df)
                        
                        # Download forecast
                        csv = forecast_df.to_csv(index=False)
                        st.download_button(
                            label="Download Forecast as CSV",
                            data=csv,
                            file_name="time_series_forecast.csv",
                            mime="text/csv"
                        )
                        
                        
                        # Model evaluation
                        st.subheader("Model Evaluation")
                        
                        evaluation_metrics = pd.DataFrame({
                            'Metric': ['AIC', 'BIC', 'Log Likelihood'],
                            'Value': [model_fit.aic, model_fit.bic, model_fit.llf]
                        })
                        
                        st.table(evaluation_metrics.set_index('Metric'))
                        
                        st.success("ARIMA model fitting and forecasting completed successfully!")
                        
                    except Exception as e:
                        st.error(f"Error fitting ARIMA model: {e}")
                        st.info("Tips: Try different p, d, q values or check if your data is appropriate for ARIMA modeling.")

            if 'model_fit' in st.session_state:

            #rolling forecast
                st.subheader("Rolling Forecast Validation")

                # Ask user if they want to perform rolling forecast validation
                perform_rolling = st.checkbox("Perform Rolling Forecast Validation")
                if perform_rolling and st.button("Run Rolling Validation"):
                    with st.spinner("Performing rolling forecast validation..."):
                        try:

                            time_series = st.session_state.time_series
                            model_fit = st.session_state.model_fit
                            p = st.session_state.p
                            d = st.session_state.d
                            q = st.session_state.q
                            target_col = st.session_state.target_col
                            # User inputs for rolling forecast
                            col1, col2 = st.columns(2)
                            with col1:
                                train_percentage = st.slider("Initial training set size (%)", 50, 95, 70)
                                step_size = st.slider("Forecast step size", 1, 10, 1)
                            with col2:
                                forecast_horizon = st.slider("Forecast horizon for each step", 1, 30, 1)
                                refit_window = st.radio("Refit strategy", ["Expanding Window", "Sliding Window"])

                            # Calculate the initial training size
                            train_size = int(len(time_series) * train_percentage / 100)

                            # Prepare for rolling forecast
                            history = time_series[:train_size].copy()
                            test = time_series[train_size:].copy()

                            # Skip if test set is too small
                            if len(test) < 5:
                                st.warning("Test set is too small for meaningful validation. Consider using a larger dataset or smaller training percentage.")
                            else:
                                predictions = []
                                confidence_intervals = []
                                actual_values = []
                                dates = []

                                # Track start time to show progress
                                start_time = pd.Timestamp.now()

                                # Progress bar
                                progress_bar = st.progress(0)

                                # Loop through test set in steps
                                for i in range(0, len(test), step_size):
                                    # Update progress
                                    progress = int((i / len(test)) * 100)
                                    progress_bar.progress(progress)

                                    # Get actual test values for this step
                                    actual = test[i:i+step_size]
                                    if len(actual) == 0:
                                        break
                                    
                                    # Fit model on current history
                                    roll_model = ARIMA(history, order=(p, d, q))
                                    roll_results = roll_model.fit()

                                    # Forecast
                                    forecast_result = roll_results.get_forecast(steps=min(forecast_horizon, len(actual)))
                                    forecast_mean = forecast_result.predicted_mean
                                    forecast_ci = forecast_result.conf_int()

                                    # Store results
                                    predictions.extend(forecast_mean.values)
                                    lower_ci = forecast_ci.iloc[:, 0].values
                                    upper_ci = forecast_ci.iloc[:, 1].values

                                    for j in range(len(forecast_mean)):
                                        if i+j < len(test):
                                            confidence_intervals.append((lower_ci[j], upper_ci[j]))
                                            actual_values.append(actual.iloc[j])
                                            dates.append(actual.index[j])

                                    # Update history based on window type
                                    if refit_window == "Expanding Window":
                                        # Add actual values to history (expanding window)
                                        history = pd.concat([history, actual])
                                    else:
                                        # Move the window forward (sliding window)
                                        history = pd.concat([history, actual]).iloc[len(actual):]

                                # Complete the progress bar
                                progress_bar.progress(100)

                                # Create DataFrame with results
                                results_df = pd.DataFrame({
                                    'Date': dates,
                                    'Actual': actual_values,
                                    'Predicted': predictions[:len(actual_values)],
                                    'Lower Bound': [ci[0] for ci in confidence_intervals],
                                    'Upper Bound': [ci[1] for ci in confidence_intervals]
                                })

                                # Calculate error metrics
                                results_df['Error'] = results_df['Actual'] - results_df['Predicted']
                                results_df['Absolute Error'] = abs(results_df['Error'])
                                results_df['Squared Error'] = results_df['Error'] ** 2
                                results_df['Percentage Error'] = (results_df['Error'] / results_df['Actual']) * 100

                                # Display results
                                st.write("#### Rolling Forecast Results")
                                st.dataframe(results_df)

                                # Calculate and display performance metrics
                                mae = results_df['Absolute Error'].mean()
                                rmse = np.sqrt(results_df['Squared Error'].mean())
                                mape = results_df['Percentage Error'].abs().mean()

                                metrics_df = pd.DataFrame({
                                    'Metric': ['Mean Absolute Error (MAE)', 'Root Mean Squared Error (RMSE)', 'Mean Absolute Percentage Error (MAPE)'],
                                    'Value': [mae, rmse, mape]
                                })
                                st.table(metrics_df.set_index('Metric'))

                                # Plot results
                                fig = go.Figure()

                                # Plot actual values
                                fig.add_trace(go.Scatter(
                                    x=results_df['Date'],
                                    y=results_df['Actual'],
                                    mode='lines',
                                    name='Actual',
                                    line=dict(color='blue')
                                ))

                                # Plot predicted values
                                fig.add_trace(go.Scatter(
                                    x=results_df['Date'],
                                    y=results_df['Predicted'],
                                    mode='lines',
                                    name='Predicted',
                                    line=dict(color='red')
                                ))

                                # Add confidence intervals
                                fig.add_trace(go.Scatter(
                                    x=results_df['Date'].tolist() + results_df['Date'].tolist()[::-1],
                                    y=results_df['Upper Bound'].tolist() + results_df['Lower Bound'].tolist()[::-1],
                                    fill='toself',
                                    fillcolor='rgba(231,107,243,0.2)',
                                    line=dict(color='rgba(255,255,255,0)'),
                                    name='95% Confidence Interval'
                                ))

                                fig.update_layout(
                                    title=f"Rolling Forecast ARIMA({p},{d},{q}) Results",
                                    xaxis_title="Date",
                                    yaxis_title=target_col,
                                    hovermode="x unified"
                                )

                                st.plotly_chart(fig)

                                # Download results
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Rolling Forecast Results as CSV",
                                    data=csv,
                                    file_name="rolling_forecast_results.csv",
                                    mime="text/csv"
                                )

                                # Elapsed time
                                elapsed_time = (pd.Timestamp.now() - start_time).total_seconds()
                                st.info(f"Rolling forecast completed in {elapsed_time:.2f} seconds.")

                        except Exception as e:
                            st.error(f"Error in rolling forecast: {e}")
                            st.info("Try different model parameters or check if your data is suitable for rolling forecast validation.")

            
    else:
        # Information about the page when no data is uploaded
        st.info("""
        ### How to use this Time Series Forecasting tool:
        
        1. **Upload your data** using the file uploader above.
        2. **Select the date column** that contains your time information.
        3. **Select the target column** you want to forecast.
        4. The app will automatically:
           - Test for stationarity
           - Display ACF and PACF plots
           - Help you identify appropriate ARIMA parameters
        5. **Set ARIMA parameters** (p, d, q) and fit the model.
        6. **Generate forecasts** for future periods.
        
        This tool is ideal for analyzing sales data, stock prices, demand forecasting, 
        and other time-dependent data series.
        """)