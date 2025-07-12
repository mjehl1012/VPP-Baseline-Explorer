# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 14:54:23 2025

@author: mjehl
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Tuple, Sequence
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Demand Response Baseline M&V", layout="wide")


###############################################################################
# Helper functions
###############################################################################

def compute_mae_rmse(y_true: pd.Series, y_pred: pd.Series) -> Tuple[float, float]:
    """
    Compute Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) between actual and predicted values.
    
    Args:
        y_true (pd.Series): Actual observed values.
        y_pred (pd.Series): Predicted values (e.g., baseline estimate).
    
    Returns:
        Tuple[float, float]: MAE and RMSE, each rounded to two decimal places.
    """
    
    mae = (y_true - y_pred).abs().mean()
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    return round(mae, 2), round(rmse, 2)


def select_proxy_days(df: pd.DataFrame, event_start: datetime, event_end: datetime, n_days: int = 3) -> Tuple[List[datetime.date], pd.DataFrame]:
    """
    Select weather-matched proxy days for a given DR event window.
    
    Args:
        df (pd.DataFrame): Input dataframe with 'Timestamp', 'OutdoorTemp', and 'OutdoorHumidity'.
        event_start (datetime): Start time of the DR event.
        event_end (datetime): End time of the DR event.
        n_days (int): Number of proxy days to return.
    
    Returns:
        Tuple[List[datetime.date], pd.DataFrame]: 
            - List of selected proxy day dates.
            - Daily weather dataframe with proxy/event labels for display.
    """
    
    # Hardcoded list of DR event dates to exclude from proxy selection
    event_dates = pd.to_datetime([
        "2024-07-02", "2024-07-04", "2024-07-08", "2024-07-09", "2024-08-05",
        "2024-08-08", "2024-09-05", "2024-09-06"
    ]).normalize()
    
    # Extract weather variables by date and event window time
    df_weather = df[["Timestamp", "OutdoorTemp", "OutdoorHumidity"]].copy()
    df_weather["Date"] = df_weather["Timestamp"].dt.date
    df_weather["Time"] = df_weather["Timestamp"].dt.time

    window_start, window_end = event_start.time(), event_end.time()
    mask_window = (df_weather["Time"] >= window_start) & (df_weather["Time"] <= window_end)
    
    # Daily average weather during the event time window
    daily_weather = (
        df_weather[mask_window]
        .groupby("Date")[["OutdoorTemp", "OutdoorHumidity"]]
        .mean()
        .dropna()
    )

    event_day = event_start.date()

    # Filter out event dates, weekends, and future dates
    exclude_dates = set(event_dates.date)
    candidates = daily_weather[~daily_weather.index.isin(exclude_dates) &
                               (pd.to_datetime(daily_weather.index).weekday < 5) &
                               (pd.to_datetime(daily_weather.index) < pd.to_datetime(event_day))].copy()
    
    # If event day's weather data is missing, fallback gracefully
    if event_day not in daily_weather.index:
        st.warning("Event day's weather data missing; proxy selection fell back.")
        return [], daily_weather

    # Compute similarity score to event day's weather
    target_weather = daily_weather.loc[event_day]
    candidates["TempDiff"] = (candidates["OutdoorTemp"] - target_weather["OutdoorTemp"]).abs()
    candidates["HumDiff"] = (candidates["OutdoorHumidity"] - target_weather["OutdoorHumidity"]).abs()
    candidates["TotalDiff"] = candidates["TempDiff"] + candidates["HumDiff"]

    # Select top N most similar days
    selected = candidates.sort_values("TotalDiff").head(n_days)

    # Label proxy and event days for visualization
    daily_weather["Label"] = ""
    daily_weather.loc[selected.index, "Label"] = "Proxy Day"
    if event_day in daily_weather.index:
        daily_weather.loc[event_day, "Label"] = "Event Day"

    # Return selected proxy days and labeled dataframe
    display_rows = selected.index.tolist() + [event_day]
    display_df = daily_weather.loc[display_rows].reset_index().rename(columns={"index": "Date"})

    return list(selected.index), display_df


def _daily_profile(load: pd.DataFrame, day: pd.Timestamp) -> pd.Series:
    """
    Extract the interval-ending load profile for a given day.
    
    Args:
        load (pd.DataFrame): DataFrame indexed by timestamp with a 'kW' column.
        day (pd.Timestamp): The day to extract the load profile for.
    
    Returns:
        pd.Series: Time series of 'kW' values for the specified day, aligned by frequency.
    """

    day = pd.to_datetime(day)
    freq = pd.infer_freq(load.index)
    start = day.normalize()
    end = start + pd.Timedelta(days=1)
    
    # Return interval-ending load values (starts at first full interval after midnight)
    return load.kW.loc[start + pd.Timedelta(freq):end]

    
def _select_lookback_days(
    load: pd.DataFrame,
    event_day: pd.Timestamp,
    n_days: int = 10,
    same_day_type: bool = True,
    exclude_days: Sequence[pd.Timestamp] | None = None,
) -> pd.DatetimeIndex:
    """
    Return an index of the *n_days* most recent qualifying days before *event_day*.

    Parameters
    ----------
    load : pd.DataFrame
        Interval data indexed by a timezone-aware Timestamp.
    event_day : pd.Timestamp
        Date of the demand response event (midnight-anchored).
    n_days : int, default 10
        Number of historical days to look back.
    same_day_type : bool, default True
        Require matching weekday/weekend type with *event_day*.
    exclude_days : sequence of pd.Timestamp, optional
        Specific days to ignore (e.g., known event days).

    Returns
    -------
    pd.DatetimeIndex
        List of prior days meeting all criteria.
    """
    
    if exclude_days is None:
        exclude_days = []

    # Normalize event day to date for comparison
    event_date = event_day.date()
    daytype = pd.Timestamp(event_date).dayofweek < 5  # True = weekday, False = weekend
    
    # Build mask for qualifying days
    mask = (
        (load.index.normalize() < pd.to_datetime(event_date))
        & (~load.index.normalize().isin(exclude_days))
    )
    if same_day_type:
        # Ensure match on weekday/weekend status
        mask &= (load.index.normalize().dayofweek < 5) == daytype

    # Select unique, normalized candidate days in reverse chronological order
    candidate_days = load.loc[mask].index.normalize().unique()
    candidate_days = candidate_days.sort_values(ascending=False)[:n_days][::-1]
    return candidate_days


###############################################################################
# Baseline generators
###############################################################################

def naive_baseline(df: pd.DataFrame, event_start: datetime, event_end: datetime) -> pd.DataFrame:
    """
    Estimate baseline using average load from the same weekday, one week prior.
    
    Args:
        df (pd.DataFrame): DataFrame with 'Timestamp' and 'kW' columns.
        event_start (datetime): Start of the DR event.
        event_end (datetime): End of the DR event.
    
    Returns:
        pd.DataFrame: Timestamp-indexed baseline values labeled as 'Baseline_kW'.
    """
    
    df = df.copy()
    df["Date"] = df["Timestamp"].dt.date
    df["Weekday"] = df["Timestamp"].dt.weekday

    event_day = event_start.date()

    # Identify the most recent same-weekday prior to the event
    baseline_day = (
        df[(df["Date"] < event_day) & (df["Weekday"] == event_start.weekday())]["Date"]
        .drop_duplicates()
        .sort_values(ascending=False)
        .head(1)
        .tolist()
    )
    
    # Extract data from the baseline day
    baseline_data = df[df["Date"].isin(baseline_day)].copy()
    baseline_data["Time"] = baseline_data["Timestamp"].dt.time

    # Compute average load per time-of-day interval
    avg_profile = baseline_data.groupby("Time")["kW"].mean().reset_index()
    
    # Combine time-of-day profile with event date
    avg_profile["Timestamp"] = [
        datetime.combine(event_start.date(), t) for t in avg_profile["Time"]
    ]
    
    # Limit to event window hours
    start_time = event_start.time()
    end_time = event_end.time()
    
    # Filter avg_profile to only keep times within the event window
    avg_profile = avg_profile[(avg_profile["Time"] >= start_time) & (avg_profile["Time"] <= end_time)]
    
    # Final output with timestamp and baseline kW
    avg_profile["Timestamp"] = [
        datetime.combine(event_start.date(), t) for t in avg_profile["Time"]
    ]
    
    return avg_profile[["Timestamp", "kW"]].rename(columns={"kW": "Baseline_kW"})




def caiso_10of10_baseline(
    load: pd.DataFrame,
    event_start: datetime,
    event_end: datetime,
    n_days: int = 10,
    exclude_days: Sequence[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """
    Compute a CAISO 10-of-10 baseline with day-of adjustment.
    
    CAISO methodology (simplified):
    1. Select the 10 most recent qualifying non-event weekdays.
    2. Stack and average their interval-level profiles to build the initial baseline.
    3. Compute a day-of adjustment factor based on the 4 intervals prior to the event.
    4. Apply the adjustment factor (bounded between 0.8 and 1.2).
    5. Clip to observed maximum kW for realism.
    
    Args:
        load (pd.DataFrame): DataFrame with a 'Timestamp' index and 'kW' column.
        event_start (datetime): Start time of the DR event.
        event_end (datetime): End time of the DR event.
        n_days (int): Number of historical days to include in the baseline.
        exclude_days (Sequence[pd.Timestamp], optional): Specific days to exclude.
    
    Returns:
        pd.DataFrame: Baseline time series with columns ['Timestamp', 'Baseline_kW'].
    """
    
    # Hardcoded list of prior DR event days and holidays to exclude
    exclude_days = pd.to_datetime([
        "2024-07-02", "2024-07-04", "2024-07-08", "2024-07-09", "2024-08-05",
        "2024-08-08", "2024-09-02", "2024-09-05", "2024-09-06"
    ]).normalize()
    
    load = load.set_index('Timestamp')
    
    # Select lookback days (weekday-matched, excludes event days)
    lookback_days = _select_lookback_days(load, event_start, n_days, True, exclude_days)

    # Build daily profiles and average across them
    profiles = pd.concat({_d: _daily_profile(load, _d) for _d in lookback_days}, axis=1)
    initial = profiles.mean(axis=1)
    initial.index = pd.to_datetime(initial.index).time
    initial = initial.groupby(initial.index).mean().sort_index()

    # Day-of adjustment factor based on 4 intervals (1 hour) before event (assumes no preconditioning phase)
    adjust_window = load.kW.loc[event_start - pd.Timedelta(minutes=15 * 4): event_start - pd.Timedelta(minutes=15 * 1)]
    adj_factor = (adjust_window.mean() / initial.loc[pd.to_datetime(adjust_window.index).time].mean())
    adj_factor = np.clip(adj_factor, 0.8, 1.2)      # CAISO constraint

    # Apply adjustment and clip to observed peak load
    baseline = initial.loc[event_start.time():event_end.time()] * adj_factor
    baseline = baseline.clip(upper=load.kW.max())
    
    # Format result
    baseline = baseline.reset_index()
    baseline.rename(columns={"index": "Time", 0:"Baseline_kW"}, inplace=True)
    baseline["Timestamp"] = [
        datetime.combine(event_start.date(), t) for t in baseline["Time"]
    ]
    
    return baseline[["Timestamp", "Baseline_kW"]]


def pjm_baseline(
    load: pd.DataFrame,
    event_start: datetime,
    event_end: datetime,
    n_days: int = 10,
    top_k: int = 5,
    exclude_days: Sequence[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """
    Compute PJM-style baseline using average of top-k days from a historical window.

    Steps:
    1. Select the most recent *n_days* matching-day-type (weekday/weekend).
    2. Compute total energy used during the event window on each day.
    3. Select the *top_k* highest energy days.
    4. Average their load profiles to create the initial baseline.
    5. Apply a same-day adjustment factor based on the 2 hours prior to the event.

    Args:
        load (pd.DataFrame): Interval load data with 'Timestamp' and 'kW'.
        event_start (datetime): Start of the DR event.
        event_end (datetime): End of the DR event.
        n_days (int): Number of historical days to consider.
        top_k (int): Number of highest energy days to average.
        exclude_days (Sequence[pd.Timestamp], optional): Dates to exclude (e.g., other event days).

    Returns:
        pd.DataFrame: Time series with ['Timestamp', 'Baseline_kW'].
    """
    
    # Hardcoded list of prior DR event days and holidays to exclude
    exclude_days = pd.to_datetime([
        "2025-07-02", "2025-07-04", "2025-07-08", "2025-07-09", "2025-08-05",
        "2025-08-08", "2025-09-02", "2025-09-05", "2025-09-06"
    ]).normalize()
    
    load = load.set_index('Timestamp')
    
    # Select historical days matching day-type and not excluded
    lookback_days = _select_lookback_days(load, event_start, n_days, True, exclude_days)

    # Compute energy used in event window for each candidate day
    energy = {}
    for d in lookback_days:
        d_start = d.replace(hour=event_start.hour, minute=event_start.minute) + pd.Timedelta(minutes=15 * 1)
        d_end = d.replace(hour=event_end.hour, minute=event_end.minute)
        seg = load.loc[d_start:d_end]
        energy[d] = seg.sum().values[0]
        
    # Select top_k days with highest energy use
    top_days = pd.Series(energy).sort_values(ascending=False).index[:top_k]

    # Average the profiles of those top days
    profiles = pd.concat({_d: _daily_profile(load, _d) for _d in top_days}, axis=1)
    baseline_initial = profiles.mean(axis=1)
    baseline_initial.index = pd.to_datetime(baseline_initial.index).time
    baseline_initial = baseline_initial.groupby(baseline_initial.index).mean().sort_index()

    # Same-day adjustment based on 2 hours (8 intervals) prior to event
    adjust_window = load.loc[event_start - pd.Timedelta(minutes=15 * 7): event_start]
    adj_factor = (adjust_window.mean() / baseline_initial.loc[pd.to_datetime(adjust_window.index).time].mean()).values[0]
    adj_factor = np.clip(adj_factor, 0.8, 1.3)

    # Apply adjustment factor and clip to observed max
    baseline = baseline_initial.loc[event_start.time():event_end.time()] * adj_factor
    baseline = baseline.clip(upper=load.kW.max())
    
    # Final formatting
    baseline = baseline.reset_index()
    baseline.rename(columns={"index": "Time", 0:"Baseline_kW"}, inplace=True)
    baseline["Timestamp"] = [
        datetime.combine(event_start.date(), t) for t in baseline["Time"]
    ]
    
    return baseline[["Timestamp", "Baseline_kW"]]


def rct_baseline(
    control_df: pd.DataFrame, event_start: datetime, event_end: datetime
) -> pd.DataFrame:
    """
    Use the control group’s observed load as the baseline (RCT method).
    
    Args:
        control_df (pd.DataFrame): Interval data from control devices with 'Timestamp' and 'kW'.
        event_start (datetime): Start of the DR event.
        event_end (datetime): End of the DR event.
    
    Returns:
        pd.DataFrame: Time series baseline for the event window with 'Baseline_KW'.
    """

    control_df = control_df.copy()
    control_df["Timestamp"] = pd.to_datetime(control_df["Timestamp"])
    
    # Slice control group data to match the event window
    control_slice = control_df[
        (control_df["Timestamp"] >= event_start) & (control_df["Timestamp"] <= event_end)
    ].copy()
    
    # Rename for standard baseline output format
    return control_slice.rename(columns={"kW": "Baseline_kW"})


def prophet_baseline(
    load: pd.DataFrame,
    event_start: datetime,
    event_end: datetime,
    train_days: int = 30
) -> pd.DataFrame:
    """
    Estimate baseline using Facebook Prophet with weather regressors.

    Fits a Prophet model to pre-event historical data and forecasts through
    the event window using outdoor temperature and humidity as external regressors.

    Args:
        load (pd.DataFrame): Interval load data with 'Timestamp', 'OutdoorTemp', 'OutdoorHumidity', and 'kW'.
        event_start (datetime): Start time of the DR event.
        event_end (datetime): End time of the DR event.
        train_days (int): Number of days to use for training the forecast model.

    Returns:
        pd.DataFrame: Forecasted baseline as ['Timestamp', 'Baseline_KW'].
    """

    # Ensure time series is complete and indexed correctly
    load = load.set_index('Timestamp')
    load = load.asfreq("15T")
    
    # Define training window and forecast horizon
    train_start = event_start - pd.Timedelta(days=train_days)
    horizon = 12        # Fixed to 3 hours (12 intervals at 15-minute frequency)

    # Prepare training data for Prophet
    history = load.loc[train_start:event_start].reset_index()
    history.columns = ['ds', 'OutdoorTemp','OutdoorHumidity','y']

    # Initialize Prophet with regressors
    m = Prophet(daily_seasonality=True, weekly_seasonality=True)
    m.add_regressor('OutdoorTemp')
    m.add_regressor('OutdoorHumidity')
    m.fit(history)

    # Prepare future dataframe with weather inputs for event window
    future = pd.date_range(start=event_start, periods=horizon, freq="15T")
    future_df = pd.DataFrame({'ds': future})
    future_df['OutdoorTemp'] = load.loc[event_start:event_end]['OutdoorTemp'].values
    future_df['OutdoorHumidity'] = load.loc[event_start:event_end]['OutdoorHumidity'].values
    
    # Forecast load
    forecast = m.predict(future_df)
    forecast_series = forecast['yhat'].copy()
    forecast_series.index = future

    # Clip predictions to observed maximum load
    baseline = forecast_series
    baseline = baseline.clip(upper=load.kW.max())
    baseline = baseline.reset_index()
    baseline.rename(columns={"index": "Timestamp", "yhat":"Baseline_kW"}, inplace=True)
    
    return baseline[["Timestamp", "Baseline_kW"]]


def arima_baseline(
    load: pd.DataFrame,
    event_start: datetime,
    event_end: datetime,
    horizon: int | None = None,
    arima_order: Tuple[int, int, int] = (2, 0, 2),
) -> pd.DataFrame:
    """
    Estimate baseline using ARIMA(p,d,q) time series forecasting.

    Fits an ARIMA model to pre-event load data and forecasts into the event window.

    Args:
        load (pd.DataFrame): Interval load data with 'Timestamp' and 'kW'.
        event_start (datetime): Start time of the DR event.
        event_end (datetime): End time of the DR event.
        horizon (int, optional): Number of forecast steps. Defaults to 12 (3 hours).
        arima_order (tuple): ARIMA model parameters (p, d, q).

    Returns:
        pd.DataFrame: Forecasted baseline as ['Timestamp', 'Baseline_KW'].
    """
    
    # Prepare input data and ensure consistent frequency
    load = load[['Timestamp', 'kW']]
    load = load.set_index('Timestamp')
    load = load.asfreq("15T")
    horizon = 12        # hard coded 3 hour event window for simplicity
    
    # Use load data prior to the event as training data
    endog = load.loc[: event_start - pd.Timedelta("15min")]
    
    # Fit ARIMA model
    model = ARIMA(endog, order=arima_order)
    fit = model.fit()
    
    # Forecast into the event window
    forecast = fit.forecast(steps=horizon)
    forecast.index = pd.date_range(event_start, periods=horizon, freq=load.index.freq)
    
    # Clip forecasted values to max observed kW
    baseline = forecast
    baseline = baseline.clip(upper=load.kW.max())
    baseline = baseline.reset_index()
    baseline.rename(columns={"index": "Timestamp", "predicted_mean":"Baseline_kW"}, inplace=True)

    return baseline[["Timestamp", "Baseline_kW"]]


###############################################################################
# Evaluation Helper for Scoring Baselines on Proxy Days
###############################################################################

def evaluate_baseline(
    baseline_name: str,
    df: pd.DataFrame,
    event_start: datetime,
    event_end: datetime,
    proxy_days: List[datetime.date],
    control_df: pd.DataFrame | None = None,
) -> Tuple[float | None, float | None]:
    
    """
    Evaluates a given baseline method by calculating the MAE and RMSE
    between the estimated and actual load during proxy event windows.
    
    Args:
        baseline_name (str): Name of the baseline method to use.
        df (pd.DataFrame): Telemetry data with 'Timestamp' and 'kW' columns.
        event_start (datetime): Start of the event window.
        event_end (datetime): End of the event window.
        proxy_days (List[datetime.date]): List of dates to test events on.
        control_df (pd.DataFrame | None): Optional control group data for RCT.
    
    Returns:
        Tuple[float | None, float | None]: Mean MAE and RMSE across proxy days.
        """

    maes = []
    rmses = []

    # Skip evaluation for RCT; impact is not estimated from proxy days
    if baseline_name == "RCT":
        return None, None

    # Iterate through each proxy day to simulate the event window
    for proxy_day in proxy_days:
        start = datetime.combine(proxy_day, event_start.time())
        end = datetime.combine(proxy_day, event_end.time())

        # Select the appropriate baseline function
        if baseline_name == "Naive":
            baseline_df = naive_baseline(df, start, end)
        elif baseline_name == "CAISO 10-of-10":
            baseline_df = caiso_10of10_baseline(df, start, end)
        elif baseline_name == "PJM 5-of-10":
            baseline_df = pjm_baseline(df, start, end)
        elif baseline_name == "Prophet":
            baseline_df = prophet_baseline(df, start, end)
        elif baseline_name == "ARIMA":
            baseline_df = arima_baseline(df, start, end)
            
        # Skip unrecognized baseline names
        else:
            continue

        # Extract actual load data during the proxy event window
        actual = df[df['Timestamp'].between(start, end)][['Timestamp', 'kW']]
        
        # Merge actual and baseline estimates on timestamp
        merged = pd.merge(actual, baseline_df, on='Timestamp', how='inner')
        
        # Skip if there's no overlap in timestamps
        if merged.empty:
            continue
        
        # Compute MAE and RMSE between actual and baseline kW
        mae, rmse = compute_mae_rmse(merged['kW'], merged['Baseline_kW'])
        maes.append(mae)
        rmses.append(rmse)
        
    # Return average error metrics if available
    if maes and rmses:
        return round(np.mean(maes), 2), round(np.mean(rmses), 2)
    
    # Fallback if no valid proxy windows were evaluated
    return None, None
    
    
    

###############################################################################
# Streamlit UI
###############################################################################

st.title("VPP Baseline Explorer")

# File upload inputs
load_file = st.file_uploader("Upload Device Telemetry Data (Timestamp, OutdoorTemp, OutdoorHumidity, kW)", type="csv")
control_file = st.file_uploader("Upload Control Group Data (Timestamp, kW)", type="csv")

if load_file:
    
    # Define event windows for dropdown
    event_windows = {
        "": ("", ""),
        "2024-08-05": ("2024-08-05 17:15:00", "2024-08-05 20:00:00"),
        "2024-08-08": ("2024-08-08 17:15:00", "2024-08-08 20:00:00"),
        "2024-09-05": ("2024-09-05 16:15:00", "2024-09-05 19:00:00"),
        "2024-09-06": ("2024-09-06 16:15:00", "2024-09-06 19:00:00")
    }
    
    # Streamlit dropdown using keys from the dictionary
    selected_day = st.selectbox("Select Event Day", list(event_windows.keys()))

    if selected_day != "":
        
        # Retrieve event window for selected day
        event_start_str, event_end_str = event_windows[selected_day]
        event_start = pd.to_datetime(event_start_str)
        event_end = pd.to_datetime(event_end_str)
    
        st.write(f"Event window (interval-ending): {event_start} to {event_end}")
        
        # Load and sanitize input data
        df = pd.read_csv(load_file, parse_dates=['Timestamp'])
        df['Timestamp'] = df['Timestamp'].dt.tz_localize(None)
    
        # Select proxy days based on weather similarity
        proxy_days, proxy_weather_df = select_proxy_days(df, event_start, event_end)
        st.subheader("Proxy Day Weather Similarity")
        if not proxy_weather_df.empty:
            st.dataframe(proxy_weather_df[['Date', 'OutdoorTemp', 'OutdoorHumidity', 'Label']])
    
        # Evaluate each baseline on the proxy days using MAE/RMSE
        results = []
        for name, func in zip(["Naive", "CAISO 10-of-10", "PJM 5-of-10", "Prophet", "ARIMA"], [naive_baseline, caiso_10of10_baseline, pjm_baseline, prophet_baseline, arima_baseline]):
            proxy_maes = []
            proxy_rmses = []
            for d in proxy_days:
                s = datetime.combine(d, event_start.time())
                e = datetime.combine(d, event_end.time())
                baseline_df = func(df, s, e)
                actual = df[df['Timestamp'].between(s, e)][['Timestamp', 'kW']]
                merged = pd.merge(actual, baseline_df, on='Timestamp', how='inner')
                if 'Baseline_KW_Raw' in merged.columns:
                    merged.rename(columns={'Baseline_KW_Raw': 'Baseline_kW'}, inplace=True)
                mae, rmse = compute_mae_rmse(merged['kW'], merged['Baseline_kW'])
                proxy_maes.append(mae)
                proxy_rmses.append(rmse)
            results.append({
                'Baseline': name,
                'MAE': round(np.mean(proxy_maes), 2),
                'RMSE': round(np.mean(proxy_rmses), 2),
                'Proxy Days Used': proxy_days
            })
        
        # Optional: Evaluate RCT if control file is uploaded
        if control_file:
            control_df = pd.read_csv(control_file, parse_dates=['Timestamp'])
            control_df['Timestamp'] = control_df['Timestamp'].dt.tz_localize(None)
            mae, rmse = None, None  # Placeholder since RCT not evaluated on proxies
            results.append({
                'Baseline': 'RCT',
                'MAE': None,
                'RMSE': None
            })
    
        # Display comparison results
        result_df = pd.DataFrame(results).sort_values(by='MAE')
        st.subheader("Baseline Comparison (on Proxy Days)")
        st.dataframe(result_df[['Baseline', 'MAE', 'RMSE']])
    
        # Baseline selection dropdown
        selected_baseline = st.selectbox("Select Baseline to Apply to Event Day", result_df['Baseline'])
    
        # Run selected baseline on event day
        if selected_baseline == 'Naive':
            final_baseline = naive_baseline(df, event_start, event_end)
        elif selected_baseline == 'CAISO 10-of-10':
            final_baseline = caiso_10of10_baseline(df, event_start, event_end)
        elif selected_baseline == 'PJM 5-of-10':
            final_baseline = pjm_baseline(df, event_start, event_end)
        elif selected_baseline == 'Prophet':
            final_baseline = prophet_baseline(df, event_start, event_end)
        elif selected_baseline == 'ARIMA':
            final_baseline = arima_baseline(df, event_start, event_end)
        elif selected_baseline == 'RCT' and control_file:
            final_baseline = rct_baseline(control_df, event_start, event_end)
        else:
            st.error("Missing control file or unsupported baseline.")
            st.stop()
    
        # Calculate event-day performance
        actual_event = df[df['Timestamp'].between(event_start, event_end)][['Timestamp', 'kW']]
        merged = pd.merge(actual_event, final_baseline, on='Timestamp', how='inner')
        mae, rmse = compute_mae_rmse(merged['kW'], merged['Baseline_kW'])
        impact = round((merged['Baseline_kW'] - merged['kW']).mean(), 2)
    
        st.subheader("Event Day Results")
        st.write(f"**Average Capacity Impact:** {impact} kW")
    
        # Plot actual vs baseline on event day
        plot_df = df[df['Timestamp'].dt.date == event_start.date()][['Timestamp', 'kW']].merge(
            final_baseline, on='Timestamp', how='left')
        plot_df['Hour'] = plot_df['Timestamp'].dt.hour + plot_df['Timestamp'].dt.minute / 60
    
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(plot_df['Hour'], plot_df['kW'], label='Actual Load')
        ax.plot(plot_df['Hour'], plot_df['Baseline_kW'], label=f'{selected_baseline} Baseline', linestyle='--')
        
        # Highlight event window
        adjusted_start = event_start - timedelta(minutes=15)
        ax.axvspan(adjusted_start.hour + adjusted_start.minute / 60,
               event_end.hour + event_end.minute / 60,
               color='gray', alpha=0.2, label='Event Window')
        ax.set_title("Event Day Load vs Baseline")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("kW")
        ax.set_xticks(range(0, 24))
        ax.legend()
        st.pyplot(fig)
