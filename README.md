# VPP Baseline Explorer

An interactive Streamlit app to automatically compare baseline estimation methods for demand response events.
Use the provided simulated thermostat telemetry and control group data to visualize load profiles, evaluate baselines, and measure event impact for a single device across a handful of events in Summer 2024.

## 🔧 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Expected File Formats

### Telemetry Data CSV
- Required Columns: `Timestamp`, `OutdoorTemp`, `OutdoorHumidity`, `kW`
- Timestamp should be in UTC or timezone-naive 15-minute intervals.

### Control Group CSV (for RCT baseline)
- Required Columns: `Timestamp`, `kW`

## Baseline Methods Included

- Naive (same-day previous week)
- CAISO 10-of-10 with adjustment
- PJM 5-of-10 with adjustment
- Prophet (with weather regressors)
- ARIMA
- RCT (if control group data is provided)

## Features

- Automatic proxy day selection based on weather similarity
- Baseline comparison and scoring on proxy days (MAE/RMSE)
- Capacity impact calculation for each method
- Load vs baseline visualizations
