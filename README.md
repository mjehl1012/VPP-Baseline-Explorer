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

## Walkthrough
- Launch publicly available Streamlit app: https://vpp-baseline-explorer.streamlit.app/
  <img width="3823" height="1963" alt="image" src="https://github.com/user-attachments/assets/d5076728-1738-4b63-bf5c-8a13c417cb65" />
- Upload device telemetry and control telemetry data found in this repo
  <img width="3763" height="1668" alt="image" src="https://github.com/user-attachments/assets/b9550134-fb5b-4cfe-88c1-c2cccb751fb5" />
- Select event day
  <img width="3757" height="1738" alt="image" src="https://github.com/user-attachments/assets/98f93cb4-719b-4c5a-9d68-d9a62632899a" />
- App will run for a moment, automatically selecting the proxy days for testing a suite of baselines, displaying the error metrics, and ranking the baselines.
  <img width="3718" height="1740" alt="image" src="https://github.com/user-attachments/assets/5266b0df-3428-4e39-bbc3-19f60a0f4975" />
- Select the baseline to apply to the event day, display the device's average capacity during the event, and visualize the load curves.
  <img width="3729" height="1904" alt="image" src="https://github.com/user-attachments/assets/cb05aece-4597-4267-b025-38b965ed7237" />
- It may take a moment to render, depending on the baseline selected.
  <img width="3750" height="1902" alt="image" src="https://github.com/user-attachments/assets/f212568e-f9b0-405c-a998-d5dbb78421fe" />
