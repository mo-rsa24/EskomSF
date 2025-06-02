

## 📄 Business Requirements Document (BRD)

### Ticket

**Implement Validation for Forecast Window Relative to Dataset Window:** https://singds.atlassian.net/browse/ESFR-6034

### Project Title:

**Forecast Window Validation for Electricity Consumption Forecasting**

### Objective:

Ensure that PowerBI frontend users select **forecast windows** that are realistic given the size of the **historical dataset window**, thereby improving model reliability, user experience, and forecast accuracy.

---

## 1. 💼 Business Requirement

### 1.1 Problem Statement:

Users frequently attempt to forecast electricity consumption far beyond what the model can reliably support based on the limited historical data selected. This leads to:

* Poor forecast accuracy
* Misleading trends in PowerBI reports
* Unnecessary load on the Databricks forecasting pipeline

### 1.2 Proposed Solution:

Introduce validation rules in the backend pipeline and expose informative feedback to the PowerBI frontend that:

* **Compare the forecast window to the historical dataset window**
* **Warn or block forecasts** when the forecast horizon is excessively large relative to the available data

---

## 2. 📊 Definitions

| Term                  | Definition                                                                                      |
| --------------------- | ----------------------------------------------------------------------------------------------- |
| **Forecast Window**   | The number of future periods (months/days) the user wants to predict                            |
| **Dataset Window**    | The span of historical data provided (i.e. number of months or years selected)                  |
| **Feasibility Ratio** | Ratio = Forecast Window / Dataset Window                                                        |
| **Threshold**         | A configurable limit (e.g., 0.5 or 50%) above which a forecast request may be denied or flagged |

---

## 3. 🔐 Functional Requirements

### FR1: Validate Input Ranges

* System must calculate:

  ```
  dataset_window_length = last_date - first_date
  forecast_window_length = forecast_end_date - forecast_start_date
  feasibility_ratio = forecast_window_length / dataset_window_length
  ```
* System must compare `feasibility_ratio` to a configured threshold (e.g., 0.5)

### FR2: Bound Enforcement

* If `feasibility_ratio` exceeds allowed threshold:

  * **Option A (Strict):** Deny the forecast and return an error message
  * **Option B (Flexible):** Allow forecast but return a warning in metadata

### FR3: PowerBI Metadata Feedback

* Return metadata with every forecast result:

  ```json
  {
    "feasibility_ratio": 0.75,
    "forecast_valid": false,
    "recommendation": "Reduce forecast window or increase historical data range"
  }
  ```

### FR4: Configurable Thresholds

* Allow thresholds to be configured globally and per model:

  ```json
  {
    "default_feasibility_threshold": 0.5,
    "model_specific": {
      "SARIMA": 0.5,
      "XGBoost": 0.7
    }
  }
  ```

---

## 4. ⚙️ Non-Functional Requirements

* Must integrate with existing Databricks notebook pipeline
* Must add < 1 second to pipeline execution
* Must store validation outcome in logging/metadata tables for traceability
* Should support custom overrides for specific customers

---

## 5. 🧪 Example Scenario

| Metric                | Value                           |
| --------------------- | ------------------------------- |
| Historical Data Range | Jan 2022 – Dec 2023 (24 months) |
| Forecast Requested    | Jan 2024 – Dec 2025 (24 months) |
| Feasibility Ratio     | 24 / 24 = 1.0                   |
| Threshold             | 0.5                             |
| Outcome               | ❌ Rejected or ⚠️ Warned         |

---

## 6. 🛠️ Implementation Suggestions

### 6.1 Python/PySpark Code Snippet

```python
from datetime import datetime

def validate_forecast_window(start_date, end_date, forecast_start, forecast_end, threshold=0.5):
    dataset_window = (end_date - start_date).days
    forecast_window = (forecast_end - forecast_start).days
    ratio = forecast_window / dataset_window if dataset_window > 0 else float('inf')
    valid = ratio <= threshold
    return {
        "forecast_valid": valid,
        "feasibility_ratio": round(ratio, 2),
        "recommendation": (
            "Forecast is acceptable." if valid 
            else "Reduce forecast horizon or increase historical data window."
        )
    }
```

---

## 7. ✅ Acceptance Criteria

| Criterion                     | Condition                            |
| ----------------------------- | ------------------------------------ |
| Rejects overly long forecasts | When feasibility\_ratio > threshold  |
| Provides user feedback        | Recommendation string returned       |
| Logs violation                | All invalid requests are logged      |
| Configurable per model        | Each model can set its own threshold |
| Fast response                 | Adds negligible latency              |

---

Let me know if you'd like a **PowerBI DAX layer suggestion**, **Databricks integration module**, or a **user-facing message design** for frontend validation.
