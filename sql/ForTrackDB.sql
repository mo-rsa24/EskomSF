-- ===============================
-- 🔍 0. IDENTIFY USER FORECAST CONFIGURATION
-- ===============================

-- This query fetches the Databrick Task ID, forecast method, and model metadata.
-- ✅ Use FIRST to identify which model configuration was triggered from the Power BI UI.
SELECT
    ufm.StartDate,
    ufm.EndDate,
    ufm.Parameters,
    ufm.Region,
    ufm.Status,
    ufm.ForecastMethodID,
    ufm.UserForecastMethodID,
    ufm.JSONCustomer as CustomerJSON,
    ufm.varJSON,
    dfm.Method AS ForecastMethodName,
    dbt.DatabrickID
FROM
    [dbo].[DataBrickTasks] AS dbt
INNER JOIN
    [dbo].[UserForecastMethod] AS ufm ON dbt.UserForecastMethodID = ufm.UserForecastMethodID
INNER JOIN
    [dbo].[DimForecastMethod] AS dfm ON ufm.ForecastMethodID = dfm.ForecastMethodID
-- WHERE ufm.UserForecastMethodID = 405
ORDER BY
    dbt.CreationDate DESC;



-- ===============================
-- 🏗️ 1. LOAD FORECAST INPUT DATA
-- ===============================

-- Load the raw time-series data for forecasting.
-- 🔁 Change 221 to the appropriate UserForecastMethodID (e.g., from query above).
SELECT * FROM dbo.PredictiveInputData(132);
-- ===============================
-- 👁️ 2. VALIDATE POD → CUSTOMER MAPPINGS
-- ===============================

-- Use this to confirm that PODs are correctly linked to Customers.
-- 🧪 Helps debug issues when forecasts fail to align by pod.
SELECT * FROM DimPOD
WHERE CustomerID IN ('7056074487','6241474325');



-- ===============================
-- 📈 3A. FETCH MODEL FORECAST RESULTS
-- ===============================

-- Shows saved forecast values per series.
-- 📌 Replace 219 with actual UserForecastMethodID used in your model.
SELECT * FROM dbo.DataBrickTasks where DatabrickID = 2;
SELECT * FROM dbo.ForecastFact
WHERE UserForecastMethodID = 11;

UPDATE UserForecastMethod SET Parameters = '(100, 30, 2, 1, 0.5, true)' WHERE UserForecastMethodID = 286;

-- ===============================
-- 📊 3B. FETCH MODEL EVALUATION METRICS
-- ===============================

-- Returns statistical performance of a forecasting run (e.g., RMSE, R²).
-- 📌 Replace 1 with actual DatabrickID from the first query.
SELECT TOP 15 *
FROM dbo.StatisticalPerformanceMetrics
WHERE DatabrickID = 1
ORDER BY ID DESC;


SELECT * FROM dbo.ForecastFact WHERE UserForecastMethodID = 404 AND PodID = 8827669849;