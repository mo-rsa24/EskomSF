# ✅ SLA Reporting
SELECT app_name, COUNT(*) AS failures
FROM profiling_logs
WHERE status = 'failed'
  AND start_time >= NOW() - INTERVAL 1 HOUR
GROUP BY app_name;



# ✅ Root Cause Clustering (error message frequency)
SELECT error, COUNT(*) AS occurrences
FROM profiling_errors
GROUP BY error
ORDER BY occurrences DESC;

# ✅ Alerting on New Unseen Tracebacks


# SAMPLE OUTPUT QUERY
SELECT
    id, module, function, message,
    duration_ms, start_time, end_time,
    error IS NOT NULL AS has_error
FROM profiling_logs
ORDER BY start_time DESC;

#  7. Sample query: recent executions with duration and error flags
SELECT
    id, module, function, message,
    duration_ms, duration_readable,
    start_time, end_time,
    error IS NOT NULL AS has_error
FROM profiling_logs
ORDER BY start_time DESC;

# ✅ 1. PySpark: Duration Summary via Spark SQL
# Query: Total duration per function (in seconds)
SELECT
  function,
  COUNT(*) AS runs,
  ROUND(SUM(duration_ms) / 1000, 2) AS total_secs,
  ROUND(AVG(duration_ms), 2) AS avg_ms,
  MAX(duration_ms) AS max_ms,
  MIN(duration_ms) AS min_ms
FROM profiling_logs
WHERE function IN (
  'load_and_prepare_data', 'clean_dataframe', 'load_ufm_config'
)
GROUP BY function
ORDER BY total_secs DESC;

# Assuming logs are stored in your PLAYGROUND.profiling_logs table:
SELECT
  function,
  COUNT(*) AS runs,
  ROUND(SUM(duration_ms) / 1000, 2) AS total_seconds,
  ROUND(AVG(duration_ms), 2) AS avg_duration_ms,
  MAX(duration_ms) AS max_ms,
  MIN(duration_ms) AS min_ms,
  MIN(start_time) AS first_logged,
  MAX(end_time) AS last_logged
FROM profiling_logs
WHERE function IN ('load_and_prepare_data', 'clean_dataframe', 'load_ufm_config')
  AND module = 'etl'
  AND status = 'completed'
GROUP BY function
ORDER BY total_seconds DESC;

# Sample Output
# | function                 | runs | total\_seconds | avg\_duration\_ms | max\_ms | min\_ms | first\_logged       | last\_logged        |
# | ------------------------ | ---- | -------------- | ----------------- | ------- | ------- | ------------------- | ------------------- |
# | load\_and\_prepare\_data | 5    | 21.44          | 4288.6            | 6200    | 3100    | 2025-05-02 08:00:01 | 2025-05-02 10:00:01 |
# | clean\_dataframe         | 5    | 5.52           | 1104.4            | 1300    | 980     | 2025-05-02 08:00:02 | 2025-05-02 10:00:02 |
# | load\_ufm\_config        | 5    | 1.21           | 242.0             | 300     | 200     | 2025-05-02 08:00:00 | 2025-05-02 10:00:00 |


# 📊 1. SQL Queries for Aggregation, Root Cause, SLA Metrics
# ✅ Query: Top Root Causes
SELECT
    component,
    error_type,
    COUNT(*) AS occurrences,
    MAX(created_at) AS last_seen
FROM profiling_errors
GROUP BY component, error_type
ORDER BY occurrences DESC;

#  ✅ Query: SLA Success vs. Failures (last 24h)
SELECT
    app_name,
    COUNT(*) AS total_runs,
    SUM(status = 'failed') AS failures,
    ROUND(SUM(status = 'failed') / COUNT(*), 3) AS failure_rate
FROM profiling_logs
WHERE start_time >= NOW() - INTERVAL 1 DAY
GROUP BY app_name
ORDER BY failure_rate DESC;

# ✅ Query: Top Slowest Functions
SELECT
    function,
    AVG(duration_ms) AS avg_ms,
    MAX(duration_ms) AS max_ms,
    COUNT(*) AS executions
FROM profiling_logs
WHERE status = 'completed'
GROUP BY function
ORDER BY avg_ms DESC
LIMIT 10;

# ✅ Query: Most Frequent Failed Functions
SELECT
    module,
    function,
    COUNT(*) AS failures
FROM profiling_logs
WHERE status = 'failed'
GROUP BY module, function
ORDER BY failures DESC;


# ✅ Average some of the results

# ✅ local(MariaDB) vs pyspark(Databricks)
