-- Alter table to add new fields for run traceability and grouping
ALTER TABLE profiling_logs
    ADD COLUMN run_id CHAR(36),
    ADD COLUMN app_name VARCHAR(255),
    ADD COLUMN category VARCHAR(100),
    ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;


-- Create indexes for fast filtering and traceability
CREATE INDEX idx_run_id ON profiling_logs(run_id);
CREATE INDEX idx_app_function ON profiling_logs(app_name, function);
CREATE INDEX idx_category ON profiling_logs(category);

ALTER TABLE profiling_logs
    ADD COLUMN forecast_method_id INT,
    ADD COLUMN forecast_method_name VARCHAR(255),
    ADD COLUMN databrick_id INT(100),
    ADD COLUMN user_forecast_method_id INT;



ALTER TABLE profiling_errors
    ADD COLUMN severity ENUM('low', 'medium', 'high', 'critical') DEFAULT 'medium',
    ADD COLUMN error_type VARCHAR(255),
    ADD COLUMN component VARCHAR(255);
