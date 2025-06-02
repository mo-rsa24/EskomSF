-- 1. Create the main profiling_logs table
CREATE TABLE IF NOT EXISTS profiling_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    module VARCHAR(255),
    function VARCHAR(255),
    message TEXT,
    start_time DATETIME,
    end_time DATETIME,
    duration_ms FLOAT,
    duration_readable VARCHAR(20),  -- e.g. "00:01:23"
    error TEXT NULL,
    traceback LONGTEXT NULL,
    hostname VARCHAR(255),
    thread_id BIGINT
);

-- 2. Add indexes for performance
CREATE INDEX idx_start_time ON profiling_logs (start_time);
CREATE INDEX idx_duration_ms ON profiling_logs (duration_ms);

-- 3. Create profiling_errors table with FK reference
CREATE TABLE IF NOT EXISTS profiling_errors (
    error_id INT AUTO_INCREMENT PRIMARY KEY,
    profiling_log_id INT NOT NULL,
    error TEXT,
    traceback LONGTEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (profiling_log_id) REFERENCES profiling_logs(id)
        ON DELETE CASCADE
);


ALTER TABLE profiling_logs
ADD COLUMN status ENUM('completed', 'failed') DEFAULT 'completed';

ALTER TABLE profiling_logs CHANGE COLUMN databrick_id databrick_task_id INT(100)

ALTER TABLE profiling_errors
MODIFY profiling_log_id INT NULL;

ALTER TABLE profiling_errors
MODIFY COLUMN error TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;

ALTER TABLE profiling_errors
MODIFY COLUMN traceback TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
