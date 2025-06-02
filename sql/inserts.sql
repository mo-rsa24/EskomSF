-- ✅ Clean slate (optional if testing locally)
use playground;

DELETE FROM profiling_errors;
DELETE FROM profiling_logs;

-- 🌱 1. Successful short execution (no error)
INSERT INTO profiling_logs (
    module, function, message, start_time, end_time, duration_ms, duration_readable,
    error, traceback, hostname, thread_id
)
VALUES (
    'etl', 'load_data', 'Read customer CSV',
    '2025-05-02 08:00:00', '2025-05-02 08:00:01', 1000.0, '00:00:01',
    NULL, NULL, 'spark-worker-01', 11111
);

SELECT LAST_INSERT_ID()

-- 🔄 2. Successful long operation (e.g., model training)
INSERT INTO profiling_logs (
    module, function, message, start_time, end_time, duration_ms, duration_readable,
    error, traceback, hostname, thread_id
)
VALUES (
    'modeling', 'train_model', 'Training random forest',
    '2025-05-02 08:30:00', '2025-05-02 08:42:30', 750000.0, '00:12:30',
    NULL, NULL, 'gpu-node-05', 22222
);

-- ⚠️ 3. Failed run with simple error and short traceback
INSERT INTO profiling_logs (
    module, function, message, start_time, end_time, duration_ms, duration_readable,
    error, traceback, hostname, thread_id
)
VALUES (
    'pipeline', 'merge_outputs', 'Failed on outer join',
    '2025-05-02 09:00:00', '2025-05-02 09:00:05', 5000.0, '00:00:05',
    'KeyError: "CustomerID not found"',
    'Traceback (most recent call last):\n  File "pipeline.py", line 83, in merge_outputs\n    merged = df1.merge(df2, on="CustomerID")\nKeyError: "CustomerID not found"',
    'core-node-01', 33333
);

-- 🔁 Related insert into profiling_errors (FK = 3)
INSERT INTO profiling_errors (
    profiling_log_id, error, traceback
)
VALUES (
    3,
    'KeyError: "CustomerID not found"',
    'Traceback (most recent call last):\n  File "pipeline.py", line 83, in merge_outputs\n    merged = df1.merge(df2, on="CustomerID")\nKeyError: "CustomerID not found"'
);

-- 🧠 4. Failed run with long traceback from deep recursion
INSERT INTO profiling_logs (
    module, function, message, start_time, end_time, duration_ms, duration_readable,
    error, traceback, hostname, thread_id
)
VALUES (
    'modeling', 'train_model', 'Failed to converge (NaNs)',
    '2025-05-02 10:15:00', '2025-05-02 10:15:45', 45000.0, '00:00:45',
    'ValueError: Input contains NaNs',
    CONCAT(
        'Traceback (most recent call last):\n',
        '  File "train.py", line 142, in train_model\n',
        '    clf.fit(X, y)\n',
        '  File "/usr/local/lib/python3.8/site-packages/sklearn/ensemble/_forest.py", line 387, in fit\n',
        '    X, y = self._validate_data(X, y, ...)\n',
        '  File "/usr/local/lib/python3.8/site-packages/sklearn/base.py", line 432, in _validate_data\n',
        '    raise ValueError("Input contains NaNs")\n',
        'ValueError: Input contains NaNs'
    ),
    'gpu-node-02', 44444
);

-- 🔁 profiling_errors insert (FK = 4)
INSERT INTO profiling_errors (
    profiling_log_id, error, traceback
)
VALUES (
    4,
    'ValueError: Input contains NaNs',
    'Traceback (most recent call last):\n  ... full traceback ...'
);

-- 🧪 5. Very short job (milliseconds-level, like a ping/test call)
INSERT INTO profiling_logs (
    module, function, message, start_time, end_time, duration_ms, duration_readable,
    error, traceback, hostname, thread_id
)
VALUES (
    'system', 'health_check', 'Ping service up',
    '2025-05-02 10:45:00.123', '2025-05-02 10:45:00.135', 12.0, '00:00:00',
    NULL, NULL, 'infra-monitor-01', 55555
);

-- 📈 6. Extremely long job (e.g., full monthly model retrain)
INSERT INTO profiling_logs (
    module, function, message, start_time, end_time, duration_ms, duration_readable,
    error, traceback, hostname, thread_id
)
VALUES (
    'batch', 'monthly_retrain', 'Retraining all customer models',
    '2025-05-01 00:00:00', '2025-05-01 12:00:00', 43200000.0, '12:00:00',
    NULL, NULL, 'batch-master-01', 66666
);



INSERT INTO profiling_logs (
    module, function, message,
    start_time, end_time, duration_ms, duration_readable,
    error, traceback, hostname, thread_id, status
)
VALUES (
    'etl', 'load_data', 'Read customer CSV',
    '2025-05-02 08:00:00', '2025-05-02 08:00:01', 1000.0, '00:00:01',
    NULL, NULL, 'spark-worker-01', 11111, 'completed'
);


INSERT INTO profiling_logs (
    module, function, message,
    start_time, end_time, duration_ms, duration_readable,
    error, traceback, hostname, thread_id, status
)
VALUES (
    'pipeline', 'merge_outputs', 'Failed on outer join',
    '2025-05-02 09:00:00', '2025-05-02 09:00:05', 5000.0, '00:00:05',
    'KeyError: "CustomerID not found"',
    'Traceback (most recent call last):\n ...',
    'core-node-01', 33333, 'failed'
);
