CREATE TABLE IF NOT EXISTS benchmark_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    benchmark_name TEXT NOT NULL,
    username TEXT NOT NULL,
    score INT NOT NULL,
    submission_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE VIEW IF NOT EXISTS top_benchmarks AS
SELECT
    username,
    benchmark_name,
    MAX(score) as max_score,
    COUNT(*) as submission_count
FROM
    benchmark_results
GROUP BY
    username, benchmark_name;

CREATE TABLE IF NOT EXISTS benchmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    is_hidden BOOLEAN DEFAULT FALSE,
    is_frozen BOOLEAN DEFAULT FALSE,
    is_archive BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS user_info (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    multiplier FLOAT NOT NULL DEFAULT 1.0
);