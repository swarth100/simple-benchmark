# Initialize SQLite database
import sqlite3
from typing import Tuple

from src.config import BenchmarkResult
from src.validation import Benchmark

MIGRATIONS_FILE = "db/migrations.sql"
BENCHMARKS_RESULT_DB: str = "db/benchmark_results.db"


def init_db():
    conn = sqlite3.connect(BENCHMARKS_RESULT_DB)
    cursor = conn.cursor()

    # We execute all migrations on init to ensure we have a consistent database setup
    # Read all SQL commands from the migrations file
    with open(MIGRATIONS_FILE, "r") as file:
        sql_script = file.read()

    # Split the script into individual SQL statements
    sql_commands = sql_script.split(";")

    # Execute each SQL statement
    for command in sql_commands:
        # Skip any empty statements (which can occur from splitting)
        if command.strip():
            cursor.execute(command)

    conn.commit()
    conn.close()


def save_benchmark_result(
    benchmark: Benchmark, username: str, benchmark_result: BenchmarkResult
):
    # Insert the result into the database
    with sqlite3.connect(BENCHMARKS_RESULT_DB) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO benchmark_results (benchmark_name, username, score)
            VALUES (?, ?, ?)
        """,
            (benchmark.function_name, username, benchmark_result.score),
        )
        conn.commit()


def get_top_benchmark_results(
    benchmark: Benchmark, username: str
) -> list[Tuple[str, ...]]:
    with sqlite3.connect(BENCHMARKS_RESULT_DB) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM top_benchmarks
            WHERE benchmark_name = ? AND username = ?
            ORDER BY max_score DESC
            """,
            (
                benchmark.function_name,
                username,
            ),
        )
        results = cursor.fetchall()
    return results
