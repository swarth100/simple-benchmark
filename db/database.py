# Initialize SQLite database
import math
import sqlite3
from typing import Tuple, Optional

from src.config import BenchmarkResult, UserRank
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
            (benchmark.function_name, username, benchmark_result.result),
        )
        conn.commit()


def get_top_benchmark_results(benchmark: Benchmark) -> list[Tuple[str, ...]]:
    with sqlite3.connect(BENCHMARKS_RESULT_DB) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM top_benchmarks
            WHERE benchmark_name = ?
            ORDER BY max_score DESC
            """,
            (benchmark.function_name,),
        )
        results = cursor.fetchall()
    return results


def get_rankings(benchmarks: list[Benchmark]) -> list[UserRank]:
    """
    Get a list of user rankings based on all available benchmarks.
    Users are scored out of 10, for each benchmark, based on the highest submitted result for each benchmark.

    :param benchmarks: List of benchmarks to compute rankings for
    :return: A list of rows with user rank, username, rankings per-benchmark and total resulting score
    """
    with sqlite3.connect(BENCHMARKS_RESULT_DB) as conn:
        cursor = conn.cursor()

        # Query for the max scores for each benchmark
        benchmark_max_scores: dict[str, float] = {}
        for benchmark in benchmarks:
            cursor.execute(
                """
                SELECT MAX(max_score) FROM top_benchmarks
                WHERE benchmark_name = ?
                """,
                (benchmark.function_name,),
            )
            # max_score could be None (if no entry present) or 0 if only invalid inputs are present
            # When read from the database it is returned as a `Optional[str]`
            max_score: float = int(cursor.fetchone()[0] or 1)
            if max_score == 0:
                max_score = 1

            # We allow for a 5% variance across repeated benchmark runs
            max_score = max_score * 0.95

            benchmark_max_scores[benchmark.function_name] = max_score

        # Query for user scores for each benchmark
        user_scores: dict[str, dict[str, float]] = {}
        for benchmark in benchmarks:
            cursor.execute(
                """
                SELECT username, max_score FROM top_benchmarks
                WHERE benchmark_name = ?
                """,
                (benchmark.function_name,),
            )
            for username, score in cursor.fetchall():
                if username not in user_scores:
                    user_scores[username] = {}
                # The scoring system is logarithmic to prevent people with extremely good
                # implementations from completely skewing the grading curve.
                # The max score is 10 and the min score is 0
                scaling_factor: float = (
                    1 / benchmark_max_scores[benchmark.function_name]
                )
                normalized_score: float = math.log(score * scaling_factor + 1, 2) * 10

                # You cannot achieve more than 10 in scoring
                normalized_score = min(normalized_score, 10.0)

                user_scores[username][benchmark.function_name] = normalized_score

        # Calculate total score and format results
        results: list[UserRank] = []
        for idx, (username, scores) in enumerate(user_scores.items()):
            total_score = sum(scores.values())

            results.append(
                UserRank(
                    rank=0,  # In human terms rank starts from 1, not 0
                    username=username,
                    scores={
                        benchmark.function_name: scores.get(benchmark.function_name, 0)
                        for benchmark in benchmarks
                    },
                    total=total_score,
                )
            )

        # Sort results based on total score
        results.sort(key=lambda x: x["total"], reverse=True)

        for idx, data in enumerate(results):
            data["rank"] = idx + 1

    return results
