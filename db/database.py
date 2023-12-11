# Initialize SQLite database
import math
import sqlite3
from typing import Tuple, Optional

from src.config import BenchmarkResult, UserRank, BenchmarkStatus
from src.validation import Benchmark, BENCHMARK_CONFIG, TBenchmark

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
    # We ensure the username is uppercase, in case the user has circumvented client-side validation
    uppercase_username: str = username.upper()

    # Insert the result into the database
    with sqlite3.connect(BENCHMARKS_RESULT_DB) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO benchmark_results (benchmark_name, username, score)
            VALUES (?, ?, ?)
        """,
            (benchmark.name, uppercase_username, benchmark_result.result),
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
            (benchmark.name,),
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
                (benchmark.name,),
            )
            # max_score could be None (if no entry present) or 0 if only invalid inputs are present
            # When read from the database it is returned as a `Optional[str]`
            max_score: float = int(cursor.fetchone()[0] or 1)
            if max_score == 0:
                max_score = 1

            # We allow for a 5% variance across repeated benchmark runs
            max_score = max_score * 0.95

            benchmark_max_scores[benchmark.name] = max_score

        # Query for user scores for each benchmark
        user_scores: dict[str, dict[str, float]] = {}
        for benchmark in benchmarks:
            cursor.execute(
                """
                SELECT username, max_score FROM top_benchmarks
                WHERE benchmark_name = ?
                """,
                (benchmark.name,),
            )
            for username, score in cursor.fetchall():
                if username not in user_scores:
                    user_scores[username] = {}
                # The scoring system is logarithmic to prevent people with extremely good
                # implementations from completely skewing the grading curve.
                # The max score is 10 and the min score is 0
                scaling_factor: float = 1 / benchmark_max_scores[benchmark.name]
                normalized_score: float = math.log(score * scaling_factor + 1, 2) * 10

                # You cannot achieve more than 10 in scoring
                normalized_score = min(normalized_score, 10.0)

                user_scores[username][benchmark.name] = normalized_score

        # Calculate total score and format results
        results: list[UserRank] = []
        for idx, (username, scores) in enumerate(user_scores.items()):
            average_score = sum(scores.values()) / len(benchmarks)

            results.append(
                UserRank(
                    rank=0,  # In human terms rank starts from 1, not 0
                    username=username,
                    scores={
                        benchmark.name: scores.get(benchmark.name, 0)
                        for benchmark in benchmarks
                    },
                    average=average_score,
                )
            )

        # Sort results based on average score
        results.sort(key=lambda x: x["average"], reverse=True)

        for idx, data in enumerate(results):
            data["rank"] = idx + 1

    return results


def toggle_benchmark_visibility(benchmark_name: str, is_hidden: bool):
    """
    Toggles the visibility of a given benchmark.
    Toggling visibility also side effects frozen status (disabled benchmarks are also frozen)

    :param benchmark_name: Name of benchmark to toggle
    :param is_hidden: Hidden status to apply
    """
    with sqlite3.connect(BENCHMARKS_RESULT_DB) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE benchmarks
            SET is_hidden = ?, is_frozen = ?, is_archive = FALSE
            WHERE name = ?
            """,
            (is_hidden, is_hidden, benchmark_name),
        )
        conn.commit()


def toggle_benchmark_frozen_state(benchmark_name: str, is_frozen: bool):
    with sqlite3.connect(BENCHMARKS_RESULT_DB) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE benchmarks
            SET is_frozen = ?
            WHERE name = ?
            """,
            (is_frozen, benchmark_name),
        )
        conn.commit()


def toggle_benchmark_archive_status(benchmark_name: str, is_archive: bool):
    with sqlite3.connect(BENCHMARKS_RESULT_DB) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE benchmarks
            SET is_archive = ?
            WHERE name = ?
            """,
            (is_archive, benchmark_name),
        )
        conn.commit()


def get_enabled_benchmarks() -> list[TBenchmark]:
    with sqlite3.connect(BENCHMARKS_RESULT_DB) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT name FROM benchmarks
            WHERE NOT is_hidden
            """
        )
        benchmark_names: list[str] = [x for x, in cursor.fetchall()]

        enabled_benchmarks: list[TBenchmark] = [
            benchmark
            for benchmark in BENCHMARK_CONFIG.benchmarks
            if benchmark.name in benchmark_names
        ]
        return enabled_benchmarks


def get_frozen_benchmarks() -> list[TBenchmark]:
    with sqlite3.connect(BENCHMARKS_RESULT_DB) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT name FROM benchmarks
            WHERE is_frozen
            """
        )
        benchmark_names: list[str] = [x for x, in cursor.fetchall()]

        frozen_benchmarks: list[TBenchmark] = [
            benchmark
            for benchmark in BENCHMARK_CONFIG.benchmarks
            if benchmark.name in benchmark_names
        ]
        return frozen_benchmarks


def get_archived_benchmarks() -> list[TBenchmark]:
    with sqlite3.connect(BENCHMARKS_RESULT_DB) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT name FROM benchmarks
            WHERE is_archive
            """
        )
        benchmark_names: list[str] = [x for x, in cursor.fetchall()]

        frozen_benchmarks: list[TBenchmark] = [
            benchmark
            for benchmark in BENCHMARK_CONFIG.benchmarks
            if benchmark.name in benchmark_names
        ]
        return frozen_benchmarks


def get_benchmark_visibility_status() -> list[BenchmarkStatus]:
    """
    Retrieve the visibility status of all available benchmarks

    :return: A list of tuples with the data (benchmark_name, is_hidden)
    """
    with sqlite3.connect(BENCHMARKS_RESULT_DB) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT name, is_hidden, is_frozen, is_archive FROM benchmarks
            ORDER BY id
            """
        )
        return [BenchmarkStatus(*row) for row in cursor.fetchall()]


def upload_benchmark_config(benchmarks: list[Benchmark]):
    """
    Upload the benchmark config to the DB to allow it to persist visibility status.
    By default visibility will be set to False for any newly added benchmarks.
    Any benchmark that no longer exist will be removed from the database as well.

    :param benchmarks: Benchmark config to upload
    """
    with sqlite3.connect(BENCHMARKS_RESULT_DB) as conn:
        cursor = conn.cursor()

        # Retrieve existing benchmarks from the database
        cursor.execute("SELECT name FROM benchmarks")
        existing_benchmarks = {row[0] for row in cursor.fetchall()}

        # Determine benchmarks to add and to remove.
        # NOTE: We care about ordering! Thus keep lists rather than sets!
        config_benchmarks = [benchmark.name for benchmark in benchmarks]
        benchmarks_to_add = [
            b for b in config_benchmarks if b not in existing_benchmarks
        ]
        benchmarks_to_remove = existing_benchmarks - set(config_benchmarks)

        # Insert new benchmarks with visibility set to hidden
        for benchmark_name in benchmarks_to_add:
            cursor.execute(
                """
                INSERT INTO benchmarks (name, is_hidden)
                VALUES (?, TRUE)
                """,
                (benchmark_name,),
            )

        # Remove benchmarks that are no longer in the config
        for benchmark_name in benchmarks_to_remove:
            cursor.execute(
                """
                DELETE FROM benchmarks
                WHERE name = ?
                """,
                (benchmark_name,),
            )

        conn.commit()
