import importlib
import os
import subprocess

import click
from typing import Optional
import uvicorn
from better_profanity import profanity

from src.execution import run_benchmark_given_config
from db.database import init_db, upload_benchmark_config
from src.benchmark.config import BENCHMARK_CONFIG, get_config

# Check if the .profanity-filter file exists and if it does we use its contents
# as an additional dictionary of possible profane words.
# Given we accept user usernames we wish to filter them for not-allowed content.
if os.path.exists(".profanity-filter"):
    with open(".profanity-filter", "r") as file:
        custom_profanity_list = file.readlines()
        # Clean up the list (remove newlines, etc.)
        custom_profanity_list = [word.strip() for word in custom_profanity_list]
        profanity.load_censor_words(custom_profanity_list)


@click.command()
@click.option(
    "--benchmark",
    "-b",
    multiple=True,
    help="Specify benchmarks to run. Can be used multiple times.",
)
@click.option(
    "--serve",
    is_flag=True,
    help="Serve the benchmark runner as a webserver on port 8421.",
)
@click.option(
    "--processes",
    default=8,
    type=int,
    help="Number of processes for the Gunicorn server.",
)
@click.option(
    "--threads",
    default=4,
    type=int,
    help="Number of threads per gunicorn worker.",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Run the app in debug mode (in-process).",
)
def main(
    benchmark: Optional[list[str]] = None,
    serve: bool = False,
    processes: int = 8,
    threads: int = 4,
    debug: bool = False,
):
    # Always initialise the database on startup
    init_db()
    # Upload the latest config to the database to ensure it's in-sync
    upload_benchmark_config(BENCHMARK_CONFIG.benchmarks)

    if serve:
        if debug:
            # Run FastAPI app in-process (useful for debugging)
            app = importlib.import_module("src.webserver").app
            uvicorn.run(
                "src.webserver:app",
                host="0.0.0.0",
                port=8421,
                timeout_keep_alive=10,
                workers=1,
            )
        else:
            # Run via Gunicorn with specified number of workers
            subprocess.run(
                [
                    "gunicorn",
                    "-w",
                    str(processes),  # Number of worker processes
                    "-k",
                    "uvicorn.workers.UvicornWorker",
                    "--threads",
                    str(threads),  # Number of threads per worker
                    "src.webserver:app",
                    "--bind",
                    "0.0.0.0:8421",
                    "--access-logfile",
                    "-",
                ]
            )
    else:
        config = get_config()

        # Filter benchmarks if specific ones are provided.
        # Mutating in-place is accepted given this is a terminal state
        if benchmark:
            config.benchmarks = [
                b
                for b in config.get_all_valid_benchmarks()
                if b.function_name in benchmark
            ]

        run_benchmark_given_config(config)


if __name__ == "__main__":
    main()
