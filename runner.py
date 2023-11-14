import os

import click
from typing import Optional
import uvicorn
from better_profanity import profanity

from src.benchmark import run_benchmark_given_config, get_config
from db.database import init_db


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
def main(benchmark: Optional[list[str]] = None, serve: bool = False):
    # Always initialise the database on startup
    init_db()

    if serve:
        uvicorn.run(
            "src.webserver:app",
            host="0.0.0.0",
            port=8421,
            timeout_keep_alive=20,
            workers=8,
        )
    else:
        config = get_config()

        # Filter benchmarks if specific ones are provided
        if benchmark:
            config.benchmarks = [
                b for b in config.benchmarks if b.function_name in benchmark
            ]

        run_benchmark_given_config(config)


if __name__ == "__main__":
    main()
