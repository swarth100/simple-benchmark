import click
from typing import Optional
import uvicorn

from benchmark import run_benchmark_given_config, get_config
from webserver import app


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
    if serve:
        uvicorn.run(
            "webserver:app", host="0.0.0.0", port=8421, timeout_keep_alive=20, workers=8
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
