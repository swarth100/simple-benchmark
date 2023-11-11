import io
import time
import importlib
import sys
import click
from typing import Dict, Tuple, Any, Optional
import uvicorn
from webserver import app

from config_validation import Config, load_config, TArg


def capture_output(func, *args, **kwargs) -> Tuple[Any, str]:
    original_stdout = sys.stdout  # Save a reference to the original standard output
    new_stdout = io.StringIO()
    sys.stdout = new_stdout  # Redirect standard output to the new StringIO object

    try:
        output = func(*args, **kwargs)
    except Exception:
        # Reraise all exceptions to allow for outer handling
        raise
    finally:
        # Whatever happens, reset standard output to its original value
        sys.stdout = original_stdout

    return output, new_stdout.getvalue().rstrip()


def run_benchmark(benchmark_config: Config):
    print("STARTING BENCHMARK")
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    try:
        ref_module = importlib.import_module(benchmark_config.reference_module)
    except ImportError:
        print(
            f"Error: Reference module '{benchmark_config.reference_module}' not found."
        )
        sys.exit(1)

    for benchmark in benchmark_config.benchmarks:
        print(">>> " + benchmark.function_name)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

        results: Dict[str, int] = {}
        errors: Dict[str, str] = {}
        try:
            ref_func = getattr(ref_module, benchmark.function_name)
        except AttributeError:
            print(
                f"Error: Function '{benchmark.function_name}' not found in "
                f"reference module '{benchmark_config.reference_module}'."
            )
            continue

        for user_module_name in [
            *benchmark_config.user_modules,
            benchmark_config.reference_module,
        ]:
            run_name: str = f"{user_module_name}.{benchmark.function_name}"

            try:
                user_module = importlib.import_module(user_module_name)
                user_func = getattr(user_module, benchmark.function_name)
            except ImportError:
                print(f"Error: User module '{user_module_name}' not found.")
                continue
            except AttributeError:
                errors[run_name] = (
                    f"Error: Function '{benchmark.function_name}' not found "
                    f"in user module '{user_module_name}'."
                )
                continue

            max_time = benchmark.max_time
            elapsed_time: float = 0
            last_valid_iteration = 0
            arg_values: Dict[str, TArg] = {
                arg.name: arg.default for arg in benchmark.args
            }

            while elapsed_time < max_time:
                start_time: float = time.perf_counter()
                try:
                    user_output, user_std_output = capture_output(
                        user_func, **arg_values
                    )
                except Exception as e:
                    errors[run_name] = f"Error while executing '{run_name}': {e}"
                    break

                # Only count the time in user code towards the benchmark, exclude all time spent in validation
                elapsed_time += time.perf_counter() - start_time

                ref_output, ref_std_output = capture_output(ref_func, **arg_values)

                if user_output != ref_output:
                    errors[run_name] = (
                        f"Mismatch in function output for '{run_name}' "
                        f"for arguments {arg_values}.\n"
                        f"Expected:\n{ref_output}\n"
                        f"Got:\n{user_output}\n"
                    )

                    break

                if user_std_output != ref_std_output:
                    errors[run_name] = (
                        f"Mismatch in print-statement output for '{run_name}' "
                        f"for arguments {arg_values}.\n"
                        f"Expected:\n{ref_std_output}\n"
                        f"Got:\n{user_std_output}\n"
                    )
                    break

                last_valid_iteration += 1

                # Increment arguments
                for arg in benchmark.args:
                    if callable(arg.increment):
                        arg_values[arg.name] = arg.increment(arg_values[arg.name])

            results[run_name] = last_valid_iteration

        # Sort the results to always list the best result first
        sorted_results = dict(
            sorted(results.items(), key=lambda item: item[1], reverse=True)
        )

        for idx, (team, result) in enumerate(sorted_results.items()):
            print(f"({idx}) [{team}]: {result}")

        print("\n❌ Execution Errors:")
        for team, error_str in errors.items():
            print(f"\n[{team}]: {error_str}")

        print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    print("\nBENCHMARK COMPLETE")


# -------------------------------------------------------------------------------------------------------------------- #
# Main benchmark running code below:
# -------------------------------------------------------------------------------------------------------------------- #


@click.command()
@click.option(
    "--benchmark",
    "-b",
    multiple=True,
    help="Specify benchmarks to run. Can be used multiple times.",
)
@click.option("--serve", is_flag=True, help="Serve the FastAPI application.")
def main(benchmark: Optional[list[str]] = None, serve: bool = False):
    if serve:
        uvicorn.run(app, host="0.0.0.0", port=8421)
    else:
        config = load_config("benchmark_config.yaml")

        # Filter benchmarks if specific ones are provided
        if benchmark:
            config.benchmarks = [
                b for b in config.benchmarks if b.function_name in benchmark
            ]

        run_benchmark(config)


if __name__ == "__main__":
    main()
