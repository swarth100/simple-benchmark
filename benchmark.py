import importlib
import io
import multiprocessing
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType
from typing import Dict, Callable, Optional, Tuple, Any, List

from config_validation import Config, TArg, load_config, Benchmark


@dataclass
class BenchmarkResult:
    name: str
    result: int = 0
    error: Optional[str] = None

    def __post_init__(self):
        if self.result is None and self.error is None:
            raise RuntimeError("Either result or error must not be None")

    @property
    def has_error(self) -> bool:
        return self.error is not None


@lru_cache
def get_config() -> Config:
    benchmark_config: Config = load_config("benchmark_config.yaml")
    return benchmark_config


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


@lru_cache
def get_reference_benchmark_function(function_name: str) -> Callable:
    benchmark_config = get_config()
    try:
        ref_module = importlib.import_module(benchmark_config.reference_module)
    except ImportError:
        print(
            f"Error: Reference module '{benchmark_config.reference_module}' not found."
        )
        sys.exit(1)

    try:
        ref_func = getattr(ref_module, function_name)
    except AttributeError:
        raise AttributeError(
            f"Error: Function '{function_name}' not found in "
            f"reference module '{ref_module.__name__}'.\n"
            f"Benchmarks cannot be run without a reference implementation!"
        )

    return ref_func


def get_benchmark_by_name(name: str) -> Optional[Benchmark]:
    benchmark_config: Config = get_config()

    for benchmark in benchmark_config.benchmarks:
        if benchmark.function_name == name:
            return benchmark
    return None


def _run_single_benchmark(
    user_module: ModuleType, benchmark: Benchmark
) -> BenchmarkResult:
    user_module_name: str = user_module.__name__
    ref_func: Callable = get_reference_benchmark_function(benchmark.function_name)
    run_name: str = f"{user_module_name}.{benchmark.function_name}"

    try:
        user_func = getattr(user_module, benchmark.function_name)
    except AttributeError:
        return BenchmarkResult(
            name=run_name,
            error=(
                f"Error: Function '{benchmark.function_name}' not found "
                f"in user module '{user_module_name}'."
            ),
        )

    max_time = benchmark.max_time
    elapsed_time: float = 0
    last_valid_iteration = 0
    arg_values: Dict[str, TArg] = {arg.name: arg.default for arg in benchmark.args}

    while elapsed_time < max_time:
        start_time: float = time.perf_counter()
        try:
            user_output, user_std_output = capture_output(user_func, **arg_values)
        except Exception as e:
            return BenchmarkResult(
                name=run_name, error=f"Error while executing '{run_name}': {e}"
            )

        # Only count the time in user code towards the benchmark, exclude all time spent in validation
        elapsed_time += time.perf_counter() - start_time

        ref_output, ref_std_output = capture_output(ref_func, **arg_values)

        if user_output != ref_output:
            return BenchmarkResult(
                name=run_name,
                result=last_valid_iteration,
                error=(
                    f"Mismatch in function output for '{run_name}' "
                    f"for arguments {arg_values}.\n"
                    f"Expected:\n{ref_output}\n"
                    f"Got:\n{user_output}\n"
                ),
            )

        if user_std_output != ref_std_output:
            return BenchmarkResult(
                name=run_name,
                result=last_valid_iteration,
                error=(
                    f"Mismatch in print-statement output for '{run_name}' "
                    f"for arguments {arg_values}.\n"
                    f"Expected:\n{ref_std_output}\n"
                    f"Got:\n{user_std_output}\n"
                ),
            )

        last_valid_iteration += 1

        # Increment arguments
        for arg in benchmark.args:
            arg_values[arg.name] = arg.apply_increment(arg_values[arg.name])

    return BenchmarkResult(name=run_name, result=last_valid_iteration)


def _run_single_benchmark_by_module_name(
    user_module_name: str, benchmark: Benchmark
) -> BenchmarkResult:
    try:
        user_module = importlib.import_module(user_module_name)
    except ImportError:
        benchmark_result: BenchmarkResult = BenchmarkResult(
            error=f"Error: User module '{user_module_name}' not found."
        )
    else:
        benchmark_result: BenchmarkResult = _run_single_benchmark(
            user_module=user_module, benchmark=benchmark
        )
    return benchmark_result


def run_single_benchmark(
    user_module: ModuleType, benchmark: Benchmark
) -> BenchmarkResult:
    """
    Execution of a single benchmark in a separate process.
    Process-level isolation prevents concurrently bottlenecks on parallel server-side execution.
    """

    def benchmark_wrapper(queue, user_module, benchmark):
        _result: BenchmarkResult = _run_single_benchmark(user_module, benchmark)
        queue.put(_result)

    queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=benchmark_wrapper, args=(queue, user_module, benchmark)
    )
    p.start()
    p.join()  # Wait for the process to finish

    # Retrieve the result from the queue
    result = queue.get()
    return result


def run_benchmark_given_config(benchmark_config: Config):
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
        task_arguments: List[Tuple[str, Benchmark]] = []

        for user_module_name in [
            *benchmark_config.user_modules,
            benchmark_config.reference_module,
        ]:
            task_arguments.append((user_module_name, benchmark))

        # We execute in parallel to ensure:
        # 1) Process-level isolation on benchmark results
        # 2) Overall faster evaluation
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            benchmark_results: List[BenchmarkResult] = pool.starmap(
                _run_single_benchmark_by_module_name, task_arguments
            )

        for benchmark_result in benchmark_results:
            if benchmark_result.has_error:
                errors[benchmark_result.name] = benchmark_result.error

            results[benchmark_result.name] = benchmark_result.result

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
