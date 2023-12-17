import importlib
import importlib
import multiprocessing
import random
from types import ModuleType
from typing import Dict, Optional, Tuple, List, Union

from faker import Faker
from func_timeout import func_set_timeout

from db.database import get_frozen_benchmarks, get_archived_benchmarks
from src.benchmark.config import Config, get_config
from src.benchmark.core import (
    Benchmark,
    TBenchmarkArgs,
)
from src.config import BenchmarkResult, BenchmarkRunInfo
from src.utils import (
    get_reference_benchmark_include,
    format_args_as_function_call,
)


def get_benchmark_by_name(
    name: str, *, include_archived: bool = False
) -> Optional[Benchmark]:
    benchmark_config: Config = get_config()

    for benchmark in benchmark_config.get_all_valid_benchmarks(
        include_archived=include_archived
    ):
        if benchmark.name == name:
            return benchmark
    return None


def is_benchmark_frozen(name: str) -> bool:
    for benchmark in get_frozen_benchmarks():
        if benchmark.name == name:
            return True
    return False


def is_benchmark_archived(name: str) -> bool:
    for benchmark in get_archived_benchmarks():
        if benchmark.name == name:
            return True
    return False


def _run_single_benchmark(
    target_module: ModuleType, benchmark: Benchmark
) -> BenchmarkResult:
    run_name: str = f"{target_module.__name__}.{benchmark.name}"
    ref_module: ModuleType = get_config().reference_module_object

    max_time: float = benchmark.max_time_seconds
    elapsed_time: float = 0
    last_valid_iteration = 0

    arg_values: TBenchmarkArgs = benchmark.default_args

    execution_details: List[Tuple[int, float]] = []

    # We seed random once to ensure that output will be consistent.
    # This applies BEFORE each benchmark execution.
    # DO NOT REMOVE, or different benchmark runs might use different argument values.
    random.seed(42)
    # We also seed faker which is used to generate random user-meaningful data
    Faker.seed(42)

    while elapsed_time < max_time:
        try:
            user_result: BenchmarkRunInfo = benchmark.run_with_arguments(
                module=target_module, arguments=arg_values
            )
            user_output, user_std_output, time_diff = user_result
        except Exception as e:
            function_call: str = benchmark.generate_python_call(arguments=arg_values)
            return BenchmarkResult(
                name=run_name,
                result=last_valid_iteration,
                error=f"Error while executing '{run_name}' for function call:\n{function_call}{e}",
            )

        # Only count the time in user code towards the benchmark, exclude all time spent in validation
        elapsed_time += time_diff

        try:
            ref_result: BenchmarkRunInfo = benchmark.run_with_arguments(
                module=ref_module, arguments=arg_values
            )
            (ref_output, ref_std_output, _) = ref_result
        except Exception as e:
            function_call: str = benchmark.generate_python_call(arguments=arg_values)
            return BenchmarkResult(
                name=run_name,
                result=last_valid_iteration,
                error=f"Error while executing '{run_name}' in the reference implementation for function call:\n"
                f"{function_call}{e}",
            )

        if user_output != ref_output:
            function_call: str = benchmark.generate_python_call(arguments=arg_values)
            return BenchmarkResult(
                name=run_name,
                result=last_valid_iteration,
                error=(
                    f"Mismatch in function output for '{run_name}' "
                    f"for function call:\n{function_call}"
                    f"Expected:\n{ref_output}\n"
                    f"Got:\n{user_output}\n"
                ),
                details=execution_details,
            )

        if user_std_output != ref_std_output:
            function_call: str = benchmark.generate_python_call(arguments=arg_values)
            return BenchmarkResult(
                name=run_name,
                result=last_valid_iteration,
                error=(
                    f"Mismatch in print-statement output for '{run_name}' "
                    f"for function call:\n{function_call}"
                    f"Expected:\n{ref_std_output}\n"
                    f"Got:\n{user_std_output}\n"
                ),
                details=execution_details,
            )

        execution_details.append((last_valid_iteration, time_diff))
        last_valid_iteration += 1

        # Increment arguments in-place in benchmark-specific ways
        benchmark.increment_args(arg_values)

    return BenchmarkResult(
        name=run_name, result=last_valid_iteration, details=execution_details
    )


def _run_single_benchmark_by_module(
    target_module: Union[str, ModuleType], benchmark: Benchmark
) -> BenchmarkResult:
    # Define the extra objects to be injected
    extra_objects = {
        include_object_name: get_reference_benchmark_include(include_object_name)
        for include_object_name in benchmark.include
    }

    try:
        if isinstance(target_module, str):
            if not target_module.startswith("/tmp"):
                # Import the module normally using the custom importer
                target_module = importlib.import_module(target_module)
            else:
                # In cases where we accept user code we write the module to a tempfile
                # This code needs to be imported specially
                spec = importlib.util.spec_from_file_location(
                    "custom_module", target_module
                )
                target_module = importlib.util.module_from_spec(spec)
                target_module.__dict__.update(extra_objects)
                spec.loader.exec_module(target_module)
    except ImportError:
        benchmark_result: BenchmarkResult = BenchmarkResult(
            name=str(target_module),
            error=f"Error: Target module '{target_module}' not found.",
        )
    except Exception as e:
        target_module_name: str = (
            target_module if isinstance(target_module, str) else target_module.__name__
        )
        benchmark_result: BenchmarkResult = BenchmarkResult(
            name=target_module_name,
            error=f"Error: Target module '{target_module_name}' raised an error when imported:\n{e}",
        )
    else:
        benchmark_result: BenchmarkResult = _run_single_benchmark(
            target_module=target_module, benchmark=benchmark
        )

    return benchmark_result


@func_set_timeout(15)
def run_benchmark_given_modules(
    target_modules: List[Union[ModuleType, str]], benchmark: Benchmark
) -> List[BenchmarkResult]:
    """
    Execution of a single benchmark in a separate process.
    Process-level isolation prevents concurrently bottlenecks on parallel server-side execution.
    We cannot allow unbound unlimited execution and must hard-terminate long-lived calls.

    :raises FunctionTimedOut: on timeout (i.e. excessively long execution)
    """

    task_arguments: List[Tuple[Union[ModuleType, str], Benchmark]] = [
        (target_module, benchmark) for target_module in target_modules
    ]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        benchmark_results: List[BenchmarkResult] = pool.starmap(
            _run_single_benchmark_by_module, task_arguments
        )

    return benchmark_results


def run_benchmark_given_config(benchmark_config: Config):
    print("STARTING BENCHMARK")
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    for benchmark in benchmark_config.get_all_valid_benchmarks():
        print(">>> " + benchmark.name)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

        results: Dict[str, int] = {}
        errors: Dict[str, str] = {}
        task_arguments: List[Tuple[str, Benchmark]] = []

        for target_module_name in [
            *benchmark_config.user_modules,
            benchmark_config.reference_module,
        ]:
            task_arguments.append((target_module_name, benchmark))

        # We execute in parallel to ensure:
        # 1) Process-level isolation on benchmark results
        # 2) Overall faster evaluation
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            benchmark_results: List[BenchmarkResult] = pool.starmap(
                _run_single_benchmark_by_module, task_arguments
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
