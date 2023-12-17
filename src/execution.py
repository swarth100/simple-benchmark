import copy
import importlib
import io
import multiprocessing
import random
import sys
import time
from functools import singledispatch
from types import ModuleType
from typing import Dict, Optional, Tuple, Any, List, Union, NamedTuple

from faker import Faker
from func_timeout import func_set_timeout

from db.database import get_frozen_benchmarks, get_archived_benchmarks
from src.benchmark.class_benchmark import ClassBenchmark
from src.benchmark.function_benchmark import FunctionBenchmark
from src.config import BenchmarkResult
from src.exceptions import ModuleAccessException
from src.utils import get_reference_benchmark_include, format_args_as_function_call
from src.benchmark.core import (
    Benchmark,
    TBenchmarkArgs,
    TArgsDict,
)
from src.benchmark.config import Config, get_config


class BenchmarkRunInfo(NamedTuple):
    return_value: Any
    std_output: str
    exec_time: float


def capture_output(func, *args, **kwargs) -> BenchmarkRunInfo:
    original_stdout = sys.stdout  # Save a reference to the original standard output
    new_stdout = io.StringIO()
    sys.stdout = new_stdout  # Redirect standard output to the new StringIO object

    try:
        start_time: float = time.perf_counter()
        output = func(*args, **kwargs)
        time_diff: float = time.perf_counter() - start_time
    except Exception:
        # Reraise all exceptions to allow for outer handling
        raise
    finally:
        # Whatever happens, reset standard output to its original value
        sys.stdout = original_stdout

    return BenchmarkRunInfo(output, new_stdout.getvalue().rstrip(), time_diff)


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
            user_result: BenchmarkRunInfo = run_benchmark_with_arguments(
                benchmark, module=target_module, arguments=arg_values
            )
            user_output, user_std_output, time_diff = user_result
        except Exception as e:
            function_call: str = format_args_as_function_call(
                func_name=benchmark.name, args_dict=arg_values
            )
            return BenchmarkResult(
                name=run_name,
                result=last_valid_iteration,
                error=f"Error while executing '{run_name}' for function call:\n{function_call}{e}",
            )

        # Only count the time in user code towards the benchmark, exclude all time spent in validation
        elapsed_time += time_diff

        try:
            ref_result: BenchmarkRunInfo = run_benchmark_with_arguments(
                benchmark, module=ref_module, arguments=arg_values
            )
            (ref_output, ref_std_output, _) = ref_result
        except Exception as e:
            function_call: str = format_args_as_function_call(
                func_name=benchmark.name, args_dict=arg_values
            )
            return BenchmarkResult(
                name=run_name,
                result=last_valid_iteration,
                error=f"Error while executing '{run_name}' in the reference implementation for function call:\n"
                f"{function_call}{e}",
            )

        if user_output != ref_output:
            function_call: str = format_args_as_function_call(
                func_name=benchmark.name, args_dict=arg_values
            )
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
            function_call: str = format_args_as_function_call(
                func_name=benchmark.name, args_dict=arg_values
            )
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


@singledispatch
def run_benchmark_with_arguments(
    benchmark: Benchmark, /, *, module: ModuleType, arguments: TBenchmarkArgs
) -> BenchmarkRunInfo:
    """
    Given a benchmark and a module, runs the benchmark with the provided arguments.
    :param benchmark: Benchmark to run
    :param module: Module to run the benchmark on
    :param arguments: Arguments to run the benchmark with

    :return: BenchmarkRunInfo containing the result of the benchmark
    :raises ModuleAccessException: If the module cannot be accessed or does not exist
    """
    raise NotImplementedError("Unsupported type of benchmark")


@run_benchmark_with_arguments.register(FunctionBenchmark)
def _(
    benchmark: FunctionBenchmark, /, *, module: ModuleType, arguments: TBenchmarkArgs
) -> BenchmarkRunInfo:
    # TODO: Remove/validate assertions
    if len(arguments) != 1:
        raise ValueError(
            "Instances of function benchmark should only be run with a single function's arguments. "
            f"Instead found {len(arguments)} arguments for functions {list(arguments.keys())}."
        )

    # After setting common fields we proceed to executing the function
    try:
        func = getattr(module, benchmark.name)
    except AttributeError:
        raise ModuleAccessException(module=module)

    # Run the reference function with the provided inputs
    valid_kwargs: TBenchmarkArgs = copy.deepcopy(
        benchmark.filter_visible_arguments(arguments)
    )
    func_args: TArgsDict = valid_kwargs[benchmark.name]
    res: BenchmarkRunInfo = capture_output(func, **func_args)

    return res


@run_benchmark_with_arguments.register(ClassBenchmark)
def _(
    benchmark: ClassBenchmark, /, *, module: ModuleType, arguments: TBenchmarkArgs
) -> Tuple[str, str]:
    try:
        klass = getattr(module, benchmark.name)
    except AttributeError:
        raise ModuleAccessException(module=module)

    # Step 0: We must deep-clone the input arguments to ensure that we don't mutate them
    valid_kwargs: TBenchmarkArgs = copy.deepcopy(
        benchmark.filter_visible_arguments(arguments)
    )

    # Step 1 is to construct the object
    init_arguments: TArgsDict = valid_kwargs[benchmark.name]
    obj = klass(**init_arguments)

    # Init arguments are specified via benchmark name, if not present we must raise
    if benchmark.name not in valid_kwargs:
        raise ValueError(
            f"Class '{benchmark.name}' is missing from the supplied arguments with keys {list(arguments.keys())}."
        )

    # Step 2 is to run the methods in the evaluation order specified
    method_evaluation_order = benchmark.generate_method_evaluation_order(init_arguments)
    res: BenchmarkRunInfo = BenchmarkRunInfo("", "", 0)
    for method in method_evaluation_order:
        # We must check if the method has arguments supplied or raise otherwise
        if method.method_name not in valid_kwargs:
            raise ValueError(
                f"Method '{method.method_name}' is missing from the arguments for class '{benchmark.name}'. "
                f"The present keys are {list(valid_kwargs.keys())}."
            )

        func_args: TArgsDict = valid_kwargs[method.method_name]
        valid_func_kwargs: TArgsDict = {
            arg.name: func_args[arg.name] for arg in method.args if not arg.hidden
        }

        method_func = getattr(obj, method.method_name)
        res: BenchmarkRunInfo = capture_output(method_func, **valid_func_kwargs)

    return res


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
