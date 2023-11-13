import importlib
import json
import tempfile
from typing import Optional, Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from db.database import save_benchmark_result, get_top_benchmark_results
from src.benchmark import (
    run_benchmark_given_modules,
    get_benchmark_by_name,
    get_config,
    capture_output,
)
from src.config import BenchmarkResult
from src.validation import BENCHMARK_CONFIG, Benchmark, Config

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Fetch benchmarks from the config
    config = BENCHMARK_CONFIG
    benchmark_names = [benchmark.function_name for benchmark in config.benchmarks]
    benchmarks_with_args = {
        benchmark.function_name: {arg.name: arg.default for arg in benchmark.args}
        for benchmark in config.benchmarks
    }

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "benchmarks": benchmark_names,
            "benchmarks_with_args": benchmarks_with_args,
        },
    )


@app.get("/fetch_leaderboard")
async def update_leaderboard(request: Request, benchmark: str, username: str):
    benchmark: Optional[Benchmark] = get_benchmark_by_name(name=benchmark)
    if benchmark is not None:
        # Fetch the updated leaderboard data based on the benchmark
        leaderboard_data = get_top_benchmark_results(benchmark=benchmark)

        # Render the leaderboard data into HTML (you can use Jinja2 here)
        return templates.TemplateResponse(
            "leaderboard_partial.html",
            {"request": request, "leaderboard": leaderboard_data, "username": username},
        )


@app.post("/sandbox")
async def run_sandbox(request: Request):
    form_data = await request.form()
    benchmark_name = form_data["benchmark"]
    user_inputs = form_data["sandbox-inputs"]

    result_data = {}

    try:
        benchmark: Optional[Benchmark] = get_benchmark_by_name(name=benchmark_name)
        if benchmark is None:
            result_data = {"error": f"Benchmark '{benchmark_name}' does not exist"}
        else:
            reference_module_name: str = get_config().reference_module
            reference_module = importlib.import_module(reference_module_name)
            reference_func = getattr(reference_module, benchmark.function_name)

            # Assume inputs are JSON and need to be converted to Python dict
            inputs_dict = json.loads(user_inputs)

            # Run the reference function with the provided inputs
            ref_output, ref_std_output = capture_output(reference_func, **inputs_dict)
            if ref_output is not None:
                result_data["output"] = ref_output
            if ref_std_output is not None:
                result_data["std_output"] = ref_std_output

    except Exception as e:
        result_data = {"error": f"Error while running sandbox: {e}"}

    return templates.TemplateResponse(
        "sandbox_result.html", {"request": request, "result": result_data}
    )


@app.post("/run_benchmark")
async def run_user_benchmark(request: Request):
    form_data = await request.form()
    benchmark_name = form_data["benchmark"]
    user_code = form_data["code"]
    username = form_data["username"]

    result_data: dict[str, Any] = {}

    # Create a temporary module for user code
    with tempfile.NamedTemporaryFile(suffix=".py") as temp:
        user_module: str = temp.name
        temp.write(user_code.encode())
        temp.flush()

        benchmark_config: Config = get_config()
        benchmark: Optional[Benchmark] = get_benchmark_by_name(name=benchmark_name)
        if benchmark is None:
            result_data = {
                "error": f"Benchmark with name '{benchmark_name}' does not exist"
            }
        else:
            # Run the benchmark
            benchmark_results: list[BenchmarkResult] = run_benchmark_given_modules(
                target_modules=[user_module, benchmark_config.reference_module],
                benchmark=benchmark,
            )

            # Given we only submit two benchmarks we can assume we get an output for each
            benchmark_result = [r for r in benchmark_results if not r.is_reference][0]
            reference_result = [r for r in benchmark_results if r.is_reference][0]

            # Persist the result to the database for subsequent collection.
            # We purposefully decide not to store cases where the user has not set a username.
            if (username is not None) and (len(username) > 0):
                save_benchmark_result(
                    benchmark=benchmark,
                    username=username,
                    benchmark_result=benchmark_result,
                )

            result_data = {"output": benchmark_result.result}

            if benchmark_result.has_error:
                result_data["error"] = benchmark_result.error
            else:
                result_data["reference"] = str(reference_result.result)

            x_cap: int = 500
            y_cap: int = 300
            n_iterations: int = len(benchmark_result.details)

            # To simplify plotting we only plot charts with at least 10 datapoints
            if n_iterations > 10:
                max_x = max(x for (x, y) in benchmark_result.details)
                max_y = max(y for (x, y) in benchmark_result.details)

                x_scaling: float = x_cap / max_x
                x_index_scaling: float = n_iterations / 10
                y_scaling: float = y_cap / max_y
                y_index_scaling: float = n_iterations / 10

                result_data["chartData"] = [
                    {"x": x * x_scaling, "y": y * y_scaling}
                    for x, y in benchmark_result.details
                ]

                tick_ranges = range(0, 10)

                x_ticks_positions = [50 * x for x in tick_ranges]
                x_ticks_labels = [
                    benchmark_result.details[int(x * x_index_scaling)][0]
                    for x in tick_ranges
                ]
                result_data["xTicks"] = list(zip(x_ticks_positions, x_ticks_labels))

                y_ticks_positions = [30 * y for y in tick_ranges]
                y_ticks_labels = [
                    "{:.4f}".format(
                        benchmark_result.details[int(y * y_index_scaling)][1]
                    )
                    for y in tick_ranges
                ]
                result_data["yTicks"] = list(zip(y_ticks_positions, y_ticks_labels))

    # Fetch top benchmark results for the current benchmark
    top_results = get_top_benchmark_results(benchmark=benchmark)
    result_data["topResults"] = top_results

    return templates.TemplateResponse(
        "benchmark_result.html",
        {"request": request, "result": result_data, "current_user": username},
    )
