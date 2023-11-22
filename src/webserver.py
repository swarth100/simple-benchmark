import importlib
import json
import tempfile
import traceback
from random import randint
from typing import Optional, Any

from better_profanity import profanity
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from func_timeout import FunctionTimedOut
from starlette.responses import FileResponse

from db.database import (
    save_benchmark_result,
    get_top_benchmark_results,
    get_rankings,
    get_benchmark_visibility_status,
    toggle_benchmark_visibility,
    toggle_benchmark_frozen_state,
    get_frozen_benchmarks,
)
from src.benchmark import (
    run_benchmark_given_modules,
    get_benchmark_by_name,
    get_config,
    capture_output,
    is_benchmark_frozen,
    run_reference_benchmark_with_arguments,
)
from src.config import BenchmarkResult, UserRank
from src.validation import BENCHMARK_CONFIG, Benchmark, Config, TArg

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/static/{file_path:path}")
async def static(file_path: str):
    # Allows for CSS updates to instantly be reflected on clients.
    # Static files may otherwise be cached by browsers.
    response = FileResponse("static/" + file_path)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Fetch benchmarks from the config
    config = BENCHMARK_CONFIG
    benchmark_names = [
        benchmark.function_name for benchmark in config.get_all_valid_benchmarks()
    ]
    benchmarks_with_args = {
        benchmark.function_name: benchmark.default_args
        for benchmark in config.get_all_valid_benchmarks()
    }

    benchmark_signatures = {
        benchmark.function_name: benchmark.generate_function_signature()
        for benchmark in config.get_all_valid_benchmarks()
    }

    frozen_benchmarks: list[str] = [
        benchmark.function_name for benchmark in get_frozen_benchmarks()
    ]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "benchmarks": benchmark_names,
            "benchmark_signatures": benchmark_signatures,
            "benchmarks_with_args": benchmarks_with_args,
            "frozen_benchmarks": frozen_benchmarks,
        },
    )


@app.get("/fetch_leaderboard")
async def update_leaderboard(request: Request, benchmark: str, username: str):
    try:
        benchmark: Optional[Benchmark] = get_benchmark_by_name(name=benchmark)
        if benchmark is not None:
            # Fetch the updated leaderboard data based on the benchmark
            leaderboard_data = get_top_benchmark_results(benchmark=benchmark)

            # Render the leaderboard data into HTML (you can use Jinja2 here)
            return templates.TemplateResponse(
                "leaderboard_partial.html",
                {
                    "request": request,
                    "leaderboard": leaderboard_data,
                    "username": username,
                },
            )
    except Exception as e:
        # Handle errors gracefully
        print(traceback.format_exc())
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "message": f"Error while fetching rankings: {e}"},
        )


@app.get("/fetch_rankings")
async def fetch_rankings(request: Request, username: str):
    try:
        rankings_data: list[UserRank] = get_rankings(
            benchmarks=BENCHMARK_CONFIG.get_all_valid_benchmarks()
        )

        benchmark_names: list[str] = [
            benchmark.function_name
            for benchmark in BENCHMARK_CONFIG.get_all_valid_benchmarks()
        ]

        # Render the rankings data into HTML using Jinja2
        return templates.TemplateResponse(
            "rankings_partial.html",
            {
                "request": request,
                "rankings": rankings_data,
                "current_user": username,
                "benchmarks": benchmark_names,
            },
        )

    except Exception as e:
        # Handle errors gracefully
        print(traceback.format_exc())
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "message": f"Error while fetching rankings: {e}"},
        )


@app.post("/run_sandbox")
async def run_sandbox(request: Request):
    try:
        form_data = await request.form()
        benchmark_name = form_data["benchmark"]
        user_inputs = form_data["sandbox-inputs"]

        result_data = {}

        try:
            benchmark: Optional[Benchmark] = get_benchmark_by_name(name=benchmark_name)

            if benchmark is None:
                result_data = {"error": f"Benchmark '{benchmark_name}' does not exist"}
            else:
                # Assume inputs are JSON and need to be converted to Python dict
                inputs_dict = json.loads(user_inputs)

                result_data["input"] = inputs_dict
                result_data["signature"] = benchmark.generate_function_signature()

                # Run the reference function with the provided inputs
                ref_output, ref_std_output = run_reference_benchmark_with_arguments(
                    benchmark=benchmark, arguments=inputs_dict
                )
                if ref_output is not None:
                    result_data["output"] = ref_output
                if (ref_std_output is not None) and (ref_std_output != ""):
                    result_data["std_output"] = ref_std_output

        except Exception as e:
            print(traceback.format_exc())
            result_data["error"] = f"Error while running sandbox: {e}"

        return templates.TemplateResponse(
            "sandbox_partial.html", {"request": request, "result": result_data}
        )
    except Exception as e:
        # Handle errors gracefully
        print(traceback.format_exc())
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "message": f"Error while running sandbox: {e}"},
        )


@app.get("/randomize_args")
async def update_leaderboard(request: Request, benchmark: str) -> dict:
    try:
        benchmark: Optional[Benchmark] = get_benchmark_by_name(name=benchmark)
        arg_values: dict[str, TArg] = {
            arg.name: arg.default_value for arg in benchmark.args
        }

        # We give a sense of randomization but only limit the extent of the range.
        # Too-large of a range and the input would not display well.
        # Random is not seeded so will still yield interesting results.
        for _ in range(randint(1, 5)):
            for arg in benchmark.args:
                arg_values[arg.name] = arg.apply_increment(
                    arg_values[arg.name], **arg_values
                )
        return {
            arg.name: arg_values[arg.name] for arg in benchmark.args if not arg.hidden
        }

    except Exception as e:
        # Handle errors gracefully
        print(traceback.format_exc())
        return {}


@app.get("/fetch_benchmark_details")
async def fetch_benchmark_details(request: Request, benchmark: str):
    try:
        benchmark: Optional[Benchmark] = get_benchmark_by_name(benchmark)

        if benchmark is None:
            raise KeyError(f"Benchmark with name '{benchmark}' is invalid")

        description = "TODO"

        example_input: dict[str, TArg] = benchmark.default_args

        example_output, example_std_output = run_reference_benchmark_with_arguments(
            benchmark=benchmark, arguments=example_input
        )
        result_data: dict[str, str] = dict()
        if example_output is not None:
            result_data["example_output"] = example_output
        if (example_std_output is not None) and (example_std_output != ""):
            result_data["example_std_output"] = example_std_output

        return templates.TemplateResponse(
            "benchmark_details_partial.html",  # Create this new template
            {
                "request": request,
                "description": description,
                "example_input": example_input,
                "result_data": result_data,
            },
        )
    except Exception as e:
        print(traceback.format_exc())
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "message": f"Error while fetching benchmark details: {e}",
            },
        )


@app.post("/run_benchmark")
async def run_user_benchmark(request: Request):
    try:
        form_data = await request.form()
        benchmark_name = form_data["benchmark"]
        user_code = form_data["code"]
        username = form_data["username"]

        # Strip all whitespace: it's NOT allowed
        username = username.strip()

        # Ensure we validate the username against profanity before persisting
        username = profanity.censor(username, "_")

        result_data: dict[str, Any] = {}

        # For frozen benchmarks we return as early as possible to prevent wasted compute.
        # All forms of submission are banned when frozen.
        if is_benchmark_frozen(benchmark_name):
            return templates.TemplateResponse("frozen.html", {"request": request})

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
                benchmark_result = [r for r in benchmark_results if not r.is_reference][
                    0
                ]
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
    except FunctionTimedOut as e:
        # Handle timeout errors gracefully
        print(traceback.format_exc())
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "message": "The benchmark timed out and was terminated.\n"
                "This is likely due to an infinite loop, "
                "an `input()` function or some other code which never terminates.\n"
                "Please refactor your code and submit again.",
            },
        )
    except Exception as e:
        # Handle errors gracefully
        print(traceback.format_exc())
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "message": f"Error while running benchmark: {e}"},
        )


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    benchmark_status: list[tuple] = get_benchmark_visibility_status()
    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "benchmark_status": benchmark_status,
        },
    )


@app.post("/toggle_benchmark_visibility")
async def toggle_benchmark(request: Request):
    form_data = await request.form()
    benchmark_name = form_data["benchmark"]
    is_hidden = form_data["is_hidden"] == "true"
    toggle_benchmark_visibility(benchmark_name, is_hidden)
    return {"success": True}


@app.post("/toggle_benchmark_frozen_state")
async def toggle_benchmark_frozen(request: Request):
    form_data = await request.form()
    benchmark_name = form_data["benchmark"]
    is_frozen = form_data["is_frozen"] == "true"
    toggle_benchmark_frozen_state(benchmark_name, is_frozen)
    return {"success": True}
