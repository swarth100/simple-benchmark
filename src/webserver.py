import importlib
import inspect
import json
import tempfile
import traceback
from random import randint
from types import ModuleType
from typing import Optional, Any

import markdown
from better_profanity import profanity
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from func_timeout import FunctionTimedOut
from pydantic import BaseModel, parse_obj_as
from starlette.responses import FileResponse

from db.database import (
    save_benchmark_result,
    get_top_benchmark_results,
    get_rankings,
    get_benchmark_visibility_status,
    toggle_benchmark_visibility,
    toggle_benchmark_frozen_state,
    get_frozen_benchmarks,
    toggle_benchmark_archive_status,
    get_archived_benchmarks,
)
from src.execution import (
    run_benchmark_given_modules,
    get_benchmark_by_name,
    is_benchmark_frozen,
    is_benchmark_archived,
)
from src.config import BenchmarkResult, UserRank, BenchmarkStatus, BenchmarkRunInfo
from src.utils import get_annotations, format_args_as_function_call
from src.benchmark.core import (
    Benchmark,
    TBenchmarkArgs,
)
from src.benchmark.config import Config, BENCHMARK_CONFIG, get_config


class PydanticEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BaseModel):
            return obj.dict()
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


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

    # We sort by difficulty to give an ascending difficulty list to users
    sorted_benchmarks = sorted(
        config.get_all_valid_benchmarks(), key=lambda b: b.difficulty
    )
    benchmark_names = [benchmark.name for benchmark in sorted_benchmarks]

    benchmarks_with_args = {
        benchmark.name: json.dumps(benchmark.example_args, cls=PydanticEncoder)
        for benchmark in config.get_all_valid_benchmarks()
    }

    benchmark_signatures = {
        benchmark.name: benchmark.generate_signature()
        for benchmark in config.get_all_valid_benchmarks(include_archived=True)
    }

    benchmark_includes = {
        benchmark.name: benchmark.generate_include_code()
        for benchmark in config.get_all_valid_benchmarks(include_archived=True)
    }

    benchmark_icons: dict[str, str] = {
        benchmark.name: benchmark.icon_unicode
        for benchmark in config.get_all_valid_benchmarks(include_archived=True)
    }

    frozen_benchmarks: list[str] = [
        benchmark.name for benchmark in get_frozen_benchmarks()
    ]

    archived_benchmarks: list[str] = [
        benchmark.name for benchmark in get_archived_benchmarks()
    ]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "benchmarks": benchmark_names,
            "benchmark_signatures": benchmark_signatures,
            "benchmark_includes": benchmark_includes,
            "benchmark_icons": benchmark_icons,
            "benchmarks_with_args": benchmarks_with_args,
            "frozen_benchmarks": frozen_benchmarks,
            "archived_benchmarks": archived_benchmarks,
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
            benchmark.name
            for benchmark in sorted(
                BENCHMARK_CONFIG.get_all_valid_benchmarks(), key=lambda b: b.difficulty
            )
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
                raw_input: dict = json.loads(user_inputs)
                inputs_dict: TBenchmarkArgs = benchmark.parse_arguments_from_dict(
                    raw_arguments=raw_input
                )

                result_data["input"] = benchmark.generate_python_call(
                    arguments=inputs_dict
                )
                result_data["signature"] = benchmark.generate_signature()

                # Sandbox is always run against the reference module
                ref_module: ModuleType = get_config().reference_module_object

                # Run the reference function with the provided inputs
                ref_result: BenchmarkRunInfo = benchmark.run_with_arguments(
                    module=ref_module, arguments=inputs_dict
                )
                (ref_output, ref_std_output, _) = ref_result
                if ref_output is not None:
                    result_data["output"] = repr(ref_output)
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
async def randomize_args(request: Request, benchmark: str) -> str:
    try:
        benchmark: Optional[Benchmark] = get_benchmark_by_name(name=benchmark)
        benchmark_arguments: TBenchmarkArgs = benchmark.default_args

        # TODO: Generalize for Classes
        # We give a sense of randomization but only limit the extent of the range.
        # Too-large of a range and the input would not display well.
        # Random is not seeded so will still yield interesting results.
        for _ in range(randint(1, 7)):
            benchmark.increment_args(benchmark_arguments)

        encoded_arguments: TBenchmarkArgs = benchmark.filter_visible_arguments(
            benchmark_arguments
        )
        return json.dumps(encoded_arguments, cls=PydanticEncoder)

    except Exception as e:
        # Handle errors gracefully
        print(traceback.format_exc())
        return "{}"


@app.get("/fetch_benchmark_details")
async def fetch_benchmark_details(request: Request, benchmark: str):
    try:
        benchmark: Optional[Benchmark] = get_benchmark_by_name(
            benchmark, include_archived=True
        )

        if benchmark is None:
            raise KeyError(f"Benchmark with name '{benchmark}' is invalid")

        description: str = markdown.markdown(
            benchmark.generate_description_md(), extensions=["nl2br"]
        )

        example_input: TBenchmarkArgs = benchmark.example_args
        pretty_printed_example_args: str = benchmark.generate_python_call(
            arguments=example_input
        )

        # Benchmark details are always fetched from the reference module
        ref_module: ModuleType = get_config().reference_module_object

        example_result: BenchmarkRunInfo = benchmark.run_with_arguments(
            module=ref_module, arguments=example_input
        )
        (example_output, example_std_output, _) = example_result

        result_data: dict[str, str] = dict()
        if example_output is not None:
            result_data["example_output"] = repr(example_output)
        if (example_std_output is not None) and (example_std_output != ""):
            result_data["example_std_output"] = example_std_output

        return templates.TemplateResponse(
            "benchmark_details_partial.html",
            {
                "request": request,
                "description": description,
                "example_input": pretty_printed_example_args,
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


@app.get("/fetch_difficulty")
async def fetch_difficulty(request: Request, benchmark: str):
    benchmark: Optional[Benchmark] = get_benchmark_by_name(
        benchmark, include_archived=True
    )
    try:
        if benchmark is None:
            raise KeyError(f"Benchmark with name '{benchmark}' is invalid")

        stars_html = benchmark.generate_difficulty_stars_html()
        return HTMLResponse(content=stars_html, media_type="text/html")
    except Exception as e:
        print(traceback.format_exc())
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "message": f"Error while fetching benchmark difficulty: {e}",
            },
        )


@app.get("/fetch_benchmark_code")
async def fetch_benchmark_code(request: Request, benchmark: str):
    try:
        if not (is_benchmark_archived(benchmark)):
            raise KeyError(f"Benchmark with name '{benchmark}' is not archived")

        reference_module_name: str = get_config().reference_module
        reference_module = importlib.import_module(reference_module_name)
        reference_func = getattr(reference_module, benchmark)
        implementation = inspect.getsource(reference_func)

        return templates.TemplateResponse(
            "archived_partial.html",
            {"request": request, "reference_implementation": implementation},
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

                result_data = {"output": benchmark_result.result}

                # Persist the result to the database for subsequent collection.
                # We purposefully decide not to store cases where the user has not set a username.
                if benchmark_result.result > 10:
                    if (username is not None) and (len(username) > 0):
                        save_benchmark_result(
                            benchmark=benchmark,
                            username=username,
                            benchmark_result=benchmark_result,
                        )
                    else:
                        result_data[
                            "warning"
                        ] = "You must set your username to submit to the leaderboards"
                else:
                    result_data[
                        "warning"
                    ] = "The benchmark result is too low and has not been submitted to the leaderboards"

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
                        "{:.1e}".format(elem)
                        if abs(
                            elem := (
                                benchmark_result.details[int(y * y_index_scaling)][1]
                            )
                        )
                        < 1e-3
                        else "{:.4f}".format(elem)
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
async def admin_page(request: Request, show_archived: bool = False):
    # Create a mapping of benchmark names to their difficulty levels
    benchmark_difficulties: dict[str, float] = {
        benchmark.name: benchmark.difficulty for benchmark in get_config().benchmarks
    }

    # Get the benchmark visibility status
    benchmark_status: list[BenchmarkStatus] = get_benchmark_visibility_status()

    # Filter out archived benchmarks if not showing archived
    if not show_archived:
        benchmark_status = [item for item in benchmark_status if not item.is_archive]

    # Sort benchmark_status by difficulty using the mapping
    benchmark_status.sort(key=lambda b: benchmark_difficulties.get(b.name, 0))

    benchmark_icons: dict[str, str] = {
        benchmark.name: benchmark.icon_unicode for benchmark in get_config().benchmarks
    }

    # Generate difficulty stars HTML for each benchmark
    difficulty_stars: dict[str, str] = {
        benchmark.name: benchmark.generate_difficulty_stars_html()
        for benchmark in BENCHMARK_CONFIG.benchmarks
    }

    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "benchmark_status": benchmark_status,
            "benchmark_icons": benchmark_icons,
            "difficulty_stars": difficulty_stars,
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


@app.post("/toggle_benchmark_archive_state")
async def toggle_benchmark_archive(request: Request):
    form_data = await request.form()
    benchmark_name = form_data["benchmark"]
    is_archive = form_data["is_archive"] == "true"
    toggle_benchmark_archive_status(benchmark_name, is_archive)
    return {"success": True}
