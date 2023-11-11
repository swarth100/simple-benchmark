import importlib
from types import ModuleType
from typing import Optional

from fastapi import FastAPI, Request
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from benchmark import (
    BenchmarkResult,
    run_single_benchmark,
    get_benchmark_by_name,
    get_config,
)
from config_validation import load_config, Benchmark, Config

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Fetch benchmarks from the config
    config = load_config("benchmark_config.yaml")
    benchmark_names = [benchmark.function_name for benchmark in config.benchmarks]

    return templates.TemplateResponse(
        "index.html", {"request": request, "benchmarks": benchmark_names}
    )


@app.post("/run_benchmark")
async def run_user_benchmark(request: Request):
    form_data = await request.form()
    benchmark_name = form_data["benchmark"]
    user_code = form_data["code"]

    result_data: dict[str, str] = {}

    # Create a temporary module for user code
    user_module = ModuleType("user_module")
    try:
        exec(user_code, user_module.__dict__)
    except Exception as e:
        result_data = {"error": f"Error in executing user code: {e}"}
    else:
        benchmark_config: Config = get_config()
        benchmark: Optional[Benchmark] = get_benchmark_by_name(name=benchmark_name)
        if benchmark is None:
            result_data = {
                "error": f"Benchmark with name '{benchmark_name}' does not exist"
            }
        else:
            # Run the benchmark
            benchmark_result: BenchmarkResult = run_single_benchmark(
                user_module=user_module, benchmark=benchmark
            )

            reference_module_name: str = benchmark_config.reference_module
            reference_module = importlib.import_module(reference_module_name)
            reference_result: BenchmarkResult = run_single_benchmark(
                user_module=reference_module, benchmark=benchmark
            )

            result_data = {
                "output": benchmark_result.result,
                "reference": reference_result.result,
            }

            if benchmark_result.has_error:
                result_data["error"] = benchmark_result.error

    return templates.TemplateResponse(
        "benchmark_result.html", {"request": request, "result": result_data}
    )
