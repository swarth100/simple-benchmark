import importlib
from types import ModuleType

from fastapi import FastAPI, Request
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from config_validation import load_config

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

    # Find the reference function and benchmark configuration
    config = load_config("benchmark_config.yaml")
    benchmark = next(
        (b for b in config.benchmarks if b.function_name == benchmark_name), None
    )
    if not benchmark:
        return {"error": f"Benchmark '{benchmark_name}' not found"}

    try:
        ref_module = importlib.import_module(config.reference_module)
        ref_func = getattr(ref_module, benchmark.function_name)
    except ImportError as e:
        return {"error": f"Reference module '{config.reference_module}' not found: {e}"}
    except AttributeError as e:
        return {
            "error": f"Function '{benchmark.function_name}' not found in reference module: {e}"
        }

    # Create a temporary module for user code
    user_module = ModuleType("user_module")
    try:
        exec(user_code, user_module.__dict__)
    except Exception as e:
        return {"error": f"Error in executing user code: {e}"}

    # Run the benchmark
    arg_values = {arg.name: arg.default for arg in benchmark.args}
    try:
        user_func = getattr(user_module, benchmark.function_name)
        user_output = user_func(**arg_values)
        ref_output = ref_func(**arg_values)
    except Exception as e:
        return {"error": f"Error while running the benchmark: {e}"}

    # Compare outputs
    if user_output != ref_output:
        return {
            "result": "Mismatch in output",
            "user_output": user_output,
            "ref_output": ref_output,
        }

    return {"result": "Success", "user_output": user_output, "ref_output": ref_output}
