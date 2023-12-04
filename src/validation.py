import importlib
import inspect
from typing import List, Union, Callable, Optional, Any

import yaml
from black import FileMode, format_str
from faker import Faker
from pydantic import BaseModel
from typing_extensions import TypeAlias

TArg: TypeAlias = Union[int, str, list]

# Global initialization to prevent redundant cost of reinitializing every time
_FAKE = Faker()


class Argument(BaseModel):
    name: str
    increment: str
    default: str
    hidden: bool = False
    description: str = ""
    example: Optional[str] = None

    def model_post_init(self, ctx: Any):
        # Set the example to equal the default if not present
        if self.example is None:
            self.example = self.default

    @property
    def increment_lambda(self) -> Callable:
        # Some libraries are required to be available for import by lambdas
        import random

        include_mapping = _get_reference_benchmark_include_mapping()

        return eval(
            self.increment,
            {"random": random, "fake": _FAKE, **include_mapping},
        )

    @property
    def takes_kwargs_in_increment(self) -> bool:
        sig = inspect.signature(self.increment_lambda)
        params = sig.parameters.values()
        return any(param.kind == param.VAR_KEYWORD for param in params)

    @property
    def example_value(self) -> TArg:
        include_mapping = _get_reference_benchmark_include_mapping()
        return eval(self.example, include_mapping)

    @property
    def default_value(self) -> TArg:
        include_mapping = _get_reference_benchmark_include_mapping()
        return eval(self.default, include_mapping)

    def apply_increment(self, argument: TArg, **kwargs) -> TArg:
        # NOTE: **kwargs will only be passed down if supported by the underlying lambda.
        #       Otherwise, they will be ignored
        if isinstance(self.increment, str) and self.increment.startswith("lambda"):
            try:
                if self.takes_kwargs_in_increment:
                    return self.increment_lambda(**kwargs)
                else:
                    return self.increment_lambda(argument)
            except Exception as e:
                raise ValueError(f"Invalid lambda function: {e}")
        raise ValueError("Invalid format for lambda function")

    def validate_default_against_type(self, arg_type: type):
        """
        Validate that the default value aligns with the argument type from the reference implementation.
        """
        default_val = self.default_value
        # We must simplify as we cannot check is_instance with typing-types
        simplified_reference_type: type = eval(arg_type.__name__)
        if not isinstance(default_val, simplified_reference_type):
            raise ValueError(
                f"Default value for {self.name} does not match expected type {arg_type.__name__}"
            )


class Benchmark(BaseModel):
    function_name: str
    max_time: int
    args: List[Argument]
    description: str
    difficulty: float
    include: Optional[list[str]] = None
    hidden: bool = False

    def model_post_init(self, __context: Any):
        if self.include is None:
            self.include = []

    @property
    def max_time_seconds(self) -> float:
        return self.max_time / 1_000

    @property
    def example_args(self) -> dict[str, TArg]:
        return {arg.name: arg.example_value for arg in self.args if not arg.hidden}

    @property
    def example_args_as_function_call(self) -> str:
        """
        Generate a string representing how to call the function with default arguments.
        """
        default_args = {
            arg.name: arg.default_value for arg in self.args if not arg.hidden
        }
        return format_args_as_function_call(self.function_name, default_args)

    def generate_function_signature(self) -> str:
        # Retrieve the argument and return type annotations
        annotations, return_type = self.get_function_annotations(BENCHMARK_CONFIG)

        # Start with the function name
        function_signature = f"def {self.function_name}("

        # Add arguments with their type annotations to the function signature
        args_with_types: list[str] = []
        for arg in self.args:
            if not arg.hidden:
                # Default to 'Any' if type is not specified
                arg_type = _format_type_hint(annotations.get(arg.name, Any))
                args_with_types.append(f"{arg.name}: {arg_type}")

        function_signature += ", ".join(args_with_types)

        # Add return type annotation
        if return_type is not None and return_type != type(None):
            return_type_annotation = _format_type_hint(return_type)
            function_signature += f") -> {return_type_annotation}:\n    ...\n"
        else:
            function_signature += "):\n    ...\n"

        return function_signature

    def get_function_annotations(
        self, config: "Config"
    ) -> tuple[dict[str, type], type]:
        """
        Retrieve argument and return type annotations from the reference implementation.
        """
        reference_module_name = config.reference_module
        reference_module = importlib.import_module(reference_module_name)
        reference_func = getattr(reference_module, self.function_name)

        annotations: dict[str, type] = dict(reference_func.__annotations__)
        return_type: type = annotations.pop("return", None)
        return annotations, return_type

    def generate_description_md(self) -> str:
        """
        Generate Markdown description with type annotations.
        """
        annotations, return_type = self.get_function_annotations(BENCHMARK_CONFIG)
        description_md = self.description + "\n<br>" + "Arguments:\n"

        for arg in self.args:
            if not arg.hidden:
                arg_type = annotations.get(arg.name, Any)
                formatted_arg_type = _format_type_hint(arg_type)
                description_md += (
                    f"- **{arg.name}**: `{formatted_arg_type}`. {arg.description}\n"
                )

        if return_type is not None:
            return_type_str = f"`{_format_type_hint(return_type)}`"
        else:
            return_type_str = "`None` (outputs via `print`)"

        description_md += f"<br>Return Type: {return_type_str}\n"
        return description_md

    def generate_difficulty_stars_html(self) -> str:
        full_stars = int(self.difficulty)
        half_star = self.difficulty - full_stars >= 0.5
        empty_stars = 5 - full_stars - int(half_star)

        color: str
        if self.difficulty >= 4:
            color = "red"
        elif self.difficulty >= 2:
            color = "orange"
        else:
            color = "green"

        stars_html = (
            f'<i class="fas fa-star" style="color: {color};"></i>' * full_stars
            + (
                f'<i class="fas fa-star-half-alt" style="color: {color};"></i>'
                if half_star
                else ""
            )
            + f'<i class="far fa-star" style="color: {color};"></i>' * empty_stars
        )
        return stars_html


class Config(BaseModel):
    reference_module: str
    user_modules: List[str]
    benchmarks: List[Benchmark]

    def get_all_valid_benchmarks(
        self, *, include_archived: bool = False
    ) -> List[Benchmark]:
        from db.database import get_enabled_benchmarks, get_archived_benchmarks

        benchmarks: list[Benchmark] = get_enabled_benchmarks()
        if include_archived:
            benchmarks.extend(get_archived_benchmarks())

        return benchmarks


# ------------------------------------------------------------------------------------------------------------------- #


def _get_reference_benchmark_include_mapping() -> dict[str, Any]:
    """
    Returns a mapping of names to objects included from the reference module.
    """
    ref_module = importlib.import_module(BENCHMARK_CONFIG.reference_module)
    include_mapping = {
        name: getattr(ref_module, name)
        for benchmark in BENCHMARK_CONFIG.benchmarks
        for name in benchmark.include
    }
    return include_mapping


def format_args_as_function_call(func_name: str, args_dict: dict) -> str:
    """
    Generate and format a string representing how to call a function with given arguments.
    """
    args_str = ", ".join(f"{key}={repr(value)}" for key, value in args_dict.items())
    function_call_str = f"{func_name}({args_str})"

    # Format using black
    mode = FileMode(line_length=80)
    formatted_str = format_str(function_call_str, mode=mode)

    return formatted_str


def _format_type_hint(type_hint: type):
    """
    Format a type hint into a readable string.
    Handles complex types like generics.
    """
    if hasattr(type_hint, "__origin__"):
        # Handle generic types (e.g., List[int], Dict[str, int])
        base = type_hint.__origin__.__name__
        args = ", ".join(_format_type_hint(arg) for arg in type_hint.__args__)
        return f"{base}[{args}]"
    elif isinstance(type_hint, type):
        # Handle simple types (e.g., int, str)
        return type_hint.__name__
    return str(type_hint)


def _load_config(file_path) -> Config:
    with open(file_path, "r") as file:
        config_data = yaml.safe_load(file)
    config = Config.parse_obj(config_data)

    # Perform validation for each benchmark to ensure a cohesive config before we continue.
    # This code might raise, and that is acceptable: it means the benchmarks are misconfigured!
    for benchmark in config.benchmarks:
        annotations, _ = benchmark.get_function_annotations(config)
        for arg in benchmark.args:
            if arg.name in annotations:
                arg_type = annotations[arg.name]
                # TODO: DO NOT COMMENT OUT VALIDATION!!
                # arg.validate_default_against_type(arg_type)

    return config


BENCHMARK_CONFIG: Config = _load_config("config/benchmark.yaml")
