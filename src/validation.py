import importlib
import inspect
from typing import List, Union, Callable, Optional, Any

import yaml
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

        return eval(self.increment, {"random": random, "fake": _FAKE})

    @property
    def takes_kwargs_in_increment(self) -> bool:
        sig = inspect.signature(self.increment_lambda)
        params = sig.parameters.values()
        return any(param.kind == param.VAR_KEYWORD for param in params)

    @property
    def example_value(self) -> TArg:
        return eval(self.example)

    @property
    def default_value(self) -> TArg:
        return eval(self.default)

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
    hidden: bool = False
    description: str

    @property
    def max_time_seconds(self) -> float:
        return self.max_time / 1_000

    @property
    def example_args(self) -> dict[str, TArg]:
        return {arg.name: arg.example_value for arg in self.args if not arg.hidden}

    def generate_function_signature(self) -> str:
        # Start with the function name
        function_signature = f"def {self.function_name}("

        # Add arguments to the function signature
        args = [arg.name for arg in self.args if not arg.hidden]
        function_signature += ", ".join(args)

        # Close the function signature and add a placeholder for function body
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
                arg.validate_default_against_type(arg_type)

    return config


BENCHMARK_CONFIG: Config = _load_config("config/benchmark.yaml")
