import inspect
from typing import List, Union, Callable

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

    @property
    def default_value(self) -> TArg:
        return eval(self.default)


class Benchmark(BaseModel):
    function_name: str
    max_time: int
    args: List[Argument]
    hidden: bool = False
    description: str

    @property
    def max_time_seconds(self) -> float:
        return self.max_time / 1_000

    def generate_function_signature(self) -> str:
        # Start with the function name
        function_signature = f"def {self.function_name}("

        # Add arguments to the function signature
        args = [arg.name for arg in self.args if not arg.hidden]
        function_signature += ", ".join(args)

        # Close the function signature and add a placeholder for function body
        function_signature += "):\n    ...\n"

        return function_signature

    def generate_description_md(self) -> str:
        description_md = ""
        for arg in self.args:
            if not arg.hidden and arg.description:
                description_md += f"- **{arg.name}**: {arg.description}\n"
        return description_md

    @property
    def default_args(self) -> dict[str, TArg]:
        return {arg.name: arg.default_value for arg in self.args if not arg.hidden}


class Config(BaseModel):
    reference_module: str
    user_modules: List[str]
    benchmarks: List[Benchmark]

    def get_all_valid_benchmarks(self) -> List[Benchmark]:
        from db.database import get_enabled_benchmarks

        return get_enabled_benchmarks()


def _load_config(file_path) -> Config:
    with open(file_path, "r") as file:
        config_data = yaml.safe_load(file)
        return Config.parse_obj(config_data)


BENCHMARK_CONFIG = _load_config("config/benchmark.yaml")
