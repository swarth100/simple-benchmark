from pydantic import BaseModel, validator
from typing import List, Union, Callable
import yaml
from typing_extensions import TypeAlias

TArg: TypeAlias = Union[int, str]


class Argument(BaseModel):
    name: str
    increment: str
    default: TArg

    def apply_increment(self, argument: int) -> int:
        if isinstance(self.increment, str) and self.increment.startswith("lambda"):
            try:
                lambda_func = eval(self.increment)
                return lambda_func(argument)
            except Exception as e:
                raise ValueError(f"Invalid lambda function: {e}")
        raise ValueError("Invalid format for lambda function")


class Benchmark(BaseModel):
    function_name: str
    max_time: int
    args: List[Argument]

    @property
    def max_time_seconds(self) -> float:
        return self.max_time / 1_000


class Config(BaseModel):
    reference_module: str
    user_modules: List[str]
    benchmarks: List[Benchmark]


def _load_config(file_path) -> Config:
    with open(file_path, "r") as file:
        config_data = yaml.safe_load(file)
        return Config.parse_obj(config_data)


BENCHMARK_CONFIG = _load_config("config/benchmark.yaml")
