from pydantic import BaseModel, validator
from typing import List, Union, Callable
import yaml
from typing_extensions import TypeAlias

TArg: TypeAlias = Union[int, str]


class Argument(BaseModel):
    name: str
    increment: Callable[[TArg], TArg]
    default: TArg

    @validator("increment", pre=True)
    def parse_lambda(cls, v):
        if isinstance(v, str) and v.startswith("lambda"):
            try:
                return eval(v)
            except Exception as e:
                raise ValueError(f"Invalid lambda function: {e}")
        raise ValueError("Invalid format for lambda function")


class Benchmark(BaseModel):
    function_name: str
    max_time: float
    args: List[Argument]


class Config(BaseModel):
    reference_module: str
    user_modules: List[str]
    benchmarks: List[Benchmark]


def load_config(file_path) -> Config:
    with open(file_path, "r") as file:
        config_data = yaml.safe_load(file)
        return Config.parse_obj(config_data)
