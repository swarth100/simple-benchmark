import importlib
from functools import lru_cache
from types import ModuleType
from typing import List, Union

import yaml
from pydantic import BaseModel

from src.benchmark.class_benchmark import ClassBenchmark
from src.benchmark.core import Benchmark
from src.benchmark.function_benchmark import FunctionBenchmark


class Config(BaseModel):
    reference_module: str
    user_modules: List[str]
    benchmarks: List[Union[FunctionBenchmark, ClassBenchmark]]

    @property
    def reference_module_object(self) -> ModuleType:
        return importlib.import_module(self.reference_module)

    def get_all_valid_benchmarks(
        self, *, include_archived: bool = False
    ) -> List[Benchmark]:
        from db.database import get_enabled_benchmarks, get_archived_benchmarks

        benchmarks: list[Benchmark] = get_enabled_benchmarks()
        if include_archived:
            benchmarks.extend(get_archived_benchmarks())

        return benchmarks


def _load_config(file_path) -> Config:
    with open(file_path, "r") as file:
        config_data = yaml.safe_load(file)
    config = Config.parse_obj(config_data)

    # Perform validation for each benchmark to ensure a cohesive config before we continue.
    # This code might raise, and that is acceptable: it means the benchmarks are misconfigured!
    # TODO: DO NOT COMMENT OUT VALIDATION!!
    # for benchmark in config.benchmarks:
    #     annotations, _ = benchmark.get_annotations(config)
    #     for arg in benchmark.args:
    #         if arg.name in annotations:
    #             arg_type = annotations[arg.name]
    #             arg.validate_default_against_type(arg_type)

    return config


BENCHMARK_CONFIG: Config = _load_config("config/benchmark.yaml")


@lru_cache
def get_config() -> Config:
    benchmark_config: Config = BENCHMARK_CONFIG
    return benchmark_config
