import abc
import importlib
import inspect
from types import ModuleType
from typing import Union, Callable, Optional, Any, Type, TYPE_CHECKING

from faker import Faker
from pydantic import BaseModel
from typing_extensions import TypeAlias

from src.config import BenchmarkRunInfo
from src.utils import (
    serialize_base_model_to_class,
    get_reference_benchmark_include,
)

if TYPE_CHECKING:
    pass

TArg: TypeAlias = Union[int, str, list]
TArgsDict: TypeAlias = dict[str, TArg]
# Every benchmark implementation is allowed to have a different structure and representation for arguments
TBenchmarkArgs: TypeAlias = dict[str, Union[TArgsDict, list[TArg], list[TArgsDict]]]

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


class Benchmark(BaseModel, abc.ABC):
    max_time: int
    description: str
    difficulty: float
    include: Optional[list[str]] = None
    hidden: bool = False

    def model_post_init(self, __context: Any):
        if self.include is None:
            self.include = []

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def example_args(self) -> TBenchmarkArgs:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def default_args(self) -> TBenchmarkArgs:
        # NOTE: Default arguments are purposefully not filtered to visible arguments
        raise NotImplementedError

    @abc.abstractmethod
    def increment_args(self, arguments: TBenchmarkArgs):
        # Increment is performed in-place
        raise NotImplementedError

    @abc.abstractmethod
    def filter_visible_arguments(self, arguments: TBenchmarkArgs) -> TBenchmarkArgs:
        raise NotImplementedError

    @abc.abstractmethod
    def run_with_arguments(
        self, *, module: ModuleType, arguments: TBenchmarkArgs
    ) -> BenchmarkRunInfo:
        """
        Given a benchmark and a module, runs the benchmark with the provided arguments.
        :param module: Module to run the benchmark on
        :param arguments: Arguments to run the benchmark with

        :return: BenchmarkRunInfo containing the result of the benchmark
        :raises ModuleAccessException: If the module cannot be accessed or does not exist
        """
        raise NotImplementedError

    @abc.abstractmethod
    def parse_arguments_from_dict(self, raw_arguments: dict) -> TBenchmarkArgs:
        """
        Given a dictionary of arguments, parses them into a valid benchmark arguments object.
        :param raw_arguments: Dictionary of arguments to parse
        :return: Valid benchmark arguments object
        """
        raise NotImplementedError

    @abc.abstractmethod
    def generate_python_call(self, arguments: TBenchmarkArgs) -> str:
        """
        Generate a string representing how to call the benchmark with given arguments.

        :param arguments: Arguments to use in the call
        :return: String representing how to call the benchmark with given arguments
        """
        raise NotImplementedError

    @abc.abstractmethod
    def generate_signature(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def generate_description_md(self) -> str:
        raise NotImplementedError

    @property
    def max_time_seconds(self) -> float:
        return self.max_time / 1_000

    @property
    def example_includes(self) -> list[str]:
        return self.include

    def generate_include_code(self) -> str:
        """
        Generate code to include from the reference module.
        """
        include_snippets: list[str] = []
        for name in self.include:
            include_object: Type[BaseModel] = get_reference_benchmark_include(name)
            include_snippets.append(serialize_base_model_to_class(include_object))
        return "\n".join(include_snippets)

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


# ------------------------------------------------------------------------------------------------------------------- #


def _get_reference_benchmark_include_mapping() -> dict[str, Any]:
    """
    Returns a mapping of names to objects included from the reference module.
    """
    from src.benchmark.config import BENCHMARK_CONFIG

    ref_module = importlib.import_module(BENCHMARK_CONFIG.reference_module)
    include_mapping = {
        name: getattr(ref_module, name)
        for benchmark in BENCHMARK_CONFIG.benchmarks
        for name in benchmark.example_includes
    }

    return include_mapping
