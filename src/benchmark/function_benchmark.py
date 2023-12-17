import copy
from types import ModuleType
from typing import List, Any

from src.benchmark.core import (
    Benchmark,
    Argument,
    TBenchmarkArgs,
    TArgsDict,
)
from src.config import BenchmarkRunInfo
from src.exceptions import ModuleAccessException

from src.utils import (
    get_function_annotations,
    _format_type_hint,
    format_arguments_as_md,
    format_args_as_function_call,
    capture_output,
)


class FunctionBenchmark(Benchmark):
    """
    A specialized Benchmark implementation for Functions.
    """

    function_name: str
    args: List[Argument]

    @property
    def name(self) -> str:
        return self.function_name

    @property
    def example_args(self) -> TBenchmarkArgs:
        function_args: TArgsDict = {arg.name: arg.example_value for arg in self.args}
        return self.filter_visible_arguments(function_args)

    @property
    def default_args(self) -> TBenchmarkArgs:
        function_args: TArgsDict = {arg.name: arg.default_value for arg in self.args}
        return function_args

    @property
    def example_args_as_python_call(self) -> str:
        filtered_args = self.filter_visible_arguments(self.default_args)
        return format_args_as_function_call(self.function_name, filtered_args)

    def increment_args(self, arguments: TBenchmarkArgs):
        for arg in self.args:
            arguments[arg.name] = arg.apply_increment(arguments[arg.name], **arguments)

    def filter_visible_arguments(self, arguments: TBenchmarkArgs) -> TBenchmarkArgs:
        visible_args: TArgsDict = {
            arg.name: arguments[arg.name] for arg in self.args if not arg.hidden
        }
        return visible_args

    def run_with_arguments(
        self, *, module: ModuleType, arguments: TBenchmarkArgs
    ) -> BenchmarkRunInfo:
        # After setting common fields we proceed to executing the function
        try:
            func = getattr(module, self.name)
        except AttributeError:
            raise ModuleAccessException(module=module)

        # Run the reference function with the provided inputs
        valid_kwargs: TBenchmarkArgs = copy.deepcopy(
            self.filter_visible_arguments(arguments)
        )
        res: BenchmarkRunInfo = capture_output(func, **valid_kwargs)

        return res

    def generate_signature(self) -> str:
        # Retrieve the argument and return type annotations
        annotations, return_type = get_function_annotations(self.name)

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

    def generate_description_md(self) -> str:
        """
        Generate Markdown description with type annotations.
        """
        annotations, return_type = get_function_annotations(self.name)
        description_md = self.description + "\n<br>" + "Arguments:\n"
        description_md += format_arguments_as_md(self.args, annotations, pre_spacing=4)

        if return_type is not None:
            return_type_str = f"`{_format_type_hint(return_type)}`"
        else:
            return_type_str = "`None` (outputs via `print`)"

        description_md += f"<br>Return Type: {return_type_str}\n"
        return description_md
