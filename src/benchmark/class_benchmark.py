from typing import List, Callable, Type

from pydantic import BaseModel

from src.benchmark.core import (
    Benchmark,
    Argument,
    TBenchmarkArgs,
    TArgsDict,
    _FAKE,
    _get_reference_benchmark_include_mapping,
)
from src.utils import (
    get_reference_benchmark_include,
    serialize_base_model_to_class,
    get_function_annotations,
    format_arguments_as_md,
    TABBED_MD_SPACING,
    _format_type_hint,
    format_args_as_function_call,
)


class Method(BaseModel):
    method_name: str
    description: str
    args: List[Argument]


class ClassBenchmark(Benchmark):
    class_name: str
    init: List[Argument]
    methods: List[Method]
    evaluation: str

    @property
    def name(self) -> str:
        return self.class_name

    @property
    def example_args(self) -> TBenchmarkArgs:
        class_args: TBenchmarkArgs = {
            self.class_name: {arg.name: arg.example_value for arg in self.init}
        }
        for method in self.methods:
            class_args[method.method_name] = {
                arg.name: arg.example_value for arg in method.args
            }
        return self.filter_visible_arguments(class_args)

    @property
    def default_args(self) -> TBenchmarkArgs:
        class_args: TBenchmarkArgs = {
            self.class_name: {arg.name: arg.default_value for arg in self.init}
        }
        for method in self.methods:
            class_args[method.method_name] = {
                arg.name: arg.default_value for arg in method.args
            }
        return class_args

    @property
    def example_args_as_python_call(self) -> str:
        # TODO: Correctly implement for classes and nested methods!
        filtered_args = self.filter_visible_arguments(self.default_args)
        return format_args_as_function_call(
            self.class_name, filtered_args[self.class_name]
        )

    def increment_args(self, arguments: TBenchmarkArgs):
        init_arguments: TArgsDict = arguments[self.class_name]
        for arg in self.init:
            init_arguments[arg.name] = arg.apply_increment(
                init_arguments[arg.name], **init_arguments
            )

        # TODO: Validate if we should generate a full MEO, or if instead we could use self.methods.
        #       The full MEO would override arguments currently.
        methods = self.generate_method_evaluation_order(init_arguments)
        for method in methods:
            method_arguments: TArgsDict = arguments[method.method_name]
            for arg in method.args:
                method_arguments[arg.name] = arg.apply_increment(
                    method_arguments[arg.name], **method_arguments
                )

    def filter_visible_arguments(self, arguments: TBenchmarkArgs) -> TBenchmarkArgs:
        class_args: TBenchmarkArgs = {
            self.class_name: {
                arg.name: arguments[self.class_name][arg.name]
                for arg in self.init
                if not arg.hidden
            }
        }
        for method in self.methods:
            class_args[method.method_name] = {
                arg.name: arguments[method.method_name][arg.name]
                for arg in method.args
                if not arg.hidden
            }
        return class_args

    @property
    def example_includes(self) -> list[str]:
        return super().example_includes + [self.class_name]

    @property
    def evaluation_lambda(self) -> Callable:
        # Some libraries are required to be available for import by lambdas
        import random

        include_mapping = _get_reference_benchmark_include_mapping()

        return eval(
            self.evaluation,
            {"random": random, "fake": _FAKE, **include_mapping},
        )

    def generate_method_evaluation_order(self, arguments: TArgsDict) -> list[Method]:
        """
        Generate the order in which methods should be evaluated and return a list of method objects for each one.
        Method objects might be duplicated as a result of this, in case a method is called more than once.
        """
        method_order: list[Method] = []
        method_names: list[str] = self.evaluation_lambda(**arguments)

        for method_name in method_names:
            for method in self.methods:
                if method.method_name == method_name:
                    method_order.append(method)

        return method_order

    def generate_signature(self) -> str:
        class_object: Type[BaseModel] = get_reference_benchmark_include(self.class_name)
        methods_to_exclude: list[str] = [method.method_name for method in self.methods]
        return serialize_base_model_to_class(
            base_model_instance=class_object, methods_to_exclude=methods_to_exclude
        )

    def generate_description_md(self) -> str:
        annotations, _ = get_function_annotations(self.name)
        description_md = self.description + "\n<br>" + "Constructor arguments:\n"
        description_md += format_arguments_as_md(self.init, annotations, pre_spacing=4)

        for method in self.methods:
            method_annotations, method_return_type = get_function_annotations(
                self.name, method_name=method.method_name
            )

            description_md += f"\n<br>Method `{method.method_name}`:\n"
            description_md += f"{TABBED_MD_SPACING}- Arguments:\n"
            description_md += format_arguments_as_md(
                method.args, method_annotations, pre_spacing=8
            )

            if method_return_type is not None:
                return_type_str = f"`{_format_type_hint(method_return_type)}`"
            else:
                return_type_str = "`None` (outputs via `print`)"

            description_md += f"{TABBED_MD_SPACING}- Return Type: {return_type_str}\n"

        return description_md
