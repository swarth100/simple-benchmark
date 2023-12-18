import copy
import time
from types import ModuleType
from typing import List, Callable, Type, Optional

from pydantic import BaseModel, parse_obj_as

from src.benchmark.core import (
    Benchmark,
    Argument,
    TBenchmarkArgs,
    TArgsDict,
    _FAKE,
    _get_reference_benchmark_include_mapping,
)
from src.config import BenchmarkRunInfo
from src.exceptions import ModuleAccessException
from src.utils import (
    get_reference_benchmark_include,
    serialize_base_model_to_class,
    get_annotations,
    format_arguments_as_md,
    TABBED_MD_SPACING,
    _format_type_hint,
    format_args_as_function_call,
    capture_output,
    PrintsToConsole,
)

# Argument name used to inject method evaluation order into argument lists
MEO_NAMES = "method_names"
MEO_ARGS = "method_arguments"


class Method(BaseModel):
    method_name: str
    description: str
    args: List[Argument]
    hidden: bool = True


class MethodEvaluationOrder(BaseModel):
    increment: str
    default: str

    @property
    def default_value(self) -> list[str]:
        include_mapping = _get_reference_benchmark_include_mapping()
        return eval(self.default, include_mapping)

    @property
    def evaluation_lambda(self) -> Callable:
        # Some libraries are required to be available for import by lambdas
        import random

        include_mapping = _get_reference_benchmark_include_mapping()

        return eval(
            self.increment,
            {"random": random, "fake": _FAKE, **include_mapping},
        )


class ClassBenchmark(Benchmark):
    class_name: str
    init: List[Argument]
    methods: List[Method]
    evaluation: MethodEvaluationOrder

    @property
    def name(self) -> str:
        return self.class_name

    @property
    def icon_unicode(self) -> str:
        # Shapes Icon
        return "f61f"

    @property
    def example_args(self) -> TBenchmarkArgs:
        class_args: TBenchmarkArgs = {
            self.class_name: {arg.name: arg.example_value for arg in self.init},
            MEO_NAMES: self.evaluation.default_value,
        }

        meo_arguments: list[TArgsDict] = []
        for method_name in class_args[MEO_NAMES]:
            for method in self.methods:
                if method.method_name == method_name:
                    meo_arguments.append(
                        {arg.name: arg.example_value for arg in method.args}
                    )
        class_args[MEO_ARGS] = meo_arguments

        return self.filter_visible_arguments(class_args)

    @property
    def default_args(self) -> TBenchmarkArgs:
        class_args: TBenchmarkArgs = {
            self.class_name: {arg.name: arg.default_value for arg in self.init},
            MEO_NAMES: self.evaluation.default_value,
        }

        meo_arguments: list[TArgsDict] = []
        for method_name in class_args[MEO_NAMES]:
            for method in self.methods:
                if method.method_name == method_name:
                    meo_arguments.append(
                        {arg.name: arg.default_value for arg in method.args}
                    )
        class_args[MEO_ARGS] = meo_arguments

        return class_args

    def increment_args(self, arguments: TBenchmarkArgs):
        init_arguments: TArgsDict = arguments[self.class_name]
        for arg in self.init:
            init_arguments[arg.name] = arg.apply_increment(
                init_arguments[arg.name], **init_arguments
            )

        # TODO: Validate if we should generate a full MEO, or if instead we could use self.methods.
        #       The full MEO would override arguments currently.
        methods = self._generate_method_evaluation_order(init_arguments)
        meo_arguments: list[TArgsDict] = []
        for method in methods:
            method_increment_arguments: TArgsDict = init_arguments

            # We purpusefully pass down `None` to guarantee it's not used in the increment
            method_arguments: TArgsDict = {
                arg.name: arg.apply_increment(
                    None, **method_increment_arguments  # type: ignore
                )
                for arg in method.args
            }
            meo_arguments.append(method_arguments)

        arguments[MEO_NAMES] = [method.method_name for method in methods]
        arguments[MEO_ARGS] = meo_arguments

    def filter_visible_arguments(self, arguments: TBenchmarkArgs) -> TBenchmarkArgs:
        class_args: TBenchmarkArgs = {
            self.class_name: {
                arg.name: arguments[self.class_name][arg.name]
                for arg in self.init
                if not arg.hidden
            },
            MEO_NAMES: arguments[MEO_NAMES],
        }

        method_args: list[TArgsDict] = []
        for method_name, method_arguments in zip(
            arguments[MEO_NAMES], arguments[MEO_ARGS]
        ):
            for method in self.methods:
                if method.method_name == method_name:
                    method_args.append(
                        {
                            arg.name: method_arguments[arg.name]
                            for arg in method.args
                            if not arg.hidden
                        }
                    )
        class_args[MEO_ARGS] = method_args

        return class_args

    def run_with_arguments(
        self, *, module: ModuleType, arguments: TBenchmarkArgs
    ) -> BenchmarkRunInfo:
        try:
            klass = getattr(module, self.name)
        except AttributeError:
            raise ModuleAccessException(module=module)

        # Step 0: We must deep-clone the input arguments to ensure that we don't mutate them
        valid_kwargs: TBenchmarkArgs = copy.deepcopy(
            self.filter_visible_arguments(arguments)
        )

        # Step 1 is to construct the object
        init_arguments: TArgsDict = valid_kwargs[self.name]
        start_time: float = time.perf_counter()
        obj = klass(**init_arguments)
        time_diff: float = time.perf_counter() - start_time

        # Class initialization time is accounted for in the total time cost
        res: BenchmarkRunInfo = BenchmarkRunInfo([], [], time_diff)

        # Init arguments are specified via benchmark name, if not present we must raise
        if self.name not in valid_kwargs:
            raise ValueError(
                f"Class '{self.name}' is missing from the supplied arguments with keys {list(arguments.keys())}."
            )

        # Step 2 is to run the methods in the evaluation order specified
        for method_name, method_args in zip(arguments[MEO_NAMES], arguments[MEO_ARGS]):
            for method in self.methods:
                if method.method_name == method_name:
                    valid_func_kwargs: TArgsDict = {
                        arg.name: method_args[arg.name]
                        for arg in method.args
                        if not arg.hidden
                    }

                    method_func = getattr(obj, method.method_name)
                    latest_res: BenchmarkRunInfo = capture_output(
                        method_func, **valid_func_kwargs
                    )

                    if latest_res.return_value[0] is not None:
                        res.return_value.extend(latest_res.return_value)

                    if len(latest_res.std_output[0]) > 0:
                        res.std_output.extend(latest_res.std_output)

                    # We must consider all cumulative runtime costs!
                    res = BenchmarkRunInfo(
                        res.return_value,
                        res.std_output,
                        res.exec_time + latest_res.exec_time,
                    )

        return res

    @property
    def example_includes(self) -> list[str]:
        return super().example_includes + [self.class_name]

    def parse_arguments_from_dict(self, raw_arguments: dict) -> TBenchmarkArgs:
        (constructor_types, _) = get_annotations(self.name)
        filtered_args = {
            self.class_name: {
                arg_name: parse_obj_as(constructor_types[arg_name], arg_value)  # type: ignore
                for arg_name, arg_value in raw_arguments[self.class_name].items()
            }
        }

        filtered_method_args: list[TArgsDict] = []
        for method_name, method_args in zip(
            raw_arguments[MEO_NAMES], raw_arguments[MEO_ARGS]
        ):
            for method in self.methods:
                if method.method_name == method_name:
                    method_types, _ = get_annotations(
                        self.name, method_name=method.method_name
                    )
                    filtered_method_args.append(
                        {
                            arg_name: parse_obj_as(method_types[arg_name], arg_value)  # type: ignore
                            for arg_name, arg_value in method_args.items()
                        }
                    )
        filtered_args[MEO_NAMES] = raw_arguments[MEO_NAMES]
        filtered_args[MEO_ARGS] = filtered_method_args
        return filtered_args

    def generate_python_call(self, arguments: TBenchmarkArgs) -> str:
        filtered_args = self.filter_visible_arguments(arguments)
        obj_name: str = self.class_name.lower()
        constructor: str = format_args_as_function_call(
            self.class_name, filtered_args[self.class_name]
        )
        output_code: str = f"{obj_name} = {constructor}"

        for method_name, method_args in zip(arguments[MEO_NAMES], arguments[MEO_ARGS]):
            output_code += format_args_as_function_call(
                f"{obj_name}.{method_name}", method_args
            )

        return output_code

    def generate_signature(self) -> str:
        class_object: Type[BaseModel] = get_reference_benchmark_include(self.class_name)
        methods_to_exclude: list[str] = [
            method.method_name for method in self.methods if method.hidden
        ]
        return serialize_base_model_to_class(
            base_model_instance=class_object, methods_to_exclude=methods_to_exclude
        )

    def generate_description_md(self) -> str:
        annotations, _ = get_annotations(self.name)
        description_md = (
            f"{self.description} \n<br>`{self.name}` constructor arguments:\n"
        )
        description_md += format_arguments_as_md(self.init, annotations, pre_spacing=4)

        for method in self.methods:
            method_annotations, method_return_type = get_annotations(
                self.name, method_name=method.method_name
            )

            description_md += f"\n<br>Method `{method.method_name}`:\n"
            method_description_md = method.description.rstrip().replace(
                "\n", f"\n{TABBED_MD_SPACING}"
            )
            description_md += TABBED_MD_SPACING + method_description_md + "\n"
            if len(method.args) > 0:
                description_md += f"{TABBED_MD_SPACING}- Arguments:\n"
                description_md += format_arguments_as_md(
                    method.args, method_annotations, pre_spacing=8
                )

            return_type_str: Optional[str] = None
            if method_return_type is PrintsToConsole:
                return_type_str = "`None` (outputs via `print`)"
            elif method_return_type is not None:
                return_type_str = f"`{_format_type_hint(method_return_type)}`"
            else:
                # We currently specifically ignore methods that do not return
                # Adding a specific None return type does not add value to the user
                pass

            if return_type_str is not None:
                description_md += (
                    f"{TABBED_MD_SPACING}- Return Type: {return_type_str}\n"
                )

        return description_md

    def _generate_method_evaluation_order(self, arguments: TArgsDict) -> list[Method]:
        """
        Generate the order in which methods should be evaluated and return a list of method objects for each one.
        Method objects might be duplicated as a result of this, in case a method is called more than once.
        """
        method_order: list[Method] = []
        method_names: list[str] = self.evaluation.evaluation_lambda(**arguments)

        for method_name in method_names:
            for method in self.methods:
                if method.method_name == method_name:
                    method_order.append(method)

        return method_order
