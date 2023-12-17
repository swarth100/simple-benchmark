import importlib
import inspect
import io
import sys
import time
from functools import lru_cache
from typing import (
    get_type_hints,
    Type,
    TYPE_CHECKING,
    Optional,
    Any,
    get_origin,
    Annotated,
    get_args,
)
from typing_extensions import TypeAlias

from black import FileMode, format_str
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from benchmarks.utils import PrintsToConsole
from src.config import BenchmarkRunInfo

if TYPE_CHECKING:
    from src.benchmark.core import Argument
    from src.benchmark.config import Config

TABBED_MD_SPACING: str = "&nbsp;" * 4


def get_annotations(
    object_name: str, *, method_name: Optional[str] = None
) -> tuple[dict[str, type], type]:
    """
    Given the name of a function to include from the reference module, resolves the python type annotations for the
    function's arguments and return type

    :param object_name: Name of the function to include
    :param method_name: (Optional) If present, the name of the method to look up the signature for
    :return: Annotations and return type of the function
    """
    from src.benchmark.config import get_config

    config: Config = get_config()

    reference_module_name = config.reference_module
    reference_module = importlib.import_module(reference_module_name)
    raw_obj = reference_object = getattr(reference_module, object_name)

    if method_name is not None:
        reference_object = getattr(reference_object, method_name)

    # We specify the current raw object as a locals reference in case there are any
    # annotations which reference the object itself (e.g., return type)
    annotations = get_type_hints(
        reference_object,
        localns={object_name: raw_obj},
        globalns=globals(),
        include_extras=True,
    )
    return_type: type = annotations.pop("return", None)

    # Check if the return type is Annotated and if it is un-pack the annotation overriding
    # the return type
    if get_origin(return_type) is Annotated:
        # Handle annotated types, we look at the RHS of the annotation always
        # This does mean we assume the LHS is annotated as None
        annotation = get_args(return_type)[1]
        if issubclass(annotation, PrintsToConsole):
            return_type = PrintsToConsole

    return annotations, return_type


def serialize_base_model_to_class(
    base_model_instance: Type[BaseModel],
    *,
    methods_to_exclude: Optional[list[str]] = None,
) -> str:
    """
    Given a pydantic BaseModel, serializes it to a class definition
    :param base_model_instance: The pydantic BaseModel to serialize
    :param methods_to_exclude: List of methods to exclude from the serialization
    :return: The stringified class definition of the BaseModel
    """
    if methods_to_exclude is None:
        methods_to_exclude = []

    class_name = base_model_instance.__name__
    fields: dict[str, FieldInfo] = base_model_instance.__dataclass_fields__  # type: ignore

    base_methods = set(dir(BaseModel))
    subclass_methods = set(dir(base_model_instance)) - base_methods

    methods = [
        func
        for func in subclass_methods
        if callable(getattr(base_model_instance, func)) and not func.startswith("_")
    ]

    init_params: str = ", ".join(
        [
            f"{name}: {get_type_hints(base_model_instance)[name].__name__}"
            for name in fields
        ]
    )
    init_body: str = "..."
    if len(fields) > 0:
        init_body = "\n        ".join([f"self.{name} = {name}" for name in fields])

    methods_str = ""
    for method in methods:
        method_func = getattr(base_model_instance, method)
        try:
            if method not in methods_to_exclude:
                method_source = inspect.getsource(method_func)
            else:
                method_signature = inspect.signature(method_func)
                method_source = f"    def {method}{method_signature}:\n        ..."
            methods_str += f"\n\n{method_source}"
        except TypeError:
            # This can happen if the method is not a regular function (e.g., built-in)
            pass

    class_def = f"""class {class_name}:
    def __init__(self, {init_params}):
        {init_body}{methods_str}
    """

    # We could have accidentally leaked a symbolic reference to the module of the class declarations
    # We remove it to ensure the class definition is valid in the user code
    class_module: str = base_model_instance.__module__
    class_def = class_def.replace(f"{class_module}.", "")

    # We format with black the class_Def to ensure a consistent output
    mode = FileMode(line_length=80)
    class_def = format_str(class_def, mode=mode)

    return class_def


def format_arguments_as_md(
    args: list["Argument"], annotations: dict[str, type], *, pre_spacing: int = 0
) -> str:
    """
    Given a list of arguments and their annotations, formats them as a markdown list
    :param args: List of arguments to format
    :param annotations: Dict of annotations
    :param pre_spacing: Number of spaces to prefix the output with
    :return: Markdown-formatted list of arguments
    """
    output_md: str = ""
    spacing: str = "&nbsp;" * pre_spacing
    for arg in args:
        if not arg.hidden:
            arg_type = annotations.get(arg.name, Any)
            formatted_arg_type = _format_type_hint(arg_type)
            output_md += f"{spacing} - **{arg.name}**: `{formatted_arg_type}`. {arg.description}\n"

    return output_md


@lru_cache
def get_reference_benchmark_include(object_name: str) -> Type[BaseModel]:
    """
    Given the name of an object to include from the reference module, resolves the python
    object and returns a reference to it

    :param object_name: Name of the object to include
    :return: Reference to the object
    """
    from src.benchmark.config import get_config

    benchmark_config = get_config()
    try:
        ref_module = importlib.import_module(benchmark_config.reference_module)
    except ImportError:
        print(
            f"Error: Reference module '{benchmark_config.reference_module}' not found."
        )
        sys.exit(1)

    try:
        ref_object = getattr(ref_module, object_name)
    except AttributeError:
        raise AttributeError(
            f"Error: Object '{object_name}' not found in "
            f"reference module '{ref_module.__name__}'.\n"
            f"Benchmarks cannot reference include objects which do not exist!"
        )

    return ref_object


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


def capture_output(func, *args, **kwargs) -> BenchmarkRunInfo:
    original_stdout = sys.stdout  # Save a reference to the original standard output
    new_stdout = io.StringIO()
    sys.stdout = new_stdout  # Redirect standard output to the new StringIO object

    try:
        start_time: float = time.perf_counter()
        output = func(*args, **kwargs)
        time_diff: float = time.perf_counter() - start_time
    except Exception:
        # Reraise all exceptions to allow for outer handling
        raise
    finally:
        # Whatever happens, reset standard output to its original value
        sys.stdout = original_stdout

    return BenchmarkRunInfo(output, new_stdout.getvalue().rstrip(), time_diff)


def is_equal_with_precision(lhs, rhs, precision=4):
    """
    Compare two values with a given precision.
    :param lhs: LHS value to compare
    :param rhs: RHS value to compare
    :param precision: (Optional) Precision to use in case of floats
    :return: True if the values are equal, False otherwise
    """
    if isinstance(lhs, float) and isinstance(rhs, float):
        return round(lhs, precision) == round(rhs, precision)
    else:
        return lhs == rhs
