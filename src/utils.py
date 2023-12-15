import importlib
import inspect
import sys
from functools import lru_cache
from typing import get_type_hints, Type, TYPE_CHECKING, Optional, Any

from pydantic import BaseModel
from pydantic.fields import FieldInfo

if TYPE_CHECKING:
    from src.validation import Config, Argument

TABBED_MD_SPACING: str = "&nbsp;" * 4


def get_function_annotations(
    object_name: str, config: "Config", *, method_name: Optional[str] = None
) -> tuple[dict[str, type], type]:
    """
    Given the name of a function to include from the reference module, resolves the python type annotations for the
    function's arguments and return type

    :param object_name: Name of the function to include
    :param config: Configuration object
    :param method_name: (Optional) If present, the name of the method to look up the signature for
    :return: Annotations and return type of the function
    """
    reference_module_name = config.reference_module
    reference_module = importlib.import_module(reference_module_name)
    reference_object = getattr(reference_module, object_name)

    if method_name is not None:
        reference_object = getattr(reference_object, method_name)

    annotations: dict[str, type] = dict(reference_object.__annotations__)
    return_type: type = annotations.pop("return", None)
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
    fields: dict[str, FieldInfo] = base_model_instance.__fields__  # type: ignore

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
    init_body: str = "\n        ".join([f"self.{name} = {name}" for name in fields])

    methods_str = ""
    for method in methods:
        method_func = getattr(base_model_instance, method)
        try:
            if method not in methods_to_exclude:
                method_source = inspect.getsource(method_func)
            else:
                method_signature = inspect.signature(method_func)
                method_source = f"    def {method}{method_signature}\n        ..."
            methods_str += f"\n\n{method_source}"
        except TypeError:
            # This can happen if the method is not a regular function (e.g., built-in)
            pass

    class_def = f"""class {class_name}:
    def __init__(self, {init_params}):
        {init_body}{methods_str}
    """

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
    from src.benchmark import get_config

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
