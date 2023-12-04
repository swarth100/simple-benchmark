import importlib
import inspect
import sys
from functools import lru_cache
from typing import get_type_hints, Type

from pydantic import BaseModel
from pydantic.fields import FieldInfo


def serialize_base_model_to_class(base_model_instance: Type[BaseModel]) -> str:
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
            method_source = inspect.getsource(method_func)
            methods_str += f"\n{method_source}\n"
        except TypeError:
            # This can happen if the method is not a regular function (e.g., built-in)
            pass

    class_def = f"""class {class_name}:
    def __init__(self, {init_params}):
        {init_body}{methods_str}
    """

    return class_def


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
