from types import ModuleType


class ModuleAccessException(Exception):
    def __init__(self, module: ModuleType):
        self.module = module
        super().__init__()

    @property
    def module_name(self) -> str:
        return self.module.__name__

    @property
    def message(self) -> str:
        return f"Cannot access module {self.module_name} as it does not exist"
