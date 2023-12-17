from typing import Annotated

from typing_extensions import TypeAlias


class PrintsToConsole:
    pass


Print: TypeAlias = Annotated[None, PrintsToConsole]
