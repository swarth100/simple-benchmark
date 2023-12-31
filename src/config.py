from dataclasses import dataclass
from typing import Optional, List, Tuple, TypedDict, NamedTuple, Any


@dataclass
class BenchmarkResult:
    name: str
    result: int = 0
    error: Optional[str] = None
    details: List[Tuple[int, float]] = None

    def __post_init__(self):
        if self.result is None and self.error is None:
            raise RuntimeError("Either result or error must not be None")
        if self.details is None:
            self.details = []

    @property
    def has_error(self) -> bool:
        return self.error is not None

    @property
    def is_reference(self) -> bool:
        return "reference" in self.name


class UserRank(TypedDict):
    rank: int
    multiplier: float
    username: str
    scores: dict[str, float]
    total: float


class BenchmarkStatus(NamedTuple):
    name: str
    is_hidden: bool
    is_frozen: bool
    is_archive: bool


class BenchmarkRunInfo(NamedTuple):
    return_value: list[Any]
    std_output: list[str]
    exec_time: float

    @property
    def return_value_repr(self) -> str:
        return "\n".join([repr(x) for x in self.return_value])

    @property
    def std_output_repr(self) -> str:
        return "\n".join(self.std_output)
