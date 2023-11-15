from dataclasses import dataclass
from typing import Optional, List, Tuple, TypedDict


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
    username: str
    scores: dict[str, float]
    total: float
