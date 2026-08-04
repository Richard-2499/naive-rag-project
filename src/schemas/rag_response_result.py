from dataclasses import field, dataclass
from typing import Any


@dataclass

class RAGResponse:

    query: str
    answer: str
    context: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)