from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrievalResult:
    """
    向量检索结果
    """
    chunk_id: str
    score: float
    text: str
    metadata: dict
    source: str
    bm25_score: Optional[float] = None
    vector_score: Optional[float] = None
