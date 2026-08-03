from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrievalResult:
    """
    向量检索结果

    Retriever:
        score = retrieval score

    Reranker:
        rerank_score = reranker score
    """
    chunk_id: str
    text: str
    metadata: dict
    source: str
    score: float
    raw_score: Optional[float] = None
    bm25_score: Optional[float] = None
    vector_score: Optional[float] = None
    rerank_score: Optional[float] = None
    rerank_source: Optional[str] = None
    is_reranked: bool = False
