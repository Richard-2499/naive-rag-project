from typing import Any


class RAGASEmbeddingAdapter:
    """
    将项目Embedding接口适配为RAGAS接口
    """

    def __init__(self, embedder: Any):
        self.embedder = embedder

    def embed_query(self, text: str) -> list[float]:
        # 复用已有embedding逻辑
        return self.embedder.embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            self.embedder.embed(text)
            for text in texts
        ]