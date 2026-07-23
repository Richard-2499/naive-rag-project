from llama_index.core.schema import TextNode

from src.bge_embedder import BGEEmbedder
from src.vector_store import VectorStore

class Retriever:
    """
    向量检索器，负责用户问题检索，从向量库中与用户问题最相关的top-k 相关文档，不负责回答
    """
    def __init__(self, vector_store: VectorStore, embedder: BGEEmbedder):
        self._vector_store = vector_store
        self._embedder = embedder

    def retrieve(self, query: str, top_k: int = 3) -> list[TextNode]:
        """
        检索并返回与 {query} 最相关的 top-k 个文档
        Args：
            query: 用户的提问
            top_k: 检索后返回相关文档的数量
        Returns:
             list[TextNode]
        """
        # query -> embedding -> vectorstore.search -> top_k_docs
        query_embedding = self._embedder.embed(query)
        return self._vector_store.search(query_embedding[0], top_k)

