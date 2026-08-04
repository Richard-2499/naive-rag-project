from src.bge_embedder import BGEEmbedder
from src.schemas.retrieval_result import RetrievalResult
from src.store.vector_store import VectorStore

class VectorRetriever:
    """
    向量检索器，负责用户问题检索，从向量库中与用户问题最相关的top-k 相关文档，不负责回答
    """
    def __init__(self, vector_store: VectorStore, embedder: BGEEmbedder):
        self._vector_store = vector_store
        self._embedder = embedder

    def retrieve(self, query: str, top_k: int = 3):
        """
        检索并返回与 {query} 最相关的 top-k 个文档
        Args：
            query: 用户的提问
            top_k: 检索后返回相关文档的数量
        Returns:


        steps:
            1. query embedding
            2. vector store search
            3. covert result to RetrievalResult
        """
        # query -> embedding -> vectorstore.search -> top_k_docs
        query_embedding = self._embedder.embed([query])[0]
        results = self._vector_store.search(query_embedding, top_k) # list[tuple[chunk_id, score]]
        retrieval_results = []
        for chunk_id, score in results:
            node = self._vector_store.get_chunk(chunk_id)
            retrieval_results.append(
                RetrievalResult(
                    chunk_id = chunk_id,
                    score = score,
                    text = node.text,
                    metadata= node.metadata,
                    source = "vector",
                    vector_score = score
                )
            )
        return retrieval_results

