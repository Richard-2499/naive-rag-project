# src/main.py
import logging
import os
import time

from bge_embedder import BGEEmbedder
from config.config_loader import load_config
from src.fusion.weighted_fusion import WeightedFusionStrategy
from src.logger import get_logger
logger = get_logger(__name__, logging.INFO)
from src.reranker.bge_reranker import BGEReranker
from src.retriever.hybrid_retriever import HybridRetriever
from src.store.bm25_store import BM25Store
from src.retriever.bm25_retriever import BM25Retriever
from src.store.vector_store import VectorStore
from src.retriever.vector_retriever import VectorRetriever
from generation import Generation, LLMClient
from rag_pipeline import RAGPipeline
from src.utils.tokenizer import chinese_tokenizer


def main():
    config = load_config()
    # 1. 初始化 Embedding
    embedder = BGEEmbedder(config["embedding"]["model"], nomalize=True)

    # 2. 初始化 VectorStore
    vector_store = VectorStore(
        save_path= config["paths"]["vector_store"]
    )
    vector_store.load()
    logger.info(f"✅ vector 载入完成，数量：{len(vector_store.vectors)}")
    bm25_store = BM25Store(
        save_path=config["paths"]["bm25_store"],
        tokenizer=chinese_tokenizer,
    )
    bm25_store.load()
    logger.info(f"✅ bm25 载入完成，数量：{len(bm25_store.corpus)}")
    # 3. 初始化 Retriever
    vector_retriever = VectorRetriever(
        embedder=embedder,
        vector_store=vector_store
    )
    bm25_retriever = BM25Retriever(
        bm25_store=bm25_store,
        tokenizer=bm25_store.tokenizer,
    )

    fusion_strategy = WeightedFusionStrategy(
        bm25_weight=0.7,
        vector_weight=0.3,
    )
    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        fusion_strategy=fusion_strategy,
        candidate_k=config["retrieval"]["hybrid"]["candidate_k"]
    )

    reranker = BGEReranker( model_name = config["reranker"]["model"] )
    # 4. 初始化 LLM
    llm = LLMClient(
        model="qwen3.7-max",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=config["llm"]["base_url"],
    )

    # 5. 初始化 Generation
    generation = Generation(llm=llm)

    # 6. 组装 RAG Pipeline
    pipeline = RAGPipeline(
        retriever=hybrid_retriever,
        generation=generation,
        reranker = reranker,
    )

    # 7. 用户问题
    query = "vxlan 中数据包转发流程"

    answer = pipeline.query(query)
    time.sleep(3)

    logger.info("=" * 80)
    logger.info("回答:")
    logger.info("=" * 80)

    time.sleep(3)
    print(answer)


if __name__ == "__main__":
    main()