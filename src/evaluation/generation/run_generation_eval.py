import asyncio
import logging
import os
import time

from openai import OpenAI
from ragas.llms import llm_factory

from config.config_loader import load_config
from src.bge_embedder import BGEEmbedder
from src.evaluation.dataset.generation_dataset import GenerationDataset
from src.evaluation.generation.evaluate_generator import GenerationEvaluator
from src.evaluation.generation.ragas_embeddings import RAGASEmbeddingAdapter
from src.evaluation.generation.run_ragas import init_metrics
from src.evaluation.report.evaluation_report import EvaluationReport
from src.fusion.weighted_fusion import WeightedFusionStrategy
from src.generation import LLMClient, Generation
from src.logger import get_logger
from src.rag_pipeline import RAGPipeline

logger = get_logger(__name__, logging.INFO)
from src.reranker.bge_reranker import BGEReranker
from src.retriever.hybrid_retriever import HybridRetriever
from src.store.bm25_store import BM25Store
from src.retriever.bm25_retriever import BM25Retriever
from src.store.vector_store import VectorStore
from src.retriever.vector_retriever import VectorRetriever

from src.utils.tokenizer import chinese_tokenizer


async def main():
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
        model=config["llm"]["model"],
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

    # 7. 初始化 evaluator
    evaluator = GenerationEvaluator(
        pipeline = pipeline,
        dataset_path = config["paths"]["generator_eval_dataset"]
    )
    client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=config["llm"]["base_url"],
        )
    evaluator_llm = llm_factory(
        "qwen3.7-max",
        client = client,
        max_tokens=2048
    )
    ragas_embeddings = RAGASEmbeddingAdapter(embedder)
    init_metrics(evaluator_llm = evaluator_llm, evaluator_embedder = ragas_embeddings)
    result = await evaluator.evaluate()
    report = EvaluationReport()
    report.save_generation_eval_result(
        result,
        output_file = config["paths"]["evaluation_results"] / "generation_eval_result.json"
    )

if __name__ == "__main__":
    asyncio.run(main())