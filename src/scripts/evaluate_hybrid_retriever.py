from src.bge_embedder import BGEEmbedder
from src.fusion.weighted_fusion import WeightedFusionStrategy
from src.retriever.hybrid_retriever import HybridRetriever
from src.retriever.vector_retriever import VectorRetriever
from src.store.vector_store import VectorStore

from config.config_loader import load_config
from src.retriever.bm25_retriever import BM25Retriever
from src.scripts.common import run_evaluation
from src.store.bm25_store import BM25Store
from src.utils.tokenizer import chinese_tokenizer

config = load_config()
def main() -> None:
    embedder = BGEEmbedder("BAAI/bge-small-zh", nomalize=True)

    vector_store = VectorStore(config["paths"]["vector_store"])
    vector_store.load()

    vector_retriever = VectorRetriever(vector_store, embedder)

    bm25_store = BM25Store(
        save_path=config["paths"]["bm25_store"],
        tokenizer=chinese_tokenizer,
    )

    bm25_store.load()

    bm25_retriever = BM25Retriever(bm25_store=bm25_store, tokenizer=chinese_tokenizer)

    hybrid_retriever = HybridRetriever(
        vector_retriever = vector_retriever,
        bm25_retriever = bm25_retriever,
        fusion_strategy = WeightedFusionStrategy(
            bm25_weight=0.3,
            vector_weight=0.7,
        ),
        candidate_k = 50
    )
    run_evaluation(
        retriever = hybrid_retriever,
        output_path = config["paths"]["evaluation_results"] / "hybrid_retriever_result.json",
        top_k = 20
    )

if __name__ == "__main__":
    main()