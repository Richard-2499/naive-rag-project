
from config.config_loader import load_config
from src.retriever.bm25_retriever import BM25Retriever
from src.evaluation.retriever.retriever_common import run_evaluation
from src.store.bm25_store import BM25Store
from src.utils.tokenizer import chinese_tokenizer

config = load_config()

def main():
    bm25_store = BM25Store(
        save_path=config["paths"]["bm25_store"],
        tokenizer=chinese_tokenizer,
    )

    bm25_store.load()

    retriever = BM25Retriever(bm25_store=bm25_store, tokenizer=chinese_tokenizer)

    run_evaluation(
        retriever = retriever,
        output_path = config["paths"]["evaluation_results"] / "bm25_retriever_result.json",
        top_k = 20
    )

if __name__ == "__main__":
    main()