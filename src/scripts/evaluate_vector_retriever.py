from src.bge_embedder import BGEEmbedder
from src.retriever.vector_retriever import VectorRetriever
from src.scripts.common import run_evaluation
from src.store.vector_store import VectorStore
from config.config_loader import load_config
config = load_config()

def main():
    embedder = BGEEmbedder("BAAI/bge-small-zh", nomalize=True)

    vector_store = VectorStore(config["paths"]["vector_store"])
    vector_store.load()

    retriever = VectorRetriever(vector_store , embedder)

    run_evaluation(
        retriever = retriever,
        output_path = config["paths"]["evaluation_results"] / "vector_retriever_result.json",
        top_k = 20
    )


if __name__  == "__main__":
    main()