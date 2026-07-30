import time

from src.store.bm25_store import BM25Store
from src.utils.tokenizer import chinese_tokenizer

from config.config_loader import load_config
config = load_config()

from src.bge_embedder import BGEEmbedder
from src.splitter import Splitter
from src.store.vector_store import VectorStore

from src.logger import get_logger
logger = get_logger(__name__)


def main():
    bm25_store = BM25Store(
        save_path=config["paths"]["bm25_store"],
        tokenizer=chinese_tokenizer,
    )

    embedder = BGEEmbedder("BAAI/bge-small-zh", nomalize=True)
    logger.info("加载chunks_store file...")
    nodes = Splitter.load_nodes_from_file(config["paths"]["chunks_store"])
    logger.info(f"chunks_store file loaded, num: {len(nodes)}")

    logger.info("bm25 store 开始保存 bm25_store file...")
    t1 = time.time()
    bm25_store.save(nodes)
    logger.info(f"bm25_store file 保存完成，耗时：{time.time() - t1}")
    logger.info("vector store 开始保存 vector_store file...")
    t1 = time.time()
    texts = [node.text for node in nodes]
    embedding_data = embedder.embed(texts)
    logger.info(f"embedding time: {time.time() - t1} 秒")
    new_vector_store = VectorStore(config["paths"]["vector_store"])
    new_vector_store.save(embedding_data, nodes)
    logger.info(f"vector store 保存完成, 耗时 {time.time() - t1} 秒")


if __name__ == "__main__":
    main()