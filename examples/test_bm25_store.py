from transformers.trainer_pt_utils import save_metrics

from src.loader import DocumentLoader
from src.splitter import Splitter
from src.store.bm25_store import BM25Store
from src.utils.tokenizer import chinese_tokenizer
from config.config_loader import load_config
config = load_config()

bm25_store = BM25Store(
    save_path=config["paths"]["bm25_store"],
    tokenizer=chinese_tokenizer,
)
docs = DocumentLoader(config["paths"]["raw_data"]).load_documents()

nodes = Splitter(config["splitter"]["chunk_size"], config["splitter"]["chunk_overlap"]).split_documents(docs)
bm25_store.save(nodes)