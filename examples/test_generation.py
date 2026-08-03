from config.config_loader import load_config
from src.bge_embedder import BGEEmbedder
from src.draft.config import QWEN_EMBEDDING_MODEL, DASHSCOPE_API_KEY, VECTOR_STORE_FILE
from src.generation import LLMClient, Generation
from src.retriever.vector_retriever import VectorRetriever
from src.store.vector_store import VectorStore

config = load_config()
vecstore = VectorStore(VECTOR_STORE_FILE)
embedder = BGEEmbedder(config["embedding"]["model"], nomalize=True)

nodes = VectorRetriever(vecstore, embedder).retrieve("JIT的3个层次")

qwen = LLMClient(QWEN_EMBEDDING_MODEL, DASHSCOPE_API_KEY)
gen = Generation(qwen)

print(gen.generate("JIT的3个层次", nodes))