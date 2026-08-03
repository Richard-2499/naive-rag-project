from config.config_loader import load_config
from src.bge_embedder import BGEEmbedder
from src.evaluation.dataset.reranker_dataset import RerankerEvalDataset
from src.evaluation.report.evaluation_report import EvaluationReport
from src.evaluation.reranker.evaluate_reranker import RerankerEvaluator
from src.fusion.weighted_fusion import WeightedFusionStrategy
from src.reranker.bge_reranker import BGEReranker
from src.reranker.reranker_factory import RerankerFactory
from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.hybrid_retriever import HybridRetriever
from src.retriever.vector_retriever import VectorRetriever
from src.store.bm25_store import BM25Store
from src.store.vector_store import VectorStore
from src.utils.tokenizer import chinese_tokenizer

config = load_config()
# 加载 reranker evaluation dataset
dataset = RerankerEvalDataset(file_path=config["paths"]["reranker_eval_dataset"])

# 创建Hybrid Retriever，用于生成reranker输入candidate集合
embedder = BGEEmbedder(config["embedding"]["model"], nomalize=True)

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
    vector_retriever=vector_retriever,
    bm25_retriever=bm25_retriever,
    fusion_strategy=WeightedFusionStrategy(
        bm25_weight=0.3,
        vector_weight=0.7,
    ),
    candidate_k=50
)

# 创建Reranker，当前实现为BGE Reranker
# reranker = RerankerFactory.create("bge")
reranker = BGEReranker( model_name = config["reranker"]["model"] )

# 创建Evaluation Pipeline
evaluator = RerankerEvaluator(
    retriever=hybrid_retriever,
    reranker=reranker,
    dataset=dataset
)

# 执行评估
result = evaluator.evaluate(candidate_k=50, rerank_k=10)

# 输出控制台结果
print(result)

# 保存评估报告
EvaluationReport.save(
    result=result,
    output_file=config["paths"]["evaluation_results"] / "reranker_eval_report.json"
)