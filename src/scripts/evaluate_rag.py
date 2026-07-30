from src.evaluation.dataset import EvaluationDataset
from src.evaluation.metrics import RetrievalMetrics
from src.evaluation.evaluator import RetrievalEvaluator
from src.evaluation.report import EvaluationReport
from src.retriever.hybrid_retriever import HybridRetriever

def main() -> None:
    dataset = (EvaluationDataset("data/evaluation.json").load())
    retriever = HybridRetriever(
        # your dependencies
    )
    evaluator = RetrievalEvaluator(
        retriever= retriever,
        metrics= RetrievalMetrics
    )
    results = evaluator.evaluate(dataset, top_k=5)
    summary = (EvaluationReport.summarize(results))
    EvaluationReport.print_report(summary)

if __name__ == "__main__":
    main()