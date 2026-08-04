import os
from typing import Any
from pydantic import BaseModel
from ragas import experiment
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import llm_factory

from config.config_loader import load_config

config = load_config()

class RAGASEvaluationResult(BaseModel):
    """
    experiment返回结果的数据结构。
    RAGAS 0.4.x要求使用结构化返回。
    """
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

METRICS: dict[str, Any] = {}

def init_metrics(evaluator_llm: Any, evaluator_embedder: Any) -> None:
    """
    初始化RAGAS metrics
    evaluator_llm由run_generation_eval.py创建，
    这里仅负责绑定。
    """
    global METRICS
    METRICS = {
    "faithfulness": Faithfulness(llm = evaluator_llm),
    "answer_relevancy": AnswerRelevancy(llm = evaluator_llm, embeddings = evaluator_embedder),
    "context_precision": ContextPrecision(llm = evaluator_llm),
    "context_recall": ContextRecall(llm = evaluator_llm),
}

@experiment(RAGASEvaluationResult)
async def evaluate_ragas(row: dict[str, Any]) -> RAGASEvaluationResult:
    """
    单条RAG结果评估。
    row:
    {
        question:"",
        answer:"",
        contexts:[],
        ground_truth:""
    }
    """
    sample = SingleTurnSample(
        user_input = row["query"],
        response = row["answer"],
        retrieved_contexts = row["context"],
        reference = row["ground_truth"]
    )
    scores: dict[str, float] = {}
    for name, metric in METRICS.items():
        # ragas 0.4.x metric 计算
        score = await metric.single_turn_ascore(sample)
        scores[name] = float(score)
    return RAGASEvaluationResult(**scores)

