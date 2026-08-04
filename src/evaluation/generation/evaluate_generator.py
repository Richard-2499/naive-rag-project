import json
from pathlib import Path

from config.config_loader import load_config
from src.evaluation.dataset.generation_dataset import GenerationDataset
from src.evaluation.generation.run_ragas import evaluate_ragas
config = load_config()
from src.logger import get_logger, get_log_level
level = get_log_level(config["logging"]["level"])
logger = get_logger(__name__, level)

class GenerationEvaluator:
    def __init__(self, pipeline, dataset_path: str):
        self.pipeline = pipeline
        self.dataset = GenerationDataset(file_path=dataset_path)

    async def evaluate(self):
        results = []
        scores = []
        dataset = self.dataset.load()
        output_file = Path(config["paths"]["evaluation_results"] / "generation_eval_timely.json")
        for index, item in enumerate(dataset, start=1):
            logger.info(f"Evaluating {index}/{len(dataset)}")
            query = item["query"]
            response = self.pipeline.query(query)
            sample = {
                "query": response.query,
                "answer": response.answer,
                "context": response.context,
                "ground_truth": item["ground_truth"]
            }
            result = await evaluate_ragas(row = sample)
            scores.append(result.model_dump())
            results.append(result)

            with output_file.open("w", encoding="utf-8") as f:
                json.dump(scores, f, ensure_ascii=False, indent=4)
        return results