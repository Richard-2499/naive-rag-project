"""
负责：
    加载评测数据，管理ground truth
设计目的： Evaluation 数据和代码分离
    1. 增加测试集
    2. A/B 测试
    3. 版本管理

 JSON格式
"""
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class EvaluationSample:
    query: str
    relevant_chunk_ids: list[str]

class EvaluationDataset:
    def __init__(self, data_path: str):
        self._path = Path(data_path)

    def load(self) -> list[EvaluationSample]:
        with self._path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return [
            EvaluationSample(
            query = item["query"], relevant_chunk_ids = item["relevant_chunk_ids"]
            ) for item in data
        ]
