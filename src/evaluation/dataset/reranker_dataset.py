import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RerankerEvalCase:
    query: str
    relevant_chunks: dict[str, int]


class RerankerEvalDataset:

    def __init__(self, file_path: str):
        self._file_path = Path(file_path)
        self.samples: list[RerankerEvalCase] = []
        self._load()

    def _load(self):

        with self._file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            sample = RerankerEvalCase(
                query = item["query"],
                relevant_chunks = item["relevant_chunks"]
            )
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

