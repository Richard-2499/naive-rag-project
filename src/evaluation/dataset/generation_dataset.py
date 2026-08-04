import json
from pathlib import Path
from typing import TypedDict


class GenerationCase(TypedDict):
    query: str
    ground_truth: str

class GenerationDataset:
    def __init__(self, file_path: str):
        self._file_path = Path(file_path)

    def load(self) -> list[GenerationCase]:
        with self._file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
            