"""
Project configuration.

This module centralizes all project configuration, including:
- Project paths
- API keys
- Model configuration
- Splitter configuration
- Retriever configuration
- LLM configuration
"""

from pathlib import Path
import os

from dotenv import load_dotenv
from src.logger import get_logger
logger = get_logger(__name__)
# Load environment variables from .env
load_dotenv()

# ==============================================================================
# Project Paths
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
VALIDATION_DIR = DATA_DIR / "evaluation"
RESULTS_DIR = PROJECT_ROOT / "results"
VECTOR_STORE_FILE = RESULTS_DIR / "vector_store.json"
QUESTION_FILE = VALIDATION_DIR / "evaluation_questions.txt"

# ==============================================================================
# API Keys
# ==============================================================================

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# ==============================================================================
# Embedding / LLM Models
# ==============================================================================

QWEN_EMBEDDING_MODEL = os.getenv(
    "QWEN_EMBEDDING_MODEL",
    "qwen3.7-text-embedding",
)

QWEN_BASE_URL = os.getenv(
    "QWEN_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ==============================================================================
# Splitter
# ==============================================================================

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 128))

# ==============================================================================
# Retriever
# ==============================================================================

SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", 3))

# ==============================================================================
# LLM
# ==============================================================================

TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 512))


def validate() -> None:
    """
    Validate required configuration.

    Raises:
        ValueError: If required environment variables are missing.
    """

    if not DASHSCOPE_API_KEY:
        raise ValueError(
            "DASHSCOPE_API_KEY is not configured.\n"
            "Please set it in your .env file or system environment."
        )

    logger.info("✅ Configuration loaded successfully.")


# Validate configuration on import.
validate()