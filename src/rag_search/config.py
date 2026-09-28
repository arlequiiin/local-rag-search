"""Настройки приложения. Любой параметр можно переопределить переменной окружения."""

import os
from pathlib import Path

# src/rag_search/config.py, переходим так как корень проекта на два уровня выше
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings:
    def __init__(self, data_dir=None, chroma_dir=None):
        # Папки можно передать явно, либо берутся из окружения
        self.data_dir = Path(data_dir or os.environ.get("RAG_DATA_DIR", PROJECT_ROOT / "data"))
        self.chroma_dir = Path(chroma_dir or os.environ.get("RAG_CHROMA_DIR", PROJECT_ROOT / "chroma_db"))
        self.collection_name = os.environ.get("RAG_COLLECTION", "documents")

        self.embedding_model = os.environ.get("RAG_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
        self.chunk_size = int(os.environ.get("RAG_CHUNK_SIZE", 1200))
        self.chunk_overlap = int(os.environ.get("RAG_CHUNK_OVERLAP", 200))
        self.top_k = int(os.environ.get("RAG_TOP_K", 5))

        self.llm_model = os.environ.get("RAG_LLM_MODEL", "qwen2.5:7b-instruct")
        self.llm_max_tokens = int(os.environ.get("RAG_LLM_MAX_TOKENS", 1024))
        self.llm_temperature = float(os.environ.get("RAG_LLM_TEMPERATURE", 0.2))

        self.docs_dir = self.data_dir / "docs"
        self.images_dir = self.data_dir / "images"

    def ensure_dirs(self) -> None:
        for path in (self.docs_dir, self.images_dir, self.chroma_dir):
            path.mkdir(parents=True, exist_ok=True)
