import hashlib
import math
import re
import struct
import zlib
from pathlib import Path

import pytest

from rag_search.config import Settings
from rag_search.pipeline import RAGPipeline
from rag_search.service import KnowledgeBase
from rag_search.store import VectorStore


class HashingEmbedder:
    """Детерминированный "мешок слов" вместо нейросети: тексты с общими словами оказываются рядом."""

    dim = 256

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in re.findall(r"\w{3,}", text.lower()):
            stem = token[:5]
            vector[int(hashlib.md5(stem.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_queries(self, texts):
        return [self._embed(t) for t in texts]

    def embed_passages(self, texts):
        return [self._embed(t) for t in texts]


class EchoLLM:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, system_prompt=""):
        self.prompts.append(prompt)
        return "Ответ по контексту"


class BrokenLLM:
    def generate(self, prompt, system_prompt=""):
        raise ConnectionError("ollama не запущена")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings(data_dir=tmp_path / "data", chroma_dir=tmp_path / "chroma")
    s.ensure_dirs()
    return s


@pytest.fixture
def pipeline() -> RAGPipeline:
    return RAGPipeline(VectorStore(), HashingEmbedder(), EchoLLM(), chunk_size=300, chunk_overlap=50)


@pytest.fixture
def kb(pipeline: RAGPipeline, settings: Settings) -> KnowledgeBase:
    return KnowledgeBase(pipeline, settings)


def png_bytes() -> bytes:
    """Минимальный валидный PNG 1×1."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
        + chunk(b"IEND", b"")
    )
