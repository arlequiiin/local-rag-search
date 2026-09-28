"""Сборка компонентов и операции над базой знаний, общие для UI, API и скриптов."""

from functools import lru_cache
from pathlib import Path

from rag_search.config import Settings
from rag_search.embeddings import SentenceTransformerEmbedder
from rag_search.llm import OllamaLLM
from rag_search.parsing import SUPPORTED_EXTENSIONS, parse_file
from rag_search.pipeline import RAGPipeline
from rag_search.store import DocumentInfo, VectorStore


class KnowledgeBase:
    def __init__(self, pipeline: RAGPipeline, settings: Settings):
        self.pipeline = pipeline
        self.settings = settings

    @property
    def store(self) -> VectorStore:
        return self.pipeline.store

    def ingest_bytes(
        self,
        filename: str,
        content: bytes,
        tags: list[str] | None = None,
        author: str = "",
        split_sections: bool = False,
    ) -> list[str]:
        """Сохраняет загруженный файл в ``data/docs`` и индексирует его."""
        # Берём только имя файла: "../../evil.txt" -> "evil.txt", чтобы нельзя было записать файл вне data/docs
        name = Path(filename).name
        if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Поддерживаются только {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        self.settings.ensure_dirs()
        path = _unique_path(self.settings.docs_dir / name)
        path.write_bytes(content)
        return self.ingest_file(path, tags=tags, author=author, split_sections=split_sections)

    def ingest_file(
        self,
        path: str | Path,
        tags: list[str] | None = None,
        author: str = "",
        split_sections: bool = False,
    ) -> list[str]:
        documents = parse_file(path, self.settings.images_dir, split_sections=split_sections)
        return [self.pipeline.index(doc, tags=tags, author=author) for doc in documents if doc.text]

    def delete(self, doc_id: str) -> bool:
        info = self.store.delete_document(doc_id)
        if info is None:
            return False
        self._remove_images(info)
        return True

    def clear(self) -> None:
        for info in self.store.list_documents():
            self._remove_images(info)
        self.store.clear()

    def image_path(self, name: str) -> Path | None:
        """Путь к картинке по имени; None, если файла нет или имя пытается выйти из папки картинок."""
        path = (self.settings.images_dir / name).resolve()
        if path.parent != self.settings.images_dir.resolve() or not path.is_file():
            return None
        return path

    def _remove_images(self, info: DocumentInfo) -> None:
        for name in info.images:
            path = self.image_path(name)
            if path is not None:
                path.unlink(missing_ok=True)


def build_knowledge_base(settings: Settings | None = None, with_llm: bool = True) -> KnowledgeBase:
    settings = settings or Settings()
    settings.ensure_dirs()
    store = VectorStore(str(settings.chroma_dir), settings.collection_name)
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    llm = OllamaLLM(settings.llm_model, settings.llm_max_tokens, settings.llm_temperature) if with_llm else None
    pipeline = RAGPipeline(
        store,
        embedder,
        llm,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
    )
    return KnowledgeBase(pipeline, settings)


# Для API: база и модели создаются один раз при первом запросе, а не на каждый запрос
@lru_cache(maxsize=1)
def get_knowledge_base() -> KnowledgeBase:
    return build_knowledge_base()


def _unique_path(path: Path) -> Path:
    """a.txt -> a_1.txt -> a_2.txt ..., чтобы новый файл не затёр старый с тем же именем."""
    candidate, n = path, 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        n += 1
    return candidate
