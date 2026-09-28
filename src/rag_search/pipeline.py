"""Индексация документов и ответы на вопросы: retrieval + генерация."""

import logging
from dataclasses import dataclass, field

from rag_search.chunking import split_text
from rag_search.embeddings import SentenceTransformerEmbedder
from rag_search.llm import SYSTEM_PROMPT, OllamaLLM, build_prompt
from rag_search.parsing import IMAGE_PLACEHOLDER_RE, Document
from rag_search.store import Hit, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class Answer:
    question: str
    hits: list[Hit]
    answer: str | None = None
    """Ответ LLM; ``None``, если LLM не настроена или недоступна - тогда показываются только найденные фрагменты."""
    error: str | None = None
    best_doc_id: str | None = None
    best_title: str | None = None
    images: list[str] = field(default_factory=list)


class RAGPipeline:
    def __init__(
        self,
        store: VectorStore,
        embedder: SentenceTransformerEmbedder,
        llm: OllamaLLM | None = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        top_k: int = 5,
    ):
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

    def index(self, document: Document, tags: list[str] | None = None, author: str = "") -> str:
        chunks = split_text(document.text, self.chunk_size, self.chunk_overlap)
        # Заголовок добавляется к каждому фрагменту: так фрагмент из середины инструкции
        # всё равно "знает", о чём она. Плейсхолдеры картинок для эмбеддинга - шум.
        texts_for_embedding = [f"{document.title}\n{IMAGE_PLACEHOLDER_RE.sub('', chunk)}" for chunk in chunks]
        embeddings = self.embedder.embed_passages(texts_for_embedding)
        return self.store.add_document(document, chunks, embeddings, tags=tags, author=author)

    def search(self, question: str, top_k: int | None = None, tag: str | None = None) -> list[Hit]:
        embedding = self.embedder.embed_queries([question])[0]
        return self.store.query(embedding, top_k=top_k or self.top_k, tag=tag)

    def ask(self, question: str, top_k: int | None = None, tag: str | None = None) -> Answer:
        hits = self.search(question, top_k=top_k, tag=tag)
        result = Answer(question=question, hits=hits)
        if not hits:
            result.answer = "В базе знаний не нашлось ничего по этому вопросу."
            return result

        # Фрагменты уже отсортированы по близости, поэтому лучший документ - документ первого фрагмента
        best = hits[0]
        result.best_doc_id = best.doc_id
        result.best_title = best.title
        result.images = best.images

        if self.llm is None:
            return result
        try:
            result.answer = self.llm.generate(build_prompt(question, format_context(hits)), SYSTEM_PROMPT)
        except Exception as exc:  # сеть, не запущенная Ollama, не скачанная модель
            logger.warning("LLM недоступна: %s", exc)
            result.error = f"Не удалось получить ответ от LLM: {exc}"
        return result


def format_context(hits: list[Hit]) -> str:
    return "\n\n---\n\n".join(f"[Источник {i}: {hit.title}]\n{hit.text}" for i, hit in enumerate(hits, 1))
