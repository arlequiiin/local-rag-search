"""Хранилище фрагментов документов в ChromaDB."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime

import chromadb

from rag_search.parsing import Document

# ChromaDB не умеет искать подстроку в метаданных, поэтому каждый тег
# хранится отдельным полем "tag:<имя>": True - по нему можно фильтровать.
TAG_KEY_PREFIX = "tag:"


@dataclass
class Hit:
    """Найденный фрагмент. score - косинусная близость: 1 - совпадение, 0 - нет связи."""

    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)

    @property
    def images(self) -> list[str]:
        return split_list(self.metadata.get("images", ""))


@dataclass
class DocumentInfo:
    doc_id: str
    title: str
    source: str
    author: str
    created_at: str
    active: bool
    tags: list[str]
    images: list[str]
    chunks: int


class VectorStore:
    """Фрагменты документов с метаданными в коллекции ChromaDB.

    Каждый фрагмент хранит doc_id своего документа, поэтому документ можно
    целиком пометить неактуальным, удалить или отфильтровать по тегу.
    Без path база создаётся в памяти (для тестов и оценки качества).
    """

    def __init__(self, path: str | None = None, collection_name: str = "documents"):
        if path:
            client = chromadb.PersistentClient(path=path)
        else:
            client = chromadb.EphemeralClient()
            # клиент в памяти общий на весь процесс, поэтому имя делаем уникальным
            collection_name = f"{collection_name}_{uuid.uuid4().hex[:8]}"
        self.collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    def add_document(
        self, document: Document, chunks: list[str], embeddings: list[list[float]], tags=None, author: str = ""
    ) -> str:
        if not chunks:
            raise ValueError(f"Документ {document.title!r} пуст - нечего индексировать")

        doc_id = uuid.uuid4().hex
        tags = sorted({t.strip() for t in tags or [] if t.strip()})
        metadata = {
            "doc_id": doc_id,
            "title": document.title,
            "source": document.source,
            "author": author,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "active": True,
            "tags": ",".join(tags),
            "images": ",".join(document.images),
            "total_chunks": len(chunks),
        }
        for tag in tags:
            metadata[TAG_KEY_PREFIX + tag] = True

        self.collection.add(
            ids=[f"{doc_id}:{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{**metadata, "chunk_index": i} for i in range(len(chunks))],
        )
        return doc_id

    def query(self, embedding: list[float], top_k: int = 5, active_only: bool = True, tag=None) -> list[Hit]:
        if self.collection.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=build_where(active_only, tag),
        )
        hits = []
        for i in range(len(result["ids"][0])):
            metadata = result["metadatas"][0][i]
            hits.append(
                Hit(
                    chunk_id=result["ids"][0][i],
                    doc_id=metadata["doc_id"],
                    title=metadata["title"],
                    text=result["documents"][0][i],
                    score=1 - result["distances"][0][i],
                    metadata=metadata,
                )
            )
        return hits

    def list_documents(self, active_only: bool = False, tag: str | None = None) -> list[DocumentInfo]:
        result = self.collection.get(where=build_where(active_only, tag))
        documents = group_documents(result["metadatas"])
        return sorted(documents, key=lambda d: d.created_at, reverse=True)

    def get_document(self, doc_id: str) -> DocumentInfo | None:
        result = self.collection.get(where={"doc_id": doc_id})
        documents = group_documents(result["metadatas"])
        return documents[0] if documents else None

    def all_tags(self) -> list[str]:
        return sorted({tag for doc in self.list_documents() for tag in doc.tags})

    def set_active(self, doc_id: str, active: bool) -> bool:
        result = self.collection.get(where={"doc_id": doc_id})
        if not result["ids"]:
            return False
        metadatas = [{**metadata, "active": active} for metadata in result["metadatas"]]
        self.collection.update(ids=result["ids"], metadatas=metadatas)
        return True

    def delete_document(self, doc_id: str) -> DocumentInfo | None:
        """Удаляет документ и возвращает его описание, чтобы можно было удалить и его картинки."""
        info = self.get_document(doc_id)
        if info is not None:
            self.collection.delete(where={"doc_id": doc_id})
        return info

    def clear(self) -> None:
        ids = self.collection.get()["ids"]
        if ids:
            self.collection.delete(ids=ids)

    def count_chunks(self) -> int:
        return self.collection.count()


def group_documents(metadatas: list[dict]) -> list[DocumentInfo]:
    """Собирает документы из фрагментов: у всех фрагментов документа одинаковый doc_id."""
    documents = {}
    for metadata in metadatas:
        doc_id = metadata["doc_id"]
        if doc_id in documents:
            documents[doc_id].chunks += 1
            continue
        documents[doc_id] = DocumentInfo(
            doc_id=doc_id,
            title=metadata["title"],
            source=metadata["source"],
            author=metadata["author"],
            created_at=metadata["created_at"],
            active=metadata["active"],
            tags=split_list(metadata["tags"]),
            images=split_list(metadata["images"]),
            chunks=1,
        )
    return list(documents.values())


def build_where(active_only: bool, tag: str | None) -> dict | None:
    """Фильтр для ChromaDB: только актуальные и/или только с тегом."""
    conditions = []
    if active_only:
        conditions.append({"active": True})
    if tag:
        conditions.append({TAG_KEY_PREFIX + tag: True})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def split_list(value: str) -> list[str]:
    """'a,b,c' -> ['a', 'b', 'c']; ChromaDB не хранит списки, поэтому храним строкой через запятую."""
    return [item.strip() for item in value.split(",") if item.strip()]
