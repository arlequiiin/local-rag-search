"""REST API. Запуск: ``uvicorn rag_search.api:app --port 8000``, документация - /docs."""

from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag_search.service import KnowledgeBase, get_knowledge_base
from rag_search.store import DocumentInfo

app = FastAPI(title="local-rag-search", version="1.0.0")

KB = Annotated[KnowledgeBase, Depends(get_knowledge_base)]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    tag: str | None = None


class HitOut(BaseModel):
    doc_id: str
    title: str
    text: str
    score: float
    images: list[str]


class AskResponse(BaseModel):
    answer: str | None
    error: str | None
    best_doc_id: str | None
    best_title: str | None
    images: list[str]
    hits: list[HitOut]


class ActivePatch(BaseModel):
    active: bool


def _hit_out(hit) -> HitOut:
    return HitOut(doc_id=hit.doc_id, title=hit.title, text=hit.text, score=round(hit.score, 4), images=hit.images)


@app.get("/health")
def health(kb: KB) -> dict:
    return {"status": "ok", "chunks": kb.store.count_chunks()}


@app.post("/search", response_model=list[HitOut])
def search(request: SearchRequest, kb: KB):
    return [_hit_out(h) for h in kb.pipeline.search(request.query, top_k=request.top_k, tag=request.tag)]


@app.post("/ask", response_model=AskResponse)
def ask(request: SearchRequest, kb: KB):
    result = kb.pipeline.ask(request.query, top_k=request.top_k, tag=request.tag)
    return AskResponse(
        answer=result.answer,
        error=result.error,
        best_doc_id=result.best_doc_id,
        best_title=result.best_title,
        images=result.images,
        hits=[_hit_out(h) for h in result.hits],
    )


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents(kb: KB, active_only: bool = False, tag: str | None = None):
    return kb.store.list_documents(active_only=active_only, tag=tag)


@app.post("/documents", response_model=list[str], status_code=201)
async def upload_document(
    kb: KB,
    file: Annotated[UploadFile, File()],
    tags: Annotated[str, Form(description="Теги через запятую")] = "",
    author: Annotated[str, Form()] = "",
    split_sections: Annotated[bool, Form(description="Разделить файл на инструкции по строкам ---")] = False,
):
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    try:
        return kb.ingest_bytes(
            file.filename or "document.txt",
            await file.read(),
            tags=tag_list,
            author=author,
            split_sections=split_sections,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.patch("/documents/{doc_id}", response_model=DocumentInfo)
def set_active(doc_id: str, patch: ActivePatch, kb: KB):
    if not kb.store.set_active(doc_id, patch.active):
        raise HTTPException(status_code=404, detail="Документ не найден")
    return kb.store.get_document(doc_id)


@app.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str, kb: KB):
    if not kb.delete(doc_id):
        raise HTTPException(status_code=404, detail="Документ не найден")


@app.get("/images/{name}")
def get_image(name: str, kb: KB):
    path = kb.image_path(name)
    if path is None:
        raise HTTPException(status_code=404, detail="Изображение не найдено")
    return FileResponse(path)
