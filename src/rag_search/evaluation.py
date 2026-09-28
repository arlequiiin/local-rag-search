"""Метрики качества поиска: Hit@k и MRR по размеченным парам "вопрос -> абзац"."""

import time

from rag_search.datasets import Passage, Question
from rag_search.parsing import Document
from rag_search.pipeline import RAGPipeline

KS = (1, 3, 5, 10)


def index_passages(pipeline: RAGPipeline, passages: list[Passage]) -> dict[str, str]:
    """Индексирует абзацы и возвращает соответствие doc_id -> passage_id."""
    doc_to_passage = {}
    for passage in passages:
        doc_id = pipeline.index(Document(title=passage.title, text=passage.text, source="sberquad"), tags=["sberquad"])
        doc_to_passage[doc_id] = passage.passage_id
    return doc_to_passage


def evaluate(pipeline: RAGPipeline, passages: list[Passage], questions: list[Question]) -> dict[str, float]:
    started = time.perf_counter()
    doc_to_passage = index_passages(pipeline, passages)
    index_seconds = time.perf_counter() - started

    hits = {k: 0 for k in KS}
    reciprocal_ranks = 0.0
    started = time.perf_counter()
    for question in questions:
        found = pipeline.search(question.question, top_k=max(KS) * 2)
        # Длинный абзац может дать несколько фрагментов - считаем места абзацев, убирая повторы
        ranked = list(dict.fromkeys(doc_to_passage[hit.doc_id] for hit in found))[: max(KS)]
        if question.passage_id in ranked:
            rank = ranked.index(question.passage_id) + 1
            reciprocal_ranks += 1 / rank
            for k in KS:
                if rank <= k:
                    hits[k] += 1
    query_ms = (time.perf_counter() - started) / len(questions) * 1000

    n = len(questions)
    metrics = {f"Hit@{k}": hits[k] / n for k in KS}
    metrics["MRR"] = reciprocal_ranks / n
    metrics["index_seconds"] = index_seconds
    metrics["query_ms"] = query_ms
    return metrics
