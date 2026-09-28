"""Командная строка: ``rag-search ingest|load-sberquad|evaluate``."""

import argparse
import sys
from pathlib import Path

from rag_search.config import Settings
from rag_search.datasets import prepare_sberquad
from rag_search.embeddings import SentenceTransformerEmbedder
from rag_search.evaluation import evaluate, index_passages
from rag_search.parsing import SUPPORTED_EXTENSIONS
from rag_search.pipeline import RAGPipeline
from rag_search.service import build_knowledge_base
from rag_search.store import VectorStore


def cmd_ingest(args: argparse.Namespace) -> None:
    kb = build_knowledge_base(with_llm=False)
    root = Path(args.path)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    total = 0
    for path in files:
        try:
            ids = kb.ingest_file(path, tags=tags, author=args.author, split_sections=args.split_sections)
        except Exception as exc:
            print(f"  ✗ {path.name}: {exc}", file=sys.stderr)
            continue
        total += len(ids)
        print(f"  ✓ {path.name}: {len(ids)}")
    print(f"Готово: файлов {len(files)}, документов {total}, фрагментов в базе {kb.store.count_chunks()}")


def cmd_load_sberquad(args: argparse.Namespace) -> None:
    settings = Settings()
    passages, _ = prepare_sberquad(settings.data_dir / "sberquad", n_passages=args.passages)
    kb = build_knowledge_base(settings, with_llm=False)
    print(f"Индексация {len(passages)} абзацев SberQuAD...")
    index_passages(kb.pipeline, passages)
    print(f"Готово, фрагментов в базе: {kb.store.count_chunks()}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    settings = Settings()
    passages, questions = prepare_sberquad(settings.data_dir / "sberquad", n_passages=args.passages)
    model = args.model or settings.embedding_model
    embedder = SentenceTransformerEmbedder(model)
    # Отдельная база в памяти: оценка не должна смешиваться с рабочей базой в chroma_db
    pipeline = RAGPipeline(
        VectorStore(), embedder, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    print(f"Модель: {model}")
    print(f"Абзацев: {len(passages)}, вопросов: {len(questions)}")
    metrics = evaluate(pipeline, passages, questions)
    print()
    for name in ("Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR"):
        print(f"{name:>7}: {metrics[name]:.3f}")
    print()
    print(f"Индексация: {metrics['index_seconds']:.1f} с, поиск: {metrics['query_ms']:.1f} мс/запрос")


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag-search", description="Локальный RAG-поиск по базе знаний")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="проиндексировать файл или папку с документами")
    ingest.add_argument("path")
    ingest.add_argument("--tags", default="", help="теги через запятую")
    ingest.add_argument("--author", default="")
    ingest.add_argument("--split-sections", action="store_true", help="делить файлы на инструкции по строкам ---")
    ingest.set_defaults(func=cmd_ingest)

    load = sub.add_parser("load-sberquad", help="загрузить демо-базу из SberQuAD")
    load.add_argument("--passages", type=int, default=1000)
    load.set_defaults(func=cmd_load_sberquad)

    ev = sub.add_parser("evaluate", help="оценить качество поиска на SberQuAD (Hit@k, MRR)")
    ev.add_argument("--passages", type=int, default=1000)
    ev.add_argument("--model", help="модель эмбеддингов (по умолчанию из настроек)")
    ev.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
