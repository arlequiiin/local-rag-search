"""Загрузка SberQuAD - открытого русскоязычного QA-датасета - для демо и оценки поиска."""

import urllib.request
from dataclasses import dataclass
from pathlib import Path

SBERQUAD_URL = (
    "https://huggingface.co/datasets/kuznetsoffandrey/sberquad/resolve/main/sberquad/validation-00000-of-00001.parquet"
)


@dataclass
class Passage:
    passage_id: str
    title: str
    text: str


@dataclass
class Question:
    question: str
    passage_id: str


def download_sberquad(target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "sberquad_validation.parquet"
    if not path.exists():
        urllib.request.urlretrieve(SBERQUAD_URL, path)
    return path


def prepare_sberquad(target_dir: Path, n_passages: int = 1000) -> tuple[list[Passage], list[Question]]:
    """Случайные n_passages разных абзацев и по одному вопросу к каждому."""
    import pandas as pd

    frame = pd.read_parquet(download_sberquad(target_dir))
    # Одна строка = один вопрос; оставляем по одной строке на абзац и берём случайные (всегда одни и те же)
    rows = frame.drop_duplicates("context").sample(n=n_passages, random_state=42)

    passages = []
    questions = []
    for i, row in enumerate(rows.itertuples()):
        passage_id = f"p{i:05d}"
        # Поле title в SberQuAD у всех записей одинаковое ("SberChallenge"), поэтому заголовок - начало абзаца
        passages.append(Passage(passage_id, title_from_text(row.context), row.context))
        questions.append(Question(row.question.strip(), passage_id))
    return passages, questions


def title_from_text(text: str, limit: int = 60) -> str:
    """Первое предложение абзаца, обрезанное до limit символов по границе слова."""
    first = text.split(".")[0].strip()
    if len(first) <= limit:
        return first
    return first[:limit].rsplit(" ", 1)[0] + "..."
