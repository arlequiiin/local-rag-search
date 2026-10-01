"""Загрузка SberQuAD - открытого русскоязычного QA-датасета - для демо и оценки поиска."""

import shutil
from dataclasses import dataclass
from pathlib import Path

SBERQUAD_REPO = "kuznetsoffandrey/sberquad"
SBERQUAD_FILE = "sberquad/validation-00000-of-00001.parquet"
SBERQUAD_URL = f"https://huggingface.co/datasets/{SBERQUAD_REPO}/resolve/main/{SBERQUAD_FILE}"


@dataclass
class Passage:
    passage_id: str
    title: str
    text: str


@dataclass
class Question:
    question: str
    passage_id: str


def is_parquet(path: Path) -> bool:
    """Файл существует и похож на целый parquet (магические байты PAR1 в начале и в конце)."""
    if not path.is_file() or path.stat().st_size < 8:
        return False
    with path.open("rb") as f:
        head = f.read(4)
        f.seek(-4, 2)
        return head == b"PAR1" and f.read(4) == b"PAR1"


def download_sberquad(target_dir: Path) -> Path:
    """Скачивает validation-часть SberQuAD в target_dir (если там ещё нет целого файла).

    Качаем через huggingface_hub: он понимает HF_TOKEN, HF_ENDPOINT (зеркало) и прокси,
    а файл в target_dir появляется только после успешной загрузки - оборванная загрузка
    не оставит пустой файл, из-за которого pandas потом падает.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "sberquad_validation.parquet"
    if is_parquet(path):
        return path
    # Пустой или битый файл от прошлой неудачной загрузки
    path.unlink(missing_ok=True)

    try:
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(repo_id=SBERQUAD_REPO, filename=SBERQUAD_FILE, repo_type="dataset")
    except Exception as exc:
        raise RuntimeError(
            f"Не удалось скачать SberQuAD: {exc}\nСкачайте файл вручную: {SBERQUAD_URL}\nи положите его сюда: {path}"
        ) from exc

    tmp = path.with_suffix(".part")
    shutil.copyfile(cached, tmp)
    if not is_parquet(tmp):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Скачанный файл SberQuAD повреждён, попробуйте ещё раз ({SBERQUAD_URL})")
    tmp.replace(path)
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


def title_from_text(text: str, limit: int = 60, min_length: int = 20) -> str:
    """Первое предложение абзаца, обрезанное до limit символов по границе слова.

    Если до первой точки слишком мало текста (инициалы "А. Н. Толстой", сокращения "фр."),
    берётся начало абзаца целиком, иначе заголовок получается вроде "А".
    """
    text = text.strip()
    first = text.split(".")[0].strip()
    if len(first) < min_length:
        first = text
    if len(first) <= limit:
        return first
    return first[:limit].rsplit(" ", 1)[0].rstrip(" .,;:-—") + "..."
