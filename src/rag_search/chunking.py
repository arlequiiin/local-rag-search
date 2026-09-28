"""Разбиение текста на перекрывающиеся фрагменты с учётом границ абзацев."""


def split_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    """Режет текст на куски длиной до ``chunk_size`` символов.

    Абзацы склеиваются, пока помещаются в лимит; слишком длинный абзац
    режется по символам.
    Каждый следующий кусок начинается с хвоста предыдущего длиной до ``overlap``,
    чтобы контекст на стыке не терялся.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть положительным")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap должен быть в диапазоне [0, chunk_size)")

    units = _split_units(text, chunk_size)
    chunks = []
    current = ""

    for unit in units:
        candidate = f"{current}\n{unit}" if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        chunks.append(current)
        tail = _tail(current, overlap)
        current = f"{tail}\n{unit}" if tail and len(tail) + 1 + len(unit) <= chunk_size else unit

    if current.strip():
        chunks.append(current)
    return chunks


def _split_units(text: str, limit: int) -> list[str]:
    """Абзацы текста; абзац длиннее лимита режется на куски по ``limit`` символов."""
    units = []
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for i in range(0, len(paragraph), limit):
            units.append(paragraph[i : i + limit])
    return units


def _tail(text: str, size: int) -> str:
    """Хвост текста длиной до ``size`` символов, обрезанный по границе слова."""
    if size == 0 or not text:
        return ""
    if len(text) <= size:
        return text
    tail = text[-size:]
    space = tail.find(" ")
    return tail[space + 1 :] if space != -1 else tail
