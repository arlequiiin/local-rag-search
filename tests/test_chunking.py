import pytest

from rag_search.chunking import split_text


def test_short_text_is_single_chunk():
    assert split_text("Первый абзац.\nВторой абзац.", chunk_size=100, overlap=10) == ["Первый абзац.\nВторой абзац."]


def test_empty_text_gives_no_chunks():
    assert split_text("  \n\n ", chunk_size=100, overlap=10) == []


def test_chunks_respect_size_limit():
    text = "\n".join(f"Абзац номер {i} с каким-то содержимым." for i in range(50))
    chunks = split_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_paragraphs_are_not_cut_in_the_middle():
    paragraphs = [f"Абзац {i}: " + "слово " * 10 for i in range(20)]
    chunks = split_text("\n".join(paragraphs), chunk_size=250, overlap=0)
    for chunk in chunks:
        for line in chunk.split("\n"):
            assert line.strip() in {p.strip() for p in paragraphs}


def test_neighbouring_chunks_overlap():
    text = "\n".join(f"Предложение {i} про настройку сервера." for i in range(30))
    chunks = split_text(text, chunk_size=200, overlap=60)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.split("\n")[0] in prev


def test_long_paragraph_is_split_by_length():
    paragraph = " ".join(f"Это предложение номер {i}." for i in range(100))
    chunks = split_text(paragraph, chunk_size=150, overlap=0)
    assert all(len(c) <= 150 for c in chunks)
    assert "".join(chunks) == paragraph


def test_very_long_word_is_split_hard():
    chunks = split_text("x" * 1000, chunk_size=300, overlap=0)
    assert all(len(c) <= 300 for c in chunks)
    assert sum(len(c) for c in chunks) == 1000


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, -1)])
def test_invalid_arguments(size, overlap):
    with pytest.raises(ValueError):
        split_text("текст", chunk_size=size, overlap=overlap)
