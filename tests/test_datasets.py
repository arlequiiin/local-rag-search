from rag_search.datasets import title_from_text


def test_title_is_first_sentence():
    assert title_from_text("Клеточное ядро содержит молекулы ДНК. Второе предложение.") == (
        "Клеточное ядро содержит молекулы ДНК"
    )


def test_long_title_is_cut_by_word():
    title = title_from_text("Фалес был первым из философов, который использовал редукционизм. Ещё.")
    assert title == "Фалес был первым из философов, который использовал..."


def test_initials_and_abbreviations_do_not_cut_title():
    assert title_from_text("А. Н. Толстой родился в 1883 году. Он писал романы.").startswith("А. Н. Толстой")
    assert title_from_text("Сам термин (фр. philosophie) появился позже. Дальше.").startswith("Сам термин (фр. phil")
