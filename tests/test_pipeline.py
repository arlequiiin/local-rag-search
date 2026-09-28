from conftest import BrokenLLM

from rag_search.parsing import Document

DOCS = [
    Document("Восстановление базы", "Чтобы восстановить базу данных из резервной копии, остановите сервер.", "a.txt"),
    Document("Печать чека", "Если принтер не печатает чек, проверьте бумагу и драйвер принтера.", "b.txt"),
    Document("Смена пароля", "Пароль пользователя меняется в разделе администрирования.", "c.txt"),
]


def index_all(pipeline, tags=None):
    return [pipeline.index(doc, tags=tags) for doc in DOCS]


def test_search_finds_relevant_document(pipeline):
    index_all(pipeline)
    hits = pipeline.search("принтер не печатает чек")
    assert hits[0].title == "Печать чека"
    assert 0 < hits[0].score <= 1.0001
    assert hits[0].score >= hits[-1].score


def test_search_on_empty_store(pipeline):
    assert pipeline.search("что угодно") == []


def test_inactive_documents_are_hidden(pipeline):
    ids = index_all(pipeline)
    pipeline.store.set_active(ids[1], False)
    assert all(h.title != "Печать чека" for h in pipeline.search("принтер чек"))
    assert not pipeline.store.get_document(ids[1]).active


def test_tag_filter(pipeline):
    pipeline.index(DOCS[0], tags=["sql"])
    pipeline.index(DOCS[1], tags=["касса", "принтер"])
    hits = pipeline.search("база данных", tag="касса")
    assert {h.title for h in hits} == {"Печать чека"}
    assert pipeline.store.all_tags() == ["sql", "касса", "принтер"]


def test_long_document_is_chunked_and_listed_once(pipeline):
    text = "\n".join(f"Шаг {i}: выполните действие номер {i} в настройках." for i in range(40))
    doc_id = pipeline.index(Document("Длинная", text, "long.txt"))
    [info] = pipeline.store.list_documents()
    assert info.doc_id == doc_id
    assert info.chunks == pipeline.store.count_chunks() > 1


def test_delete_document(pipeline):
    ids = index_all(pipeline)
    assert pipeline.store.delete_document(ids[0]).title == "Восстановление базы"
    assert pipeline.store.delete_document(ids[0]) is None
    assert len(pipeline.store.list_documents()) == 2


def test_ask_uses_llm_with_context(pipeline):
    index_all(pipeline)
    answer = pipeline.ask("как восстановить базу из резервной копии")
    assert answer.answer == "Ответ по контексту"
    assert answer.best_title == "Восстановление базы"
    assert "резервной копии" in pipeline.llm.prompts[0]


def test_ask_without_llm_returns_hits(pipeline):
    index_all(pipeline)
    pipeline.llm = None
    answer = pipeline.ask("смена пароля")
    assert answer.answer is None and answer.error is None
    assert answer.hits[0].title == "Смена пароля"


def test_ask_survives_llm_failure(pipeline):
    index_all(pipeline)
    pipeline.llm = BrokenLLM()
    answer = pipeline.ask("смена пароля")
    assert answer.answer is None
    assert "ollama не запущена" in answer.error
    assert answer.hits
