"""Веб-интерфейс. Запуск: ``streamlit run app/streamlit_app.py``."""

import streamlit as st

from rag_search.parsing import IMAGE_PLACEHOLDER_RE
from rag_search.service import KnowledgeBase, build_knowledge_base

st.set_page_config(page_title="local-rag-search", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Загрузка модели эмбеддингов...")
def knowledge_base() -> KnowledgeBase:
    return build_knowledge_base()


def show_image(kb: KnowledgeBase, name: str, width: int | None = None) -> None:
    path = kb.image_path(name)
    if path is None:
        st.caption(f"🖼️ {name} (файл не найден)")
    else:
        st.image(str(path), width=width)


def render_with_images(kb: KnowledgeBase, text: str) -> None:
    """Выводит текст, подставляя картинки на место плейсхолдеров ``[[image: ...]]``."""
    parts = IMAGE_PLACEHOLDER_RE.split(text)
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                st.markdown(part.strip())
        else:
            show_image(kb, part.strip(), width=480)


def search_tab(kb: KnowledgeBase, top_k: int, tag: str | None) -> None:
    with st.form("search"):
        question = st.text_input("Вопрос", placeholder="Например: как восстановить базу из резервной копии?")
        use_llm = st.toggle("Сгенерировать ответ через LLM", value=True)
        submitted = st.form_submit_button("Найти", type="primary")
    if not submitted or not question.strip():
        return

    if use_llm:
        with st.spinner("Поиск и генерация ответа..."):
            result = kb.pipeline.ask(question, top_k=top_k, tag=tag)
        hits = result.hits
        st.subheader("Ответ")
        if result.error:
            st.warning(result.error + " Ниже - найденные фрагменты.")
        if result.answer:
            render_with_images(kb, result.answer)
            shown = set(IMAGE_PLACEHOLDER_RE.findall(result.answer))
            # Картинки лучшего документа, которые LLM не вставила в ответ, показываем ниже в ряд
            rest = [kb.image_path(img) for img in result.images if img not in shown]
            rest = [str(path) for path in rest if path is not None]
            if rest:
                st.markdown(f"**Изображения из {result.best_title}**")
                st.image(rest, width=300)
    else:
        with st.spinner("Поиск..."):
            hits = kb.pipeline.search(question, top_k=top_k, tag=tag)

    st.subheader("Источники")
    if not hits:
        st.info("Ничего не найдено.")
    for i, hit in enumerate(hits, 1):
        with st.expander(f"{i}. {hit.title} - близость {hit.score:.2f}", expanded=not use_llm and i == 1):
            render_with_images(kb, hit.text)
            st.caption(
                f"Фрагмент {hit.metadata['chunk_index'] + 1}/{hit.metadata['total_chunks']}"
                f", файл: {hit.metadata['source']}"
            )


def upload_tab(kb: KnowledgeBase) -> None:
    files = st.file_uploader("Файлы (.docx, .md, .txt)", type=["docx", "md", "txt"], accept_multiple_files=True)
    split_sections = st.radio(
        "Структура файла",
        options=[False, True],
        format_func=lambda v: "Разделить на инструкции по строкам ---" if v else "Один файл - одна инструкция",
        horizontal=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        author = st.text_input("Автор")
    with col2:
        existing = st.multiselect("Теги", options=kb.store.all_tags())
    extra = st.text_input("Новые теги через запятую")
    tags = existing + [t.strip() for t in extra.split(",") if t.strip()]

    if st.button("Загрузить", type="primary", disabled=not files):
        for file in files:
            try:
                with st.spinner(f"Индексация {file.name}..."):
                    ids = kb.ingest_bytes(
                        file.name, file.getvalue(), tags=tags, author=author, split_sections=split_sections
                    )
                st.success(f"{file.name}: добавлено инструкций - {len(ids)}")
            except Exception as exc:
                st.error(f"{file.name}: {exc}")


def manage_tab(kb: KnowledgeBase) -> None:
    documents = kb.store.list_documents()
    active = sum(d.active for d in documents)
    c1, c2, c3 = st.columns(3)
    c1.metric("Актуальных", active)
    c2.metric("Неактуальных", len(documents) - active)
    c3.metric("Фрагментов", kb.store.count_chunks())

    c1, c2, c3 = st.columns([2, 1, 1])
    query = c1.text_input("Фильтр по названию")
    tag = c2.selectbox("Тег", ["Все", *kb.store.all_tags()])
    show_inactive = c3.checkbox("Показывать неактуальные")

    shown = [
        d
        for d in documents
        if (show_inactive or d.active) and (tag == "Все" or tag in d.tags) and query.lower() in d.title.lower()
    ]
    st.caption(f"Показано: {len(shown)} из {len(documents)}")

    for doc in shown[:200]:
        with st.expander(f"{'✅' if doc.active else '⛔'} {doc.title}"):
            st.write(f"**Файл:** {doc.source}, **Фрагментов:** {doc.chunks}, **Автор:** {doc.author or '-'}")
            st.write(f"**Добавлено:** {doc.created_at}, **Теги:** {', '.join(doc.tags) or '-'}")
            b1, b2, _ = st.columns([1, 1, 3])
            label = "Пометить неактуальным" if doc.active else "Вернуть в поиск"
            if b1.button(label, key=f"toggle_{doc.doc_id}"):
                kb.store.set_active(doc.doc_id, not doc.active)
                st.rerun()
            if b2.button("Удалить", key=f"delete_{doc.doc_id}"):
                kb.delete(doc.doc_id)
                st.rerun()
    if len(shown) > 200:
        st.caption("Показаны первые 200 - уточните фильтр.")

    with st.expander("Опасная зона"):
        confirm = st.text_input("Чтобы удалить всю базу, введите 'удалить'")
        if st.button("Очистить базу", disabled=confirm.strip().lower() != "удалить"):
            kb.clear()
            st.rerun()


def main() -> None:
    st.title("🔎 Поиск по базе знаний")
    try:
        kb = knowledge_base()
    except Exception as exc:
        st.error(f"Не удалось инициализировать систему: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Настройки поиска")
        top_k = st.slider("Сколько фрагментов искать", 1, 10, kb.settings.top_k)
        tag = st.selectbox("Искать только по тегу", ["Все", *kb.store.all_tags()])
        st.divider()
        st.caption(f"Эмбеддинги: `{kb.settings.embedding_model}`")
        st.caption(f"LLM: `{kb.settings.llm_model}` (Ollama)")

    search, upload, manage = st.tabs(["Поиск", "Загрузка", "База знаний"])
    with search:
        search_tab(kb, top_k, None if tag == "Все" else tag)
    with upload:
        upload_tab(kb)
    with manage:
        manage_tab(kb)


main()
