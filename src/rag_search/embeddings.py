"""Эмбеддинги на sentence-transformers с учётом соглашений моделей семейства E5."""


class SentenceTransformerEmbedder:
    """Обёртка над SentenceTransformer.

    Модели E5 обучены с префиксами ``query:`` и ``passage:``, и авторы рекомендуют
    их использовать, поэтому для таких моделей префиксы добавляются автоматически.
    Векторы нормализуются, чтобы косинусная близость в ChromaDB была корректной.
    """

    def __init__(self, model_name: str):
        # Для тестов скипаем импорт torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._use_e5_prefixes = "e5" in model_name.lower()

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, "query: ")

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, "passage: ")

    def _encode(self, texts: list[str], prefix: str) -> list[list[float]]:
        if self._use_e5_prefixes:
            texts = [prefix + t for t in texts]
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=len(texts) > 256)
        return vectors.tolist()
