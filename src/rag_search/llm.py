"""Генерация ответа локальной LLM через Ollama."""

SYSTEM_PROMPT = """Ты - ассистент, который отвечает на вопросы по базе знаний.

Правила:
1. Используй только информацию из переданного контекста. Ничего не придумывай.
2. Если в контексте нет ответа, так и скажи: "В базе знаний нет информации по этому вопросу".
3. Если вопрос слишком общий или не относится к базе знаний, попроси его уточнить.
4. Отвечай по делу и структурированно: для инструкций используй нумерованные шаги.
5. Если в контексте есть плейсхолдеры изображений вида [[image: имя_файла]], вставляй их
   в ответ в тех местах, где они поясняют шаг. Не пиши "см. изображение" без плейсхолдера.
6. Отвечай на языке вопроса."""


class OllamaLLM:
    """Клиент Ollama. Адрес сервера берётся из переменной ``OLLAMA_HOST`` (по умолчанию localhost:11434)."""

    def __init__(self, model: str, max_tokens: int = 1024, temperature: float = 0.2):
        # Импорт здесь, чтобы тестам не нужен был пакет ollama
        import ollama

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = ollama.Client()

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat(
            model=self.model,
            messages=messages,
            options={"num_predict": self.max_tokens, "temperature": self.temperature},
        )
        return response["message"]["content"]


def build_prompt(question: str, context: str) -> str:
    return f"""Контекст из базы знаний:
{context}

---

Вопрос: {question}

Ответ:"""
