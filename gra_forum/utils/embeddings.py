"""
Работа с эмбеддингами для метрик.
"""

from typing import List, Optional
import numpy as np


class EmbeddingProvider:
    """Провайдер эмбеддингов для текстов."""
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        dimension: int = 384,
        use_local: bool = False,
        local_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model_name = model_name
        self.dimension = dimension
        self.use_local = use_local
        self.local_model = local_model
        self._local_model_instance = None
        self._openai_client = None
    
    def _get_openai_client(self):
        """Ленивая инициализация OpenAI клиента."""
        if self._openai_client is None:
            try:
                import openai
                self._openai_client = openai.AsyncOpenAI()
            except ImportError:
                raise ImportError("OpenAI package not installed. Install with: pip install openai")
        return self._openai_client
    
    def _get_local_model(self):
        """Ленивая инициализация локальной модели."""
        if self._local_model_instance is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._local_model_instance = SentenceTransformer(self.local_model)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self._local_model_instance
    
    async def get_embedding(self, text: str) -> List[float]:
        """Получает эмбеддинг для текста."""
        if self.use_local:
            return await self._get_local_embedding(text)
        else:
            return await self._get_openai_embedding(text)
    
    async def _get_openai_embedding(self, text: str) -> List[float]:
        """Получает эмбеддинг через OpenAI API."""
        client = self._get_openai_client()
        response = await client.embeddings.create(
            model=self.model_name,
            input=text
        )
        return response.data[0].embedding
    
    async def _get_local_embedding(self, text: str) -> List[float]:
        """Получает эмбеддинг через локальную модель."""
        model = self._get_local_model()
        embedding = model.encode(text)
        return embedding.tolist()
    
    def get_embedding_sync(self, text: str) -> List[float]:
        """Синхронная версия получения эмбеддинга (для тестов)."""
        if self.use_local:
            model = self._get_local_model()
            embedding = model.encode(text)
            return embedding.tolist()
        else:
            # Для OpenAI используем случайный вектор в синхронном режиме
            np.random.seed(hash(text) % 2**32)
            return np.random.randn(self.dimension).tolist()


# Глобальный провайдер
_default_provider: Optional[EmbeddingProvider] = None


def get_embedding_provider() -> EmbeddingProvider:
    """Возвращает глобальный провайдер эмбеддингов."""
    global _default_provider
    if _default_provider is None:
        _default_provider = EmbeddingProvider()
    return _default_provider


def set_embedding_provider(provider: EmbeddingProvider):
    """Устанавливает глобальный провайдер эмбеддингов."""
    global _default_provider
    _default_provider = provider
