"""
Конфигурация GRA Forum.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class GRAConfig:
    """Глобальная конфигурация системы."""
    
    # Веса для функционала J
    weights: Dict[str, float] = field(default_factory=lambda: {
        "conflict": 0.35,
        "vacuity": 0.30,
        "redundancy": 0.25,
        "noise": 0.10
    })
    
    # Пороги для метрик Φ
    conflict_threshold: float = 0.85
    redundancy_threshold: float = 0.92
    
    # Защита discovery
    discovery_protection: float = 0.3
    
    # Параметры оркестратора
    max_rounds: int = 5
    convergence_threshold: float = 0.05
    
    # API ключи (загружаются из переменных окружения)
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    deepseek_api_key: Optional[str] = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY"))
    perplexity_api_key: Optional[str] = field(default_factory=lambda: os.getenv("PERPLEXITY_API_KEY"))
    kimi_api_key: Optional[str] = field(default_factory=lambda: os.getenv("KIMI_API_KEY"))
    
    # Настройки эмбеддингов
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 384
    use_local_embeddings: bool = False
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует конфигурацию в словарь."""
        return {
            "weights": self.weights,
            "conflict_threshold": self.conflict_threshold,
            "redundancy_threshold": self.redundancy_threshold,
            "discovery_protection": self.discovery_protection,
            "max_rounds": self.max_rounds,
            "convergence_threshold": self.convergence_threshold,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "use_local_embeddings": self.use_local_embeddings,
            "local_embedding_model": self.local_embedding_model,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GRAConfig":
        """Создаёт конфигурацию из словаря."""
        return cls(**data)


# Глобальный экземпляр конфигурации
config = GRAConfig()
