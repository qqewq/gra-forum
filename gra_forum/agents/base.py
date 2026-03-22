"""
Level 0: Базовый интерфейс для всех агентов.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class AgentType(Enum):
    """Тип агента."""
    TEXT_LLM = "text_llm"
    PHYSICAL_SIM = "physical_sim"


@dataclass
class Source:
    """Источник утверждения (URL, DOI, расчёт, эксперимент)."""
    type: str  # "url", "doi", "calculation", "experiment", "assumption"
    reference: str
    confidence: float = 1.0


@dataclass
class Claim:
    """Извлечённое утверждение из ответа агента."""
    text: str
    embedding: Optional[List[float]] = None
    sources: List[Source] = field(default_factory=list)
    confidence: float = 0.5
    is_verifiable: bool = False


@dataclass
class AgentReply:
    """Структурированный ответ агента."""
    agent_id: str
    raw_text: str
    claims: List[Claim]
    metadata: Dict[str, Any] = field(default_factory=dict)  # latency, model_version, tokens_used и т.д.


class BaseAgent(ABC):
    """Универсальный интерфейс для любого агента Level 0."""
    
    def __init__(self, agent_id: str, agent_type: AgentType):
        self.agent_id = agent_id
        self.agent_type = agent_type
    
    @abstractmethod
    async def answer(self, question: str, context: Optional[str] = None) -> AgentReply:
        """
        Ответить на вопрос с учётом опционального контекста (история дискуссии).
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Возвращает способности агента для выбора оркестратором."""
        pass
