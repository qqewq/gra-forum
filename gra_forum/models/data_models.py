"""
Pydantic-модели данных для GRA Forum.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class SourceModel(BaseModel):
    """Модель источника утверждения."""
    type: str = Field(..., description="Тип источника: url, doi, calculation, experiment, assumption")
    reference: str = Field(..., description="Ссылка или идентификатор источника")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ClaimModel(BaseModel):
    """Модель утверждения."""
    text: str = Field(..., description="Текст утверждения")
    embedding: Optional[List[float]] = Field(default=None, description="Векторное представление")
    sources: List[SourceModel] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    is_verifiable: bool = Field(default=False)


class AgentReplyModel(BaseModel):
    """Модель ответа агента."""
    agent_id: str
    raw_text: str
    claims: List[ClaimModel]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class DebateRound(BaseModel):
    """Модель раунда дискуссии."""
    round_number: int
    question: str
    replies: List[AgentReplyModel]
    metrics: Dict[str, float] = Field(default_factory=dict)
    J_value: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=datetime.now)


class DebateState(BaseModel):
    """Модель состояния дискуссии."""
    rounds: List[DebateRound] = Field(default_factory=list)
    agent_ids: List[str] = Field(default_factory=list)
    current_round: int = Field(default=0)
    converged: bool = Field(default=False)
    
    def add_round(self, round_data: DebateRound):
        """Добавляет раунд в историю."""
        self.rounds.append(round_data)
        self.current_round = round_data.round_number
        
        # Обновляем список агентов
        for reply in round_data.replies:
            if reply.agent_id not in self.agent_ids:
                self.agent_ids.append(reply.agent_id)
    
    def get_latest_round(self) -> Optional[DebateRound]:
        """Возвращает последний раунд."""
        if self.rounds:
            return self.rounds[-1]
        return None
    
    def get_J_trajectory(self) -> List[float]:
        """Возвращает траекторию J."""
        return [r.J_value for r in self.rounds]
