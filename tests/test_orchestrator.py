"""
Тесты для DebateOrchestrator.
"""

import pytest
import asyncio
from gra_forum.orchestrator.orchestrator import DebateOrchestrator, RoundResult
from gra_forum.core.gra_core import GRACore
from gra_forum.agents.base import BaseAgent, AgentReply, Claim, AgentType


class MockAgent(BaseAgent):
    """Мок-агент для тестирования."""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id, AgentType.TEXT_LLM)
        self.call_count = 0
    
    async def answer(self, question: str, context: str = None) -> AgentReply:
        self.call_count += 1
        return AgentReply(
            agent_id=self.agent_id,
            raw_text=f"Response from {self.agent_id} to: {question[:30]}",
            claims=[
                Claim(
                    text=f"Claim from {self.agent_id}",
                    embedding=[0.1] * 384,
                    sources=[],
                    confidence=0.7
                )
            ],
            metadata={"test": True}
        )
    
    def get_capabilities(self) -> dict:
        return {"test": True}


class TestDebateOrchestrator:
    """Тесты для DebateOrchestrator."""
    
    @pytest.fixture
    def mock_agents(self):
        return [
            MockAgent("agent_1"),
            MockAgent("agent_2")
        ]
    
    @pytest.fixture
    def core(self):
        return GRACore()
    
    @pytest.fixture
    def orchestrator(self, mock_agents, core):
        return DebateOrchestrator(
            agents=mock_agents,
            core=core,
            max_rounds=2,
            convergence_threshold=0.05
        )
    
    @pytest.mark.asyncio
    async def test_run_debate(self, orchestrator):
        """Тест запуска дискуссии."""
        history = await orchestrator.run_debate("Test question")
        
        assert len(history) > 0
        assert len(history) <= 2  # max_rounds
        
        for result in history:
            assert isinstance(result, RoundResult)
            assert result.round_number > 0
            assert len(result.replies) == 2  # 2 агента
    
    def test_get_context_for_empty_history(self, orchestrator):
        """Тест получения контекста при пустой истории."""
        context = orchestrator._get_context_for("agent_1")
        assert context == ""
    
    def test_check_convergence_insufficient(self, orchestrator):
        """Тест проверки сходимости с недостаточно данных."""
        assert orchestrator._check_convergence() is False
    
    def test_format_next_question(self, orchestrator):
        """Тест форматирования вопроса."""
        from gra_forum.core.gra_core import DebatePlan
        
        plan = DebatePlan(
            focus_question="Focus on this",
            target_agents=["agent_1"],
            conflict_nodes=["topic_a"],
            strategy="attack",
            expected_phi_reduction={"conflict": 0.2}
        )
        
        question = orchestrator._format_next_question(plan)
        assert "Focus on this" in question
        assert "agent_1" in question
        assert "topic_a" in question
    
    def test_get_debate_log_empty(self, orchestrator):
        """Тест получения лога при пустой истории."""
        log = orchestrator.get_debate_log()
        assert "agents" in log
        assert "rounds" in log
        assert len(log["rounds"]) == 0
    
    def test_agents_dict(self, orchestrator, mock_agents):
        """Тест словаря агентов."""
        assert len(orchestrator.agents) == 2
        assert "agent_1" in orchestrator.agents
        assert "agent_2" in orchestrator.agents
