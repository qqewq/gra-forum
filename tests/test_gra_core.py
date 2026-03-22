"""
Тесты для GRA Core.
"""

import pytest
import numpy as np
from gra_forum.core.gra_core import GRACore, DebatePlan, DebateState
from gra_forum.core.metrics import FoamMetrics


class TestDebateState:
    """Тесты для DebateState."""
    
    def test_initial_state(self):
        """Тест начального состояния."""
        state = DebateState()
        assert state.current_round_replies == []
        assert state.agent_ids == set()
    
    def test_add_round(self):
        """Тест добавления раунда."""
        from gra_forum.agents.base import AgentReply, Claim
        
        state = DebateState()
        reply = AgentReply(
            agent_id="test_agent",
            raw_text="Test response",
            claims=[Claim(text="Test claim")]
        )
        state.add_round([reply], "Test question")
        
        assert len(state.current_round_replies) == 1
        assert "test_agent" in state.agent_ids


class TestGRACore:
    """Тесты для GRACore."""
    
    @pytest.fixture
    def core(self):
        return GRACore()
    
    @pytest.fixture
    def sample_metrics(self):
        return FoamMetrics(
            conflict=0.3,
            vacuity=0.2,
            redundancy=0.1,
            noise=0.05,
            discovery_score=0.5
        )
    
    def test_default_weights(self, core):
        """Тест весов по умолчанию."""
        assert core.weights["conflict"] == 0.35
        assert core.weights["vacuity"] == 0.30
        assert core.weights["redundancy"] == 0.25
        assert core.weights["noise"] == 0.10
    
    def test_compute_J(self, core, sample_metrics):
        """Тест вычисления J."""
        J = core.compute_J(sample_metrics)
        expected = (
            0.35 * 0.3 +
            0.30 * 0.2 +
            0.25 * 0.1 +
            0.10 * 0.05
        )
        assert abs(J - expected) < 1e-6
    
    def test_compute_J_with_discovery_penalty(self, core):
        """Тест штрафа за низкий discovery score."""
        metrics = FoamMetrics(
            conflict=0.1,
            vacuity=0.1,
            redundancy=0.1,
            noise=0.0,
            discovery_score=0.1  # Ниже protection threshold
        )
        J = core.compute_J(metrics)
        base_J = (
            0.35 * 0.1 +
            0.30 * 0.1 +
            0.25 * 0.1 +
            0.10 * 0.0
        )
        assert J > base_J  # Должен быть штраф
    
    def test_propose_next_round_conflict(self, core):
        """Тест планирования при высоком conflict."""
        state = DebateState()
        metrics = FoamMetrics(
            conflict=0.8,  # Высокий
            vacuity=0.2,
            redundancy=0.1,
            noise=0.0
        )
        
        plan = core.propose_next_round(state, metrics)
        assert plan.strategy == "attack"
        assert "conflict" in plan.expected_phi_reduction
    
    def test_propose_next_round_vacuity(self, core):
        """Тест планирования при высоком vacuity."""
        state = DebateState()
        metrics = FoamMetrics(
            conflict=0.2,
            vacuity=0.8,  # Высокий
            redundancy=0.1,
            noise=0.0
        )
        
        plan = core.propose_next_round(state, metrics)
        assert plan.strategy == "verify"
        assert "vacuity" in plan.expected_phi_reduction
    
    def test_propose_next_round_redundancy(self, core):
        """Тест планирования при высоком redundancy."""
        state = DebateState()
        metrics = FoamMetrics(
            conflict=0.2,
            vacuity=0.1,
            redundancy=0.8,  # Высокий
            noise=0.0
        )
        
        plan = core.propose_next_round(state, metrics)
        assert plan.strategy == "synthesize"
        assert "redundancy" in plan.expected_phi_reduction
    
    def test_get_embedding(self, core):
        """Тест получения эмбеддинга."""
        embedding = core._get_embedding("test text")
        assert len(embedding) == 384
        assert isinstance(embedding, list)
    
    def test_is_decreasing_true(self, core):
        """Тест определения убывающей последовательности."""
        values = [1.0, 0.8, 0.6, 0.4, 0.2]
        assert core._is_decreasing(values) is True
    
    def test_is_decreasing_false(self, core):
        """Тест определения неубывающей последовательности."""
        values = [0.2, 0.4, 0.6, 0.8, 1.0]
        assert core._is_decreasing(values) is False
    
    def test_optimization_trajectory_insufficient(self, core):
        """Тест траектории с недостаточно данных."""
        trajectory = core.get_optimization_trajectory()
        assert trajectory["status"] == "insufficient_data"
