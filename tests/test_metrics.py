"""
Тесты для метрик пены Φ.
"""

import pytest
import numpy as np
from gra_forum.core.metrics import FoamMetrics, PhiCalculator


class TestFoamMetrics:
    """Тесты для FoamMetrics."""
    
    def test_basic_creation(self):
        """Тест создания FoamMetrics."""
        metrics = FoamMetrics(
            conflict=0.5,
            vacuity=0.3,
            redundancy=0.2,
            noise=0.1
        )
        assert metrics.conflict == 0.5
        assert metrics.vacuity == 0.3
        assert metrics.redundancy == 0.2
        assert metrics.noise == 0.1
    
    def test_default_discovery(self):
        """Тест значений по умолчанию."""
        metrics = FoamMetrics(0, 0, 0, 0)
        assert metrics.discovery_score == 0.0
        assert metrics.total_claims == 0


class TestPhiCalculator:
    """Тесты для PhiCalculator."""
    
    @pytest.fixture
    def calculator(self):
        return PhiCalculator()
    
    @pytest.fixture
    def sample_claims(self):
        """Создаёт тестовые claims с эмбеддингами."""
        np.random.seed(42)
        return [
            {
                "agent_id": "agent_1",
                "text": "AI can solve complex problems",
                "embedding": np.random.randn(384).tolist(),
                "sources": [{"type": "url", "reference": "https://example.com"}],
                "is_verifiable": True
            },
            {
                "agent_id": "agent_2",
                "text": "AI cannot solve all problems",
                "embedding": np.random.randn(384).tolist(),
                "sources": [],
                "is_verifiable": False
            }
        ]
    
    def test_compute_all_empty(self, calculator):
        """Тест с пустым списком claims."""
        metrics = calculator.compute_all([], {})
        assert metrics.conflict == 0
        assert metrics.vacuity == 0
        assert metrics.redundancy == 0
    
    def test_compute_vacuity_no_sources(self, calculator):
        """Тест vacuity для claim без источников."""
        claims = [{
            "agent_id": "agent_1",
            "text": "Short text",
            "embedding": np.random.randn(384).tolist(),
            "sources": [],
            "is_verifiable": False
        }]
        vacuity = calculator._compute_vacuity(claims)
        assert vacuity > 0.5  # Должно быть высоким
    
    def test_compute_vacuity_with_sources(self, calculator):
        """Тест vacuity для claim с источниками."""
        claims = [{
            "agent_id": "agent_1",
            "text": "This is a longer text with 123 numbers for verification",
            "embedding": np.random.randn(384).tolist(),
            "sources": [{"type": "doi", "reference": "10.1000/xyz"}],
            "is_verifiable": True
        }]
        vacuity = calculator._compute_vacuity(claims)
        assert vacuity < 0.5  # Должно быть низким
    
    def test_compute_redundancy_same_agent(self, calculator):
        """Тест redundancy для claims от одного агента."""
        embedding = np.random.randn(384).tolist()
        claims = [
            {
                "agent_id": "agent_1",
                "text": "Same idea",
                "embedding": embedding,
                "sources": [{"type": "url", "reference": "ref1"}]
            },
            {
                "agent_id": "agent_1",
                "text": "Same idea repeated",
                "embedding": embedding,
                "sources": [{"type": "url", "reference": "ref1"}]
            }
        ]
        # Устанавливаем низкий порог для теста
        calculator.redundancy_threshold = 0.5
        redundancy = calculator._compute_redundancy(claims)
        assert redundancy == 1.0  # Полная избыточность
    
    def test_discovery_score(self, calculator):
        """Тест discovery score."""
        np.random.seed(42)
        claims = [
            {"agent_id": f"agent_{i}", "text": f"Claim {i}", "embedding": np.random.randn(384).tolist()}
            for i in range(5)
        ]
        score = calculator._compute_discovery_score(claims)
        assert 0 <= score <= 1
    
    def test_is_bad_duplicate_same_agent(self, calculator):
        """Тест определения плохого дубля от одного агента."""
        c1 = {"agent_id": "agent_1", "sources": []}
        c2 = {"agent_id": "agent_1", "sources": []}
        assert calculator._is_bad_duplicate(c1, c2) is True
    
    def test_is_bad_duplicate_different_agents(self, calculator):
        """Тест определения дубля от разных агентов."""
        c1 = {"agent_id": "agent_1", "sources": [{"reference": "ref1"}]}
        c2 = {"agent_id": "agent_2", "sources": [{"reference": "ref2"}]}
        assert calculator._is_bad_duplicate(c1, c2) is False
