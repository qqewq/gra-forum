"""
GRA Core: Ядро системы для минимизации аргументационной пены.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .metrics import PhiCalculator, FoamMetrics


@dataclass
class DebatePlan:
    """План следующего раунда, сгенерированный GRA-ядром."""
    focus_question: str
    target_agents: List[str]  # Кого спрашивать
    conflict_nodes: List[str]   # Какие противоречия разрешать
    strategy: str               # "attack", "synthesize", "verify"
    expected_phi_reduction: Dict[str, float]  # Ожидаемое снижение Φ


class DebateState:
    """Состояние дискуссии для передачи в GRA-ядро."""
    
    def __init__(self):
        self.current_round_replies: List[Any] = []
        self.agent_ids: set = set()
        self.agent_capabilities: Dict[str, Dict] = {}
        self.round_history: List[Dict] = []
    
    def add_round(self, replies: List[Any], question: str):
        """Добавляет раунд в историю."""
        self.current_round_replies = replies
        for reply in replies:
            self.agent_ids.add(reply.agent_id)
        self.round_history.append({
            "question": question,
            "replies": replies
        })
    
    def find_conflict_pairs(self) -> List[Tuple[str, str, str]]:
        """Находит пары агентов с конфликтующими позициями."""
        # Упрощённая реализация
        conflicts = []
        if len(self.agent_ids) >= 2:
            agents = list(self.agent_ids)
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    conflicts.append((agents[i], agents[j], "topic"))
        return conflicts
    
    def find_most_vacuous_claims(self, top_k: int = 3) -> List[Dict]:
        """Находит утверждения с минимумом источников."""
        vacuous = []
        for reply in self.current_round_replies:
            for claim in reply.claims:
                sources_count = len(claim.sources) if claim.sources else 0
                vacuous.append({
                    "agent_id": reply.agent_id,
                    "text": claim.text,
                    "sources_count": sources_count
                })
        vacuous.sort(key=lambda x: x["sources_count"])
        return vacuous[:top_k]
    
    def cluster_claims(self) -> List[List[Dict]]:
        """Кластеризует утверждения по сходству."""
        # Упрощённая реализация — каждый агент в своём кластере
        clusters = []
        agent_claims: Dict[str, List[Dict]] = {}
        for reply in self.current_round_replies:
            if reply.agent_id not in agent_claims:
                agent_claims[reply.agent_id] = []
            for claim in reply.claims:
                agent_claims[reply.agent_id].append({
                    "agent_id": reply.agent_id,
                    "text": claim.text
                })
        for claims in agent_claims.values():
            clusters.append(claims)
        return clusters


class GRACore:
    """
    Level 2: GRA-ядро.
    Управляет глобальным функционалом J и стратегией минимизации пены.
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        phi_calculator: Optional[PhiCalculator] = None,
        discovery_protection: float = 0.3  # Минимальный discovery score
    ):
        # Веса для J = w_c*Φ_c + w_v*Φ_v + w_r*Φ_r + w_n*Φ_n
        self.weights = weights or {
            "conflict": 0.35,
            "vacuity": 0.30,
            "redundancy": 0.25,
            "noise": 0.10
        }
        self.phi_calc = phi_calculator or PhiCalculator()
        self.discovery_protection = discovery_protection
        self.phi_history: List[FoamMetrics] = []
    
    def compute_phi(self, state: DebateState) -> FoamMetrics:
        """
        Вычисляет все метрики пены Φ для текущего состояния дискуссии.
        """
        # Извлекаем claims из состояния
        claims = []
        for reply in state.current_round_replies:
            for claim in reply.claims:
                claims.append({
                    "agent_id": reply.agent_id,
                    "text": claim.text,
                    "embedding": claim.embedding or self._get_embedding(claim.text),
                    "sources": [{"type": s.type, "reference": s.reference} 
                               for s in (claim.sources or [])],
                    "is_verifiable": claim.is_verifiable
                })
        
        metrics = self.phi_calc.compute_all(claims, state.agent_capabilities)
        self.phi_history.append(metrics)
        return metrics
    
    def compute_J(self, metrics: FoamMetrics) -> float:
        """
        Глобальный функционал: сумма взвешенных Φ.
        
        Ключевое условие: если discovery_score падает ниже порога,
        добавляем штраф (не даём убить конструктивный спор).
        """
        J = (
            self.weights["conflict"] * metrics.conflict +
            self.weights["vacuity"] * metrics.vacuity +
            self.weights["redundancy"] * metrics.redundancy +
            self.weights["noise"] * metrics.noise
        )
        
        # Защита discovery: если новые смыслы исчезают — штрафуем
        if metrics.discovery_score < self.discovery_protection:
            penalty = (self.discovery_protection - metrics.discovery_score) * 2.0
            J += penalty
        
        return float(J)
    
    def propose_next_round(
        self,
        state: DebateState,
        current_phi: FoamMetrics
    ) -> DebatePlan:
        """
        Генерирует план следующего раунда для минимизации J.
        
        Стратегия:
        1. Если высокий Φ_conflict → направить агентов спорить по конкретным узлам
        2. Если высокий Φ_vacuity → требовать источники/расчёты
        3. Если высокий Φ_redundancy → перенаправить агентов на разные аспекты
        """
        # Анализируем, какая Φ доминирует
        phis = {
            "conflict": current_phi.conflict,
            "vacuity": current_phi.vacuity,
            "redundancy": current_phi.redundancy
        }
        dominant = max(phis, key=phis.get)
        
        # Выбор стратегии
        if dominant == "conflict":
            return self._plan_conflict_resolution(state, current_phi)
        elif dominant == "vacuity":
            return self._plan_verification_challenge(state, current_phi)
        else:  # redundancy
            return self._plan_diversification(state, current_phi)
    
    def _plan_conflict_resolution(
        self,
        state: DebateState,
        phi: FoamMetrics
    ) -> DebatePlan:
        """Стратегия: столкнуть агентов с противоположными позициями."""
        # Находим пары агентов с конфликтующими claims
        conflict_pairs = state.find_conflict_pairs()
        
        if not conflict_pairs:
            # Если явных конфликтов нет, но Φ_conflict высок — 
            # это скрытые конфликты, нужна провокация
            return DebatePlan(
                focus_question="Challenge: Identify hidden assumptions in previous claims. "
                              "Explicitly state what you disagree with and why.",
                target_agents=list(state.agent_ids),
                conflict_nodes=["hidden_assumptions"],
                strategy="attack",
                expected_phi_reduction={"conflict": 0.2, "vacuity": 0.0}
            )
        
        agent_a, agent_b, topic = conflict_pairs[0]
        
        return DebatePlan(
            focus_question=f"Direct response required: Agent {agent_a} claims X, "
                          f"Agent {agent_b} claims not-X regarding '{topic}'. "
                          f"Provide evidence or concede.",
            target_agents=[agent_a, agent_b],
            conflict_nodes=[topic],
            strategy="attack",
            expected_phi_reduction={"conflict": 0.3, "vacuity": 0.1}
        )
    
    def _plan_verification_challenge(
        self,
        state: DebateState,
        phi: FoamMetrics
    ) -> DebatePlan:
        """Стратегия: потребовать источники для самых пустых утверждений."""
        # Находим claims с минимумом источников
        vacuous_claims = state.find_most_vacuous_claims(top_k=3)
        
        target_agents = list(set(c["agent_id"] for c in vacuous_claims))
        
        claims_text = "; ".join([c["text"][:50] + "..." for c in vacuous_claims])
        
        return DebatePlan(
            focus_question=f"Verification required: Provide sources, calculations, "
                          f"or experimental evidence for: {claims_text}",
            target_agents=target_agents,
            conflict_nodes=[],
            strategy="verify",
            expected_phi_reduction={"conflict": 0.0, "vacuity": 0.4}
        )
    
    def _plan_diversification(
        self,
        state: DebateState,
        phi: FoamMetrics
    ) -> DebatePlan:
        """Стратегия: развести агентов на разные аспекты проблемы."""
        # Находим кластеры похожих claims
        clusters = state.cluster_claims()
        
        # Выбираем агентов, которые "эхоят" друг друга
        redundant_agents = []
        for cluster in clusters:
            if len(cluster) > 1:
                redundant_agents.extend([c["agent_id"] for c in cluster])
        
        # Переназначаем роли
        aspects = ["mechanism", "evidence", "implications", "limitations"]
        
        return DebatePlan(
            focus_question=f"Avoid repetition. Each agent focuses on ONE aspect: "
                          f"{', '.join(aspects[:len(state.agent_ids)])}. "
                          f"Bring new information, not agreement.",
            target_agents=list(state.agent_ids),
            conflict_nodes=[],
            strategy="synthesize",
            expected_phi_reduction={"redundancy": 0.3, "conflict": 0.1}
        )
    
    def _get_embedding(self, text: str) -> List[float]:
        """Получает эмбеддинг текста (заглушка для MVP)."""
        # В реальности: вызов OpenAI/text-embedding-3-small или локальной модели
        # Для MVP: случайный вектор (не использовать в продакшене!)
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(384).tolist()
    
    def get_optimization_trajectory(self) -> Dict:
        """Анализирует, как менялись Φ и J."""
        if len(self.phi_history) < 2:
            return {"status": "insufficient_data"}
        
        return {
            "J_trend": "decreasing" if self._is_decreasing([self.compute_J(p) for p in self.phi_history]) else "stable",
            "phi_trends": {
                name: "decreasing" if self._is_decreasing([getattr(p, name) for p in self.phi_history]) else "stable"
                for name in ["conflict", "vacuity", "redundancy"]
            },
            "discovery_protected": all(p.discovery_score > self.discovery_protection for p in self.phi_history)
        }
    
    def _is_decreasing(self, values: List[float]) -> bool:
        """Проверяет монотонное убывание (с допуском на шум)."""
        if len(values) < 2:
            return False
        # Линейная регрессия: отрицательный наклон?
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope < -0.01  # Допуск на шум
