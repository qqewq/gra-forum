"""
Level 1: Форум-оркестратор.
Управляет потоком данных между агентами и GRA-ядром.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

from ..agents.base import BaseAgent, AgentReply
from ..core.gra_core import GRACore, DebatePlan, DebateState
from ..core.metrics import FoamMetrics


@dataclass
class RoundResult:
    """Результат одного раунда дискуссии."""
    round_number: int
    question: str
    replies: List[AgentReply]
    metrics: FoamMetrics
    J_value: float
    timestamp: datetime = field(default_factory=datetime.now)


class DebateOrchestrator:
    """
    Level 1: Форум-оркестратор.
    Управляет потоком данных между агентами и GRA-ядром.
    """
    
    def __init__(
        self,
        agents: List[BaseAgent],
        core: GRACore,
        max_rounds: int = 5,
        convergence_threshold: float = 0.05
    ):
        self.agents = {a.agent_id: a for a in agents}
        self.core = core
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold
        self.history: List[RoundResult] = []
        self.state: DebateState = DebateState()
    
    async def run_debate(self, initial_question: str) -> List[RoundResult]:
        """
        Главный цикл дискуссии.
        """
        current_question = initial_question
        
        for round_num in range(1, self.max_rounds + 1):
            print(f"\n=== Round {round_num} ===")
            print(f"Question: {current_question[:100]}...")
            
            # 1. Сбор ответов агентов (параллельно)
            replies = await self._gather_replies(current_question)
            
            # 2. Обновление состояния дискуссии
            self.state.add_round(replies, current_question)
            
            # 3. Расчёт метрик пены Φ (передаём в GRA-ядро)
            metrics = self.core.compute_phi(self.state)
            
            # 4. Расчёт глобального функционала J
            J = self.core.compute_J(metrics)
            
            # 5. Сохранение результата раунда
            result = RoundResult(
                round_number=round_num,
                question=current_question,
                replies=replies,
                metrics=metrics,
                J_value=J
            )
            self.history.append(result)
            
            print(f"J = {J:.4f} (conflict={metrics.conflict:.3f}, "
                  f"vacuity={metrics.vacuity:.3f}, redundancy={metrics.redundancy:.3f})")
            
            # 6. Проверка сходимости
            if self._check_convergence():
                print("Convergence reached.")
                break
            
            # 7. Планирование следующего раунда (GRA-ядро)
            if round_num < self.max_rounds:
                plan: DebatePlan = self.core.propose_next_round(self.state, metrics)
                current_question = self._format_next_question(plan)
        
        return self.history
    
    async def _gather_replies(self, question: str) -> List[AgentReply]:
        """Параллельный сбор ответов от всех агентов."""
        import asyncio
        tasks = [
            agent.answer(question, self._get_context_for(agent.agent_id))
            for agent in self.agents.values()
        ]
        return await asyncio.gather(*tasks)
    
    def _get_context_for(self, agent_id: str) -> str:
        """Формирует контекст (историю) для конкретного агента."""
        if not self.history:
            return ""
        
        # Последние 2 раунда в компактной форме
        recent = self.history[-2:]
        context_parts = []
        for r in recent:
            other_claims = [
                f"{rep.agent_id}: {c.text[:100]}"
                for rep in r.replies 
                for c in rep.claims 
                if rep.agent_id != agent_id
            ]
            context_parts.append(f"Round {r.round_number}:\n" + "\n".join(other_claims))
        
        return "\n\n".join(context_parts)
    
    def _format_next_question(self, plan: DebatePlan) -> str:
        """Форматирует план GRA в текст вопроса для агентов."""
        parts = [plan.focus_question]
        
        if plan.target_agents:
            parts.append(f"Address to: {', '.join(plan.target_agents)}")
        
        if plan.conflict_nodes:
            parts.append(f"Resolve conflicts: {plan.conflict_nodes}")
        
        return "\n".join(parts)
    
    def _check_convergence(self) -> bool:
        """Проверяет, стабилизировались ли метрики."""
        if len(self.history) < 2:
            return False
        
        prev_J = self.history[-2].J_value
        curr_J = self.history[-1].J_value
        return abs(curr_J - prev_J) / (abs(prev_J) + 1e-6) < self.convergence_threshold
    
    def get_debate_log(self) -> Dict[str, Any]:
        """Экспорт полного лога для анализа."""
        return {
            "agents": list(self.agents.keys()),
            "rounds": [
                {
                    "round": r.round_number,
                    "J": r.J_value,
                    "metrics": {
                        "conflict": r.metrics.conflict,
                        "vacuity": r.metrics.vacuity,
                        "redundancy": r.metrics.redundancy,
                        "noise": r.metrics.noise
                    },
                    "claims_count": sum(len(rep.claims) for rep in r.replies)
                }
                for r in self.history
            ]
        }
    
    def plot_J(self, save_path: Optional[str] = None):
        """Визуализация динамики J по раундам."""
        import matplotlib.pyplot as plt
        
        rounds = [r.round_number for r in self.history]
        J_values = [r.J_value for r in self.history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, J_values, 'b-o', linewidth=2, markersize=8)
        plt.xlabel('Round')
        plt.ylabel('Global Functional J')
        plt.title('GRA Forum: Foam Reduction Dynamics')
        plt.grid(True, alpha=0.3)
        
        # Аннотации метрик
        for i, r in enumerate(self.history):
            plt.annotate(
                f"Φc={r.metrics.conflict:.2f}",
                (r.round_number, r.J_value),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=8
            )
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
