#!/usr/bin/env python3
"""
Пример: Дискуссия ИИ-агентов по проблеме масштабирования когнитивности.
Реализация за 2-4 недели (1 разработчик).
"""

import asyncio
import os
from dotenv import load_dotenv

# Импорты gra-forum
from gra_forum.orchestrator.orchestrator import DebateOrchestrator
from gra_forum.agents import RoleBasedAgent, PhysicalAgent
from gra_forum.core import GRACore, PhiCalculator

load_dotenv()


async def main():
    # === 1. Инициализация агентов (Level 0) ===
    
    # Агент 1: GRA-оптимист (ищет синтез)
    optimist = RoleBasedAgent(
        agent_id="gra_optimist",
        role="gra_optimist",
        llm_backend="openai",  # или любой другой
        api_key=os.getenv("OPENAI_API_KEY", "stub-key")
    )
    
    # Агент 2: Физик-скептик (проверяет ограничения)
    skeptic = RoleBasedAgent(
        agent_id="physics_skeptic",
        role="physics_skeptic",
        llm_backend="anthropic",
        api_key=os.getenv("ANTHROPIC_API_KEY", "stub-key")
    )
    
    # Агент 3: Методолог (проверяет формальную корректность)
    methodologist = RoleBasedAgent(
        agent_id="methodologist",
        role="methodologist",
        llm_backend="deepseek",
        api_key=os.getenv("DEEPSEEK_API_KEY", "stub-key")
    )
    
    # Stub физического агента (без реального NVIDIA Modulus)
    physical_stub = PhysicalAgent(agent_id="nvidia_sim_stub")
    
    agents = [optimist, skeptic, methodologist, physical_stub]
    
    # === 2. Инициализация GRA-ядра (Level 2) ===
    
    core = GRACore(
        weights={
            "conflict": 0.40,    # Высокий вес — противоречия критичны
            "vacuity": 0.35,     # Источники важны
            "redundancy": 0.20,
            "noise": 0.05
        },
        discovery_protection=0.25  # Не убивать спор ради снижения J
    )
    
    # === 3. Инициализация оркестратора (Level 1) ===
    
    orchestrator = DebateOrchestrator(
        agents=agents,
        core=core,
        max_rounds=3,
        convergence_threshold=0.1
    )
    
    # === 4. Запуск дискуссии ===
    
    question = (
        "Can exponential growth in the number of AI agents lead to "
        "exponential growth in system cognitive capacity? "
        "Consider: (1) communication overhead, (2) emergent collective intelligence, "
        "(3) physical constraints on computation."
    )
    
    print("=" * 60)
    print("GRA FORUM: Initializing Debate")
    print(f"Question: {question}")
    print(f"Agents: {[a.agent_id for a in agents]}")
    print("=" * 60)
    
    # Запуск (3 раунда)
    history = await orchestrator.run_debate(question)
    
    # === 5. Анализ результатов ===
    
    print("\n" + "=" * 60)
    print("DEBATE SUMMARY")
    print("=" * 60)
    
    for r in history:
        print(f"\nRound {r.round_number}:")
        print(f"  J = {r.J_value:.4f}")
        print(f"  Φ_conflict = {r.metrics.conflict:.3f}")
        print(f"  Φ_vacuity = {r.metrics.vacuity:.3f}")
        print(f"  Φ_redundancy = {r.metrics.redundancy:.3f}")
        print(f"  Discovery = {r.metrics.discovery_score:.3f}")
        print(f"  Claims: {r.metrics.total_claims}")
    
    # Проверка успеха GRA
    trajectory = core.get_optimization_trajectory()
    print(f"\nOptimization trajectory: {trajectory}")
    
    # Визуализация
    orchestrator.plot_J(save_path="debate_J_trajectory.png")
    
    # Экспорт лога
    log = orchestrator.get_debate_log()
    import json
    with open("debate_log.json", "w") as f:
        json.dump(log, f, indent=2)
    
    print("\nDebate log saved to debate_log.json")
    print("J trajectory plot saved to debate_J_trajectory.png")


if __name__ == "__main__":
    asyncio.run(main())
