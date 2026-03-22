"""
GRA Forum: AI Agent Debate Orchestrator

Пакет для оркестрации многокруговых дискуссий ИИ-агентов
с минимизацией аргументационной "пены" (foam).
"""

__version__ = "0.1.0"
__author__ = "GRA Forum Team"

from .agents.base import BaseAgent, AgentReply, Claim, Source, AgentType
from .orchestrator.orchestrator import DebateOrchestrator, RoundResult
from .core.gra_core import GRACore, DebatePlan
from .core.metrics import FoamMetrics, PhiCalculator

__all__ = [
    "BaseAgent",
    "AgentReply",
    "Claim",
    "Source",
    "AgentType",
    "DebateOrchestrator",
    "RoundResult",
    "GRACore",
    "DebatePlan",
    "FoamMetrics",
    "PhiCalculator",
]
