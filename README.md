https://orcid.org/my-orcid?orcid=0009-0004-1872-1153
https://doi.org/10.5281/zenodo.19158686
# GRA Forum: AI Agent Debate Orchestrator

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**GRA Forum** — это оркестратор спора ИИ-агентов, реализующий принципы **Gradient Reduction of Argumentative foam (GRA)**. Система управляет многокруговыми дискуссиями между агентами с различными ролями, минимизируя "пену" (foam) — избыточные противоречия, пустоту и избыточность в аргументации.

## Архитектура

Проект построен по трёхуровневой схеме:

```
┌─────────────────────────────────────────────────────────────┐
│ Level 0: Конкретные агенты (Perplexity, Kimi, DeepSeek)    │
│         • Отвечают на вопросы                               │
│         • Извлекают структурированные утверждения (claims) │
└─────────────────────────┬───────────────────────────────────┘
                          │ AgentReply (claims + embeddings)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Level 1: Форум-оркестратор                                  │
│         • Управляет потоком раундов                         │
│         • Передаёт данные в GRA-ядро                        │
│         • Формирует контекст для агентов                    │
└─────────────────────────┬───────────────────────────────────┘
                          │ DebateState
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Level 2: GRA-ядро                                           │
│         • Вычисляет метрики пены Φ                          │
│         • Минимизирует глобальный функционал J              │
│         • Планирует следующие раунды                        │
└─────────────────────────────────────────────────────────────┘
```

## Установка

```bash
# Клонирование репозитория
git clone https://github.com/yourusername/gra-forum.git
cd gra-forum

# Установка зависимостей
pip install -r requirements.txt

# Или установка как пакет
pip install -e .
```

## Быстрый старт

```python
import asyncio
from gra_forum.orchestrator import DebateOrchestrator
from gra_forum.agents import RoleBasedAgent
from gra_forum.core import GRACore

async def main():
    # Инициализация агентов
    optimist = RoleBasedAgent(
        agent_id="gra_optimist",
        role="gra_optimist",
        llm_backend="openai",
        api_key="your-api-key"
    )
    
    skeptic = RoleBasedAgent(
        agent_id="physics_skeptic",
        role="physics_skeptic",
        llm_backend="anthropic",
        api_key="your-api-key"
    )
    
    # Инициализация GRA-ядра
    core = GRACore(weights={
        "conflict": 0.40,
        "vacuity": 0.35,
        "redundancy": 0.20,
        "noise": 0.05
    })
    
    # Инициализация оркестратора
    orchestrator = DebateOrchestrator(
        agents=[optimist, skeptic],
        core=core,
        max_rounds=3
    )
    
    # Запуск дискуссии
    question = "Can exponential growth in AI agents lead to exponential cognitive capacity?"
    history = await orchestrator.run_debate(question)
    
    # Визуализация результатов
    orchestrator.plot_J(save_path="debate_results.png")

if __name__ == "__main__":
    asyncio.run(main())
```

## Метрики пены Φ

GRA-ядро вычисляет четыре ключевых метрики:

| Метрика | Описание | Метод вычисления |
|---------|----------|------------------|
| **Φ_conflict** | Непрояснённые противоречия | Антикорреляция эмбеддингов между claims разных агентов |
| **Φ_vacuity** | Пустота (отсутствие опоры) | Отсутствие источников, невозможность верификации |
| **Φ_redundancy** | Избыточные дубли | Высокое сходство эмбеддингов без добавочной ценности |
| **Φ_noise** | Явные ошибки | Внешняя валидация (заглушка в MVP) |

Глобальный функционал:
```
J = w_c·Φ_conflict + w_v·Φ_vacuity + w_r·Φ_redundancy + w_n·Φ_noise
```

## Стратегии оркестратора

На основе доминирующей метрики Φ ядро выбирает стратегию:

- **Φ_conflict ↑** → `attack`: Столкнуть агентов с противоположными позициями
- **Φ_vacuity ↑** → `verify`: Потребовать источники и расчёты
- **Φ_redundancy ↑** → `synthesize`: Развести агентов на разные аспекты

## Структура проекта

```
gra-forum/
├── gra_forum/
│   ├── core/              # GRA-ядро и метрики
│   ├── orchestrator/      # Форум-оркестратор
│   ├── agents/            # Базовые агенты и реализации
│   ├── models/            # Pydantic-модели данных
│   └── utils/             # Утилиты (эмбеддинги)
├── examples/              # Примеры использования
├── tests/                 # Тесты
└── docs/                  # Документация
```

## Лицензия

MIT License — см. [LICENSE](LICENSE)

## Citation

```bibtex
@software{gra_forum,
  title = {GRA Forum: AI Agent Debate Orchestrator},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/gra-forum}
}
```
