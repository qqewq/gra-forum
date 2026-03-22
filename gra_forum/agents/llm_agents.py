"""
Реализации текстовых LLM-агентов.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from .base import BaseAgent, AgentReply, AgentType, Claim, Source


class LLMAgent(BaseAgent):
    """Базовый класс для LLM-агентов с разными бэкендами."""
    
    def __init__(self, agent_id: str, api_config: Dict[str, Any], role_prompt: str):
        super().__init__(agent_id, AgentType.TEXT_LLM)
        self.api_config = api_config
        self.role_prompt = role_prompt  # Системный промпт (роль агента)
        self.client = None  # Инициализируется в подклассах
    
    def _extract_claims(self, text: str) -> List[Claim]:
        """Извлекает структурированные утверждения из сырого текста."""
        # Упрощённая реализация: разбиваем на предложения
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
        claims = []
        for sent in sentences[:5]:  # Берём первые 5 значимых предложений
            claims.append(Claim(
                text=sent,
                embedding=self._get_embedding(sent),
                sources=[Source(type="llm_output", reference=f"{self.agent_id}_generation")],
                confidence=0.7
            ))
        return claims
    
    def _get_embedding(self, text: str) -> List[float]:
        """Получает эмбеддинг текста (заглушка)."""
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(384).tolist()
    
    def _get_metadata(self) -> Dict[str, Any]:
        """Метаданные ответа."""
        return {
            "model": self.api_config.get("provider", "unknown"),
            "agent_id": self.agent_id
        }
    
    async def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """Вызов API (заглушка для базового класса)."""
        # В реальных подклассах: реальный вызов API
        return f"[Stub response from {self.agent_id}] Based on: {messages[-1]['content'][:50]}..."
    
    async def answer(self, question: str, context: Optional[str] = None) -> AgentReply:
        messages = [{"role": "system", "content": self.role_prompt}]
        if context:
            messages.append({"role": "user", "content": f"Context: {context}"})
        messages.append({"role": "user", "content": question})
        
        # API call...
        response = await self._call_api(messages)
        claims = self._extract_claims(response)
        
        return AgentReply(
            agent_id=self.agent_id,
            raw_text=response,
            claims=claims,
            metadata=self._get_metadata()
        )
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "can_search": False,
            "can_reason": True,
            "max_context": 4096,
            "provider": self.api_config.get("provider", "unknown")
        }


class PerplexityAgent(LLMAgent):
    """Агент с доступом к поиску (Perplexity-подобный)."""
    
    def __init__(self, api_key: str):
        super().__init__(
            agent_id="perplexity_1",
            api_config={"base_url": "https://api.perplexity.ai", "api_key": api_key, "provider": "perplexity"},
            role_prompt="You are a research assistant with real-time search access. "
                       "Always cite sources. Be precise and factual."
        )
    
    def _extract_claims(self, text: str) -> List[Claim]:
        # Парсит [1], [2] ссылки и извлекает URL из контекста
        claims = []
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
        for sent in sentences[:5]:
            # Ищем ссылки в формате [1], [2]
            sources = []
            if '[' in sent and ']' in sent:
                sources.append(Source(type="url", reference="https://example.com/source"))
            claims.append(Claim(
                text=sent,
                embedding=self._get_embedding(sent),
                sources=sources if sources else [Source(type="llm_output", reference="perplexity")],
                confidence=0.8,
                is_verifiable=len(sources) > 0
            ))
        return claims
    
    def get_capabilities(self) -> Dict[str, Any]:
        caps = super().get_capabilities()
        caps["can_search"] = True
        return caps


class KimiAgent(LLMAgent):
    """Агент с длинным контекстом (Kimi-подобный)."""
    
    def __init__(self, api_key: str):
        super().__init__(
            agent_id="kimi_1",
            api_config={"base_url": "https://api.moonshot.cn", "api_key": api_key, "provider": "kimi"},
            role_prompt="You are a long-context analyst. "
                       "Focus on deep reasoning and consistency across large texts."
        )
    
    def get_capabilities(self) -> Dict[str, Any]:
        caps = super().get_capabilities()
        caps["max_context"] = 200000  # Kimi известен длинным контекстом
        return caps


class DeepSeekAgent(LLMAgent):
    """Агент с акцентом на математику/код (DeepSeek-подобный)."""
    
    def __init__(self, api_key: str):
        super().__init__(
            agent_id="deepseek_1",
            api_config={"base_url": "https://api.deepseek.com", "api_key": api_key, "provider": "deepseek"},
            role_prompt="You are a mathematical and logical analyst. "
                       "Provide formal reasoning when possible. "
                       "Flag assumptions explicitly."
        )
    
    def _extract_claims(self, text: str) -> List[Claim]:
        claims = []
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
        for sent in sentences[:5]:
            # Проверяем на формальные утверждения (содержащие числа, формулы)
            is_verifiable = any(c.isdigit() for c in sent) or '=' in sent
            claims.append(Claim(
                text=sent,
                embedding=self._get_embedding(sent),
                sources=[Source(type="calculation", reference="deepseek_analysis")],
                confidence=0.85 if is_verifiable else 0.7,
                is_verifiable=is_verifiable
            ))
        return claims
    
    def get_capabilities(self) -> Dict[str, Any]:
        caps = super().get_capabilities()
        caps["math_reasoning"] = True
        return caps


class RoleBasedAgent(LLMAgent):
    """Агент с конкретной ролью в дискуссии (GRA-оптимист, скептик и т.д.)."""
    
    ROLES = {
        "gra_optimist": "You believe GRA principles can resolve contradictions. "
                       "Look for synthesis and constructive conflicts.",
        "physics_skeptic": "You apply physical constraints and conservation laws. "
                          "Challenge assumptions violating thermodynamics/complexity theory.",
        "methodologist": "You focus on formal validity, metrics, and epistemology. "
                        "Check definitions and measurement procedures."
    }
    
    def __init__(self, agent_id: str, role: str, llm_backend: str, api_key: str):
        if role not in self.ROLES:
            raise ValueError(f"Unknown role: {role}. Available: {list(self.ROLES.keys())}")
        
        super().__init__(
            agent_id=agent_id,
            api_config={"provider": llm_backend, "api_key": api_key},
            role_prompt=self.ROLES[role]
        )
        self.role = role
    
    def get_capabilities(self) -> Dict[str, Any]:
        caps = super().get_capabilities()
        caps["role"] = self.role
        return caps
