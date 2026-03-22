"""
Функционалы пены Φ для GRA-ядра.
"""

from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class FoamMetrics:
    """Контейнер для всех метрик пены."""
    conflict: float      # Φ_conflict
    vacuity: float       # Φ_vacuity  
    redundancy: float    # Φ_redundancy
    noise: float         # Φ_noise (опционально)
    
    # Дополнительные информативные метрики
    discovery_score: float = 0.0  # Новые смыслы (не должно падать!)
    total_claims: int = 0


class PhiCalculator:
    """
    Вычисление функционалов пены Φ на основе структурированных данных диалога.
    """
    
    def __init__(
        self,
        conflict_threshold: float = 0.85,   # Косинусное расстояние для "противоположности"
        redundancy_threshold: float = 0.92,  # Для дублей
        vacuity_source_weight: float = 0.5
    ):
        self.conflict_threshold = conflict_threshold
        self.redundancy_threshold = redundancy_threshold
        self.vacuity_source_weight = vacuity_source_weight
    
    def compute_all(
        self,
        claims: List[Dict],  # [{"agent_id": str, "text": str, "embedding": [...], "sources": [...]}, ...]
        agent_capabilities: Optional[Dict[str, Dict]] = None
    ) -> FoamMetrics:
        """
        Основной метод расчёта всех Φ.
        """
        if not claims:
            return FoamMetrics(0, 0, 0, 0)
        
        # 1. Φ_conflict: непрояснённые противоречия
        phi_conflict = self._compute_conflict(claims)
        
        # 2. Φ_vacuity: пустота (отсутствие опоры)
        phi_vacuity = self._compute_vacuity(claims)
        
        # 3. Φ_redundancy: дубли без добавочной ценности
        phi_redundancy = self._compute_redundancy(claims)
        
        # 4. Φ_noise: явные ошибки (заглушка для MVP)
        phi_noise = 0.0  # Требует внешнего валидатора
        
        # 5. Discovery score (защита конструктивного спора)
        discovery = self._compute_discovery_score(claims)
        
        return FoamMetrics(
            conflict=phi_conflict,
            vacuity=phi_vacuity,
            redundancy=phi_redundancy,
            noise=phi_noise,
            discovery_score=discovery,
            total_claims=len(claims)
        )
    
    def _compute_conflict(self, claims: List[Dict]) -> float:
        """
        Φ_conflict = (число пар (A, не-A) без разрешения) / (всего пар)
        
        Детекция: 
        - Высокое косинусное расстояние между эмбеддингами (> threshold)
        - Отсутствие общих агентов, пытавшихся разрешить (cross-examination)
        - Ни один агент не предоставил источников для обоих утверждений
        """
        n = len(claims)
        if n < 2:
            return 0.0
        
        conflict_pairs = 0
        total_pairs = 0
        
        embeddings = np.array([c["embedding"] for c in claims])
        similarities = cosine_similarity(embeddings)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Проверяем, что утверждения от разных агентов
                if claims[i]["agent_id"] == claims[j]["agent_id"]:
                    continue
                
                total_pairs += 1
                sim = similarities[i][j]
                
                # Антикорреляция (противоположные утверждения)
                if sim < -self.conflict_threshold:
                    # Проверяем, есть ли разрешение
                    has_resolution = self._check_resolution(claims[i], claims[j])
                    if not has_resolution:
                        conflict_pairs += 1
        
        return conflict_pairs / total_pairs if total_pairs > 0 else 0.0
    
    def _check_resolution(self, claim1: Dict, claim2: Dict) -> bool:
        """
        Проверяет, было ли противоречие разрешено:
        - Есть ли общий агент, комментировавший оба утверждения?
        - Есть ли цитирование одного утверждения в источниках другого?
        """
        # Упрощённая реализация: проверяем пересечение источников
        sources1 = set(s["reference"] for s in claim1.get("sources", []))
        sources2 = set(s["reference"] for s in claim2.get("sources", []))
        
        # Если есть общие источники — возможно, разрешено
        return len(sources1 & sources2) > 0
    
    def _compute_vacuity(self, claims: List[Dict]) -> float:
        """
        Φ_vacuity = средняя "пустота" утверждений
        
        Пустота утверждения:
        - Нет источников (URL, DOI, расчёт)
        - Нет флага is_verifiable
        - Высокая степень абстрактности (embedding близок к общим концептам)
        """
        if not claims:
            return 0.0
        
        vacuity_scores = []
        
        for claim in claims:
            score = 0.0
            
            # Нет источников
            sources = claim.get("sources", [])
            if not sources:
                score += 0.6
            
            # Нет проверяемых фактов
            if not claim.get("is_verifiable", False):
                score += 0.3
            
            # Абстрактность (расстояние до конкретных эмбеддингов)
            # Упрощение: если текст короткий и без чисел — абстрактный
            text = claim.get("text", "")
            if len(text) < 50 and not any(c.isdigit() for c in text):
                score += 0.1
            
            vacuity_scores.append(min(score, 1.0))
        
        return float(np.mean(vacuity_scores))
    
    def _compute_redundancy(self, claims: List[Dict]) -> float:
        """
        Φ_redundancy = "плохие" дубли / все дубли
        
        "Плохой" дубль:
        - Высокое сходство эмбеддингов (> threshold)
        - От одного и того же агента ИЛИ
        - Без добавочной информации (новых источников, уточнений)
        """
        if len(claims) < 2:
            return 0.0
        
        embeddings = np.array([c["embedding"] for c in claims])
        similarities = cosine_similarity(embeddings)
        
        redundant_pairs = 0
        total_similar_pairs = 0
        
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                sim = similarities[i][j]
                
                if sim > self.redundancy_threshold:
                    total_similar_pairs += 1
                    
                    # Проверяем "плохость" дубля
                    if self._is_bad_duplicate(claims[i], claims[j]):
                        redundant_pairs += 1
        
        return redundant_pairs / total_similar_pairs if total_similar_pairs > 0 else 0.0
    
    def _is_bad_duplicate(self, c1: Dict, c2: Dict) -> bool:
        """Дубль "плохой", если не добавляет новой информации."""
        # От одного агента — плохо (эхо)
        if c1["agent_id"] == c2["agent_id"]:
            return True
        
        # Нет новых источников
        s1 = set(s["reference"] for s in c1.get("sources", []))
        s2 = set(s["reference"] for s in c2.get("sources", []))
        if s2.issubset(s1) or s1.issubset(s2):
            return True
        
        return False
    
    def _compute_discovery_score(self, claims: List[Dict]) -> float:
        """
        Discovery score: насколько много новых, непохожих утверждений.
        Защищает конструктивный спор от чрезмерной минимизации J.
        """
        if len(claims) < 2:
            return 1.0
        
        embeddings = np.array([c["embedding"] for c in claims])
        # Среднее попарное расстояние (разнообразие)
        dist_matrix = 1 - cosine_similarity(embeddings)
        # Берём только верхний треугольник
        upper_tri = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        return float(np.mean(upper_tri))
