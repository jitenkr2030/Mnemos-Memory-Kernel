"""
Importance Scorer for Mnemos Recall Engine

This module implements importance scoring for memories. Each memory is scored
based on multiple factors to determine its relevance and priority for recall.

Importance scoring considers:
- Intent type (decisions are more important than general ideas)
- Entity count and types (mentions of people, dates, numbers)
- Temporal factors (recent memories, time-sensitive content)
- Content characteristics (length, specificity)
- Evolution context (memories that are part of evolution chains)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
import re

from ..kernel.memory_node import MemoryNode, MemoryIntent, Entity, EntityType


@dataclass
class ImportanceScore:
    """
    Represents the importance score breakdown for a memory.
    
    The importance score is a composite of multiple factors, each with
    its own weight and contribution to the final score.
    
    Attributes:
        total: Final importance score (0.0-1.0)
        intent_score: Score from intent type
        entity_score: Score from entity mentions
        recency_score: Score from temporal factors
        content_score: Score from content characteristics
        evolution_score: Score from evolution context
        factors: Breakdown of contributing factors
    """
    total: float
    intent_score: float
    entity_score: float
    recency_score: float
    content_score: float
    evolution_score: float
    factors: Dict[str, float]


class ImportanceScorer:
    """
    Calculates importance scores for memory nodes.
    
    The importance scorer assigns scores based on multiple heuristics.
    Higher scores indicate memories that are likely to be more relevant
    for recall, especially for decision-making and action tracking.
    
    Attributes:
        weights: Configurable weights for each scoring factor
        recent_decay_days: Days after which recency impact decreases
    """
    
    # Default weights for each scoring factor
    DEFAULT_WEIGHTS = {
        "intent": 0.30,       # Intent type is most important
        "entity": 0.25,       # Entity mentions indicate specificity
        "recency": 0.20,      # Recent memories are more relevant
        "content": 0.15,      # Content characteristics matter
        "evolution": 0.10,    # Evolution context adds value
    }
    
    # Intent importance ranking (higher = more important)
    INTENT_WEIGHTS = {
        MemoryIntent.DECISION: 1.0,     # Decisions are highest priority
        MemoryIntent.ACTION: 0.9,        # Actions are very important
        MemoryIntent.QUESTION: 0.7,      # Questions indicate open items
        MemoryIntent.REFLECTION: 0.6,    # Reflections show learning
        MemoryIntent.IDEA: 0.5,          # Ideas are important but not urgent
        MemoryIntent.UNKNOWN: 0.3,       # Unknown intent is least important
    }
    
    # Entity type importance
    ENTITY_TYPE_WEIGHTS = {
        EntityType.DATE: 0.8,       # Dates are time-sensitive
        EntityType.NUMBER: 0.7,     # Numbers indicate specificity
        EntityType.PERSON: 0.6,     # People mentions are relevant
        EntityType.ORGANIZATION: 0.5,
        EntityType.LAW: 0.7,        # Legal references are important
        EntityType.LOCATION: 0.4,
        EntityType.PRODUCT: 0.5,
        EntityType.CONCEPT: 0.4,
        EntityType.UNKNOWN: 0.2,
    }
    
    # Content characteristics that increase importance
    IMPORTANT_PATTERNS = [
        r'\b(important|crucial|critical|essential|urgent)\b',
        r'\b(decided|decided to|must|should|need to)\b',
        r'\b(deadline|due date|by tomorrow|by next week)\b',
        r'\b(\$[\d,]+|₹[\d,]+|€[\d,]+|£[\d,]+)\b',  # Money amounts
        r'\b(\d+%|percent|percentage)\b',           # Percentages
        r'\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b',  # Dates
    ]
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        recent_decay_days: int = 30
    ):
        """
        Initialize the importance scorer.
        
        Args:
            weights: Optional custom weights for scoring factors
            recent_decay_days: Days after which recency impact decreases
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.recent_decay_days = recent_decay_days
    
    def score(self, memory: MemoryNode) -> ImportanceScore:
        """
        Calculate the importance score for a memory.
        
        This is the main entry point for scoring. It calculates scores
        for each factor and combines them into a final importance score.
        
        Args:
            memory: The memory node to score
            
        Returns:
            ImportanceScore with breakdown of factors
        """
        # Calculate individual factor scores
        intent_score = self._score_intent(memory)
        entity_score = self._score_entities(memory)
        recency_score = self._score_recency(memory)
        content_score = self._score_content(memory)
        evolution_score = self._score_evolution(memory)
        
        # Calculate weighted total
        total = (
            intent_score * self.weights["intent"] +
            entity_score * self.weights["entity"] +
            recency_score * self.weights["recency"] +
            content_score * self.weights["content"] +
            evolution_score * self.weights["evolution"]
        )
        
        # Normalize to 0-1 range
        total = min(1.0, max(0.0, total))
        
        # Build factors breakdown
        factors = {
            "intent": intent_score,
            "entities": entity_score,
            "recency": recency_score,
            "content": content_score,
            "evolution": evolution_score,
            "intent_weight": self.weights["intent"],
            "entity_weight": self.weights["entity"],
            "recency_weight": self.weights["recency"],
            "content_weight": self.weights["content"],
            "evolution_weight": self.weights["evolution"],
        }
        
        return ImportanceScore(
            total=total,
            intent_score=intent_score,
            entity_score=entity_score,
            recency_score=recency_score,
            content_score=content_score,
            evolution_score=evolution_score,
            factors=factors
        )
    
    def _score_intent(self, memory: MemoryNode) -> float:
        """
        Score based on intent type.
        
        Decisions and actions are more important than general ideas
        because they indicate commitment and follow-up requirements.
        
        Args:
            memory: The memory node
            
        Returns:
            Score from 0.0 to 1.0
        """
        return self.INTENT_WEIGHTS.get(memory.intent, 0.3)
    
    def _score_entities(self, memory: MemoryNode) -> float:
        """
        Score based on entity mentions.
        
        Memories with more entities, especially important types like
        dates and numbers, are considered more specific and important.
        
        Args:
            memory: The memory node
            
        Returns:
            Score from 0.0 to 1.0
        """
        if not memory.entities:
            return 0.0
        
        total_score = 0.0
        for entity in memory.entities:
            type_weight = self.ENTITY_TYPE_WEIGHTS.get(entity.type, 0.2)
            # Confidence affects the score
            confidence_weight = entity.confidence
            total_score += type_weight * confidence_weight
        
        # Normalize by number of entities (more entities = higher specificity)
        # Cap at 5 entities for normalization
        normalized = total_score / min(len(memory.entities), 5)
        
        return min(1.0, normalized)
    
    def _score_recency(self, memory: MemoryNode) -> float:
        """
        Score based on recency.
        
        Recent memories are more relevant than old ones. The score
        decays over time based on the recent_decay_days setting.
        
        Args:
            memory: The memory node
            
        Returns:
            Score from 0.0 to 1.0
        """
        now = datetime.utcnow()
        age = now - memory.timestamp
        
        # Very recent (last 24 hours) - highest score
        if age < timedelta(hours=24):
            return 1.0
        
        # Within the decay period - linear decay
        if age < timedelta(days=self.recent_decay_days):
            days_old = age.total_seconds() / (24 * 3600)
            return 1.0 - (days_old / self.recent_decay_days) * 0.5
        
        # Beyond decay period - lower baseline score
        return 0.2
    
    def _score_content(self, memory: MemoryNode) -> float:
        """
        Score based on content characteristics.
        
        Content that contains specific numbers, dates, or urgency
        indicators is considered more important.
        
        Args:
            memory: The memory node
            
        Returns:
            Score from 0.0 to 1.0
        """
        text = memory.raw_text.lower()
        score = 0.0
        
        # Check for important patterns
        for pattern in self.IMPORTANT_PATTERNS:
            if re.search(pattern, text):
                score += 0.2
        
        # Length consideration (longer content may be more specific)
        word_count = len(text.split())
        if word_count >= 5 and word_count <= 50:
            score += 0.3  # Sweet spot for specificity
        elif word_count > 50:
            score += 0.2  # Longer but may be verbose
        
        # Specificity bonus for numbers
        number_count = len(re.findall(r'\d+', text))
        if number_count >= 1 and number_count <= 3:
            score += 0.2  # Contains specific numbers
        elif number_count > 3:
            score += 0.1  # May be a list
        
        return min(1.0, score)
    
    def _score_evolution(self, memory: MemoryNode) -> float:
        """
        Score based on evolution context.
        
        Memories that are part of evolution chains (have references
        to other memories or are referenced by others) are considered
        more important as they represent evolving thought.
        
        Args:
            memory: The memory node
            
        Returns:
            Score from 0.0 to 1.0
        """
        # Has outgoing evolution references
        outgoing_count = len(memory.evolution_ref)
        
        # Score based on evolution involvement
        if outgoing_count >= 3:
            return 1.0  # Highly connected
        elif outgoing_count >= 2:
            return 0.7  # Moderately connected
        elif outgoing_count >= 1:
            return 0.5  # Has some connection
        else:
            return 0.3  # Isolated memory
    
    def batch_score(self, memories: List[MemoryNode]) -> Dict[str, ImportanceScore]:
        """
        Score multiple memories at once.
        
        Args:
            memories: List of memory nodes to score
            
        Returns:
            Dictionary mapping memory ID to ImportanceScore
        """
        return {memory.id: self.score(memory) for memory in memories}
    
    def get_top_memories(
        self,
        memories: List[MemoryNode],
        limit: int = 10,
        min_score: float = 0.0
    ) -> List[Tuple[MemoryNode, ImportanceScore]]:
        """
        Get memories sorted by importance score.
        
        Args:
            memories: List of memories to score
            limit: Maximum number of results
            min_score: Minimum score threshold
            
        Returns:
            List of (memory, score) tuples sorted by score
        """
        scored = []
        for memory in memories:
            score = self.score(memory)
            if score.total >= min_score:
                scored.append((memory, score))
        
        # Sort by total score descending
        scored.sort(key=lambda x: x[1].total, reverse=True)
        
        return scored[:limit]
