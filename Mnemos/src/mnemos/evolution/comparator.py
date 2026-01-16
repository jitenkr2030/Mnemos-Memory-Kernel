"""
Evolution Comparator for Mnemos

This module provides the comparison logic for detecting relationships
between memories. It classifies relationships as REPEATS, CONTRADICTS,
UPDATES, SUPPORTS, or QUESTIONS.

The comparator uses both rule-based logic and LLM-based analysis to
determine the nature of the relationship between two memories.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
import re


class RelationshipType(Enum):
    """
    Types of relationships between two memories.
    
    These types describe how one memory relates to another, going beyond
    the simple LinkType to capture the semantic relationship.
    """
    REPETITION = "repetition"       # Same content repeated
    CONTRADICTION = "contradiction" # Direct opposition
    EVOLUTION = "evolution"         # Natural progression/refinement
    SUPPORT = "support"             # Provides evidence
    QUESTIONING = "questioning"     # Challenges prior claim
    UNRELATED = "unrelated"         # No meaningful relationship


@dataclass
class RelationshipResult:
    """
    Result of comparing two memories.
    
    Contains the relationship type, confidence, and optional explanation.
    """
    source_id: str
    target_id: str
    relationship: RelationshipType
    confidence: float
    explanation: str = ""
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship": self.relationship.value,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "created_at": self.created_at.isoformat()
        }


class EvolutionComparator:
    """
    Compares memories to detect their evolutionary relationship.
    
    The comparator takes two memories and determines how they relate:
    - Repetition: Same idea restated
    - Contradiction: Opposing views
    - Evolution: Refinement or progression
    - Support: One validates the other
    - Questioning: One challenges the other
    
    The initial implementation uses rule-based heuristics. A more
    sophisticated version would integrate LLM analysis for better
    semantic understanding.
    
    Attributes:
        use_llm: Whether to use LLM for comparison
        llm_provider: Optional LLM provider name
    """
    
    # Keywords indicating contradiction
    CONTRADICTION_PATTERNS = [
        (r"\b(hate|despise|loathe)\b", r"\b(love|like|adore)\b"),
        (r"\b(stop|quit|avoid)\b", r"\b(start|begin|continue)\b"),
        (r"\b(wrong|incorrect)\b", r"\b(right|correct)\b"),
        (r"\b(never|not)\b", r"\b(always)\b"),
        (r"\b(bad|terrible|awful)\b", r"\b(good|great|excellent)\b"),
        (r"\b(increase|raise|higher)\b", r"\b(decrease|lower|reduce)\b"),
        (r"\b(expensive|costly)\b", r"\b(cheap|affordable)\b"),
    ]
    
    # Keywords indicating support
    SUPPORT_PATTERNS = [
        r"\b(yes|agreed|correct|right)\b",
        r"\b(proves|demonstrates|shows)\b",
        r"\b(exactly|precisely)\b",
        r"\b(that's true|indeed)\b",
    ]
    
    # Keywords indicating questioning/challenge
    QUESTIONING_PATTERNS = [
        r"\b(but|however|yet)\b.*\b(why|how)\b",
        r"\b(are you sure|i doubt)\b",
        r"\b(what about|consider)\b",
        r"\b(disagree|hmm|not sure)\b",
    ]
    
    def __init__(self, use_llm: bool = False, llm_provider: Optional[str] = None):
        """
        Initialize the evolution comparator.
        
        Args:
            use_llm: Whether to use LLM fallback for comparison
            llm_provider: Optional LLM provider name
        """
        self.use_llm = use_llm
        self.llm_provider = llm_provider
    
    def compare(
        self,
        source_text: str,
        target_text: str,
        source_id: str,
        target_id: str,
        source_intent: str = "",
        target_intent: str = ""
    ) -> RelationshipResult:
        """
        Compare two memories and determine their relationship.
        
        This is the main entry point for comparison. It applies
        rule-based logic first, then falls back to LLM if enabled.
        
        Args:
            source_text: Text content of the newer memory
            target_text: Text content of the older memory
            source_id: ID of the newer memory
            target_id: ID of the older memory
            source_intent: Intent type of the newer memory
            target_intent: Intent type of the older memory
            
        Returns:
            RelationshipResult with the detected relationship
        """
        source_lower = source_text.lower()
        target_lower = target_text.lower()
        
        # Check for exact or near-exact repetition
        repetition_score = self._check_repetition(source_text, target_text)
        if repetition_score > 0.9:
            return RelationshipResult(
                source_id=source_id,
                target_id=target_id,
                relationship=RelationshipType.REPETITION,
                confidence=repetition_score,
                explanation="The content is nearly identical to a previous memory"
            )
        
        # Check for contradiction
        contradiction_score = self._check_contradiction(source_lower, target_lower)
        if contradiction_score > 0.5:
            return RelationshipResult(
                source_id=source_id,
                target_id=target_id,
                relationship=RelationshipType.CONTRADICTION,
                confidence=contradiction_score,
                explanation="This memory contradicts a previous statement"
            )
        
        # Check for support
        support_score = self._check_support(source_lower, target_lower)
        if support_score > 0.5:
            return RelationshipResult(
                source_id=source_id,
                target_id=target_id,
                relationship=RelationshipType.SUPPORT,
                confidence=support_score,
                explanation="This memory supports or validates a previous statement"
            )
        
        # Check for questioning
        questioning_score = self._check_questioning(source_lower, target_lower)
        if questioning_score > 0.5:
            return RelationshipResult(
                source_id=source_id,
                target_id=target_id,
                relationship=RelationshipType.QUESTIONING,
                confidence=questioning_score,
                explanation="This memory questions or challenges a previous statement"
            )
        
        # Check for evolution/refinement
        evolution_score = self._check_evolution(source_lower, target_lower)
        if evolution_score > 0.4:
            return RelationshipResult(
                source_id=source_id,
                target_id=target_id,
                relationship=RelationshipType.EVOLUTION,
                confidence=evolution_score,
                explanation="This memory represents an evolution or refinement of prior thinking"
            )
        
        # Default to unrelated if no pattern matches
        return RelationshipResult(
            source_id=source_id,
            target_id=target_id,
            relationship=RelationshipType.UNRELATED,
            confidence=0.3,
            explanation="No significant relationship detected"
        )
    
    def _check_repetition(self, text1: str, text2: str) -> float:
        """
        Check if texts represent repetition.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Repetition score between 0.0 and 1.0
        """
        # Normalize texts
        norm1 = re.sub(r'\s+', ' ', text1.lower().strip())
        norm2 = re.sub(r'\s+', ' ', text2.lower().strip())
        
        # Exact match
        if norm1 == norm2:
            return 1.0
        
        # Check word overlap
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1 & words2)
        union = len(words1 | words2)
        
        # High overlap indicates repetition
        if union > 0:
            overlap_ratio = overlap / union
            # Also check if one is subset of another
            subset_ratio = min(len(words1), len(words2)) / max(len(words1), len(words2))
            
            if overlap_ratio > 0.8 and subset_ratio > 0.8:
                return 0.95
        
        return 0.0
    
    def _check_contradiction(self, text1: str, text2: str) -> float:
        """
        Check if texts contradict each other.
        
        Args:
            text1: First text (normalized, lowercase)
            text2: Second text (normalized, lowercase)
            
        Returns:
            Contradiction score between 0.0 and 1.0
        """
        score = 0.0
        
        for pattern1, pattern2 in self.CONTRADICTION_PATTERNS:
            # Check for contradiction in both directions
            # Direction 1: pattern1 in text1 AND pattern2 in text2
            has_p1_in_t1 = bool(re.search(pattern1, text1))
            has_p2_in_t2 = bool(re.search(pattern2, text2))
            
            # Direction 2: pattern2 in text1 AND pattern1 in text2
            has_p2_in_t1 = bool(re.search(pattern2, text1))
            has_p1_in_t2 = bool(re.search(pattern1, text2))
            
            # If we have contradiction in either direction
            if (has_p1_in_t1 and has_p2_in_t2) or (has_p2_in_t1 and has_p1_in_t2):
                score = max(score, 0.7)
            
            # Also check for negation patterns
            if "not " in text1 or " never " in text1 or " don't " in text1:
                if "always " in text2 or " definitely " in text2:
                    score = max(score, 0.6)
        
        # Check for explicit disagreement words
        disagree_words = ["disagree", "opposite", "contrary", "wrong", "incorrect"]
        if any(word in text1 for word in disagree_words):
            score = max(score, 0.5)
        
        return min(1.0, score)
    
    def _check_support(self, text1: str, text2: str) -> float:
        """
        Check if text1 supports text2.
        
        Args:
            text1: First text (normalized, lowercase)
            text2: Second text (normalized, lowercase)
            
        Returns:
            Support score between 0.0 and 1.0
        """
        score = 0.0
        
        for pattern in self.SUPPORT_PATTERNS:
            if re.search(pattern, text1):
                score = max(score, 0.6)
        
        # Check for agreement indicators
        agreement_indicators = [
            "that's right", "i agree", "exactly right",
            "proves it", "as i said", "i was right"
        ]
        
        for indicator in agreement_indicators:
            if indicator in text1.lower():
                score = max(score, 0.7)
        
        return min(1.0, score)
    
    def _check_questioning(self, text1: str, text2: str) -> float:
        """
        Check if text1 questions or challenges text2.
        
        Args:
            text1: First text (normalized, lowercase)
            text2: Second text (normalized, lowercase)
            
        Returns:
            Questioning score between 0.0 and 1.0
        """
        score = 0.0
        
        for pattern in self.QUESTIONING_PATTERNS:
            if re.search(pattern, text1):
                score = max(score, 0.6)
        
        # Check for explicit doubt or challenge
        challenge_words = ["but", "however", "really", "sure", "doubt"]
        question_markers = ["?", "why", "how come"]
        
        challenge_count = sum(1 for word in challenge_words if word in text1)
        has_question = any(marker in text1 for marker in question_markers)
        
        if challenge_count >= 2:
            score = max(score, 0.5)
        
        if has_question and challenge_count >= 1:
            score = max(score, 0.6)
        
        return min(1.0, score)
    
    def _check_evolution(self, text1: str, text2: str) -> float:
        """
        Check if text1 represents evolution from text2.
        
        Evolution is indicated by refinement, updating, or progression
        of the same topic.
        
        Args:
            text1: First text (normalized, lowercase)
            text2: Second text (normalized, lowercase)
            
        Returns:
            Evolution score between 0.0 and 1.0
        """
        score = 0.0
        
        # Evolution indicators
        evolution_words = [
            "updated", "revised", "refined", "improved",
            "decided", "concluded", "realized", "decided to"
        ]
        
        for word in evolution_words:
            if word in text1:
                score = max(score, 0.5)
        
        # Time-based evolution
        time_words = ["now", "today", "currently", "finally"]
        for word in time_words:
            if word in text1:
                score = max(score, 0.4)
        
        # Check for specific numeric updates
        number_pattern = r'\d+'
        numbers1 = set(re.findall(number_pattern, text1))
        numbers2 = set(re.findall(number_pattern, text2))
        
        if numbers1 and numbers2 and numbers1 != numbers2:
            # Numbers changed, might be an update
            score = max(score, 0.45)
        
        return min(1.0, score)
    
    def compare_with_llm(
        self,
        source_text: str,
        target_text: str,
        source_id: str,
        target_id: str
    ) -> RelationshipResult:
        """
        Use LLM to compare two memories.
        
        This is a placeholder for LLM-based comparison. In production,
        this would call an external LLM API for more accurate semantic
        understanding of the relationship.
        
        Args:
            source_text: Text content of the newer memory
            target_text: Text content of the older memory
            source_id: ID of the newer memory
            target_id: ID of the older memory
            
        Returns:
            RelationshipResult from LLM analysis
        """
        # Placeholder: Use rule-based as fallback
        return self.compare(source_text, target_text, source_id, target_id)
    
    def batch_compare(
        self,
        pairs: List[Tuple[str, str, str, str]]
    ) -> List[RelationshipResult]:
        """
        Compare multiple pairs of memories.
        
        Args:
            pairs: List of (source_text, target_text, source_id, target_id) tuples
            
        Returns:
            List of RelationshipResults
        """
        results = []
        for source_text, target_text, source_id, target_id in pairs:
            result = self.compare(source_text, target_text, source_id, target_id)
            results.append(result)
        
        return results
