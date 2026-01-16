"""
Intent Classifier for Mnemos Memory Kernel

This module provides intent classification for memory nodes. Every memory
must have a clear purpose, and the intent classifier determines what that
purpose is based on the text content.

The classifier starts with a rule-based approach for common patterns and
provides a fallback mechanism for integration with LLM-based classification
when more sophisticated understanding is needed.
"""

import re
from typing import Optional, Tuple
from enum import Enum

from ..kernel.memory_node import MemoryIntent


class PatternRule:
    """
    Represents a classification pattern rule.
    
    Rules are defined as a regex pattern and a list of candidate intents.
    When a pattern matches, the first matching intent in the list is returned.
    """
    def __init__(self, pattern: str, intents: list[MemoryIntent], weight: float = 1.0):
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.intents = intents
        self.weight = weight
    
    def matches(self, text: str) -> bool:
        """Check if the text matches this pattern rule."""
        return bool(self.pattern.search(text))


class IntentClassifier:
    """
    Classifies memory nodes by their intent.
    
    The classifier uses a two-stage approach:
    1. Rule-based classification for common patterns
    2. LLM-based classification as a fallback for complex cases
    
    This design prioritizes speed and consistency for common cases while
    maintaining flexibility for edge cases through the LLM integration.
    
    Attributes:
        rules: List of pattern rules for classification
        use_llm: Whether to use LLM fallback for unclassified text
    """
    
    def __init__(self, use_llm: bool = False, llm_provider: Optional[str] = None):
        """
        Initialize the intent classifier.
        
        Args:
            use_llm: Whether to use LLM fallback for classification
            llm_provider: Optional LLM provider name for fallback
        """
        self.rules: list[PatternRule] = []
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self._initialize_rules()
    
    def _initialize_rules(self) -> None:
        """Initialize the default pattern rules for classification."""
        
        # Decision patterns - explicit commitments and choices
        self.rules.extend([
            PatternRule(r'\b(i will|i am going to|we will|decided to|decision:|going to)\b', 
                       [MemoryIntent.DECISION]),
            PatternRule(r'\b(remember to|don\'t forget|make sure to|ensure that)\b', 
                       [MemoryIntent.ACTION]),
        ])
        
        # Idea patterns - creative and exploratory thinking (before action to prioritize)
        self.rules.extend([
            PatternRule(r'\b(i think|i believe|i feel like|my opinion is)\b', 
                       [MemoryIntent.IDEA]),
            PatternRule(r'\b(what if|imagine if|suppose we|could we)\b', 
                       [MemoryIntent.IDEA]),
            PatternRule(r'\b(interesting|cool|awesome|this would be great)\b', 
                       [MemoryIntent.IDEA]),
            PatternRule(r'\b(maybe|perhaps|possibly|it might work)\b', 
                       [MemoryIntent.IDEA]),
        ])
        
        # Action patterns - tasks and follow-ups
        self.rules.extend([
            PatternRule(r'\b(todo:|task:|action:|@todo|@action)\b', 
                       [MemoryIntent.ACTION]),
            PatternRule(r'\b(add to (my )?list|put this on my list|remind me to)\b', 
                       [MemoryIntent.ACTION]),
            PatternRule(r'\b(book|schedule|set up|安排|schedule a|book a)\b', 
                       [MemoryIntent.ACTION]),
            # Generic action words - checked after more specific patterns
            PatternRule(r'\b(should|must|need to|have to|gotta)\b', 
                       [MemoryIntent.ACTION]),
        ])
        
        # Question patterns - inquiries and information gaps
        self.rules.extend([
            PatternRule(r'\b(\?|how do i|what is|why does|when will|where can|who knows)\b', 
                       [MemoryIntent.QUESTION]),
            PatternRule(r'\b(i wonder|i\'m wondering|i was wondering)\b', 
                       [MemoryIntent.QUESTION]),
            PatternRule(r'\b(any idea|does anyone know|can someone explain)\b', 
                       [MemoryIntent.QUESTION]),
        ])
        
        # Reflection patterns - retrospective analysis
        self.rules.extend([
            PatternRule(r'\b(i learned|i realized|i noticed|i discovered)\b', 
                       [MemoryIntent.REFLECTION]),
            PatternRule(r'\b(looking back|in retrospect|when i think about it)\b', 
                       [MemoryIntent.REFLECTION]),
            PatternRule(r'\b(that didn\'t work|i was wrong|i should have)\b', 
                       [MemoryIntent.REFLECTION]),
            PatternRule(r'\b(i regret|i appreciate|i\'m grateful for)\b', 
                       [MemoryIntent.REFLECTION]),
        ])
    
    def classify(self, text: str) -> Tuple[MemoryIntent, float]:
        """
        Classify the intent of the given text.
        
        This method applies the rule-based classification first. If no rules
        match and LLM fallback is enabled, it attempts LLM classification.
        Otherwise, it returns UNKNOWN with low confidence.
        
        Args:
            text: The text to classify
            
        Returns:
            A tuple of (intent, confidence) where confidence is a float between 0 and 1
        """
        if not text or not text.strip():
            return MemoryIntent.UNKNOWN, 0.0
        
        # Apply pattern rules in order of specificity (most specific first)
        for rule in self.rules:
            if rule.matches(text):
                # Calculate confidence based on rule weight and match quality
                match = rule.pattern.search(text)
                match_ratio = len(match.group(0)) / len(text) if match else 0.1
                confidence = min(0.95, rule.weight * (0.6 + match_ratio))
                return rule.intents[0], confidence
        
        # If no rules matched and LLM is available, try LLM fallback
        if self.use_llm:
            return self._classify_with_llm(text)
        
        # Default to UNKNOWN for unclassified text
        return MemoryIntent.UNKNOWN, 0.3
    
    def _classify_with_llm(self, text: str) -> Tuple[MemoryIntent, float]:
        """
        Classify text using an LLM as fallback.
        
        This is a placeholder for LLM integration. In a production system,
        this would call an external LLM API to classify the text.
        
        Args:
            text: The text to classify
            
        Returns:
            A tuple of (intent, confidence) from LLM classification
        """
        # Placeholder for LLM integration
        # In production, this would call OpenAI, Anthropic, or local LLM
        return MemoryIntent.UNKNOWN, 0.5
    
    def add_rule(self, pattern: str, intents: list[MemoryIntent], weight: float = 1.0) -> None:
        """
        Add a custom classification rule.
        
        This allows the classifier to be extended with domain-specific
        patterns for better classification accuracy.
        
        Args:
            pattern: Regex pattern to match
            intents: List of possible intents (first matching is returned)
            weight: Weight for confidence calculation
        """
        self.rules.append(PatternRule(pattern, intents, weight))
    
    def batch_classify(self, texts: list[str]) -> list[Tuple[MemoryIntent, float]]:
        """
        Classify multiple texts in batch.
        
        This method is more efficient than individual classifications when
        processing multiple memories, as it can leverage batch processing
        capabilities of the underlying classification system.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of (intent, confidence) tuples
        """
        return [self.classify(text) for text in texts]
    
    def get_classification_stats(self) -> dict:
        """
        Get statistics about the classifier configuration.
        
        Returns:
            Dictionary with rule counts and configuration info
        """
        intent_counts: dict[MemoryIntent, int] = {}
        for rule in self.rules:
            for intent in rule.intents:
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        return {
            "total_rules": len(self.rules),
            "rules_by_intent": {i.value: c for i, c in intent_counts.items()},
            "llm_enabled": self.use_llm,
            "llm_provider": self.llm_provider
        }
