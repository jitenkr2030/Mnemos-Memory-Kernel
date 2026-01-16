"""
Mnemos Recall Engine - Layer 3

This module implements the Recall Engine for Layer 3, providing intelligent
memory retrieval through natural language query parsing, importance scoring,
and insight generation.

The recall engine transforms simple keyword searches into semantic queries
that understand intent, context, and temporal relationships.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
import re
from collections import Counter

from ..kernel.memory_node import MemoryNode, MemoryIntent
from ..storage.memory_store import MemoryStore


class QueryType(Enum):
    """
    Types of recall queries.
    
    Different query types require different retrieval strategies.
    """
    KEYWORD = "keyword"           # Simple text search
    TEMPORAL = "temporal"         # Time-based queries (last week, today)
    INTENT = "intent"             # Filter by intent type
    TOPIC = "topic"               # Filter by topic
    ENTITY = "entity"             # Filter by entity (people, places)
    COMPOSITE = "composite"       # Multiple criteria combined
    NATURAL_LANGUAGE = "natural_language"  # Full natural language


class TemporalQualifier(Enum):
    """
    Time qualifiers for temporal queries.
    """
    TODAY = "today"
    YESTERDAY = "yesterday"
    THIS_WEEK = "this_week"
    LAST_WEEK = "last_week"
    THIS_MONTH = "this_month"
    LAST_MONTH = "last_month"
    THIS_YEAR = "this_year"
    RECENT = "recent"
    OLD = "old"


@dataclass
class ParsedQuery:
    """
    Represents a parsed and understood query.
    
    The query parser transforms natural language queries into structured
    ParsedQuery objects that can be efficiently executed against the memory store.
    
    Attributes:
        raw_query: The original query text
        query_type: Type of query detected
        keywords: Extracted keywords for search
        temporal_qualifier: Optional time filter
        start_time: Optional explicit start time
        end_time: Optional explicit end time
        intent_filter: Optional intent type filter
        topic_filter: Optional topic filter
        entity_filter: Optional entity filter
        min_importance: Minimum importance score (0.0-1.0)
        limit: Maximum results to return
        sort_by: Sort order for results
    """
    raw_query: str
    query_type: QueryType = QueryType.KEYWORD
    keywords: List[str] = field(default_factory=list)
    temporal_qualifier: Optional[TemporalQualifier] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    intent_filter: Optional[MemoryIntent] = None
    topic_filter: Optional[str] = None
    entity_filter: Optional[str] = None
    min_importance: float = 0.0
    limit: int = 20
    sort_by: str = "relevance"  # relevance, recency, importance
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "raw_query": self.raw_query,
            "query_type": self.query_type.value,
            "keywords": self.keywords,
            "temporal_qualifier": self.temporal_qualifier.value if self.temporal_qualifier else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "intent_filter": self.intent_filter.value if self.intent_filter else None,
            "topic_filter": self.topic_filter,
            "entity_filter": self.entity_filter,
            "min_importance": self.min_importance,
            "limit": self.limit,
            "sort_by": self.sort_by
        }


class QueryParser:
    """
    Parses natural language queries into structured query objects.
    
    The query parser understands various query formats:
    - Simple keywords: "pricing discussion"
    - Temporal queries: "what did I decide today", "last week"
    - Intent filters: "my decisions about pricing", "questions about the project"
    - Composite queries: "recent decisions about pricing"
    
    Attributes:
        store: Reference to memory store for entity/topic lookups
    """
    
    # Temporal patterns
    TEMPORAL_PATTERNS = [
        (r"\b(today)\b", TemporalQualifier.TODAY),
        (r"\b(yesterday)\b", TemporalQualifier.YESTERDAY),
        (r"\b(this week|this week)\b", TemporalQualifier.THIS_WEEK),
        (r"\b(last week)\b", TemporalQualifier.LAST_WEEK),
        (r"\b(this month)\b", TemporalQualifier.THIS_MONTH),
        (r"\b(last month)\b", TemporalQualifier.LAST_MONTH),
        (r"\b(this year)\b", TemporalQualifier.THIS_YEAR),
        (r"\b(recently|recent)\b", TemporalQualifier.RECENT),
    ]
    
    # Intent indicators
    INTENT_INDICATORS = {
        MemoryIntent.DECISION: ["decision", "decided", "chose", "choice", "will", "must"],
        MemoryIntent.QUESTION: ["question", "asked", "wondering", "how", "what", "why", "when"],
        MemoryIntent.IDEA: ["idea", "think", "believe", "suggest", "propose"],
        MemoryIntent.ACTION: ["remember", "action", "todo", "task", "reminder", "do this"],
        MemoryIntent.REFLECTION: ["realized", "learned", "reflect", "thought", "noticed"],
    }
    
    def __init__(self, store: Optional[MemoryStore] = None):
        """
        Initialize the query parser.
        
        Args:
            store: Optional reference to memory store for context
        """
        self.store = store
    
    def parse(self, query: str) -> ParsedQuery:
        """
        Parse a natural language query.
        
        This is the main entry point for query parsing. It analyzes the
        query text and extracts all relevant filters and criteria.
        
        Args:
            query: The natural language query to parse
            
        Returns:
            ParsedQuery with extracted criteria
        """
        query_lower = query.lower().strip()
        
        # Initialize parsed query
        parsed = ParsedQuery(raw_query=query)
        
        # Detect query type
        if self._is_temporal_query(query_lower):
            parsed.query_type = QueryType.TEMPORAL
            parsed.temporal_qualifier = self._extract_temporal_qualifier(query_lower)
        elif self._has_intent_indicator(query_lower):
            parsed.query_type = QueryType.INTENT
            parsed.intent_filter = self._extract_intent_filter(query_lower)
        elif self._is_composite_query(query):
            parsed.query_type = QueryType.COMPOSITE
        else:
            # Check if it looks like a natural language question
            if query_lower.startswith(("what", "how", "when", "where", "who", "why", "did i", "have i")):
                parsed.query_type = QueryType.NATURAL_LANGUAGE
            else:
                parsed.query_type = QueryType.KEYWORD
        
        # Extract keywords
        parsed.keywords = self._extract_keywords(query)
        
        # Extract temporal bounds
        start, end = self._extract_temporal_bounds(query_lower)
        parsed.start_time = start
        parsed.end_time = end
        
        # Extract topic if mentioned
        parsed.topic_filter = self._extract_topic_filter(query)
        
        # Extract entity if mentioned
        parsed.entity_filter = self._extract_entity_filter(query)
        
        # Set temporal qualifier from patterns if not already set
        if parsed.temporal_qualifier is None:
            for pattern, qualifier in self.TEMPORAL_PATTERNS:
                if re.search(pattern, query_lower):
                    parsed.temporal_qualifier = qualifier
                    break
        
        # Apply temporal qualifier if set
        if parsed.temporal_qualifier:
            start_q, end_q = self._get_temporal_range(parsed.temporal_qualifier)
            if not parsed.start_time:
                parsed.start_time = start_q
            if not parsed.end_time:
                parsed.end_time = end_q
        
        return parsed
    
    def _is_temporal_query(self, query: str) -> bool:
        """Check if query has temporal focus."""
        temporal_words = ["today", "yesterday", "this week", "last week", "this month",
                         "last month", "this year", "recent", "recently", "ago", "before"]
        return any(word in query for word in temporal_words)
    
    def _has_intent_indicator(self, query: str) -> bool:
        """Check if query indicates a specific intent filter."""
        query_lower = query.lower()
        for intent_type, indicators in self.INTENT_INDICATORS.items():
            for indicator in indicators:
                if indicator in query_lower:
                    return True
        return False
    
    def _is_composite_query(self, query: str) -> bool:
        """Check if query combines multiple criteria."""
        # Count number of filter indicators
        indicators = 0
        query_lower = query.lower()
        
        if self._is_temporal_query(query_lower):
            indicators += 1
        if self._has_intent_indicator(query_lower):
            indicators += 1
        if "about" in query_lower or "regarding" in query_lower:
            indicators += 1
        
        return indicators >= 2
    
    def _extract_temporal_qualifier(self, query: str) -> Optional[TemporalQualifier]:
        """Extract temporal qualifier from query."""
        for pattern, qualifier in self.TEMPORAL_PATTERNS:
            if re.search(pattern, query):
                return qualifier
        return None
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract search keywords from query."""
        # Remove common question words and fillers
        stop_words = {
            "what", "did", "i", "do", "have", "my", "the", "a", "an",
            "about", "regarding", "concerning", "related", "to", "of",
            "in", "on", "at", "for", "with", "this", "that", "these",
            "those", "is", "are", "was", "were", "be", "been", "being",
            "make", "made"
        }
        
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords
    
    def _extract_temporal_bounds(self, query: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Extract explicit time bounds from query."""
        # ISO date pattern
        iso_match = re.search(r'(\d{4}-\d{2}-\d{2})', query)
        if iso_match:
            date = datetime.fromisoformat(iso_match.group(1))
            return date, date + timedelta(days=1)
        
        return None, None
    
    def _extract_topic_filter(self, query: str) -> Optional[str]:
        """Extract topic filter from query."""
        # Pattern: "about TOPIC" or "regarding TOPIC"
        patterns = [
            r'\babout\s+([a-zA-Z\s]+?)(?:\?|that|what|$)',
            r'\bregarding\s+([a-zA-Z\s]+?)(?:\?|that|what|$)',
            r'\bon\s+([a-zA-Z\s]+?)(?:\?|that|what|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                topic = match.group(1).strip()
                if topic and len(topic) > 2:
                    return topic.lower()
        
        return None
    
    def _extract_entity_filter(self, query: str) -> Optional[str]:
        """Extract entity filter from query."""
        # Look for named entities mentioned
        # This is a simple implementation - production would use NER
        person_patterns = [r'\b([A-Z][a-z]+)\b']
        
        for pattern in person_patterns:
            match = re.search(pattern, query)
            if match:
                entity = match.group(1)
                # Filter out common words
                if entity.lower() not in {"what", "when", "where", "who", "why", "how", "this", "that"}:
                    return entity
        
        return None
    
    def _extract_intent_filter(self, query: str) -> Optional[MemoryIntent]:
        """Extract intent filter from query."""
        query_lower = query.lower()
        
        for intent_type, indicators in self.INTENT_INDICATORS.items():
            for indicator in indicators:
                if indicator in query_lower:
                    # Check for negation
                    if f"not {indicator}" in query_lower or f"don't {indicator}" in query_lower:
                        continue
                    return intent_type
        
        return None
    
    def _get_temporal_range(self, qualifier: TemporalQualifier) -> Tuple[datetime, datetime]:
        """Get the time range for a temporal qualifier."""
        now = datetime.utcnow()
        
        ranges = {
            TemporalQualifier.TODAY: (
                datetime.combine(now.date(), datetime.min.time()),
                datetime.combine(now.date(), datetime.max.time())
            ),
            TemporalQualifier.YESTERDAY: (
                datetime.combine(now.date() - timedelta(days=1), datetime.min.time()),
                datetime.combine(now.date() - timedelta(days=1), datetime.max.time())
            ),
            TemporalQualifier.THIS_WEEK: (
                now - timedelta(days=now.weekday()),
                now
            ),
            TemporalQualifier.LAST_WEEK: (
                now - timedelta(days=now.weekday() + 7),
                now - timedelta(days=now.weekday() + 1)
            ),
            TemporalQualifier.THIS_MONTH: (
                datetime(now.year, now.month, 1),
                now
            ),
            TemporalQualifier.LAST_MONTH: (
                datetime(now.year, now.month - 1 if now.month > 1 else 12,
                         1 if now.month > 1 else 31),
                datetime(now.year, now.month, 1) - timedelta(days=1)
            ),
            TemporalQualifier.THIS_YEAR: (
                datetime(now.year, 1, 1),
                now
            ),
            TemporalQualifier.RECENT: (
                now - timedelta(days=7),
                now
            ),
        }
        
        return ranges.get(qualifier, (now - timedelta(days=7), now))
