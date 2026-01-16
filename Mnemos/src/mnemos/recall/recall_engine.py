"""
Mnemos Recall Engine - Layer 3 Core

This module implements the main Recall Engine that orchestrates all
Layer 3 functionality: query parsing, memory resolution, importance
scoring, and insight generation.

The recall engine provides intelligent memory retrieval that goes beyond
simple keyword matching to understand query intent, rank results by
importance, and generate contextual insights.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

from ..kernel.memory_node import MemoryNode, MemoryIntent
from ..storage.memory_store import MemoryStore
from .query_parser import QueryParser, ParsedQuery, QueryType
from .importance_scorer import ImportanceScorer, ImportanceScore
from .insight_generator import InsightGenerator, Insight, InsightCollection, InsightType


@dataclass
class RecallResult:
    """
    Represents a complete recall result with all metadata.
    
    A recall result contains the matched memories along with their
    importance scores, relevance information, and any generated insights.
    
    Attributes:
        memories: List of matching memory nodes
        scores: Dictionary mapping memory ID to importance score
        query: The parsed query that produced these results
        insights: Optional insights generated from the results
        total_found: Total number of matching memories (before limit)
        execution_time_ms: Time taken to execute the query
    """
    memories: List[MemoryNode]
    scores: Dict[str, ImportanceScore]
    query: ParsedQuery
    insights: Optional[InsightCollection] = None
    total_found: int = 0
    execution_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "query": self.query.to_dict(),
            "total_found": self.total_found,
            "returned_count": len(self.memories),
            "execution_time_ms": self.execution_time_ms,
            "memories": [
                {
                    "memory": m.to_summary(),
                    "importance_score": self.scores[m.id].total,
                    "score_breakdown": self.scores[m.id].factors
                }
                for m in self.memories
            ],
            "insights": self.insights.to_dict() if self.insights else None
        }
    
    def get_top_memories(self, count: int = 5) -> List[MemoryNode]:
        """Get the top N memories by importance score."""
        sorted_memories = sorted(
            self.memories,
            key=lambda m: self.scores.get(m.id, ImportanceScore(0, 0, 0, 0, 0, 0, {})).total,
            reverse=True
        )
        return sorted_memories[:count]


class RecallEngine:
    """
    Main orchestrator for intelligent memory recall.
    
    The recall engine combines query parsing, memory resolution, importance
    scoring, and insight generation to provide comprehensive memory retrieval.
    
    Key features:
    - Natural language query understanding
    - Multi-factor importance scoring
    - Contextual insight generation
    - Configurable result ranking and limits
    
    Attributes:
        store: Memory storage backend
        query_parser: Query parsing component
        importance_scorer: Importance scoring component
        insight_generator: Insight generation component
    """
    
    def __init__(
        self,
        storage_dir: str = "./data",
        enable_insights: bool = True,
        default_limit: int = 20,
        min_importance_threshold: float = 0.0,
        store: Optional[MemoryStore] = None
    ):
        """
        Initialize the recall engine.
        
        Args:
            storage_dir: Directory for memory storage
            enable_insights: Whether to generate insights for queries
            default_limit: Default maximum results per query
            min_importance_threshold: Minimum importance score to return
            store: Optional MemoryStore instance (uses shared store if provided)
        """
        # Use provided store or create a new one
        if store is not None:
            self.store = store
        else:
            self.store = MemoryStore(storage_dir)
        self.query_parser = QueryParser(self.store)
        self.importance_scorer = ImportanceScorer()
        self.insight_generator = InsightGenerator()
        
        self.enable_insights = enable_insights
        self.default_limit = default_limit
        self.min_importance_threshold = min_importance_threshold
    
    def recall(
        self,
        query: str,
        limit: Optional[int] = None,
        generate_insights: Optional[bool] = None,
        include_scores: bool = True
    ) -> RecallResult:
        """
        Execute a recall query and return results.
        
        This is the main entry point for memory retrieval. The method:
        1. Parses the natural language query
        2. Resolves memories matching the query criteria
        3. Scores memories by importance
        4. Optionally generates insights
        5. Returns ranked results
        
        Args:
            query: Natural language query string
            limit: Maximum results to return (uses default if not specified)
            generate_insights: Whether to generate insights (uses enable_insights if not specified)
            include_scores: Whether to include importance scores
            
        Returns:
            RecallResult with memories, scores, and optional insights
        """
        import time
        start_time = time.time()
        
        # Parse the query
        parsed = self.query_parser.parse(query)
        
        # Apply limit override if specified
        if limit is not None:
            parsed.limit = limit
        else:
            parsed.limit = self.default_limit
        
        # Resolve memories
        memories = self._resolve_query(parsed)
        total_found = len(memories)
        
        # Score by importance
        scores = self.importance_scorer.batch_score(memories)
        
        # Filter by minimum importance
        if self.min_importance_threshold > 0:
            memories = [m for m in memories if scores[m.id].total >= self.min_importance_threshold]
        
        # Sort by configured sort order
        memories = self._sort_results(memories, scores, parsed.sort_by)
        
        # Apply limit
        memories = memories[:parsed.limit]
        
        # Generate insights if enabled
        insights = None
        if generate_insights if generate_insights is not None else self.enable_insights:
            if len(memories) >= 2:  # Need at least 2 memories for insights
                insights = self.insight_generator.generate_insights(memories, query)
        
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return RecallResult(
            memories=memories,
            scores=scores,
            query=parsed,
            insights=insights,
            total_found=total_found,
            execution_time_ms=execution_time
        )
    
    def _resolve_query(self, parsed: ParsedQuery) -> List[MemoryNode]:
        """
        Resolve a parsed query to matching memories.
        
        This method executes the parsed query against the memory store,
        applying all specified filters and criteria.
        
        Args:
            parsed: ParsedQuery with query criteria
            
        Returns:
            List of matching memory nodes
        """
        memories: List[MemoryNode] = []
        
        # Apply filters based on query type
        if parsed.query_type == QueryType.TEMPORAL:
            memories = self._query_temporal(parsed)
        elif parsed.query_type == QueryType.INTENT:
            memories = self._query_by_intent(parsed)
        elif parsed.query_type == QueryType.TOPIC:
            memories = self._query_by_topic(parsed)
        elif parsed.query_type == QueryType.COMPOSITE:
            memories = self._query_composite(parsed)
        else:
            # Default: keyword search with optional filters
            memories = self._query_keyword(parsed)
        
        return memories
    
    def _query_keyword(self, parsed: ParsedQuery) -> List[MemoryNode]:
        """Query by keywords with optional filters."""
        memories = list(self.store.iter_all())
        
        # Apply keyword filter
        if parsed.keywords:
            filtered = []
            for memory in memories:
                text_lower = memory.raw_text.lower()
                # Check if all keywords are present
                if all(kw.lower() in text_lower for kw in parsed.keywords):
                    filtered.append(memory)
            memories = filtered
        
        # Apply temporal filter
        memories = self._apply_temporal_filter(memories, parsed)
        
        # Apply intent filter
        if parsed.intent_filter:
            memories = [m for m in memories if m.intent == parsed.intent_filter]
        
        # Apply topic filter
        if parsed.topic_filter:
            memories = [m for m in memories if parsed.topic_filter in m.topics]
        
        return memories
    
    def _query_temporal(self, parsed: ParsedQuery) -> List[MemoryNode]:
        """Query by temporal criteria."""
        memories = list(self.store.iter_all())
        memories = self._apply_temporal_filter(memories, parsed)
        
        # Apply keyword filter if present
        if parsed.keywords:
            memories = [
                m for m in memories
                if any(kw.lower() in m.raw_text.lower() for kw in parsed.keywords)
            ]
        
        return memories
    
    def _query_by_intent(self, parsed: ParsedQuery) -> List[MemoryNode]:
        """Query by intent type."""
        memories = list(self.store.iter_all())
        
        # Filter by intent
        if parsed.intent_filter:
            memories = [m for m in memories if m.intent == parsed.intent_filter]
        else:
            # Try to detect intent from query
            intent = self._detect_intent_from_keywords(parsed.keywords)
            if intent:
                memories = [m for m in memories if m.intent == intent]
        
        # Apply temporal filter
        memories = self._apply_temporal_filter(memories, parsed)
        
        return memories
    
    def _query_by_topic(self, parsed: ParsedQuery) -> List[MemoryNode]:
        """Query by topic."""
        if parsed.topic_filter:
            memories = self.store.query_by_topic(parsed.topic_filter)
        else:
            memories = list(self.store.iter_all())
        
        memories = self._apply_temporal_filter(memories, parsed)
        
        return memories
    
    def _query_composite(self, parsed: ParsedQuery) -> List[MemoryNode]:
        """Query with multiple combined criteria."""
        memories = list(self.store.iter_all())
        
        # Apply all applicable filters
        if parsed.keywords:
            memories = [
                m for m in memories
                if any(kw.lower() in m.raw_text.lower() for kw in parsed.keywords)
            ]
        
        memories = self._apply_temporal_filter(memories, parsed)
        
        if parsed.intent_filter:
            memories = [m for m in memories if m.intent == parsed.intent_filter]
        
        if parsed.topic_filter:
            memories = [m for m in memories if parsed.topic_filter in m.topics]
        
        return memories
    
    def _apply_temporal_filter(
        self,
        memories: List[MemoryNode],
        parsed: ParsedQuery
    ) -> List[MemoryNode]:
        """Apply temporal filters to memories."""
        if not parsed.start_time and not parsed.end_time:
            return memories
        
        filtered = []
        for memory in memories:
            if parsed.start_time and memory.timestamp < parsed.start_time:
                continue
            if parsed.end_time and memory.timestamp > parsed.end_time:
                continue
            filtered.append(memory)
        
        return filtered
    
    def _detect_intent_from_keywords(self, keywords: List[str]) -> Optional[MemoryIntent]:
        """Detect intent type from keywords."""
        keyword_str = " ".join(keywords).lower()
        
        if any(w in keyword_str for w in ["decision", "decided", "will", "chose"]):
            return MemoryIntent.DECISION
        elif any(w in keyword_str for w in ["question", "wonder", "how", "what", "why"]):
            return MemoryIntent.QUESTION
        elif any(w in keyword_str for w in ["idea", "think", "believe", "suggest"]):
            return MemoryIntent.IDEA
        elif any(w in keyword_str for w in ["remember", "todo", "action", "reminder"]):
            return MemoryIntent.ACTION
        elif any(w in keyword_str for w in ["realized", "learned", "reflect"]):
            return MemoryIntent.REFLECTION
        
        return None
    
    def _sort_results(
        self,
        memories: List[MemoryNode],
        scores: Dict[str, ImportanceScore],
        sort_by: str
    ) -> List[MemoryNode]:
        """Sort results by the specified criteria."""
        if sort_by == "recency":
            return sorted(memories, key=lambda m: m.timestamp, reverse=True)
        elif sort_by == "importance":
            return sorted(
                memories,
                key=lambda m: scores.get(m.id, ImportanceScore(0, 0, 0, 0, 0, 0, {})).total,
                reverse=True
            )
        else:  # relevance or default
            # Use importance as proxy for relevance
            return sorted(
                memories,
                key=lambda m: scores.get(m.id, ImportanceScore(0, 0, 0, 0, 0, 0, {})).total,
                reverse=True
            )
    
    def get_memory_with_context(
        self,
        memory_id: str,
        include_evolution: bool = True,
        include_insights: bool = True
    ) -> Dict[str, Any]:
        """
        Get a single memory with full context.
        
        This method retrieves a specific memory along with related information
        such as evolution links and contextual insights.
        
        Args:
            memory_id: The memory ID to retrieve
            include_evolution: Whether to include evolution chain
            include_insights: Whether to generate insights
            
        Returns:
            Dictionary with memory and context information
        """
        memory = self.store.retrieve(memory_id)
        
        if not memory:
            return {"error": "Memory not found"}
        
        score = self.importance_scorer.score(memory)
        
        result = {
            "memory": memory.to_dict(),
            "importance_score": score.total,
            "score_breakdown": score.factors
        }
        
        if include_evolution:
            evolution = self.store.query_by_evolution(memory_id)
            result["evolution_chain"] = [m.to_summary() for m in evolution]
            result["evolution_count"] = len(evolution)
        
        if include_insights:
            # Get recent related memories for insight generation
            related = self._get_related_memories(memory)
            if related:
                insights = self.insight_generator.generate_insights(
                    [memory] + related[:10],
                    f"Context for: {memory.raw_text[:50]}"
                )
                result["insights"] = insights.to_dict()
        
        return result
    
    def _get_related_memories(self, memory: MemoryNode) -> List[MemoryNode]:
        """Get memories related to the given memory."""
        all_memories = list(self.store.iter_all())
        related = []
        
        for m in all_memories:
            if m.id == memory.id:
                continue
            
            # Check for topic overlap
            if set(memory.topics) & set(m.topics):
                related.append(m)
                continue
            
            # Check for entity overlap
            memory_entities = {e.value.lower() for e in memory.entities}
            m_entities = {e.value.lower() for e in m.entities}
            if memory_entities & m_entities:
                related.append(m)
        
        return related
    
    def search_similar(self, memory_id: str, limit: int = 5) -> List[MemoryNode]:
        """
        Find memories similar to a given memory.
        
        Args:
            memory_id: The reference memory ID
            limit: Maximum results to return
            
        Returns:
            List of similar memories
        """
        memory = self.store.retrieve(memory_id)
        
        if not memory:
            return []
        
        # Get all other memories
        all_memories = list(self.store.iter_all())
        
        # Score each by similarity
        scored = []
        for m in all_memories:
            if m.id == memory_id:
                continue
            
            similarity = self._calculate_similarity(memory, m)
            if similarity > 0.2:  # Threshold for similarity
                scored.append((m, similarity))
        
        # Sort by similarity and return top results
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:limit]]
    
    def _calculate_similarity(
        self,
        memory1: MemoryNode,
        memory2: MemoryNode
    ) -> float:
        """Calculate similarity between two memories."""
        score = 0.0
        
        # Topic overlap
        if memory1.topics and memory2.topics:
            topic_overlap = len(set(memory1.topics) & set(memory2.topics))
            topic_union = len(set(memory1.topics) | set(memory2.topics))
            if topic_union > 0:
                score += 0.4 * (topic_overlap / topic_union)
        
        # Entity matching
        entities1 = {e.value.lower() for e in memory1.entities}
        entities2 = {e.value.lower() for e in memory2.entities}
        if entities1 and entities2:
            entity_overlap = len(entities1 & entities2)
            entity_union = len(entities1 | entities2)
            if entity_union > 0:
                score += 0.3 * (entity_overlap / entity_union)
        
        # Intent matching
        if memory1.intent == memory2.intent:
            score += 0.2
        
        # Text similarity
        words1 = set(memory1.raw_text.lower().split())
        words2 = set(memory2.raw_text.lower().split())
        if words1 and words2:
            word_overlap = len(words1 & words2)
            word_union = len(words1 | words2)
            if word_union > 0:
                score += 0.1 * (word_overlap / word_union)
        
        return min(1.0, score)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get recall engine statistics.
        
        Returns:
            Dictionary with recall engine statistics
        """
        store_stats = self.store.get_stats()
        
        return {
            "total_memories": store_stats["total_memories"],
            "storage_dir": store_stats["storage_dir"],
            "insights_enabled": self.enable_insights,
            "default_limit": self.default_limit,
            "min_importance_threshold": self.min_importance_threshold
        }
