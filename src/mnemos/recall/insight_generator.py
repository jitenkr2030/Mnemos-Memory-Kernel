"""
Insight Generator for Mnemos Recall Engine

This module provides insight generation capabilities for recalled memories.
Insights synthesize information from multiple memories to provide
contextual understanding and patterns that aren't visible in individual memories.

Insight types include:
- Theme extraction (common topics across memories)
- Decision tracking (evolution of decisions over time)
- Question resolution (which questions have been answered)
- Action status (pending vs completed actions)
- Pattern detection (recurring topics, people, or themes)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
from collections import Counter

from ..kernel.memory_node import MemoryNode, MemoryIntent
from ..evolution.linker import EvolutionLinker, LinkType


class InsightType(Enum):
    """
    Types of insights that can be generated.
    """
    THEME = "theme"                     # Common topics/themes
    DECISION_TRACKER = "decision"       # Decision evolution
    QUESTION_STATUS = "question"        # Question resolution
    ACTION_ITEMS = "action"             # Action items status
    PATTERN = "pattern"                 # Recurring patterns
    ENTITY_FOCUS = "entity"             # Entity-centric insights
    TEMPORAL_PATTERN = "temporal"       # Time-based patterns
    SUMMARY = "summary"                 # General summary


@dataclass
class Insight:
    """
    Represents a synthesized insight from memories.
    
    Insights are higher-level observations that go beyond individual
    memories to provide contextual understanding.
    
    Attributes:
        type: Type of insight
        title: Short descriptive title
        description: Detailed explanation of the insight
        evidence: Supporting memories or data points
        confidence: Confidence in the insight (0.0-1.0)
        created_at: When this insight was generated
        related_memory_ids: IDs of memories that informed this insight
    """
    type: InsightType
    title: str
    description: str
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.7
    created_at: datetime = field(default_factory=datetime.utcnow)
    related_memory_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": self.type.value,
            "title": self.title,
            "description": self.description,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "related_memory_ids": self.related_memory_ids
        }


@dataclass
class InsightCollection:
    """
    Collection of insights generated from a memory query.
    
    A collection contains multiple insights of different types,
    providing comprehensive understanding of the recalled memories.
    
    Attributes:
        query: The query that triggered this insight collection
        insights: List of generated insights
        memory_count: Number of memories analyzed
        generated_at: When these insights were generated
    """
    query: str
    insights: List[Insight] = field(default_factory=list)
    memory_count: int = 0
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "query": self.query,
            "insights": [i.to_dict() for i in self.insights],
            "memory_count": self.memory_count,
            "generated_at": self.generated_at.isoformat()
        }
    
    def get_by_type(self, insight_type: InsightType) -> List[Insight]:
        """Get insights of a specific type."""
        return [i for i in self.insights if i.type == insight_type]


class InsightGenerator:
    """
    Generates insights from collections of memories.
    
    The insight analyzer takes a set of recalled memories and produces
    synthesized insights that provide deeper understanding. This includes
    identifying themes, tracking decisions, and detecting patterns.
    
    Attributes:
        linker: Optional evolution linker for relationship analysis
    """
    
    # Theme extraction keywords
    THEME_INDICATORS = {
        "pricing": ["price", "cost", "pricing", "charge", "fee", "$", "₹", "€"],
        "timeline": ["deadline", "schedule", "when", "date", "time", "by"],
        "budget": ["budget", "spend", "expense", "money", "fund", "cost"],
        "team": ["team", "people", "person", "who", "colleague", "member"],
        "technology": ["python", "code", "api", "system", "tech", "software", "implement"],
        "strategy": ["plan", "strategy", "approach", "direction", "goal", "objective"],
        "meeting": ["meet", "discuss", "talk", "chat", "conversation", "call"],
    }
    
    def __init__(self, linker: Optional[EvolutionLinker] = None):
        """
        Initialize the insight generator.
        
        Args:
            linker: Optional evolution linker for relationship analysis
        """
        self.linker = linker
    
    def generate_insights(
        self,
        memories: List[MemoryNode],
        query: str = ""
    ) -> InsightCollection:
        """
        Generate insights from a collection of memories.
        
        This is the main entry point for insight generation. It analyzes
        the memories and produces multiple types of insights.
        
        Args:
            memories: List of memories to analyze
            query: Optional query context for the insights
            
        Returns:
            InsightCollection with synthesized insights
        """
        collection = InsightCollection(query=query, memory_count=len(memories))
        
        if not memories:
            collection.insights.append(Insight(
                type=InsightType.SUMMARY,
                title="No Memories Found",
                description="No memories were found to generate insights from.",
                confidence=1.0
            ))
            return collection
        
        # Generate different types of insights
        collection.insights.extend(self._extract_themes(memories))
        collection.insights.extend(self._track_decisions(memories))
        collection.insights.extend(self._analyze_questions(memories))
        collection.insights.extend(self._summarize_actions(memories))
        collection.insights.extend(self._detect_patterns(memories))
        collection.insights.extend(self._extract_entity_focus(memories))
        
        # Sort insights by confidence
        collection.insights.sort(key=lambda i: i.confidence, reverse=True)
        
        return collection
    
    def _extract_themes(self, memories: List[MemoryNode]) -> List[Insight]:
        """
        Extract common themes from memories.
        
        Args:
            memories: List of memories to analyze
            
        Returns:
            List of theme insights
        """
        if len(memories) < 2:
            return []
        
        # Count topic mentions
        topic_counter = Counter()
        for memory in memories:
            for topic in memory.topics:
                topic_counter[topic] += 1
        
        # Find dominant themes
        if not topic_counter:
            # Fall back to keyword analysis
            text_all = " ".join(m.raw_text.lower() for m in memories)
            for theme, indicators in self.THEME_INDICATORS.items():
                count = sum(1 for ind in indicators if ind in text_all)
                if count >= 2:
                    topic_counter[theme] = count
        
        if not topic_counter:
            return []
        
        # Create theme insight for top themes
        insights = []
        top_themes = topic_counter.most_common(3)
        
        if len(top_themes) == 1:
            single = top_themes[0]
            insights.append(Insight(
                type=InsightType.THEME,
                title=f"Primary Theme: {single[0]}",
                description=f"This collection focuses primarily on '{single[0]}' "
                           f"with {single[1]} mention(s) across the memories.",
                evidence=[f"Topic mentioned {single[1]} times"],
                confidence=min(1.0, single[1] / len(memories) + 0.3),
                related_memory_ids=[m.id for m in memories[:5]]
            ))
        else:
            themes_str = ", ".join(f"'{t[0]}'" for t in top_themes)
            insights.append(Insight(
                type=InsightType.THEME,
                title=f"Main Themes: {themes_str}",
                description=f"The memories cover multiple themes: {themes_str}. "
                           f"This indicates a diverse set of topics discussed.",
                evidence=[f"{t[0]}: {t[1]} mentions" for t in top_themes],
                confidence=0.7,
                related_memory_ids=[m.id for m in memories[:5]]
            ))
        
        return insights
    
    def _track_decisions(self, memories: List[MemoryNode]) -> List[Insight]:
        """
        Track decision evolution across memories.
        
        Args:
            memories: List of memories to analyze
            
        Returns:
            List of decision tracking insights
        """
        decisions = [m for m in memories if m.intent == MemoryIntent.DECISION]
        
        if not decisions:
            return []
        
        insights = []
        
        if len(decisions) == 1:
            insights.append(Insight(
                type=InsightType.DECISION_TRACKER,
                title="One Decision Made",
                description=f"A single decision was recorded: \"{decisions[0].raw_text[:100]}...\"",
                evidence=[decisions[0].raw_text],
                confidence=1.0,
                related_memory_ids=[decisions[0].id]
            ))
        else:
            # Check for evolution in decisions
            decision_texts = [d.raw_text for d in decisions]
            
            # Look for keyword changes (e.g., price changes)
            price_changes = self._detect_changes(decision_texts, ["$", "₹", "price", "cost"])
            
            if price_changes:
                insights.append(Insight(
                    type=InsightType.DECISION_TRACKER,
                    title="Decision Updates Detected",
                    description=f"Found {len(price_changes)} related decisions with potential updates.",
                    evidence=price_changes[:3],
                    confidence=0.8,
                    related_memory_ids=[d.id for d in decisions[:3]]
                ))
            else:
                insights.append(Insight(
                    type=InsightType.DECISION_TRACKER,
                    title=f"{len(decisions)} Decisions Made",
                    description=f"Multiple decisions were recorded in this collection. "
                               f"Review each decision for follow-up actions.",
                    evidence=[d.raw_text[:80] for d in decisions[:3]],
                    confidence=0.75,
                    related_memory_ids=[d.id for d in decisions[:3]]
                ))
        
        return insights
    
    def _analyze_questions(self, memories: List[MemoryNode]) -> List[Insight]:
        """
        Analyze question status in memories.
        
        Args:
            memories: List of memories to analyze
            
        Returns:
            List of question status insights
        """
        questions = [m for m in memories if m.intent == MemoryIntent.QUESTION]
        
        if not questions:
            return []
        
        insights = []
        
        # Categorize questions
        open_questions = []
        answered_patterns = ["yes", "no", "correct", "decided", "will", "should"]
        
        for q in questions:
            # Check if there's a related answer in other memories
            has_answer = False
            for m in memories:
                if m.id != q.id:
                    text_lower = m.raw_text.lower()
                    if any(pattern in text_lower for pattern in answered_patterns):
                        if any(word in text_lower for word in q.raw_text.lower().split()[:3]):
                            has_answer = True
                            break
            
            if not has_answer:
                open_questions.append(q)
        
        if open_questions:
            insights.append(Insight(
                type=InsightType.QUESTION_STATUS,
                title=f"{len(open_questions)} Open Question(s)",
                description=f"Found {len(open_questions)} question(s) that may not have been addressed yet. "
                           f"Consider following up on these topics.",
                evidence=[q.raw_text[:100] for q in open_questions[:3]],
                confidence=0.7,
                related_memory_ids=[q.id for q in open_questions[:3]]
            ))
        
        if len(questions) > len(open_questions):
            answered = len(questions) - len(open_questions)
            insights.append(Insight(
                type=InsightType.QUESTION_STATUS,
                title=f"{answered} Question(s) Potentially Answered",
                description=f"{answered} question(s) appear to have related answers or decisions.",
                confidence=0.6,
                related_memory_ids=[q.id for q in questions[:5]]
            ))
        
        return insights
    
    def _summarize_actions(self, memories: List[MemoryNode]) -> List[Insight]:
        """
        Summarize action items from memories.
        
        Args:
            memories: List of memories to analyze
            
        Returns:
            List of action item insights
        """
        actions = [m for m in memories if m.intent == MemoryIntent.ACTION]
        
        if not actions:
            return []
        
        insights = []
        
        # Check for urgency indicators
        urgent_patterns = ["tomorrow", "today", "urgent", "asap", "immediately", "deadline"]
        urgent_actions = []
        
        for action in actions:
            text_lower = action.raw_text.lower()
            if any(pattern in text_lower for pattern in urgent_patterns):
                urgent_actions.append(action)
        
        if urgent_actions:
            insights.append(Insight(
                type=InsightType.ACTION_ITEMS,
                title="Urgent Actions Pending",
                description=f"Found {len(urgent_actions)} action(s) with urgent or time-sensitive language.",
                evidence=[a.raw_text[:80] for a in urgent_actions[:3]],
                confidence=0.9,
                related_memory_ids=[a.id for a in urgent_actions[:3]]
            ))
        
        # General action summary
        insights.append(Insight(
            type=InsightType.ACTION_ITEMS,
            title=f"{len(actions)} Action Item(s)",
            description=f"Total of {len(actions)} action(s) or reminders recorded. "
                       f"{len(urgent_actions)} marked as urgent.",
            evidence=[a.raw_text[:80] for a in actions[:3]],
            confidence=0.8,
            related_memory_ids=[a.id for a in actions[:3]]
        ))
        
        return insights
    
    def _detect_patterns(self, memories: List[MemoryNode]) -> List[Insight]:
        """
        Detect recurring patterns in memories.
        
        Args:
            memories: List of memories to analyze
            
        Returns:
            List of pattern insights
        """
        if len(memories) < 3:
            return []
        
        # Check for temporal patterns
        timestamps = [m.timestamp for m in memories]
        timestamps.sort()
        
        insights = []
        
        # Check for concentrated activity
        if len(timestamps) >= 2:
            time_span = timestamps[-1] - timestamps[0]
            if time_span < timedelta(hours=2):
                insights.append(Insight(
                    type=InsightType.TEMPORAL_PATTERN,
                    title="Concentrated Activity",
                    description=f"All {len(memories)} memories were recorded within {time_span.total_seconds()/60:.0f} minutes. "
                               f"This suggests a focused discussion or work session.",
                    confidence=0.85,
                    related_memory_ids=[m.id for m in memories[:5]]
                ))
        
        # Check for recurring topics
        all_topics = []
        for memory in memories:
            all_topics.extend(memory.topics)
        
        topic_counts = Counter(all_topics)
        recurring = [t for t, c in topic_counts.items() if c >= 2]
        
        if recurring:
            insights.append(Insight(
                type=InsightType.PATTERN,
                title="Recurring Topics",
                description=f"Topics \"{', '.join(recurring)}\" appeared multiple times, "
                           f"indicating sustained focus on these areas.",
                evidence=[f"'{t}' appeared {topic_counts[t]} times" for t in recurring[:3]],
                confidence=0.75,
                related_memory_ids=[m.id for m in memories[:5]]
            ))
        
        return insights
    
    def _extract_entity_focus(self, memories: List[MemoryNode]) -> List[Insight]:
        """
        Extract entity-centric insights.
        
        Args:
            memories: List of memories to analyze
            
        Returns:
            List of entity focus insights
        """
        # Collect all entities
        entity_counter = Counter()
        for memory in memories:
            for entity in memory.entities:
                entity_counter[(entity.type.value, entity.value)] += 1
        
        if not entity_counter:
            return []
        
        insights = []
        
        # Most mentioned entities
        top_entities = entity_counter.most_common(5)
        
        if top_entities:
            entities_by_type = {}
            for (etype, value), count in top_entities:
                if etype not in entities_by_type:
                    entities_by_type[etype] = []
                entities_by_type[etype].append((value, count))
            
            if len(entities_by_type) == 1:
                etype = list(entities_by_type.keys())[0]
                entities = entities_by_type[etype]
                if len(entities) == 1:
                    insights.append(Insight(
                        type=InsightType.ENTITY_FOCUS,
                        title=f"Key {etype.title()}: {entities[0][0]}",
                        description=f"This {etype} was mentioned multiple times, "
                                   f"suggesting it's a key focus area.",
                        confidence=0.7,
                        related_memory_ids=[m.id for m in memories[:3]]
                    ))
        
        return insights
    
    def _detect_changes(self, texts: List[str], keywords: List[str]) -> List[str]:
        """
        Detect potential changes in a list of texts.
        
        Args:
            texts: List of text strings to analyze
            keywords: Keywords that indicate changeable topics
            
        Returns:
            List of text snippets that may indicate changes
        """
        changes = []
        for text in texts:
            text_lower = text.lower()
            if any(kw.lower() in text_lower for kw in keywords):
                changes.append(text[:100])
        return changes
