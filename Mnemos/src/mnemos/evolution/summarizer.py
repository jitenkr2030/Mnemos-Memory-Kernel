"""
Temporal Summarizer for Mnemos

This module provides temporal summary generation for Layer 2. It
aggregates memories over time periods (daily, weekly, monthly) and
generates coherent narrative summaries of knowledge evolution.

The summarizer groups memories by topic, identifies key themes, and
produces synthesized narratives that capture the essence of what
was discussed or decided over the specified time period.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
from collections import Counter
from pathlib import Path
from ..kernel.memory_node import MemoryNode, MemoryIntent


class SummaryPeriod(Enum):
    """
    Time periods for summary generation.
    """
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class TemporalSummary:
    """
    Represents a synthesized summary of memories over a time period.
    
    Temporal summaries capture the evolution of knowledge rather than
    just listing memories. They identify themes, track changes, and
    provide a coherent narrative of thought progression.
    
    Attributes:
        period: Type of summary (daily, weekly, monthly)
        start_date: Start of the summary period
        end_date: End of the summary period
        content: The synthesized summary text
        key_topics: List of main topics discussed
        key_decisions: List of decisions made
        key_questions: List of questions raised
        memory_count: Number of memories included
        source_ids: IDs of source memories
        created_at: When this summary was generated
    """
    period: SummaryPeriod
    start_date: datetime
    end_date: datetime
    content: str
    key_topics: List[str] = field(default_factory=list)
    key_decisions: List[str] = field(default_factory=list)
    key_questions: List[str] = field(default_factory=list)
    memory_count: int = 0
    source_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "period": self.period.value,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "content": self.content,
            "key_topics": self.key_topics,
            "key_decisions": self.key_decisions,
            "key_questions": self.key_questions,
            "memory_count": self.memory_count,
            "source_ids": self.source_ids,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalSummary":
        """Create from dictionary representation."""
        return cls(
            period=SummaryPeriod(data["period"]),
            start_date=datetime.fromisoformat(data["start_date"]) if isinstance(data["start_date"], str) else data["start_date"],
            end_date=datetime.fromisoformat(data["end_date"]) if isinstance(data["end_date"], str) else data["end_date"],
            content=data["content"],
            key_topics=data.get("key_topics", []),
            key_decisions=data.get("key_decisions", []),
            key_questions=data.get("key_questions", []),
            memory_count=data.get("memory_count", 0),
            source_ids=data.get("source_ids", []),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.utcnow())
        )


class TemporalSummarizer:
    """
    Generates temporal summaries of memories.
    
    The summarizer takes a collection of memories within a time range
    and produces a coherent narrative that captures:
    - Main topics discussed
    - Decisions made
    - Questions raised
    - Evolution of thinking over time
    
    In production, this would use LLM-based generation for more
    sophisticated narratives. The current implementation uses
    template-based synthesis.
    
    Attributes:
        storage_dir: Directory for storing summaries
        use_llm: Whether to use LLM for summary generation
    """
    
    def __init__(
        self,
        storage_dir: str = "./data",
        use_llm: bool = False
    ):
        """
        Initialize the temporal summarizer.
        
        Args:
            storage_dir: Directory for storing summaries
            use_llm: Whether to use LLM for generation
        """
        self.storage_dir = storage_dir
        self.use_llm = use_llm
        self.summaries_file = f"{storage_dir}/summaries.json"
    
    def generate_summary(
        self,
        memories: List[MemoryNode],
        period: SummaryPeriod,
        start_date: datetime,
        end_date: datetime
    ) -> TemporalSummary:
        """
        Generate a temporal summary from memories.
        
        This is the main entry point for summary generation. It
        aggregates memories and produces a synthesized narrative.
        
        Args:
            memories: List of memories to summarize
            period: Type of summary (daily, weekly, monthly)
            start_date: Start of the summary period
            end_date: End of the summary period
            
        Returns:
            TemporalSummary with the synthesized content
        """
        if not memories:
            return TemporalSummary(
                period=period,
                start_date=start_date,
                end_date=end_date,
                content="No memories recorded during this period.",
                memory_count=0
            )
        
        # Analyze memories
        topic_counts = self._count_topics(memories)
        intent_counts = self._count_intents(memories)
        decisions = self._extract_decisions(memories)
        questions = self._extract_questions(memories)
        
        # Generate narrative
        content = self._synthesize_narrative(
            memories=memories,
            period=period,
            topic_counts=topic_counts,
            intent_counts=intent_counts,
            decisions=decisions,
            questions=questions
        )
        
        # Extract top topics
        top_topics = [topic for topic, count in topic_counts.most_common(5)]
        
        # Get memory IDs
        memory_ids = [m.id for m in memories]
        
        return TemporalSummary(
            period=period,
            start_date=start_date,
            end_date=end_date,
            content=content,
            key_topics=top_topics,
            key_decisions=[d.raw_text for d in decisions[:5]],
            key_questions=[q.raw_text for q in questions[:5]],
            memory_count=len(memories),
            source_ids=memory_ids
        )
    
    def _count_topics(self, memories: List[MemoryNode]) -> Counter:
        """Count occurrences of each topic."""
        all_topics = []
        for memory in memories:
            all_topics.extend(memory.topics)
        return Counter(all_topics)
    
    def _count_intents(self, memories: List[MemoryNode]) -> Counter:
        """Count occurrences of each intent type."""
        return Counter(m.intent for m in memories)
    
    def _extract_decisions(self, memories: List[MemoryNode]) -> List[MemoryNode]:
        """Extract memories with DECISION intent."""
        return [m for m in memories if m.intent == MemoryIntent.DECISION]
    
    def _extract_questions(self, memories: List[MemoryNode]) -> List[MemoryNode]:
        """Extract memories with QUESTION intent."""
        return [m for m in memories if m.intent == MemoryIntent.QUESTION]
    
    def _synthesize_narrative(
        self,
        memories: List[MemoryNode],
        period: SummaryPeriod,
        topic_counts: Counter,
        intent_counts: Counter,
        decisions: List[MemoryNode],
        questions: List[MemoryNode]
    ) -> str:
        """
        Synthesize a narrative from the analyzed memories.
        
        This template-based approach creates a coherent summary.
        In production, this would use LLM-based generation.
        
        Args:
            memories: List of source memories
            period: Summary period type
            topic_counts: Topic frequency counts
            intent_counts: Intent frequency counts
            decisions: List of decision memories
            questions: List of question memories
            
        Returns:
            Synthesized narrative text
        """
        period_name = period.value.capitalize()
        
        lines = []
        lines.append(f"## {period_name} Summary")
        lines.append("")
        
        # Overview
        lines.append(f"**Overview**: {len(memories)} memories recorded")
        
        if topic_counts:
            top_topics = [t for t, _ in topic_counts.most_common(3)]
            topics_str = ", ".join(f"'{t}'" for t in top_topics)
            lines.append(f"**Main Topics**: {topics_str}")
        
        lines.append("")
        
        # Activity breakdown
        lines.append("### Activity Breakdown")
        lines.append("")
        
        if intent_counts:
            for intent, count in intent_counts.most_common():
                intent_name = intent.value.capitalize()
                lines.append(f"- {intent_name}: {count}")
            lines.append("")
        
        # Key decisions
        if decisions:
            lines.append("### Key Decisions")
            lines.append("")
            for decision in decisions[:3]:
                lines.append(f"- {decision.raw_text}")
            lines.append("")
        
        # Open questions
        if questions:
            lines.append("### Open Questions")
            lines.append("")
            for question in questions[:3]:
                lines.append(f"- {question.raw_text}")
            lines.append("")
        
        # Topics section
        if topic_counts:
            lines.append("### Topic Evolution")
            lines.append("")
            for topic, count in topic_counts.most_common(5):
                lines.append(f"- **{topic}**: {count} mentions")
            lines.append("")
        
        # Activity timeline (sample recent memories)
        recent_memories = sorted(memories, key=lambda m: m.timestamp, reverse=True)[:5]
        if recent_memories:
            lines.append("### Recent Activity")
            lines.append("")
            for memory in recent_memories:
                timestamp = memory.timestamp.strftime("%Y-%m-%d %H:%M")
                preview = memory.raw_text[:80] + "..." if len(memory.raw_text) > 80 else memory.raw_text
                lines.append(f"- *{timestamp}* ({memory.intent.value}): {preview}")
        
        return "\n".join(lines)
    
    def generate_daily_summary(
        self,
        memories: List[MemoryNode],
        date: Optional[datetime] = None
    ) -> TemporalSummary:
        """
        Generate a daily summary.
        
        Args:
            memories: Memories from the day
            date: The date to summarize (defaults to today)
            
        Returns:
            Daily TemporalSummary
        """
        target_date = date or datetime.utcnow().date()
        start_date = datetime.combine(target_date, datetime.min.time())
        end_date = datetime.combine(target_date, datetime.max.time())
        
        return self.generate_summary(
            memories=memories,
            period=SummaryPeriod.DAILY,
            start_date=start_date,
            end_date=end_date
        )
    
    def generate_weekly_summary(
        self,
        memories: List[MemoryNode],
        end_date: Optional[datetime] = None
    ) -> TemporalSummary:
        """
        Generate a weekly summary.
        
        Args:
            memories: Memories from the week
            end_date: End of the week (defaults to today)
            
        Returns:
            Weekly TemporalSummary
        """
        end = end_date or datetime.utcnow()
        start_date = end - timedelta(days=7)
        
        return self.generate_summary(
            memories=memories,
            period=SummaryPeriod.WEEKLY,
            start_date=start_date,
            end_date=end
        )
    
    def generate_monthly_summary(
        self,
        memories: List[MemoryNode],
        end_date: Optional[datetime] = None
    ) -> TemporalSummary:
        """
        Generate a monthly summary.
        
        Args:
            memories: Memories from the month
            end_date: End of the month (defaults to today)
            
        Returns:
            Monthly TemporalSummary
        """
        end = end_date or datetime.utcnow()
        start_date = end - timedelta(days=30)
        
        return self.generate_summary(
            memories=memories,
            period=SummaryPeriod.MONTHLY,
            start_date=start_date,
            end_date=end
        )
    
    def save_summary(self, summary: TemporalSummary) -> bool:
        """
        Save a summary to storage.
        
        Args:
            summary: The summary to save
            
        Returns:
            True if save was successful
        """
        import json
        
        try:
            # Load existing summaries
            summaries = []
            if hasattr(self, '_summaries_cache'):
                summaries = self._summaries_cache
            elif Path(self.summaries_file).exists():
                with open(self.summaries_file, 'r') as f:
                    summaries = json.load(f)
            
            # Add new summary
            summaries.append(summary.to_dict())
            
            # Save
            with open(self.summaries_file, 'w') as f:
                json.dump(summaries, f, indent=2)
            
            # Update cache
            self._summaries_cache = summaries
            
            return True
        except (IOError, OSError):
            return False
    
    def load_summaries(
        self,
        period: Optional[SummaryPeriod] = None,
        limit: int = 10
    ) -> List[TemporalSummary]:
        """
        Load saved summaries.
        
        Args:
            period: Optional filter by period type
            limit: Maximum number of summaries to return
            
        Returns:
            List of TemporalSummaries
        """
        import json
        
        if not Path(self.summaries_file).exists():
            return []
        
        try:
            with open(self.summaries_file, 'r') as f:
                data = json.load(f)
            
            summaries = [TemporalSummary.from_dict(s) for s in data]
            
            # Filter by period if specified
            if period:
                summaries = [s for s in summaries if s.period == period]
            
            # Sort by creation date and limit
            summaries.sort(key=lambda s: s.created_at, reverse=True)
            return summaries[:limit]
        except (json.JSONDecodeError, KeyError):
            return []
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get statistics about generated summaries.
        
        Returns:
            Dictionary with summary statistics
        """
        summaries = self.load_summaries(limit=1000)
        
        if not summaries:
            return {
                "total_summaries": 0,
                "by_period": {},
                "avg_memory_count": 0.0
            }
        
        by_period: Dict[str, int] = {}
        total_memories = 0
        
        for summary in summaries:
            period_name = summary.period.value
            by_period[period_name] = by_period.get(period_name, 0) + 1
            total_memories += summary.memory_count
        
        return {
            "total_summaries": len(summaries),
            "by_period": by_period,
            "avg_memory_count": total_memories / len(summaries) if summaries else 0
        }
