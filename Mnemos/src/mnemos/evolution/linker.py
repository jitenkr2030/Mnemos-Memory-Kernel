"""
Evolution Linker for Mnemos

This module provides the core linking functionality for Layer 2. It searches
past memories on related topics and automatically creates evolution links
between them.

The linker uses semantic similarity (embeddings) and topic matching to find
relevant past memories, then passes candidate pairs to the comparator for
relationship classification.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
import json
from pathlib import Path

from ..kernel.memory_node import MemoryNode


class LinkType(Enum):
    """
    Types of evolutionary relationships between memories.
    
    These types describe how a new memory relates to past memories.
    The linker creates potential links; the comparator refines them.
    
    Semantic Categories:
    - Positive Evolution: REFINES, REINFORCES, SUPPORTS (knowledge building)
    - Negative/Corrective: CONTRADICTS, CORRECTS (knowledge revision)
    - Neutral: RELATES_TO, REPEATS (knowledge organization)
    - Challenging: QUESTIONS (knowledge inquiry)
    """
    
    # Positive Evolution - Knowledge Building
    REFINES = "refines"              # Adds detail or nuance to an existing memory
    REINFORCES = "reinforces"        # Confirms or strengthens a previous memory
    
    # Corrective - Knowledge Revision  
    CORRECTS = "corrects"            # Fixes an error or misconception in a previous memory
    CONTRADICTS = "contradicts"     # Direct opposition to a previous statement
    
    # Neutral - Knowledge Organization
    RELATES_TO = "relates_to"        # General topical relationship
    REPEATS = "repeats"              # Similar content repeated
    UPDATES = "updates"              # Supersedes or replaces previous information
    
    # Challenging - Knowledge Inquiry
    SUPPORTS = "supports"            # Provides evidence for a previous memory
    QUESTIONS = "questions"          # Challenges or questions a prior claim
    
    @property
    def is_positive_evolution(self) -> bool:
        """Check if this link type represents positive knowledge building."""
        return self in (self.REFINES, self.REINFORCES, self.SUPPORTS)
    
    @property
    def is_corrective(self) -> bool:
        """Check if this link type represents knowledge revision."""
        return self in (self.CORRECTS, self.CONTRADICTS)
    
    @property
    def strength_impact(self) -> float:
        """
        Returns the expected impact on memory strength.
        
        Positive evolution types should strengthen the target memory.
        Corrective types may weaken or flag the target memory.
        """
        if self.is_positive_evolution:
            return 0.15  # Moderate strengthening
        elif self == self.CORRECTS:
            return -0.20  # Significant weakening (correction)
        elif self == self.CONTRADICTS:
            return -0.10  # Mild weakening (flagged for review)
        else:
            return 0.0  # Neutral impact


@dataclass
class MemoryLink:
    """
    Represents a link between two memories in the evolution graph.
    
    MemoryLinks form the edges of the memory evolution graph, enabling
    traversal and analysis of how knowledge develops over time.
    
    Attributes:
        source_id: UUID of the newer memory
        target_id: UUID of the older memory
        link_type: Type of evolutionary relationship
        strength: Semantic similarity strength (0.0-1.0)
        created_at: When this link was created
        context: Optional explanation of the relationship
        bidirectional: Whether the relationship flows both ways
    """
    source_id: str
    target_id: str
    link_type: LinkType
    strength: float = 0.5
    created_at: datetime = field(default_factory=datetime.utcnow)
    context: str = ""
    bidirectional: bool = False  # If True, represents bidirectional relationship
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert link to dictionary representation."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "link_type": self.link_type.value,
            "strength": self.strength,
            "created_at": self.created_at.isoformat(),
            "context": self.context,
            "bidirectional": self.bidirectional
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryLink":
        """Create link from dictionary representation."""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            link_type=LinkType(data["link_type"]),
            strength=data.get("strength", 0.5),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.utcnow()),
            context=data.get("context", ""),
            bidirectional=data.get("bidirectional", False)
        )


class EvolutionLinker:
    """
    Links new memories to relevant past memories.
    
    The linker is triggered after memory ingestion. It searches for
    semantically similar or topically related memories and creates
    candidate links for the comparator to classify.
    
    This implementation uses simple text-based similarity for the
    initial version. In production, embedding-based similarity would
    provide better results.
    
    Attributes:
        storage_dir: Directory for storing link data
        similarity_threshold: Minimum similarity to create a link
        max_links_per_memory: Maximum links to create per memory
        lookback_days: How far back to search for related memories
    """
    
    def __init__(
        self,
        storage_dir: str = "./data",
        similarity_threshold: float = 0.3,
        max_links_per_memory: int = 10,
        lookback_days: int = 365
    ):
        """
        Initialize the evolution linker.
        
        Args:
            storage_dir: Directory for storing link data
            similarity_threshold: Minimum similarity score to consider linking
            max_links_per_memory: Maximum number of links per memory
            lookback_days: How many days back to search for related memories
        """
        self.storage_dir = Path(storage_dir)
        self.links_dir = self.storage_dir / "links"
        self.links_dir.mkdir(parents=True, exist_ok=True)
        
        self.similarity_threshold = similarity_threshold
        self.max_links_per_memory = max_links_per_memory
        self.lookback_days = lookback_days
        
        # Simple word-based similarity cache
        self._word_index: Dict[str, List[str]] = {}
    
    def find_related_memories(
        self,
        memory: MemoryNode,
        all_memories: List[MemoryNode]
    ) -> List[Tuple[MemoryNode, float]]:
        """
        Find memories related to the given memory.
        
        This method searches for memories that share topics, entities,
        or have high text similarity. The search is bounded by the
        lookback period to maintain relevance.
        
        Args:
            memory: The memory to find relations for
            all_memories: List of all existing memories to search
            
        Returns:
            List of (memory, similarity_score) tuples, sorted by score
        """
        related = []
        cutoff_date = datetime.utcnow() - timedelta(days=self.lookback_days)
        
        for past_memory in all_memories:
            # Skip if same memory
            if past_memory.id == memory.id:
                continue
            
            # Skip if too old
            if past_memory.timestamp < cutoff_date:
                continue
            
            # Calculate similarity
            similarity = self._calculate_similarity(memory, past_memory)
            
            if similarity >= self.similarity_threshold:
                related.append((past_memory, similarity))
        
        # Sort by similarity and limit
        related.sort(key=lambda x: x[1], reverse=True)
        return related[:self.max_links_per_memory]
    
    def _calculate_similarity(
        self,
        memory1: MemoryNode,
        memory2: MemoryNode
    ) -> float:
        """
        Calculate similarity between two memories.
        
        This simple implementation uses word overlap and topic matching.
        A more sophisticated version would use embeddings.
        
        Args:
            memory1: First memory
            memory2: Second memory
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        score = 0.0
        
        # Topic overlap (40% weight)
        if memory1.topics and memory2.topics:
            topic_overlap = len(set(memory1.topics) & set(memory2.topics))
            topic_union = len(set(memory1.topics) | set(memory2.topics))
            if topic_union > 0:
                score += 0.4 * (topic_overlap / topic_union)
        
        # Entity matching (30% weight)
        entities1 = {e.value.lower() for e in memory1.entities}
        entities2 = {e.value.lower() for e in memory2.entities}
        if entities1 and entities2:
            entity_overlap = len(entities1 & entities2)
            entity_union = len(entities1 | entities2)
            if entity_union > 0:
                score += 0.3 * (entity_overlap / entity_union)
        
        # Intent matching (20% weight)
        if memory1.intent == memory2.intent:
            score += 0.2
        
        # Text similarity using simple word overlap (10% weight)
        words1 = set(memory1.raw_text.lower().split())
        words2 = set(memory2.raw_text.lower().split())
        if words1 and words2:
            word_overlap = len(words1 & words2)
            word_union = len(words1 | words2)
            if word_union > 0:
                score += 0.1 * (word_overlap / word_union)
        
        return min(1.0, score)
    
    def create_link(
        self,
        source_memory: MemoryNode,
        target_memory: MemoryNode,
        link_type: LinkType,
        strength: float,
        context: str = ""
    ) -> MemoryLink:
        """
        Create and persist a memory link.
        
        Args:
            source_memory: The newer memory
            target_memory: The older memory
            link_type: Type of relationship
            strength: Similarity/relationship strength
            context: Optional explanation
            
        Returns:
            The created MemoryLink
        """
        link = MemoryLink(
            source_id=source_memory.id,
            target_id=target_memory.id,
            link_type=link_type,
            strength=strength,
            context=context
        )
        
        # Persist the link
        link_file = self.links_dir / f"{source_memory.id}_{target_memory.id}.json"
        with open(link_file, 'w') as f:
            json.dump(link.to_dict(), f, indent=2)
        
        # Update source memory's evolution_ref
        source_memory.add_evolution_ref(target_memory.id)
        
        return link
    
    def get_links_for_memory(self, memory_id: str) -> List[MemoryLink]:
        """
        Get all links where this memory is the source.
        
        Args:
            memory_id: The memory ID to get links for
            
        Returns:
            List of MemoryLinks
        """
        links = []
        
        if not self.links_dir.exists():
            return links
        
        for link_file in self.links_dir.glob(f"{memory_id}_*.json"):
            with open(link_file, 'r') as f:
                data = json.load(f)
                links.append(MemoryLink.from_dict(data))
        
        return links
    
    def get_reverse_links(self, memory_id: str) -> List[MemoryLink]:
        """
        Get all links where this memory is the target.
        
        Args:
            memory_id: The memory ID to get links for
            
        Returns:
            List of MemoryLinks
        """
        links = []
        
        if not self.links_dir.exists():
            return links
        
        for link_file in self.links_dir.glob(f"*_{memory_id}.json"):
            with open(link_file, 'r') as f:
                data = json.load(f)
                links.append(MemoryLink.from_dict(data))
        
        return links
    
    def get_all_links(self) -> List[MemoryLink]:
        """
        Get all stored links.
        
        Returns:
            List of all MemoryLinks
        """
        links = []
        
        if not self.links_dir.exists():
            return links
        
        for link_file in self.links_dir.glob("*.json"):
            with open(link_file, 'r') as f:
                data = json.load(f)
                links.append(MemoryLink.from_dict(data))
        
        return links
    
    def get_link_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored links.
        
        Returns:
            Dictionary with link statistics
        """
        links = self.get_all_links()
        
        if not links:
            return {
                "total_links": 0,
                "by_type": {},
                "avg_strength": 0.0
            }
        
        by_type: Dict[str, int] = {}
        total_strength = 0.0
        
        for link in links:
            link_type = link.link_type.value
            by_type[link_type] = by_type.get(link_type, 0) + 1
            total_strength += link.strength
        
        return {
            "total_links": len(links),
            "by_type": by_type,
            "avg_strength": total_strength / len(links)
        }
    
    def clear_all_links(self) -> None:
        """Remove all stored links."""
        if self.links_dir.exists():
            for link_file in self.links_dir.glob("*.json"):
                link_file.unlink()
