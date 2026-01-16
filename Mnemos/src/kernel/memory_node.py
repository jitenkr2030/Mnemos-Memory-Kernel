"""
Mnemos Memory Kernel - Core Data Structures

This module defines the fundamental MemoryNode structure that serves as the
atomic unit of memory in the Mnemos system. Every piece of captured knowledge
is represented as a MemoryNode with strict schema enforcement.

The MemoryNode is intentionally minimal to ensure the kernel remains lightweight
and extensible. Additional capabilities are layered on top through the plugin
architecture rather than core schema expansion.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
import uuid


class MemoryIntent(Enum):
    """
    Enumeration of possible memory intents.
    
    Each memory must have a clear purpose. This classification enables
    downstream processing to understand how to handle and prioritize
    different types of captured knowledge.
    """
    IDEA = "idea"
    DECISION = "decision"
    QUESTION = "question"
    REFLECTION = "reflection"
    ACTION = "action"
    UNKNOWN = "unknown"


class EntityType(Enum):
    """
    Enumeration of entity types that can be extracted from memories.
    
    Entities represent concrete, named elements within the text that
    can be cross-referenced and queried specifically.
    """
    PERSON = "person"
    ORGANIZATION = "organization"
    LAW = "law"
    NUMBER = "number"
    DATE = "date"
    LOCATION = "location"
    PRODUCT = "product"
    CONCEPT = "concept"
    UNKNOWN = "unknown"


@dataclass
class Entity:
    """
    Structured representation of an extracted entity.
    
    Entities are named elements extracted from the raw text that provide
    concrete anchors for knowledge. Each entity has a type and value,
    with optional metadata for additional context.
    """
    type: EntityType
    value: str
    confidence: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            "type": self.type.value,
            "value": self.value,
            "confidence": self.confidence,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """Create entity from dictionary representation."""
        return cls(
            type=EntityType(data["type"]),
            value=data["value"],
            confidence=data.get("confidence", 0.8),
            metadata=data.get("metadata", {})
        )


@dataclass
class MemoryNode:
    """
    The fundamental atomic unit of memory in Mnemos.
    
    A MemoryNode represents a single, coherent thought or piece of knowledge
    captured from a voice transcript. It serves as the foundational data
    structure that all other layers of the system build upon.
    
    Attributes:
        id: Unique identifier for this memory node
        timestamp: When this memory was captured
        raw_text: The verbatim transcript content
        intent: The purpose/category of this memory
        topics: Semantic cluster identifiers this memory relates to
        entities: Extracted named elements from the text
        confidence: System certainty in classifications
        evolution_ref: Links to related past memory nodes
    """
    raw_text: str
    timestamp: datetime
    intent: MemoryIntent
    topics: List[str] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    confidence: float = 0.7
    evolution_ref: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Validate the memory node after initialization."""
        if not self.raw_text or not self.raw_text.strip():
            raise ValueError("MemoryNode must have non-empty raw_text")
        
        if not isinstance(self.timestamp, datetime):
            raise ValueError("MemoryNode must have a valid datetime timestamp")
        
        if not isinstance(self.intent, MemoryIntent):
            raise ValueError("MemoryNode must have a valid MemoryIntent")
        
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert memory node to dictionary representation.
        
        This serialization format is used for storage and API responses.
        The format is designed to be compatible with various storage backends.
        """
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "raw_text": self.raw_text,
            "intent": self.intent.value,
            "topics": self.topics,
            "entities": [e.to_dict() for e in self.entities],
            "confidence": self.confidence,
            "evolution_ref": self.evolution_ref,
            "embedding": self.embedding,
            "created_at": datetime.utcnow().isoformat()
        }
    
    def to_summary(self) -> Dict[str, Any]:
        """
        Create a summary representation for lightweight display.
        
        This excludes the embedding and other heavy data for efficient
        transmission to UI layers or summary generation.
        """
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "intent": self.intent.value,
            "topics": self.topics,
            "confidence": self.confidence,
            "text_preview": self.raw_text[:100] + "..." if len(self.raw_text) > 100 else self.raw_text
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryNode":
        """
        Create a memory node from dictionary representation.
        
        Handles the conversion from stored format back to MemoryNode object.
        """
        entities = [
            Entity.from_dict(e) for e in data.get("entities", [])
        ]
        
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data["timestamp"], str) else data["timestamp"],
            raw_text=data["raw_text"],
            intent=MemoryIntent(data["intent"]),
            topics=data.get("topics", []),
            entities=entities,
            confidence=data.get("confidence", 0.7),
            evolution_ref=data.get("evolution_ref", []),
            embedding=data.get("embedding")
        )
    
    def add_evolution_ref(self, memory_id: str) -> None:
        """
        Add a reference to a related past memory node.
        
        This establishes the evolution chain that enables the system to
        trace how thinking has developed over time.
        """
        if memory_id not in self.evolution_ref:
            self.evolution_ref.append(memory_id)
    
    def add_topic(self, topic_id: str) -> None:
        """
        Add a topic to this memory node.
        
        Topics are semantic cluster identifiers that group related memories
        regardless of the specific vocabulary used.
        """
        if topic_id not in self.topics:
            self.topics.append(topic_id)
    
    def add_entity(self, entity: Entity) -> None:
        """
        Add an entity to this memory node.
        
        Entities are extracted named elements that provide concrete anchors
        for querying and cross-referencing knowledge.
        """
        self.entities.append(entity)
    
    @property
    def text_length(self) -> int:
        """Return the length of the raw text content."""
        return len(self.raw_text)
    
    @property
    def has_entities(self) -> bool:
        """Check if this memory node contains any entities."""
        return len(self.entities) > 0
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if this memory has high classification confidence."""
        return self.confidence >= 0.8
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"MemoryNode(id={self.id[:8]}..., intent={self.intent.value}, "
            f"topics={self.topics[:3]}..., timestamp={self.timestamp.isoformat()})"
        )
