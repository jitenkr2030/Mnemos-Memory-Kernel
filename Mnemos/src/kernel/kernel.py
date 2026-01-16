"""
Mnemos Kernel - Core Memory Processing Engine

The kernel is the central orchestrator of the Mnemos system. It coordinates
the flow of data through the memory processing pipeline, from raw transcript
ingestion to memory storage and retrieval.

This module implements the core Layer 1 functionality of the Mnemos architecture:
- Memory ingestion from VoiceInk transcripts
- Intent classification
- Semantic topic processing
- Memory storage and retrieval

The kernel is designed to be minimal and focused, with additional capabilities
layered on through the plugin architecture defined in Layer 4.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
import uuid

from .memory_node import MemoryNode, MemoryIntent, Entity, EntityType
from ..classifier.intent_classifier import IntentClassifier
from ..storage.memory_store import MemoryStore


class TranscriptInput:
    """
    Represents input data from VoiceInk.
    
    This class defines the expected structure of data flowing from VoiceInk
    into the Mnemos kernel. VoiceInk handles speech recognition, audio capture,
    and language decoding, passing clean transcripts with metadata to Mnemos.
    
    Attributes:
        text: The transcribed text content
        timestamp: When the speech was captured
        duration: Duration of the speech segment in seconds
        app_context: Name of the active application when speech was captured
        window_title: Title of the active window
    """
    
    def __init__(
        self,
        text: str,
        timestamp: Optional[datetime] = None,
        duration: Optional[float] = None,
        app_context: Optional[str] = None,
        window_title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.text = text
        self.timestamp = timestamp or datetime.utcnow()
        self.duration = duration
        self.app_context = app_context
        self.window_title = window_title
        self.metadata = metadata or {}
    
    def validate(self) -> bool:
        """Validate that the input has required fields."""
        return bool(self.text and self.text.strip())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
            "duration": self.duration,
            "app_context": self.app_context,
            "window_title": self.window_title,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptInput":
        """Create from dictionary representation."""
        return cls(
            text=data["text"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else data.get("timestamp"),
            duration=data.get("duration"),
            app_context=data.get("app_context"),
            window_title=data.get("window_title"),
            metadata=data.get("metadata", {})
        )


class MnemosKernel:
    """
    The core processing kernel for Mnemos.
    
    The kernel orchestrates the memory processing pipeline:
    1. Receives transcripts from VoiceInk
    2. Classifies memory intent
    3. Extracts entities and topics
    4. Stores structured memories
    5. Provides recall capabilities
    
    This is the central component that all other layers build upon.
    The kernel intentionally maintains a minimal surface area, delegating
    specialized processing to dedicated modules.
    
    Attributes:
        store: Memory storage backend
        classifier: Intent classification module
        config: Kernel configuration
    """
    
    def __init__(
        self,
        storage_dir: str = "./data",
        enable_llm_classification: bool = False,
        llm_provider: Optional[str] = None
    ):
        """
        Initialize the Mnemos kernel.
        
        Args:
            storage_dir: Directory for memory storage
            enable_llm_classification: Whether to use LLM fallback for intent
            llm_provider: Optional LLM provider name
        """
        self.store = MemoryStore(storage_dir)
        self.classifier = IntentClassifier(
            use_llm=enable_llm_classification,
            llm_provider=llm_provider
        )
        self.config = {
            "storage_dir": storage_dir,
            "llm_enabled": enable_llm_classification,
            "llm_provider": llm_provider
        }
    
    def ingest(self, transcript_input: TranscriptInput) -> MemoryNode:
        """
        Process and store a transcript as a memory node.
        
        This is the primary entry point for adding new memories to the system.
        The method applies the full processing pipeline:
        1. Validates input
        2. Classifies intent
        3. Extracts entities (basic)
        4. Creates and stores the memory node
        
        Args:
            transcript_input: The transcript data from VoiceInk
            
        Returns:
            The created MemoryNode
            
        Raises:
            ValueError: If input validation fails
        """
        if not transcript_input.validate():
            raise ValueError("Invalid transcript input: missing required fields")
        
        # Step 1: Classify intent
        intent, confidence = self.classifier.classify(transcript_input.text)
        
        # Step 2: Create memory node
        memory = MemoryNode(
            raw_text=transcript_input.text,
            timestamp=transcript_input.timestamp,
            intent=intent,
            confidence=confidence
        )
        
        # Step 3: Extract entities (basic implementation)
        entities = self._extract_entities(transcript_input.text)
        for entity in entities:
            memory.add_entity(entity)
        
        # Step 4: Store the memory
        self.store.store(memory)
        
        return memory
    
    def _extract_entities(self, text: str) -> List[Entity]:
        """
        Extract named entities from text.
        
        This is a basic implementation using pattern matching.
        In production, this would use NER (Named Entity Recognition)
        models for better accuracy.
        
        Args:
            text: The text to extract entities from
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Basic pattern-based entity extraction
        import re
        
        # Email patterns
        emails = re.findall(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', text)
        for email in emails:
            entities.append(Entity(
                type=EntityType.CONCEPT,
                value=email,
                confidence=0.9,
                metadata={"format": "email"}
            ))
        
        # URL patterns
        urls = re.findall(r'https?://[^\s<>"]+', text)
        for url in urls:
            entities.append(Entity(
                type=EntityType.CONCEPT,
                value=url,
                confidence=0.85,
                metadata={"format": "url"}
            ))
        
        # Currency patterns
        currencies = re.findall(r'[$€£¥₹]\d+(?:,\d{3})*(?:\.\d{2})?', text)
        for currency in currencies:
            entities.append(Entity(
                type=EntityType.NUMBER,
                value=currency,
                confidence=0.9,
                metadata={"format": "currency"}
            ))
        
        # Percentage patterns
        percentages = re.findall(r'\d+(?:\.\d+)?%', text)
        for pct in percentages:
            entities.append(Entity(
                type=EntityType.NUMBER,
                value=pct,
                confidence=0.9,
                metadata={"format": "percentage"}
            ))
        
        # Date patterns
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # ISO format
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # US format
            r'\d{1,2}-\d{1,2}-\d{2,4}',  # EU format
        ]
        for pattern in date_patterns:
            dates = re.findall(pattern, text)
            for date in dates:
                entities.append(Entity(
                    type=EntityType.DATE,
                    value=date,
                    confidence=0.8
                ))
        
        return entities
    
    def recall(
        self,
        query: Optional[str] = None,
        topic: Optional[str] = None,
        intent: Optional[MemoryIntent] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 20
    ) -> List[MemoryNode]:
        """
        Query memories based on various criteria.
        
        This is the primary interface for memory retrieval. It supports
        multiple query dimensions that can be combined for precise recall.
        
        Args:
            query: Optional text search query
            topic: Optional topic filter
            intent: Optional intent filter
            start_time: Optional start of time range
            end_time: Optional end of time range
            limit: Maximum number of results
            
        Returns:
            List of matching memory nodes
        """
        results: List[MemoryNode] = []
        
        # Apply topic filter first (most selective)
        if topic:
            results = self.store.query_by_topic(topic)
        elif intent:
            results = self.store.query_by_intent(intent)
        elif start_time or end_time:
            results = self.store.query_by_time_range(
                start_time or datetime.min,
                end_time or datetime.utcnow()
            )
        else:
            results = self.store.query_recent(limit)
        
        # Apply additional filters
        if query:
            query_lower = query.lower()
            results = [
                m for m in results 
                if query_lower in m.raw_text.lower()
            ]
        
        return results[:limit]
    
    def recall_evolution(self, memory_id: str) -> List[MemoryNode]:
        """
        Retrieve the evolution chain for a memory.
        
        This follows the evolution references to trace how thinking
        about a topic has developed over time.
        
        Args:
            memory_id: The starting memory ID
            
        Returns:
            List of memories in evolution order
        """
        return self.store.query_by_evolution(memory_id)
    
    def get_memory(self, memory_id: str) -> Optional[MemoryNode]:
        """
        Retrieve a specific memory by ID.
        
        Args:
            memory_id: The unique memory identifier
            
        Returns:
            The memory node if found, None otherwise
        """
        return self.store.retrieve(memory_id)
    
    def update_memory(self, memory: MemoryNode) -> bool:
        """
        Update an existing memory.
        
        Args:
            memory: The memory node with updated data
            
        Returns:
            True if update was successful
        """
        return self.store.update(memory)
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory.
        
        Args:
            memory_id: The ID of the memory to delete
            
        Returns:
            True if deletion was successful
        """
        return self.store.delete(memory_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get system statistics.
        
        Returns:
            Dictionary with storage and processing statistics
        """
        store_stats = self.store.get_stats()
        classifier_stats = self.classifier.get_classification_stats()
        
        return {
            "storage": store_stats,
            "classifier": classifier_stats,
            "kernel_version": "0.1.0"
        }
    
    def get_recent_memories(self, count: int = 10) -> List[MemoryNode]:
        """
        Get the most recent memories.
        
        Args:
            count: Number of recent memories to return
            
        Returns:
            List of recent memory nodes
        """
        return self.store.query_recent(count)
    
    def clear_all(self) -> bool:
        """
        Clear all stored memories.
        
        This is a destructive operation that removes all data.
        Use with caution.
        
        Returns:
            True if operation was successful
        """
        import shutil
        
        if self.store.storage_dir.exists():
            shutil.rmtree(self.store.storage_dir)
            self.store.storage_dir.mkdir(parents=True, exist_ok=True)
            self.store.memories_dir.mkdir(parents=True, exist_ok=True)
            self.store._load_indexes()
            return True
        
        return False
