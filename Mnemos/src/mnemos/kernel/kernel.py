"""
Mnemos Kernel - Core Memory Processing Engine

The kernel is the central orchestrator of the Mnemos system. It coordinates
the flow of data through the memory processing pipeline, from raw transcript
ingestion to memory storage and retrieval.

This module implements Layer 1 (Memory Kernel), Layer 2 (Evolution Intelligence),
Layer 3 (Recall Engine), and Layer 4 (Domain Constraints):
- Memory ingestion from VoiceInk transcripts
- Intent classification
- Entity and topic extraction
- Evolution linking and conflict detection
- Temporal summarization
- Memory storage and retrieval
- Intelligent recall with query parsing and importance scoring
- Domain constraint validation and enforcement
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import uuid

from .memory_node import MemoryNode, MemoryIntent, Entity, EntityType
from ..classifier.intent_classifier import IntentClassifier
from ..storage.memory_store import MemoryStore
from ..evolution.linker import EvolutionLinker, MemoryLink, LinkType
from ..evolution.comparator import EvolutionComparator, RelationshipResult, RelationshipType
from ..evolution.summarizer import TemporalSummarizer, TemporalSummary, SummaryPeriod
from ..recall import RecallEngine, RecallResult
from ..constraints import (
    ConstraintEngine,
    ConstraintEngineResult,
    ConstraintRegistry,
    BaseConstraint,
)


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
    4. Validates against domain constraints (Layer 4)
    5. Stores structured memories
    6. Triggers evolution linking (Layer 2)
    7. Provides intelligent recall (Layer 3)
    
    This is the central component that all other layers build upon.
    The kernel intentionally maintains a minimal surface area, delegating
    specialized processing to dedicated modules.
    
    Attributes:
        store: Memory storage backend
        classifier: Intent classification module
        linker: Evolution linking module (Layer 2)
        comparator: Evolution comparison module (Layer 2)
        summarizer: Temporal summarization module (Layer 2)
        recall_engine: Intelligent recall engine (Layer 3)
        constraint_engine: Domain constraint engine (Layer 4)
        config: Kernel configuration
    """
    
    def __init__(
        self,
        storage_dir: str = "./data",
        enable_llm_classification: bool = False,
        llm_provider: Optional[str] = None,
        enable_evolution: bool = True,
        enable_recall: bool = True,
        recall_insights: bool = True,
        recall_limit: int = 20,
        enable_constraints: bool = True,
        constraints_fail_on_error: bool = False
    ):
        """
        Initialize the Mnemos kernel.
        
        Args:
            storage_dir: Directory for memory storage
            enable_llm_classification: Whether to use LLM fallback for intent
            llm_provider: Optional LLM provider name
            enable_evolution: Whether to enable Layer 2 evolution intelligence
            enable_recall: Whether to enable Layer 3 recall engine
            recall_insights: Whether to generate insights during recall
            recall_limit: Default limit for recall results
            enable_constraints: Whether to enable Layer 4 domain constraints
            constraints_fail_on_error: Whether to fail on constraint errors
        """
        self.store = MemoryStore(storage_dir)
        self.classifier = IntentClassifier(
            use_llm=enable_llm_classification,
            llm_provider=llm_provider
        )
        
        # Layer 2: Evolution Intelligence
        self.enable_evolution = enable_evolution
        if enable_evolution:
            self.linker = EvolutionLinker(storage_dir=storage_dir)
            self.comparator = EvolutionComparator(use_llm=False)
            self.summarizer = TemporalSummarizer(storage_dir=storage_dir)
        
        # Layer 3: Recall Engine
        self.enable_recall = enable_recall
        if enable_recall:
            self.recall_engine = RecallEngine(
                storage_dir=storage_dir,
                enable_insights=recall_insights,
                default_limit=recall_limit,
                store=self.store  # Pass the shared store
            )
        
        # Layer 4: Domain Constraints
        self.enable_constraints = enable_constraints
        if enable_constraints:
            self.constraint_engine = ConstraintEngine(
                storage_dir=storage_dir,
                enabled=True,
                fail_on_error=constraints_fail_on_error
            )
        
        self.config = {
            "storage_dir": storage_dir,
            "llm_enabled": enable_llm_classification,
            "llm_provider": llm_provider,
            "evolution_enabled": enable_evolution,
            "recall_enabled": enable_recall,
            "recall_insights": recall_insights,
            "recall_limit": recall_limit,
            "constraints_enabled": enable_constraints,
            "constraints_fail_on_error": constraints_fail_on_error
        }
    
    def ingest(self, transcript_input: TranscriptInput) -> MemoryNode:
        """
        Process and store a transcript as a memory node.
        
        This is the primary entry point for adding new memories to the system.
        The method applies the full processing pipeline:
        1. Validates input
        2. Classifies intent
        3. Extracts entities (basic)
        4. Validates against domain constraints (Layer 4)
        5. Stores the memory node
        6. Triggers evolution linking (Layer 2)
        
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
        
        # Step 4: Validate against domain constraints (Layer 4)
        constraint_result = None
        if self.enable_constraints:
            can_proceed, constraint_result = self.constraint_engine.before_ingest(memory)
            if not can_proceed:
                raise ValueError(
                    f"Memory failed constraint validation: {constraint_result.error_count} errors"
                )
        
        # Step 5: Store the memory
        self.store.store(memory)
        
        # Step 6: Trigger evolution linking (Layer 2)
        if self.enable_evolution:
            self._process_evolution(memory)
        
        # Step 7: Post-ingest constraint processing (Layer 4)
        if self.enable_constraints:
            post_result = self.constraint_engine.after_ingest(memory)
            if post_result and post_result.violation_count > 0:
                # Log violations but don't fail
                memory.metadata["constraint_warnings"] = post_result.violation_count
                memory.metadata["constraint_result"] = post_result.to_dict()
        
        return memory
    
    def _process_evolution(self, memory: MemoryNode) -> None:
        """
        Process evolution intelligence for a new memory.
        
        This method:
        1. Finds related past memories
        2. Compares relationships
        3. Creates evolution links
        
        Args:
            memory: The newly created memory
        """
        # Get all existing memories
        all_memories = list(self.store.iter_all())
        
        if not all_memories:
            return
        
        # Find related memories
        related = self.linker.find_related_memories(memory, all_memories)
        
        for related_memory, similarity in related:
            # Compare memories to determine relationship
            result = self.comparator.compare(
                source_text=memory.raw_text,
                target_text=related_memory.raw_text,
                source_id=memory.id,
                target_id=related_memory.id,
                source_intent=memory.intent.value,
                target_intent=related_memory.intent.value
            )
            
            # Determine link type from relationship
            link_type = self._relationship_to_link_type(result.relationship)
            
            # Create the evolution link
            self.linker.create_link(
                source_memory=memory,
                target_memory=related_memory,
                link_type=link_type,
                strength=similarity,
                context=result.explanation
            )
    
    def _relationship_to_link_type(self, relationship: RelationshipType) -> LinkType:
        """Convert RelationshipType to LinkType."""
        mapping = {
            RelationshipType.REPETITION: LinkType.REPEATS,
            RelationshipType.CONTRADICTION: LinkType.CONTRADICTS,
            RelationshipType.EVOLUTION: LinkType.UPDATES,
            RelationshipType.SUPPORT: LinkType.SUPPORTS,
            RelationshipType.QUESTIONING: LinkType.QUESTIONS,
            RelationshipType.UNRELATED: LinkType.RELATES_TO,
        }
        return mapping.get(relationship, LinkType.RELATES_TO)
    
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
    
    # =========================================================================
    # Constraint Management (Layer 4)
    # =========================================================================
    
    def add_constraint(self, constraint: BaseConstraint) -> bool:
        """
        Add a domain constraint to the kernel.
        
        Args:
            constraint: The constraint to add
            
        Returns:
            True if constraint was added successfully
        """
        if not self.enable_constraints:
            return False
        return self.constraint_engine.register_constraint(constraint)
    
    def remove_constraint(self, name: str) -> bool:
        """
        Remove a constraint by name.
        
        Args:
            name: Name of the constraint to remove
            
        Returns:
            True if constraint was removed
        """
        if not self.enable_constraints:
            return False
        return self.constraint_engine.unregister_constraint(name)
    
    def validate_memory(self, memory_id: str) -> Optional[ConstraintEngineResult]:
        """
        Validate a stored memory against constraints.
        
        Args:
            memory_id: ID of the memory to validate
            
        Returns:
            ConstraintEngineResult or None if memory not found
        """
        if not self.enable_constraints:
            return None
        
        memory = self.store.retrieve(memory_id)
        if not memory:
            return None
        
        return self.constraint_engine.validate_memory(memory)
    
    def get_constraint_status(self) -> Dict[str, Any]:
        """
        Get the status of domain constraints.
        
        Returns:
            Dictionary with constraint status
        """
        if not self.enable_constraints:
            return {
                "enabled": False,
                "message": "Constraint engine is disabled"
            }
        
        return self.constraint_engine.get_engine_status()
    
    def enable_constraints(self) -> None:
        """Enable the constraint engine."""
        self.enable_constraints = True
        if hasattr(self, 'constraint_engine'):
            self.constraint_engine.enable()
    
    def disable_constraints(self) -> None:
        """Disable the constraint engine."""
        self.enable_constraints = False
        if hasattr(self, 'constraint_engine'):
            self.constraint_engine.disable()
    
    # =========================================================================
    # Recall Methods (Layer 3)
    # =========================================================================
    
    def recall(
        self,
        query: str,
        limit: Optional[int] = None,
        generate_insights: bool = True
    ) -> RecallResult:
        """
        Execute an intelligent recall query.
        
        This is the primary interface for Layer 3 memory retrieval.
        The method uses natural language understanding to parse queries,
        ranks results by importance, and optionally generates insights.
        
        Args:
            query: Natural language query string
            limit: Maximum results to return
            generate_insights: Whether to generate contextual insights
            
        Returns:
            RecallResult with matching memories, scores, and insights
        """
        if not self.enable_recall:
            # Fall back to basic recall
            from .kernel import TranscriptInput
            memories = self._basic_recall(query, limit)
            from ..recall.importance_scorer import ImportanceScorer
            scorer = ImportanceScorer()
            scores = scorer.batch_score(memories)
            return RecallResult(
                memories=memories,
                scores=scores,
                query=None,
                total_found=len(memories)
            )
        
        return self.recall_engine.recall(
            query=query,
            limit=limit,
            generate_insights=generate_insights
        )
    
    def _basic_recall(
        self,
        query: str,
        limit: Optional[int] = None
    ) -> List[MemoryNode]:
        """Basic keyword-based recall (fallback when recall is disabled)."""
        results: List[MemoryNode] = []
        query_lower = query.lower()
        
        # Simple keyword matching
        all_memories = list(self.store.iter_all())
        for memory in all_memories:
            if query_lower in memory.raw_text.lower():
                results.append(memory)
        
        return results[:limit] if limit else results
    
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
    
    # =========================================================================
    # Memory Management Methods
    # =========================================================================
    
    def get_memory(self, memory_id: str) -> Optional[MemoryNode]:
        """
        Retrieve a specific memory by ID.
        
        Args:
            memory_id: The unique memory identifier
            
        Returns:
            The memory node if found, None otherwise
        """
        return self.store.retrieve(memory_id)
    
    def get_memory_with_context(
        self,
        memory_id: str,
        include_evolution: bool = True,
        include_insights: bool = True
    ) -> Dict[str, Any]:
        """
        Get a memory with full context including insights.
        
        Args:
            memory_id: The memory ID to retrieve
            include_evolution: Whether to include evolution chain
            include_insights: Whether to generate insights
            
        Returns:
            Dictionary with memory and context information
        """
        if not self.enable_recall:
            memory = self.get_memory(memory_id)
            if not memory:
                return {"error": "Memory not found"}
            return {"memory": memory.to_dict()}
        
        return self.recall_engine.get_memory_with_context(
            memory_id=memory_id,
            include_evolution=include_evolution,
            include_insights=include_insights
        )
    
    def search_similar(self, memory_id: str, limit: int = 5) -> List[MemoryNode]:
        """
        Find memories similar to a given memory.
        
        Args:
            memory_id: The reference memory ID
            limit: Maximum results to return
            
        Returns:
            List of similar memories
        """
        if not self.enable_recall:
            return []
        
        return self.recall_engine.search_similar(memory_id, limit)
    
    def get_memory_links(self, memory_id: str) -> List[MemoryLink]:
        """
        Get evolution links for a memory.
        
        Args:
            memory_id: The memory ID to get links for
            
        Returns:
            List of MemoryLinks where this memory is the source
        """
        if not self.enable_evolution:
            return []
        return self.linker.get_links_for_memory(memory_id)
    
    def get_conflicts(self) -> List[Dict[str, Any]]:
        """
        Get all detected contradictions.
        
        Returns:
            List of conflict information dictionaries
        """
        if not self.enable_evolution:
            return []
        
        conflicts = []
        links = self.linker.get_all_links()
        
        for link in links:
            if link.link_type == LinkType.CONTRADICTS:
                source = self.store.retrieve(link.source_id)
                target = self.store.retrieve(link.target_id)
                
                if source and target:
                    conflicts.append({
                        "source_id": link.source_id,
                        "target_id": link.target_id,
                        "source_text": source.raw_text,
                        "target_text": target.raw_text,
                        "source_timestamp": source.timestamp.isoformat(),
                        "target_timestamp": target.timestamp.isoformat(),
                        "strength": link.strength,
                        "context": link.context
                    })
        
        return conflicts
    
    # =========================================================================
    # Summary Methods (Layer 2)
    # =========================================================================
    
    def generate_summary(
        self,
        start_time: datetime,
        end_time: datetime,
        period: SummaryPeriod = SummaryPeriod.WEEKLY
    ) -> TemporalSummary:
        """
        Generate a temporal summary.
        
        Args:
            start_time: Start of the summary period
            end_time: End of the summary period
            period: Type of summary to generate
            
        Returns:
            TemporalSummary with synthesized content
        """
        if not self.enable_evolution:
            return TemporalSummary(
                period=period,
                start_date=start_time,
                end_date=end_time,
                content="Evolution intelligence is disabled."
            )
        
        # Get memories in the time range
        memories = self.store.query_by_time_range(start_time, end_time)
        
        # Generate summary
        summary = self.summarizer.generate_summary(
            memories=memories,
            period=period,
            start_date=start_time,
            end_date=end_time
        )
        
        # Save the summary
        self.summarizer.save_summary(summary)
        
        return summary
    
    def generate_daily_summary(self, date: Optional[datetime] = None) -> TemporalSummary:
        """
        Generate a daily summary.
        
        Args:
            date: The date to summarize (defaults to today)
            
        Returns:
            Daily TemporalSummary
        """
        if not self.enable_evolution:
            target_date = (date or datetime.utcnow()).date()
            return TemporalSummary(
                period=SummaryPeriod.DAILY,
                start_date=datetime.combine(target_date, datetime.min.time()),
                end_date=datetime.combine(target_date, datetime.max.time()),
                content="Evolution intelligence is disabled."
            )
        
        # Get memories from the day
        target_date = (date or datetime.utcnow()).date()
        start_time = datetime.combine(target_date, datetime.min.time())
        end_time = datetime.combine(target_date, datetime.max.time())
        memories = self.store.query_by_time_range(start_time, end_time)
        
        # Generate and save
        summary = self.summarizer.generate_daily_summary(memories, target_date)
        self.summarizer.save_summary(summary)
        
        return summary
    
    def generate_weekly_summary(self, end_date: Optional[datetime] = None) -> TemporalSummary:
        """
        Generate a weekly summary.
        
        Args:
            end_date: End of the week (defaults to today)
            
        Returns:
            Weekly TemporalSummary
        """
        if not self.enable_evolution:
            end = end_date or datetime.utcnow()
            return TemporalSummary(
                period=SummaryPeriod.WEEKLY,
                start_date=end - timedelta(days=7),
                end_date=end,
                content="Evolution intelligence is disabled."
            )
        
        # Get memories from the week
        end = end_date or datetime.utcnow()
        start_time = end - timedelta(days=7)
        memories = self.store.query_by_time_range(start_time, end)
        
        # Generate and save
        summary = self.summarizer.generate_weekly_summary(memories, end)
        self.summarizer.save_summary(summary)
        
        return summary
    
    def get_summaries(
        self,
        period: Optional[SummaryPeriod] = None,
        limit: int = 10
    ) -> List[TemporalSummary]:
        """
        Get saved summaries.
        
        Args:
            period: Optional filter by period type
            limit: Maximum number of summaries to return
            
        Returns:
            List of TemporalSummaries
        """
        if not self.enable_evolution:
            return []
        return self.summarizer.load_summaries(period=period, limit=limit)
    
    # =========================================================================
    # CRUD Operations
    # =========================================================================
    
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
        
        stats = {
            "storage": store_stats,
            "classifier": classifier_stats,
            "kernel_version": "0.4.0",
            "layer_2_enabled": self.enable_evolution,
            "layer_3_enabled": self.enable_recall,
            "layer_4_enabled": self.enable_constraints
        }
        
        # Add evolution stats if enabled
        if self.enable_evolution:
            stats["evolution"] = {
                "links": self.linker.get_link_stats(),
                "summaries": self.summarizer.get_summary_stats()
            }
        
        # Add recall stats if enabled
        if self.enable_recall:
            stats["recall"] = self.recall_engine.get_stats()
        
        # Add constraint stats if enabled
        if self.enable_constraints:
            stats["constraints"] = self.constraint_engine.get_validation_stats()
        
        return stats
    
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
        Clear all stored memories and evolution data.
        
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
            
            # Clear evolution data
            if self.enable_evolution:
                self.linker.clear_all_links()
            
            return True
        
        return False
