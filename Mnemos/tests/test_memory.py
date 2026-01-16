"""
Basic tests for Mnemos Memory Kernel

These tests verify the core functionality of the memory kernel,
including memory node creation, intent classification, and storage.
"""

import pytest
from datetime import datetime
import tempfile
import shutil

from mnemos import MemoryNode, MemoryIntent, Entity, EntityType, MnemosKernel, TranscriptInput
from mnemos import IntentClassifier, MemoryStore


class TestMemoryNode:
    """Tests for MemoryNode data structure."""
    
    def test_create_memory_node(self):
        """Test creating a basic memory node."""
        memory = MemoryNode(
            raw_text="I think we should increase the budget",
            timestamp=datetime.utcnow(),
            intent=MemoryIntent.IDEA,
            confidence=0.9
        )
        
        assert memory.id is not None
        assert memory.raw_text == "I think we should increase the budget"
        assert memory.intent == MemoryIntent.IDEA
        assert memory.confidence == 0.9
    
    def test_memory_node_with_entities(self):
        """Test creating a memory node with entities."""
        memory = MemoryNode(
            raw_text="Meeting with John on January 15th",
            timestamp=datetime.utcnow(),
            intent=MemoryIntent.IDEA,
            confidence=0.8
        )
        
        memory.add_entity(Entity(
            type=EntityType.PERSON,
            value="John",
            confidence=0.9
        ))
        
        memory.add_entity(Entity(
            type=EntityType.DATE,
            value="January 15th",
            confidence=0.85
        ))
        
        assert len(memory.entities) == 2
        assert any(e.value == "John" for e in memory.entities)
    
    def test_memory_node_serialization(self):
        """Test memory node serialization to dict."""
        memory = MemoryNode(
            raw_text="Test memory content",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            intent=MemoryIntent.QUESTION,
            confidence=0.95
        )
        
        data = memory.to_dict()
        
        assert data["raw_text"] == "Test memory content"
        assert data["intent"] == "question"
        assert data["confidence"] == 0.95
    
    def test_intent_enum_values(self):
        """Test that all intent enum values exist."""
        assert MemoryIntent.IDEA.value == "idea"
        assert MemoryIntent.DECISION.value == "decision"
        assert MemoryIntent.QUESTION.value == "question"
        assert MemoryIntent.REFLECTION.value == "reflection"
        assert MemoryIntent.ACTION.value == "action"


class TestIntentClassifier:
    """Tests for IntentClassifier."""
    
    def test_classify_decision(self):
        """Test classifying a decision statement."""
        classifier = IntentClassifier()
        
        intent, confidence = classifier.classify("I have decided to cancel the project")
        
        assert intent == MemoryIntent.DECISION
        assert confidence > 0.5
    
    def test_classify_question(self):
        """Test classifying a question."""
        classifier = IntentClassifier()
        
        intent, confidence = classifier.classify("How do I implement this feature?")
        
        assert intent == MemoryIntent.QUESTION
        assert confidence > 0.5
    
    def test_classify_action(self):
        """Test classifying an action item."""
        classifier = IntentClassifier()
        
        intent, confidence = classifier.classify("Remember to call John tomorrow")
        
        assert intent == MemoryIntent.ACTION
        assert confidence > 0.5
    
    def test_classify_idea(self):
        """Test classifying an idea."""
        classifier = IntentClassifier()
        
        intent, confidence = classifier.classify("I think we should explore this direction")
        
        assert intent == MemoryIntent.IDEA
        assert confidence > 0.5
    
    def test_classify_reflection(self):
        """Test classifying a reflection."""
        classifier = IntentClassifier()
        
        intent, confidence = classifier.classify("I realized that my initial approach was wrong")
        
        assert intent == MemoryIntent.REFLECTION
        assert confidence > 0.5
    
    def test_classification_confidence_range(self):
        """Test that confidence is always between 0 and 1."""
        classifier = IntentClassifier()
        test_texts = [
            "I have decided to proceed with option A",
            "What is the best way to handle this?",
            "Remember to send the email",
            "I think this might work",
            "Looking back, I should have done differently"
        ]
        
        for text in test_texts:
            intent, confidence = classifier.classify(text)
            assert 0 <= confidence <= 1


class TestMemoryStore:
    """Tests for MemoryStore."""
    
    def setup_method(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_store_and_retrieve(self):
        """Test storing and retrieving a memory."""
        store = MemoryStore(self.temp_dir)
        
        memory = MemoryNode(
            raw_text="Test memory",
            timestamp=datetime.utcnow(),
            intent=MemoryIntent.IDEA,
            confidence=0.9
        )
        
        store.store(memory)
        
        retrieved = store.retrieve(memory.id)
        
        assert retrieved is not None
        assert retrieved.id == memory.id
        assert retrieved.raw_text == "Test memory"
    
    def test_query_by_intent(self):
        """Test querying memories by intent."""
        store = MemoryStore(self.temp_dir)
        
        for i in range(5):
            memory = MemoryNode(
                raw_text=f"Decision {i}",
                timestamp=datetime.utcnow(),
                intent=MemoryIntent.DECISION,
                confidence=0.9
            )
            store.store(memory)
        
        for i in range(3):
            memory = MemoryNode(
                raw_text=f"Idea {i}",
                timestamp=datetime.utcnow(),
                intent=MemoryIntent.IDEA,
                confidence=0.9
            )
            store.store(memory)
        
        decisions = store.query_by_intent(MemoryIntent.DECISION)
        ideas = store.query_by_intent(MemoryIntent.IDEA)
        
        assert len(decisions) == 5
        assert len(ideas) == 3
    
    def test_query_recent(self):
        """Test querying recent memories."""
        store = MemoryStore(self.temp_dir)
        
        from datetime import timedelta
        
        # Store memories at different times
        for i in range(10):
            memory = MemoryNode(
                raw_text=f"Memory {i}",
                timestamp=datetime.utcnow() - timedelta(hours=i),
                intent=MemoryIntent.IDEA,
                confidence=0.9
            )
            store.store(memory)
        
        recent = store.query_recent(5)
        
        assert len(recent) == 5
        # Most recent should be first
        assert "Memory 0" in recent[0].raw_text
    
    def test_update_memory(self):
        """Test updating an existing memory."""
        store = MemoryStore(self.temp_dir)
        
        memory = MemoryNode(
            raw_text="Original text",
            timestamp=datetime.utcnow(),
            intent=MemoryIntent.IDEA,
            confidence=0.9
        )
        store.store(memory)
        
        # Update the memory
        memory.raw_text = "Updated text"
        result = store.update(memory)
        
        assert result is True
        
        retrieved = store.retrieve(memory.id)
        assert retrieved.raw_text == "Updated text"
    
    def test_delete_memory(self):
        """Test deleting a memory."""
        store = MemoryStore(self.temp_dir)
        
        memory = MemoryNode(
            raw_text="To be deleted",
            timestamp=datetime.utcnow(),
            intent=MemoryIntent.IDEA,
            confidence=0.9
        )
        store.store(memory)
        
        result = store.delete(memory.id)
        
        assert result is True
        
        retrieved = store.retrieve(memory.id)
        assert retrieved is None
    
    def test_get_stats(self):
        """Test getting storage statistics."""
        store = MemoryStore(self.temp_dir)
        
        for i in range(5):
            memory = MemoryNode(
                raw_text=f"Memory {i}",
                timestamp=datetime.utcnow(),
                intent=MemoryIntent.IDEA,
                confidence=0.9
            )
            store.store(memory)
        
        stats = store.get_stats()
        
        assert stats["total_memories"] == 5
        assert "idea" in stats["by_intent"]
        assert stats["by_intent"]["idea"] == 5


class TestMnemosKernel:
    """Integration tests for MnemosKernel."""
    
    def setup_method(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ingest_transcript(self):
        """Test ingesting a transcript creates a memory."""
        kernel = MnemosKernel(
            storage_dir=self.temp_dir,
            enable_evolution=False,
            enable_recall=False
        )
        
        transcript = TranscriptInput(
            text="I have decided to use Python for this project",
            timestamp=datetime.utcnow()
        )
        
        memory = kernel.ingest(transcript)
        
        assert memory is not None
        assert memory.id is not None
        assert memory.intent == MemoryIntent.DECISION
        assert memory.confidence > 0.5
    
    def test_get_memory(self):
        """Test retrieving a specific memory."""
        kernel = MnemosKernel(
            storage_dir=self.temp_dir,
            enable_evolution=False,
            enable_recall=False
        )
        
        transcript = TranscriptInput(text="Test memory content")
        memory = kernel.ingest(transcript)
        
        retrieved = kernel.get_memory(memory.id)
        
        assert retrieved is not None
        assert retrieved.id == memory.id
    
    def test_get_recent_memories(self):
        """Test getting recent memories."""
        kernel = MnemosKernel(
            storage_dir=self.temp_dir,
            enable_evolution=False,
            enable_recall=False
        )
        
        # Create several memories
        for i in range(5):
            transcript = TranscriptInput(text=f"Memory {i}")
            kernel.ingest(transcript)
        
        recent = kernel.get_recent_memories(3)
        
        assert len(recent) == 3
    
    def test_delete_memory(self):
        """Test deleting a memory through the kernel."""
        kernel = MnemosKernel(
            storage_dir=self.temp_dir,
            enable_evolution=False,
            enable_recall=False
        )
        
        transcript = TranscriptInput(text="To be deleted")
        memory = kernel.ingest(transcript)
        
        result = kernel.delete_memory(memory.id)
        
        assert result is True
        
        retrieved = kernel.get_memory(memory.id)
        assert retrieved is None
    
    def test_get_stats(self):
        """Test getting kernel statistics."""
        kernel = MnemosKernel(
            storage_dir=self.temp_dir,
            enable_evolution=False,
            enable_recall=False
        )
        
        # Create some memories
        for i in range(3):
            kernel.ingest(TranscriptInput(text=f"Memory {i}"))
        
        stats = kernel.get_stats()
        
        assert stats["storage"]["total_memories"] == 3
    
    def test_clear_all(self):
        """Test clearing all memories."""
        kernel = MnemosKernel(
            storage_dir=self.temp_dir,
            enable_evolution=False,
            enable_recall=False
        )
        
        # Create some memories
        for i in range(5):
            kernel.ingest(TranscriptInput(text=f"Memory {i}"))
        
        result = kernel.clear_all()
        
        assert result is True
        assert kernel.get_stats()["storage"]["total_memories"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
