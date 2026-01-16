"""
Basic tests for Mnemos Memory Kernel

These tests verify the core functionality of the memory kernel,
including memory node creation, intent classification, and storage.
"""

import pytest
from datetime import datetime
import tempfile
import shutil

from src.kernel.memory_node import MemoryNode, MemoryIntent, Entity, EntityType
from src.kernel.kernel import MnemosKernel, TranscriptInput
from src.classifier.intent_classifier import IntentClassifier
from src.storage.memory_store import MemoryStore


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
        
        assert memory.raw_text == "I think we should increase the budget"
        assert memory.intent == MemoryIntent.IDEA
        assert memory.confidence == 0.9
        assert memory.id is not None
        assert len(memory.id) > 0
    
    def test_memory_node_validation_empty_text(self):
        """Test that empty text raises an error."""
        with pytest.raises(ValueError):
            MemoryNode(
                raw_text="",
                timestamp=datetime.utcnow(),
                intent=MemoryIntent.IDEA
            )
    
    def test_memory_node_to_dict(self):
        """Test serialization to dictionary."""
        memory = MemoryNode(
            raw_text="Test memory",
            timestamp=datetime.utcnow(),
            intent=MemoryIntent.QUESTION,
            confidence=0.8
        )
        
        data = memory.to_dict()
        
        assert "id" in data
        assert "timestamp" in data
        assert "raw_text" in data
        assert data["intent"] == "question"
        assert data["confidence"] == 0.8
    
    def test_memory_node_from_dict(self):
        """Test deserialization from dictionary."""
        original = MemoryNode(
            raw_text="Decision to launch",
            timestamp=datetime.utcnow(),
            intent=MemoryIntent.DECISION,
            confidence=0.95
        )
        
        data = original.to_dict()
        restored = MemoryNode.from_dict(data)
        
        assert restored.id == original.id
        assert restored.raw_text == original.raw_text
        assert restored.intent == original.intent
        assert restored.confidence == original.confidence
    
    def test_add_evolution_ref(self):
        """Test adding evolution references."""
        memory = MemoryNode(
            raw_text="Updated thinking on pricing",
            timestamp=datetime.utcnow(),
            intent=MemoryIntent.IDEA
        )
        
        memory.add_evolution_ref("prev-memory-id-1")
        memory.add_evolution_ref("prev-memory-id-2")
        
        assert len(memory.evolution_ref) == 2
        assert "prev-memory-id-1" in memory.evolution_ref
    
    def test_add_topic(self):
        """Test adding topics."""
        memory = MemoryNode(
            raw_text="Discussion about architecture",
            timestamp=datetime.utcnow(),
            intent=MemoryIntent.IDEA
        )
        
        memory.add_topic("architecture")
        memory.add_topic("design")
        memory.add_topic("architecture")  # Duplicate
        
        assert len(memory.topics) == 2
        assert "architecture" in memory.topics
        assert "design" in memory.topics


class TestIntentClassifier:
    """Tests for intent classification."""
    
    def setup_method(self):
        """Set up classifier for each test."""
        self.classifier = IntentClassifier()
    
    def test_classify_decision(self):
        """Test classifying decision statements."""
        intent, confidence = self.classifier.classify("I will send the email tomorrow")
        
        assert intent == MemoryIntent.DECISION
        assert confidence > 0.5
    
    def test_classify_action(self):
        """Test classifying action statements."""
        intent, confidence = self.classifier.classify("Remember to call John")
        
        assert intent == MemoryIntent.ACTION
        assert confidence > 0.5
    
    def test_classify_question(self):
        """Test classifying questions."""
        intent, confidence = self.classifier.classify("How do I implement this feature?")
        
        assert intent == MemoryIntent.QUESTION
        assert confidence > 0.5
    
    def test_classify_idea(self):
        """Test classifying idea statements."""
        intent, confidence = self.classifier.classify("I think we should explore this direction")
        
        assert intent == MemoryIntent.IDEA
        assert confidence > 0.5
    
    def test_classify_reflection(self):
        """Test classifying reflection statements."""
        intent, confidence = self.classifier.classify("I realized that my initial approach was wrong")
        
        assert intent == MemoryIntent.REFLECTION
        assert confidence > 0.5
    
    def test_batch_classify(self):
        """Test batch classification."""
        texts = [
            "I will complete this task",
            "What is the deadline?",
            "I think this is a good idea"
        ]
        
        results = self.classifier.batch_classify(texts)
        
        assert len(results) == 3
        assert all(confidence > 0 for _, confidence in results)


class TestMemoryStore:
    """Tests for memory storage."""
    
    def setup_method(self):
        """Set up temporary storage for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.store = MemoryStore(self.temp_dir)
    
    def teardown_method(self):
        """Clean up after each test."""
        shutil.rmtree(self.temp_dir)
    
    def test_store_and_retrieve(self):
        """Test storing and retrieving a memory."""
        memory = MemoryNode(
            raw_text="Test memory content",
            timestamp=datetime.utcnow(),
            intent=MemoryIntent.IDEA,
            confidence=0.8
        )
        
        success = self.store.store(memory)
        assert success is True
        
        retrieved = self.store.retrieve(memory.id)
        assert retrieved is not None
        assert retrieved.raw_text == memory.raw_text
        assert retrieved.intent == memory.intent
    
    def test_query_by_topic(self):
        """Test querying by topic."""
        memory = MemoryNode(
            raw_text="Discussion about pricing",
            timestamp=datetime.utcnow(),
            intent=MemoryIntent.IDEA,
            topics=["pricing", "strategy"]
        )
        
        self.store.store(memory)
        
        results = self.store.query_by_topic("pricing")
        assert len(results) == 1
        assert results[0].id == memory.id
    
    def test_query_by_intent(self):
        """Test querying by intent."""
        memory = MemoryNode(
            raw_text="A decision was made",
            timestamp=datetime.utcnow(),
            intent=MemoryIntent.DECISION
        )
        
        self.store.store(memory)
        
        results = self.store.query_by_intent(MemoryIntent.DECISION)
        assert len(results) == 1
        assert results[0].id == memory.id
    
    def test_query_recent(self):
        """Test querying recent memories."""
        for i in range(5):
            memory = MemoryNode(
                raw_text=f"Memory {i}",
                timestamp=datetime.utcnow(),
                intent=MemoryIntent.IDEA
            )
            self.store.store(memory)
        
        results = self.store.query_recent(3)
        assert len(results) == 3
    
    def test_delete_memory(self):
        """Test deleting a memory."""
        memory = MemoryNode(
            raw_text="To be deleted",
            timestamp=datetime.utcnow(),
            intent=MemoryIntent.IDEA
        )
        
        self.store.store(memory)
        
        success = self.store.delete(memory.id)
        assert success is True
        
        retrieved = self.store.retrieve(memory.id)
        assert retrieved is None
    
    def test_count(self):
        """Test memory count."""
        assert self.store.count() == 0
        
        memory = MemoryNode(
            raw_text="Test",
            timestamp=datetime.utcnow(),
            intent=MemoryIntent.IDEA
        )
        self.store.store(memory)
        
        assert self.store.count() == 1


class TestMnemosKernel:
    """Tests for the main kernel."""
    
    def setup_method(self):
        """Set up temporary kernel for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.kernel = MnemosKernel(self.temp_dir)
    
    def teardown_method(self):
        """Clean up after each test."""
        shutil.rmtree(self.temp_dir)
    
    def test_ingest_transcript(self):
        """Test ingesting a transcript."""
        transcript = TranscriptInput(
            text="I think the pricing should be ₹999",
            timestamp=datetime.utcnow()
        )
        
        memory = self.kernel.ingest(transcript)
        
        assert memory is not None
        assert memory.raw_text == "I think the pricing should be ₹999"
        assert memory.intent == MemoryIntent.IDEA
    
    def test_ingest_and_recall(self):
        """Test ingesting and then recalling a memory."""
        transcript = TranscriptInput(
            text="Remember to schedule the meeting",
            timestamp=datetime.utcnow()
        )
        
        self.kernel.ingest(transcript)
        
        memories = self.kernel.recall()
        assert len(memories) == 1
        assert memories[0].intent == MemoryIntent.ACTION
    
    def test_recall_with_query(self):
        """Test recall with text query."""
        self.kernel.ingest(TranscriptInput(text="Pricing discussion", timestamp=datetime.utcnow()))
        self.kernel.ingest(TranscriptInput(text="Architecture design", timestamp=datetime.utcnow()))
        
        results = self.kernel.recall(query="pricing")
        assert len(results) == 1
        assert "pricing" in results[0].raw_text.lower()
    
    def test_get_stats(self):
        """Test getting system statistics."""
        self.kernel.ingest(TranscriptInput(text="Test 1", timestamp=datetime.utcnow()))
        self.kernel.ingest(TranscriptInput(text="Test 2", timestamp=datetime.utcnow()))
        
        stats = self.kernel.get_stats()
        
        assert stats["storage"]["total_memories"] == 2
    
    def test_get_recent_memories(self):
        """Test getting recent memories."""
        for i in range(3):
            self.kernel.ingest(TranscriptInput(text=f"Memory {i}", timestamp=datetime.utcnow()))
        
        recent = self.kernel.get_recent_memories(2)
        assert len(recent) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
