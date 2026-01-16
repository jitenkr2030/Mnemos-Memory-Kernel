"""
Memory Storage Layer for Mnemos

This module provides persistent storage for memory nodes. It implements
a storage abstraction that can work with different backends while
maintaining a consistent interface for the kernel layer.

The storage layer handles:
- CRUD operations for memory nodes
- Topic-based querying
- Temporal range queries
- Evolution chain navigation
"""

import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any, Iterator
from pathlib import Path

from ..kernel.memory_node import MemoryNode, MemoryIntent


class MemoryStore:
    """
    Persistent storage for memory nodes.
    
    This storage implementation uses a file-based JSON format for
    simplicity and portability. In production, this would be replaced
    with PostgreSQL + pgvector or a similar database solution.
    
    The store maintains indexes for efficient querying by:
    - Memory ID (primary key)
    - Timestamp (temporal queries)
    - Topics (semantic clustering)
    - Intent (classification filtering)
    
    Attributes:
        storage_dir: Directory path for storing memory data
        primary_index: Mapping of memory IDs to storage locations
        topic_index: Mapping of topic IDs to memory IDs
        temporal_index: Sorted list of (timestamp, memory_id) tuples
    """
    
    def __init__(self, storage_dir: str = "./data"):
        """
        Initialize the memory store.
        
        Args:
            storage_dir: Directory path for storing memory data
        """
        self.storage_dir = Path(storage_dir)
        self.memories_dir = self.storage_dir / "memories"
        self.index_file = self.storage_dir / "index.json"
        
        # Ensure directories exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.memories_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize indexes
        self._load_indexes()
    
    def _load_indexes(self) -> None:
        """Load or initialize the storage indexes."""
        self.primary_index: Dict[str, Dict[str, Any]] = {}
        self.topic_index: Dict[str, List[str]] = {}
        self.temporal_index: List[tuple] = []
        
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    data = json.load(f)
                    self.primary_index = data.get("primary_index", {})
                    self.topic_index = data.get("topic_index", {})
                    self.temporal_index = [
                        (datetime.fromisoformat(ts), mid) 
                        for ts, mid in data.get("temporal_index", [])
                    ]
            except (json.JSONDecodeError, KeyError) as e:
                # Index file is corrupted, start fresh
                self._save_indexes()
    
    def _save_indexes(self) -> None:
        """Persist the indexes to disk."""
        temporal_data = [
            (ts.isoformat(), mid) for ts, mid in self.temporal_index
        ]
        
        data = {
            "primary_index": self.primary_index,
            "topic_index": self.topic_index,
            "temporal_index": temporal_data,
            "last_updated": datetime.utcnow().isoformat()
        }
        
        with open(self.index_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def store(self, memory: MemoryNode) -> bool:
        """
        Store a memory node.
        
        This method persists the memory node and updates all relevant
        indexes for efficient querying.
        
        Args:
            memory: The memory node to store
            
        Returns:
            True if storage was successful, False otherwise
        """
        try:
            # Save memory data to file
            memory_file = self.memories_dir / f"{memory.id}.json"
            with open(memory_file, 'w') as f:
                json.dump(memory.to_dict(), f, indent=2)
            
            # Update primary index
            self.primary_index[memory.id] = {
                "file": f"{memory.id}.json",
                "timestamp": memory.timestamp.isoformat(),
                "intent": memory.intent.value,
                "topics": memory.topics,
                "confidence": memory.confidence
            }
            
            # Update topic index
            for topic in memory.topics:
                if topic not in self.topic_index:
                    self.topic_index[topic] = []
                if memory.id not in self.topic_index[topic]:
                    self.topic_index[topic].append(memory.id)
            
            # Update temporal index
            self.temporal_index.append((memory.timestamp, memory.id))
            self.temporal_index.sort(key=lambda x: x[0])
            
            # Persist indexes
            self._save_indexes()
            
            return True
            
        except (IOError, OSError) as e:
            return False
    
    def retrieve(self, memory_id: str) -> Optional[MemoryNode]:
        """
        Retrieve a memory node by ID.
        
        Args:
            memory_id: The unique identifier of the memory
            
        Returns:
            The memory node if found, None otherwise
        """
        if memory_id not in self.primary_index:
            return None
        
        memory_file = self.memories_dir / f"{memory_id}.json"
        if not memory_file.exists():
            return None
        
        try:
            with open(memory_file, 'r') as f:
                data = json.load(f)
                return MemoryNode.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None
    
    def query_by_topic(self, topic: str) -> List[MemoryNode]:
        """
        Query memory nodes by topic.
        
        Args:
            topic: The topic identifier to search for
            
        Returns:
            List of memory nodes matching the topic
        """
        if topic not in self.topic_index:
            return []
        
        memories = []
        for memory_id in self.topic_index[topic]:
            memory = self.retrieve(memory_id)
            if memory:
                memories.append(memory)
        
        return memories
    
    def query_by_intent(self, intent: MemoryIntent) -> List[MemoryNode]:
        """
        Query memory nodes by intent.
        
        Args:
            intent: The intent type to filter by
            
        Returns:
            List of memory nodes with matching intent
        """
        memories = []
        for memory_id, index_data in self.primary_index.items():
            if index_data.get("intent") == intent.value:
                memory = self.retrieve(memory_id)
                if memory:
                    memories.append(memory)
        
        return memories
    
    def query_by_time_range(
        self, 
        start_time: datetime, 
        end_time: Optional[datetime] = None
    ) -> List[MemoryNode]:
        """
        Query memory nodes within a time range.
        
        Args:
            start_time: Start of the time range
            end_time: End of the time range (defaults to now)
            
        Returns:
            List of memory nodes within the time range
        """
        end = end_time or datetime.utcnow()
        
        memories = []
        for timestamp, memory_id in self.temporal_index:
            if start_time <= timestamp <= end:
                memory = self.retrieve(memory_id)
                if memory:
                    memories.append(memory)
        
        return memories
    
    def query_recent(self, count: int = 10) -> List[MemoryNode]:
        """
        Query the most recent memory nodes.
        
        Args:
            count: Number of recent memories to return
            
        Returns:
            List of recent memory nodes, most recent first
        """
        memories = []
        for timestamp, memory_id in reversed(self.temporal_index[-count:]):
            memory = self.retrieve(memory_id)
            if memory:
                memories.append(memory)
        
        return memories
    
    def query_by_evolution(self, memory_id: str) -> List[MemoryNode]:
        """
        Query memories related through evolution chain.
        
        This follows the evolution references to find connected memories
        that represent the development of a thought over time.
        
        Args:
            memory_id: The starting memory ID
            
        Returns:
            List of connected memories in evolution order
        """
        memory = self.retrieve(memory_id)
        if not memory:
            return []
        
        memories = [memory]
        
        # Follow evolution references forward
        for ref_id in memory.evolution_ref:
            ref_memory = self.retrieve(ref_id)
            if ref_memory:
                memories.append(ref_memory)
        
        return memories
    
    def update(self, memory: MemoryNode) -> bool:
        """
        Update an existing memory node.
        
        Args:
            memory: The memory node with updated data
            
        Returns:
            True if update was successful, False otherwise
        """
        if memory.id not in self.primary_index:
            return False
        
        return self.store(memory)
    
    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory node.
        
        Args:
            memory_id: The ID of the memory to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if memory_id not in self.primary_index:
            return False
        
        # Remove from primary index
        del self.primary_index[memory_id]
        
        # Remove from topic index
        for topic, memory_ids in self.topic_index.items():
            if memory_id in memory_ids:
                memory_ids.remove(memory_id)
        
        # Remove from temporal index
        self.temporal_index = [
            (ts, mid) for ts, mid in self.temporal_index 
            if mid != memory_id
        ]
        
        # Delete memory file
        memory_file = self.memories_dir / f"{memory_id}.json"
        if memory_file.exists():
            memory_file.unlink()
        
        # Save updated indexes
        self._save_indexes()
        
        return True
    
    def count(self) -> int:
        """Return the total number of stored memories."""
        return len(self.primary_index)
    
    def count_by_intent(self) -> Dict[str, int]:
        """Return the count of memories grouped by intent."""
        counts: Dict[str, int] = {}
        for index_data in self.primary_index.values():
            intent = index_data.get("intent", "unknown")
            counts[intent] = counts.get(intent, 0) + 1
        return counts
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        return {
            "total_memories": self.count(),
            "by_intent": self.count_by_intent(),
            "topic_count": len(self.topic_index),
            "storage_dir": str(self.storage_dir),
            "memory_files": len(list(self.memories_dir.glob("*.json")))
        }
    
    def iter_all(self) -> Iterator[MemoryNode]:
        """
        Iterate over all stored memories.
        
        Yields:
            MemoryNode objects in temporal order
        """
        for timestamp, memory_id in self.temporal_index:
            memory = self.retrieve(memory_id)
            if memory:
                yield memory
