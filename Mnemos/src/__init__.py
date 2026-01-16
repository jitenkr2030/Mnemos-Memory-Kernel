"""
Mnemos - Memory Kernel for Personal Knowledge Management

A lightweight, extensible memory system that transforms voice transcripts
into structured, evolving memories with semantic understanding.
"""

__version__ = "0.3.0"
__author__ = "Mnemos Development Team"

from .kernel.memory_node import MemoryNode, MemoryIntent, EntityType
from .kernel.kernel import MnemosKernel, TranscriptInput
from .storage.memory_store import MemoryStore
from .classifier.intent_classifier import IntentClassifier
from .recall import RecallEngine, RecallResult, InsightCollection

__all__ = [
    "MemoryNode",
    "MemoryIntent", 
    "EntityType",
    "MnemosKernel",
    "TranscriptInput",
    "MemoryStore",
    "IntentClassifier",
    "RecallEngine",
    "RecallResult",
    "InsightCollection",
]
