"""
Evolution Intelligence Module for Mnemos

This module provides Layer 2 functionality for understanding how memories
relate to each other over time. It includes evolution linking, conflict
detection, and temporal summarization.

Submodules:
- linker: Finds and creates links between related memories
- comparator: Detects repetition, change, and contradiction
- summarizer: Generates temporal summaries
"""

from .linker import EvolutionLinker, MemoryLink, LinkType
from .comparator import EvolutionComparator, RelationshipType
from .summarizer import TemporalSummarizer, SummaryPeriod

__all__ = [
    "EvolutionLinker",
    "MemoryLink", 
    "LinkType",
    "EvolutionComparator",
    "RelationshipType",
    "TemporalSummarizer",
    "SummaryPeriod",
]
