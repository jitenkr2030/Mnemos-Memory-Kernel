"""
FastAPI Web Interface for Mnemos

This module provides a REST API for interacting with the Mnemos kernel.
The API enables external applications to ingest transcripts, query memories,
and manage the memory system.

The API is designed to be minimal and focused on the core kernel operations,
with additional endpoints added as needed for specific use cases.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .kernel import MnemosKernel, TranscriptInput
from .memory_node import MemoryNode, MemoryIntent


# Pydantic models for API request/response
class TranscriptRequest(BaseModel):
    """Request model for ingesting a transcript."""
    text: str = Field(..., min_length=1, description="The transcribed text content")
    timestamp: Optional[datetime] = Field(None, description="When the speech was captured")
    duration: Optional[float] = Field(None, ge=0, description="Duration of speech in seconds")
    app_context: Optional[str] = Field(None, description="Active application during speech")
    window_title: Optional[str] = Field(None, description="Active window title")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class MemoryResponse(BaseModel):
    """Response model for memory data."""
    id: str
    timestamp: datetime
    intent: str
    topics: List[str]
    confidence: float
    text_preview: str
    raw_text: Optional[str] = None
    
    @classmethod
    def from_memory(cls, memory: MemoryNode, include_raw: bool = False) -> "MemoryResponse":
        """Create response from memory node."""
        return cls(
            id=memory.id,
            timestamp=memory.timestamp,
            intent=memory.intent.value,
            topics=memory.topics,
            confidence=memory.confidence,
            text_preview=memory.raw_text[:100] + "..." if len(memory.raw_text) > 100 else memory.raw_text,
            raw_text=memory.raw_text if include_raw else None
        )


class QueryRequest(BaseModel):
    """Request model for querying memories."""
    query: Optional[str] = Field(None, description="Text search query")
    topic: Optional[str] = Field(None, description="Filter by topic")
    intent: Optional[str] = Field(None, description="Filter by intent (idea, decision, question, reflection, action)")
    start_time: Optional[datetime] = Field(None, description="Start of time range")
    end_time: Optional[datetime] = Field(None, description="End of time range")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum results")


class StatsResponse(BaseModel):
    """Response model for system statistics."""
    total_memories: int
    by_intent: Dict[str, int]
    topic_count: int
    storage_dir: str


# Create FastAPI application
app = FastAPI(
    title="Mnemos API",
    description="Memory Kernel API for personal knowledge management",
    version="0.1.0"
)

# Global kernel instance
kernel: Optional[MnemosKernel] = None


def get_kernel() -> MnemosKernel:
    """Get or create the kernel instance."""
    global kernel
    if kernel is None:
        kernel = MnemosKernel()
    return kernel


@app.on_event("startup")
async def startup_event():
    """Initialize kernel on startup."""
    get_kernel()


@app.post("/ingest", response_model=MemoryResponse, status_code=201)
async def ingest_transcript(request: TranscriptRequest):
    """
    Ingest a transcript and create a memory node.
    
    This endpoint receives text from VoiceInk (or any other source),
    processes it through the memory kernel, and returns the created
    memory node.
    """
    kernel = get_kernel()
    
    transcript_input = TranscriptInput(
        text=request.text,
        timestamp=request.timestamp,
        duration=request.duration,
        app_context=request.app_context,
        window_title=request.window_title,
        metadata=request.metadata
    )
    
    try:
        memory = kernel.ingest(transcript_input)
        return MemoryResponse.from_memory(memory, include_raw=True)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/memories", response_model=List[MemoryResponse])
async def query_memories(
    query: Optional[str] = Query(None, description="Text search query"),
    topic: Optional[str] = Query(None, description="Filter by topic"),
    intent: Optional[str] = Query(None, description="Filter by intent type"),
    start_time: Optional[datetime] = Query(None, description="Start of time range (ISO format)"),
    end_time: Optional[datetime] = Query(None, description="End of time range (ISO format)"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results")
):
    """
    Query memories based on various criteria.
    
    This endpoint supports flexible querying by combining filters.
    Multiple filters are applied with AND logic.
    """
    kernel = get_kernel()
    
    # Parse intent filter
    intent_enum = None
    if intent:
        try:
            intent_enum = MemoryIntent(intent)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid intent: {intent}. Valid values: idea, decision, question, reflection, action"
            )
    
    memories = kernel.recall(
        query=query,
        topic=topic,
        intent=intent_enum,
        start_time=start_time,
        end_time=end_time,
        limit=limit
    )
    
    return [MemoryResponse.from_memory(m) for m in memories]


@app.get("/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(memory_id: str):
    """
    Retrieve a specific memory by ID.
    """
    kernel = get_kernel()
    memory = kernel.get_memory(memory_id)
    
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    return MemoryResponse.from_memory(memory, include_raw=True)


@app.get("/memories/{memory_id}/evolution")
async def get_memory_evolution(memory_id: str):
    """
    Get the evolution chain for a memory.
    
    Returns the memory and all related memories that trace the
    development of a thought over time.
    """
    kernel = get_kernel()
    memories = kernel.recall_evolution(memory_id)
    
    if not memories:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    return {
        "origin_memory": memories[0].id,
        "chain_length": len(memories),
        "memories": [MemoryResponse.from_memory(m, include_raw=True) for m in memories]
    }


@app.get("/recent", response_model=List[MemoryResponse])
async def get_recent_memories(count: int = Query(10, ge=1, le=100)):
    """
    Get the most recent memories.
    """
    kernel = get_kernel()
    memories = kernel.get_recent_memories(count)
    return [MemoryResponse.from_memory(m) for m in memories]


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get system statistics.
    """
    kernel = get_kernel()
    stats = kernel.get_stats()
    
    return StatsResponse(
        total_memories=stats["storage"]["total_memories"],
        by_intent=stats["storage"]["by_intent"],
        topic_count=stats["storage"]["topic_count"],
        storage_dir=stats["storage"]["storage_dir"]
    )


@app.delete("/memories/{memory_id}", status_code=204)
async def delete_memory(memory_id: str):
    """
    Delete a memory by ID.
    """
    kernel = get_kernel()
    success = kernel.delete_memory(memory_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Memory not found")


@app.delete("/clear", status_code=204)
async def clear_all_memories():
    """
    Delete all stored memories.
    
    WARNING: This is a destructive operation that cannot be undone.
    """
    kernel = get_kernel()
    kernel.clear_all()


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "service": "mnemos"}


@app.get("/")
async def root():
    """
    API root endpoint with basic information.
    """
    return {
        "service": "Mnemos Memory Kernel",
        "version": "0.1.0",
        "endpoints": {
            "ingest": "POST /ingest",
            "query": "GET /memories",
            "recent": "GET /recent",
            "stats": "GET /stats",
            "health": "GET /health"
        }
    }
