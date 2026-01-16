"""
FastAPI Web Interface for Mnemos

This module provides a REST API for interacting with the Mnemos kernel.
The API enables external applications to ingest transcripts, query memories,
and manage the memory system with Layer 2 Evolution Intelligence support.

The API is designed to be minimal and focused on the core kernel operations,
with additional endpoints for evolution intelligence.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from ..kernel import MnemosKernel, TranscriptInput
from ..kernel.memory_node import MemoryNode, MemoryIntent
from ..evolution.summarizer import SummaryPeriod


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
    layer_2_enabled: bool = True


class MemoryLinkResponse(BaseModel):
    """Response model for memory evolution links."""
    source_id: str
    target_id: str
    link_type: str
    strength: float
    context: str
    created_at: datetime
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryLinkResponse":
        """Create from dictionary."""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            link_type=data["link_type"],
            strength=data["strength"],
            context=data.get("context", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.utcnow())
        )


class ConflictResponse(BaseModel):
    """Response model for detected contradictions."""
    source_id: str
    target_id: str
    source_text: str
    target_text: str
    source_timestamp: str
    target_timestamp: str
    strength: float
    context: str


class SummaryResponse(BaseModel):
    """Response model for temporal summaries."""
    period: str
    start_date: datetime
    end_date: datetime
    content: str
    key_topics: List[str]
    key_decisions: List[str]
    key_questions: List[str]
    memory_count: int
    created_at: datetime
    
    @classmethod
    def from_summary(cls, summary) -> "SummaryResponse":
        """Create from TemporalSummary object."""
        return cls(
            period=summary.period.value,
            start_date=summary.start_date,
            end_date=summary.end_date,
            content=summary.content,
            key_topics=summary.key_topics,
            key_decisions=summary.key_decisions,
            key_questions=summary.key_questions,
            memory_count=summary.memory_count,
            created_at=summary.created_at
        )


class SummaryGenerateRequest(BaseModel):
    """Request model for generating summaries."""
    start_time: datetime = Field(..., description="Start of summary period")
    end_time: datetime = Field(..., description="End of summary period")
    period: str = Field(default="weekly", description="Summary period type: daily, weekly, monthly")


# Create FastAPI application
app = FastAPI(
    title="Mnemos API",
    description="Memory Kernel API for personal knowledge management with Evolution Intelligence",
    version="0.2.0"
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
    memory node. Evolution linking is automatically triggered.
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


@app.get("/memories/{memory_id}/links", response_model=List[MemoryLinkResponse])
async def get_memory_links(memory_id: str):
    """
    Get evolution links for a specific memory.
    
    Returns all links where this memory is the source in new relationships.
    """
    kernel = get_kernel()
    links = kernel.get_memory_links(memory_id)
    
    return [MemoryLinkResponse.from_dict(l.to_dict()) for l in links]


@app.get("/memories/conflicts", response_model=List[ConflictResponse])
async def get_conflicts():
    """
    Get all detected contradictions.
    
    Returns memory pairs that have been identified as contradicting
    each other through evolution intelligence.
    """
    kernel = get_kernel()
    conflicts = kernel.get_conflicts()
    
    return [
        ConflictResponse(
            source_id=c["source_id"],
            target_id=c["target_id"],
            source_text=c["source_text"],
            target_text=c["target_text"],
            source_timestamp=c["source_timestamp"],
            target_timestamp=c["target_timestamp"],
            strength=c["strength"],
            context=c.get("context", "")
        )
        for c in conflicts
    ]


@app.post("/evolution/summarize", response_model=SummaryResponse)
async def generate_summary(request: SummaryGenerateRequest):
    """
    Generate a temporal summary for a specific time range.
    
    Creates a synthesized narrative of knowledge evolution over
    the specified period.
    """
    kernel = get_kernel()
    
    # Parse period
    try:
        period = SummaryPeriod(request.period)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid period: {request.period}. Valid values: daily, weekly, monthly"
        )
    
    summary = kernel.generate_summary(
        start_time=request.start_time,
        end_time=request.end_time,
        period=period
    )
    
    return SummaryResponse.from_summary(summary)


@app.get("/evolution/summaries", response_model=List[SummaryResponse])
async def get_summaries(
    period: Optional[str] = Query(None, description="Filter by period type"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results")
):
    """
    Get previously generated summaries.
    
    Returns saved temporal summaries that can be used to review
    knowledge evolution over time.
    """
    kernel = get_kernel()
    
    # Parse period filter
    period_enum = None
    if period:
        try:
            period_enum = SummaryPeriod(period)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid period: {period}. Valid values: daily, weekly, monthly"
            )
    
    summaries = kernel.get_summaries(period=period_enum, limit=limit)
    
    return [SummaryResponse.from_summary(s) for s in summaries]


@app.get("/evolution/daily-summary")
async def get_daily_summary(date: Optional[datetime] = Query(None, description="Date for summary")):
    """
    Generate and retrieve a daily summary.
    
    Creates a summary of memories from the specified date (or today).
    """
    kernel = get_kernel()
    summary = kernel.generate_daily_summary(date)
    
    return SummaryResponse.from_summary(summary)


@app.get("/evolution/weekly-summary")
async def get_weekly_summary(end_date: Optional[datetime] = Query(None, description="End date for week")):
    """
    Generate and retrieve a weekly summary.
    
    Creates a summary of memories from the past week.
    """
    kernel = get_kernel()
    summary = kernel.generate_weekly_summary(end_date)
    
    return SummaryResponse.from_summary(summary)


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
    Get system statistics including evolution intelligence metrics.
    """
    kernel = get_kernel()
    stats = kernel.get_stats()
    
    return StatsResponse(
        total_memories=stats["storage"]["total_memories"],
        by_intent=stats["storage"]["by_intent"],
        topic_count=stats["storage"]["topic_count"],
        storage_dir=stats["storage"]["storage_dir"],
        layer_2_enabled=stats.get("layer_2_enabled", True)
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
    Delete all stored memories and evolution data.
    
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
        "version": "0.2.0",
        "layer_2": "Evolution Intelligence enabled",
        "endpoints": {
            "ingest": "POST /ingest",
            "query": "GET /memories",
            "memory": "GET /memories/{id}",
            "evolution": "GET /memories/{id}/evolution",
            "links": "GET /memories/{id}/links",
            "conflicts": "GET /memories/conflicts",
            "summarize": "POST /evolution/summarize",
            "summaries": "GET /evolution/summaries",
            "daily_summary": "GET /evolution/daily-summary",
            "weekly_summary": "GET /evolution/weekly-summary",
            "recent": "GET /recent",
            "stats": "GET /stats",
            "health": "GET /health"
        }
    }
