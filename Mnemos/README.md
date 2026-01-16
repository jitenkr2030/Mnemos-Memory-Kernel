# Mnemos Memory Kernel

<div align="center">

**A memory kernel that evolves. It does not merely store data—it understands how knowledge changes over time.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-47%20passing-green.svg)](https://github.com/mnemos-project/mnemos)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

---

## What is Mnemos?

Mnemos is a **memory kernel** designed to transform raw voice transcripts into structured, evolving memories. Built as the intelligence layer for [VoiceInk](https://github.com/Beingpax/VoiceInk), Mnemos takes clean transcripts with timestamps and app context, then applies semantic understanding to create a knowledge graph that grows and adapts over time.

Unlike simple note-taking apps or databases, Mnemos understands **context**, **temporal relationships**, and **knowledge evolution**. When you say something today that contradicts what you said last week, Mnemos notices. When a topic resurfaces after months of silence, Mnemos connects the threads.

## Architecture

Mnemos follows a **4-layer architecture** designed for incremental development:

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 4: Domain Constraints (Plugins)                       │
│  Validators, truth rules, specialized entity extraction      │
└─────────────────────────────────────────────────────────────┘
                          ▲
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3: Recall Engine                                     │
│  Query → Memory resolution, importance scoring, insights     │
└─────────────────────────────────────────────────────────────┘
                          ▲
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2: Time & Evolution Intelligence                     │
│  Evolution linking, temporal summaries, conflict detection  │
└─────────────────────────────────────────────────────────────┘
                          ▲
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: Memory Kernel (Core)                              │
│  MemoryNode schema, intent classification, topic clustering │
└─────────────────────────────────────────────────────────────┘
                          ▲
┌─────────────────────────────────────────────────────────────┐
│  LAYER 0: Input (VoiceInk)                                  │
│  Speech recognition, audio capture, language decoding       │
└─────────────────────────────────────────────────────────────┘
```

## Core Concepts

### MemoryNode

The fundamental atomic unit of Mnemos. Every captured thought becomes a MemoryNode with the following structure:

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Unique identifier for this memory |
| `timestamp` | datetime | When this memory was captured |
| `raw_text` | string | The verbatim transcript content |
| `intent` | enum | IDEA, DECISION, QUESTION, REFLECTION, ACTION |
| `topics` | list[string] | Semantic cluster identifiers |
| `entities` | list[Entity] | Extracted named elements (people, numbers, dates) |
| `confidence` | float | Classification certainty (0.0-1.0) |
| `evolution_ref` | list[UUID] | Links to related past memories |

### Intent Classification

Every memory must have a clear purpose. Mnemos classifies utterances into five intent types:

- **IDEA**: Creative or exploratory thinking ("I think we should explore this direction")
- **DECISION**: Commitments and choices made ("We will charge ₹999")
- **QUESTION**: Inquiries or information gaps ("How do I implement this feature?")
- **REFLECTION**: Retrospective analysis ("I realized that my initial approach was wrong")
- **ACTION**: Tasks and follow-ups ("Remember to call John tomorrow")

### Evolution Intelligence (Layer 2)

The killer feature. Mnemos doesn't just store memories—it connects them:

1. **Evolution Linking**: When a new memory is created, Mnemos searches past memories on the same topic and automatically creates links.

2. **Conflict Detection**: Contradictions are identified and flagged. "I hate spinach" after "I love spinach" creates a CONTRADICTS link.

3. **Repetition Recognition**: Repeated statements are linked, not duplicated. The system learns what's already been said.

4. **Temporal Summaries**: Auto-generated daily, weekly, and monthly synthesis of knowledge evolution.

### Recall Engine (Layer 3)

Intelligent memory retrieval that goes beyond simple keyword search:

1. **Natural Language Querying**: Ask questions like "What decisions did I make today?" and get relevant results.

2. **Importance Scoring**: Results are ranked by importance, not just relevance. Decisions and actions surface first.

3. **Contextual Insights**: Automatically generated insights reveal themes, patterns, and connections across your memories.

4. **Similar Memory Discovery**: Find related memories based on semantic similarity.

## Quick Start

### Installation

```bash
git clone https://github.com/mnemos-project/mnemos.git
cd mnemos
pip install -r requirements.txt
```

### Basic Usage

```python
from mnemos import MnemosKernel, TranscriptInput

# Initialize the kernel
kernel = MnemosKernel(storage_dir="./data")

# Ingest a transcript from VoiceInk
transcript = TranscriptInput(
    text="I think we should increase the pricing to ₹999",
    timestamp=datetime.utcnow(),
    app_context="Slack",
    window_title="pricing-discussion"
)

# Create a memory
memory = kernel.ingest(transcript)

print(f"Memory created: {memory.id}")
print(f"Intent: {memory.intent.value}")
print(f"Confidence: {memory.confidence:.2f}")

# Intelligent recall with natural language query
result = kernel.recall(query="What decisions did I make about pricing?")
for item in result.memories:
    print(f"- {item.raw_text[:80]}...")
    print(f"  Importance Score: {result.scores[item.id].total:.2f}")

# View generated insights
if result.insights:
    for insight in result.insights.insights:
        print(f"\n[{insight.type.value}] {insight.title}")
        print(f"  {insight.description}")
```

### Running the API Server

```bash
python main.py --mode server --port 8000
```

The API will be available at `http://localhost:8000`.

### Interactive Shell

```bash
python main.py --mode shell
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingest` | Ingest a transcript and create a memory |
| `POST` | `/recall` | Execute intelligent recall query |
| `GET` | `/recall` | Execute recall query (GET method) |
| `GET` | `/memories/similar/{id}` | Find similar memories |
| `GET` | `/memories/{id}/context` | Get memory with full context |
| `GET` | `/memories` | Query memories with filters |
| `GET` | `/memories/{id}` | Retrieve a specific memory |
| `GET` | `/memories/{id}/evolution` | Get evolution chain for a memory |
| `GET` | `/memories/{id}/links` | Get evolution links |
| `GET` | `/memories/conflicts` | Retrieve all detected contradictions |
| `GET` | `/recent` | Get the most recent memories |
| `POST` | `/evolution/summarize` | Generate a temporal summary |
| `GET` | `/evolution/summaries` | Retrieve past summaries |
| `GET` | `/stats` | Get system statistics |
| `GET` | `/health` | Health check endpoint |

### Example: Intelligent Recall via API

```bash
# Natural language query
curl -X POST "http://localhost:8000/recall" \
  -H "Content-Type: application/json" \
  -d '{"query": "What decisions did I make today about pricing?", "limit": 10, "generate_insights": true}'

# Response includes:
# - Matched memories ranked by importance
# - Importance score breakdown
# - Auto-generated insights about themes and patterns
```

### Example: Memory Context

```bash
# Get full context for a memory
curl "http://localhost:8000/memories/{memory_id}/context"

# Returns:
# - Memory data
# - Importance score and breakdown
# - Evolution chain
# - Contextual insights
```

## Configuration

Mnemos can be configured via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MNEMOS_STORAGE_DIR` | Directory for memory storage | `./data` |
| `MNEMOS_LLM_ENABLED` | Enable LLM-based classification | `false` |
| `MNEMOS_LLM_PROVIDER` | LLM provider (openai, anthropic, local) | `null` |
| `MNEMOS_EMBEDDING_MODEL` | Model for generating embeddings | `all-MiniLM-L6-v2` |
| `MNEMOS_LOG_LEVEL` | Logging level (debug, info, warning, error) | `info` |
| `MNEMOS_API_HOST` | API server host | `0.0.0.0` |
| `MNEMOS_API_PORT` | API server port | `8000` |

## Project Structure

```
mnemos/
├── main.py                    # Entry point for server and shell modes
├── requirements.txt           # Python dependencies
├── src/
│   ├── __init__.py           # Package exports
│   ├── config.py             # Configuration management
│   ├── kernel/
│   │   ├── __init__.py      # Kernel module exports
│   │   ├── memory_node.py   # MemoryNode dataclass (core ABI)
│   │   └── kernel.py        # MnemosKernel orchestrator
│   ├── classifier/
│   │   ├── __init__.py      # Classifier exports
│   │   └── intent_classifier.py  # Rule-based intent classification
│   ├── storage/
│   │   ├── __init__.py      # Storage exports
│   │   └── memory_store.py  # File-based JSON storage
│   ├── api/
│   │   ├── __init__.py      # API exports
│   │   └── api.py           # FastAPI REST endpoints
│   ├── evolution/           # Layer 2: Evolution Intelligence
│   │   ├── __init__.py
│   │   ├── linker.py        # Semantic memory linking
│   │   ├── comparator.py    # Conflict/repetition detection
│   │   └── summarizer.py    # Temporal summary generation
│   └── recall/              # Layer 3: Recall Engine
│       ├── __init__.py
│       ├── query_parser.py  # Natural language query parsing
│       ├── importance_scorer.py  # Memory importance scoring
│       ├── insight_generator.py  # Insight generation
│       └── recall_engine.py # Main recall orchestrator
└── tests/
    ├── __init__.py
    ├── test_memory.py       # Layer 1 tests
    ├── test_evolution.py    # Layer 2 tests
    └── test_recall.py       # Layer 3 tests
```

## Layer 2: Evolution Intelligence

### Evolution Linking

When a new memory is created, the Evolution Linker searches past memories on related topics and automatically establishes connections:

```python
# First memory
kernel.ingest(TranscriptInput(text="I think we should use Python for the backend"))

# Later memory on the same topic
kernel.ingest(TranscriptInput(text="Python is the right choice for this project"))

# System automatically links these with RELATES_TO or UPDATES
```

### Conflict Detection

Mnemos identifies contradictions between memories:

```python
# Memory 1
kernel.ingest(TranscriptInput(text="I love coffee in the morning"))

# Contradicting memory
kernel.ingest(TranscriptInput(text="I've decided to stop drinking coffee"))

# Conflict is detected and linked
conflicts = kernel.get_conflicts()
# Returns: [{"memory1_id": "...", "memory2_id": "...", "type": "CONTRADICTION"}]
```

### Temporal Summaries

Generate synthesized summaries of knowledge evolution:

```python
# Generate a weekly summary
summary = kernel.generate_summary(
    start_time=datetime.utcnow() - timedelta(days=7),
    end_time=datetime.utcnow(),
    period="weekly"
)

# Returns: "Over the last week, you focused primarily on X and Y.
#          There was a shift in opinion regarding Z..."
```

## Layer 3: Recall Engine

### Natural Language Querying

The recall engine understands natural language queries:

```python
# Query patterns
result = kernel.recall("What decisions did I make today?")
result = kernel.recall("My questions about the project")
result = kernel.recall("Recent ideas about pricing")
result = kernel.recall("What was I working on last week?")
```

### Importance Scoring

Every recalled memory includes an importance score based on:

- **Intent Type**: Decisions and actions score higher than ideas
- **Entity Mentions**: Memories with dates, numbers, and specific entities rank higher
- **Recency**: Recent memories are prioritized
- **Content Characteristics**: Specific language (urgency, amounts) increases score
- **Evolution Context**: Memories that are part of evolution chains are valued higher

```python
result = kernel.recall("pricing decisions")

for memory in result.memories:
    score = result.scores[memory.id]
    print(f"Memory: {memory.raw_text[:50]}...")
    print(f"  Total Score: {score.total:.2f}")
    print(f"  Intent Score: {score.intent_score:.2f}")
    print(f"  Recency Score: {score.recency_score:.2f}")
```

### Contextual Insights

The recall engine automatically generates insights from your memories:

```python
result = kernel.recall("my work this week", generate_insights=True)

if result.insights:
    for insight in result.insights.insights:
        print(f"[{insight.type.value}] {insight.title}")
        print(f"  {insight.description}")
        
        if insight.evidence:
            print(f"  Evidence: {insight.evidence}")
```

Insight types include:
- **THEME**: Common topics across memories
- **DECISION_TRACKER**: Decision evolution and changes
- **QUESTION_STATUS**: Open and answered questions
- **ACTION_ITEMS**: Pending action items with urgency
- **PATTERN**: Recurring patterns detected
- **TEMPORAL_PATTERN**: Time-based activity patterns

### Similar Memory Discovery

Find related memories based on semantic similarity:

```python
# Get similar memories
memory = kernel.get_memory("specific-memory-id")
similar = kernel.search_similar(memory.id, limit=5)

for s in similar:
    print(f"Similar: {s.raw_text[:80]}...")
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

All 47 tests pass, covering:
- MemoryNode creation and validation
- Intent classification (20+ patterns)
- Storage CRUD operations
- Query functionality
- Kernel ingestion pipeline
- Evolution linking and conflict detection
- Temporal summarization
- Query parsing and understanding
- Importance scoring
- Insight generation
- Recall engine orchestration

## Contributing

Mnemos is designed to be extended. To add a new domain constraint (Layer 4):

1. Create a new module in `src/constraints/`
2. Implement validators for your domain (e.g., GST validation for accounting)
3. Register validators with the kernel configuration
4. The kernel will automatically apply constraints during ingestion

## Layer 4: Domain Constraints (Coming Soon)

Future Layer 4 will enable domain-specific validation and rules:

```python
# Example: Accounting constraints
kernel.add_constraint(GSTValidator())
kernel.add_constraint(InvoiceValidator())
kernel.add_constraint(DateConsistencyValidator())
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [VoiceInk](https://github.com/Beingpax/VoiceInk) for the input layer
- [Sentence-Transformers](https://www.sbert.net/) for embedding models
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework

---

<div align="center">

**Mnemos: Because memory is not storage. It's evolution.**

</div>
