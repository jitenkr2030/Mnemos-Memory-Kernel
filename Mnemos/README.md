# Mnemos Memory Kernel

<div align="center">

**A memory kernel that evolves. It does not merely store data—it understands how knowledge changes over time.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-23%20passing-green.svg)](https://github.com/mnemos-project/mnemos)
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
│  Query → Memory resolution, importance scoring              │
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

# Query memories
results = kernel.recall(query="pricing")
for memory in results:
    print(f"- {memory.raw_text[:80]}...")
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
| `GET` | `/memories` | Query memories with filters |
| `GET` | `/memories/{id}` | Retrieve a specific memory |
| `GET` | `/memories/{id}/evolution` | Get evolution chain for a memory |
| `GET` | `/memories/conflicts` | Retrieve all detected contradictions |
| `GET` | `/recent` | Get the most recent memories |
| `POST` | `/evolution/summarize` | Generate a temporal summary |
| `GET` | `/evolution/summaries` | Retrieve past summaries |
| `GET` | `/stats` | Get system statistics |
| `GET` | `/health` | Health check endpoint |

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
│   └── evolution/           # Layer 2: Evolution Intelligence
│       ├── __init__.py
│       ├── linker.py        # Semantic memory linking
│       ├── comparator.py    # Conflict/repetition detection
│       └── summarizer.py    # Temporal summary generation
└── tests/
    ├── __init__.py
    └── test_memory.py       # 23 passing tests
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

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

All 23 tests pass, covering:
- MemoryNode creation and validation
- Intent classification (20+ patterns)
- Storage CRUD operations
- Query functionality
- Kernel ingestion pipeline

## Contributing

Mnemos is designed to be extended. To add a new domain constraint (Layer 4):

1. Create a new module in `src/constraints/`
2. Implement validators for your domain (e.g., GST validation for accounting)
3. Register validators with the kernel configuration
4. The kernel will automatically apply constraints during ingestion

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
