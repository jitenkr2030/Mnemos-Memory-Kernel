# Mnemos Memory Kernel

<div align="center">

**A memory kernel that evolves. It does not merely store data—it understands how knowledge changes over time.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-142%20passing-green.svg)](https://github.com/mnemos-project/mnemos)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

---

## What is Mnemos?

Mnemos is a **memory kernel** designed to transform raw voice transcripts into structured, evolving memories. Built as the intelligence layer for [VoiceInk](https://github.com/Beingpax/VoiceInk), Mnemos takes clean transcripts with timestamps and app context, then applies semantic understanding to create a knowledge graph that grows and adapts over time.

Unlike simple note-taking apps or databases, Mnemos understands **context**, **temporal relationships**, and **knowledge evolution**. When you say something today that contradicts what you said last week, Mnemos notices. When a topic resurfaces after months of silence, Mnemos connects the threads.

## What Mnemos Is / Is Not

### Mnemos IS:

- **A Memory Kernel**: A foundational infrastructure layer for storing, organizing, and retrieving memories
- **Deterministic**: Predictable behavior where the same inputs produce the same outputs
- **Extensible**: Designed to be embedded in larger systems, not to serve end-users directly
- **Evolving Graph**: Treats memories as nodes in a temporal knowledge graph, not static records
- **Privacy-Focused**: Stores data locally with no external dependencies or cloud services required

### Mnemos IS NOT:

- **An LLM or AI System**: While it can integrate with LLMs, the core memory operations are rule-based and deterministic
- **A Chatbot or Assistant**: It provides memory infrastructure, not conversational interfaces
- **A Vector Database**: While it supports semantic similarity, its primary value is in evolution tracking and temporal reasoning
- **A "Magic Box"**: Every operation is auditable and explainable

### Systems That Can Embed Mnemos

Mnemos is designed as infrastructure for applications that need memory capabilities:

- **Voice Assistants**: Capture, store, and recall conversation context
- **Research Tools**: Track evolving understanding of complex topics over time
- **Legal/Compliance Systems**: Maintain audit trails of decisions and their justifications
- **Personal Knowledge Management**: Build a second brain that understands how your knowledge grows
- **Decision Tracking Systems**: Record and trace the evolution of important choices

## Architecture

Mnemos follows a **4-layer architecture** designed for incremental development:

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 4: Domain Constraints (Plugins)                       │
│  Validators, truth rules, specialized entity extraction      │
│  GST, Invoice, Date, Email, URL, Currency, Phone validators  │
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

### Domain Constraints (Layer 4)

Mnemos includes a pluggable constraint system for domain-specific validation:

- **GSTValidator**: Validates Indian Goods and Services Tax identification numbers
- **InvoiceValidator**: Validates invoice patterns, amounts, and date consistency
- **DateConsistencyValidator**: Checks for temporal logical errors
- **EmailValidator**: Validates email addresses with disposable domain filtering
- **URLValidator**: Validates website URLs
- **CurrencyValidator**: Validates currency amounts with proper formatting
- **PhoneNumberValidator**: Validates international phone numbers
- **BusinessRuleValidator**: Framework for custom business logic

Domain constraints ensure data integrity before memories are stored, making Mnemos suitable for compliance-critical applications.

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
| `GET` | `/constraints` | List registered domain constraints |
| `GET` | `/constraints/status` | Get constraint engine status |
| `POST` | `/constraints/{id}/validate` | Validate a stored memory |

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
Mnemos/
├── main.py                    # Entry point for server and shell modes
├── requirements.txt           # Python dependencies
├── src/
│   └── mnemos/
│       ├── __init__.py       # Package exports
│       ├── config.py         # Configuration management
│       ├── kernel/
│       │   ├── __init__.py   # Kernel module exports
│       │   ├── memory_node.py  # MemoryNode dataclass (core ABI)
│       │   └── kernel.py     # MnemosKernel orchestrator
│       ├── classifier/
│       │   ├── __init__.py   # Classifier exports
│       │   └── intent_classifier.py  # Rule-based intent classification
│       ├── storage/
│       │   ├── __init__.py   # Storage exports
│       │   └── memory_store.py  # File-based JSON storage
│       ├── api/
│       │   ├── __init__.py   # API exports
│       │   └── api.py        # FastAPI REST endpoints
│       ├── evolution/        # Layer 2: Evolution Intelligence
│       │   ├── __init__.py
│       │   ├── linker.py     # Semantic memory linking
│       │   ├── comparator.py # Conflict/repetition detection
│       │   └── summarizer.py # Temporal summary generation
│       ├── recall/           # Layer 3: Recall Engine
│       │   ├── __init__.py
│       │   ├── query_parser.py  # Natural language query parsing
│       │   ├── importance_scorer.py  # Memory importance scoring
│       │   ├── insight_generator.py  # Insight generation
│       │   └── recall_engine.py # Main recall orchestrator
│       └── constraints/      # Layer 4: Domain Constraints
│           ├── __init__.py
│           ├── base.py       # Constraint base classes
│           ├── constraint_engine.py  # Constraint registry and engine
│           └── validators.py # Domain-specific validators
└── tests/
    ├── __init__.py
    ├── test_memory.py       # Layer 1 tests
    ├── test_evolution.py    # Layer 2 tests
    ├── test_recall.py       # Layer 3 tests
    └── test_constraints.py  # Layer 4 tests
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

## Layer 4: Domain Constraints

Domain constraints enable domain-specific validation before memories are stored. This is critical for compliance, data integrity, and specialized applications.

### Why Domain Constraints Matter

Domain constraints transform Mnemos from a general-purpose memory system into a specialized knowledge engine. Without constraints, any data can enter the memory system, leading to:

- **Data Quality Issues**: Invalid formats, inconsistent data, or nonsensical entries
- **Compliance Risks**: Lack of validation for regulated industries (finance, healthcare, legal)
- **Poor Query Results**: Garbage in, garbage out—bad data produces bad insights
- **Maintenance Burden**: Manual cleanup of invalid or duplicate memories

With domain constraints, Mnemos:

- **Enforces Data Integrity**: Invalid data is rejected or flagged before storage
- **Enables Compliance**: Built-in validators for GST, invoices, dates, and more
- **Improves Query Quality**: Only valid, meaningful memories enter the system
- **Reduces Maintenance**: Automatic validation prevents data quality issues

### Built-in Validators

Mnemos includes several validators for common domains:

```python
from mnemos import GSTValidator, InvoiceValidator, EmailValidator, URLValidator

# Configure kernel with constraints
kernel = MnemosKernel(
    storage_dir="./data",
    enable_constraints=True
)

# Validators are automatically applied during ingestion
kernel.add_constraint(GSTValidator())
kernel.add_constraint(InvoiceValidator())
kernel.add_constraint(EmailValidator())
kernel.add_constraint(URLValidator())
```

### Creating Custom Constraints

You can create custom constraints by implementing the BaseConstraint interface:

```python
from mnemos import BaseConstraint, ConstraintResult, ConstraintType

class CustomValidator(BaseConstraint):
    name = "custom_validator"
    description = "Validates custom business rules"
    constraint_type = ConstraintType.BUSINESS_RULE
    
    def validate(self, memory: MemoryNode) -> ConstraintResult:
        # Your validation logic here
        if "invalid" in memory.raw_text.lower():
            return ConstraintResult(
                passed=False,
                message="Memory contains invalid content",
                severity=ValidationSeverity.ERROR
            )
        return ConstraintResult(passed=True)

kernel.add_constraint(CustomValidator())
```

### Constraint Severity Levels

- **VIOLATION**: Prevents memory storage
- **WARNING**: Allows storage but flags for review
- **INFO**: Informational feedback only

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

All 142 tests pass, covering:

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
- Domain constraint validation
- Constraint engine registration and execution

## Flagship Use Case: Research & Decision Tracking

Mnemos excels at tracking how understanding evolves over time. Here's a complete example demonstrating the Research & Decision Tracking use case:

### Scenario: Researching a Technical Decision

```python
from mnemos import MnemosKernel, TranscriptInput
from datetime import datetime

kernel = MnemosKernel(storage_dir="./research_data")

# Phase 1: Initial Research (Facts)
# We capture factual information we learn during research
kernel.ingest(TranscriptInput(
    text="PostgreSQL supports JSONB data type for semi-structured data",
    timestamp=datetime.utcnow()
))

# Phase 2: Exploration (Beliefs/Ideas)
# We capture our initial thoughts and hypotheses
kernel.ingest(TranscriptInput(
    text="NoSQL databases like MongoDB might be faster for document storage",
    timestamp=datetime.utcnow()
))

# Phase 3: Evaluation (Decisions)
# We capture our evaluation results as we test alternatives
kernel.ingest(TranscriptInput(
    text="After testing, JSONB in Postgres is faster than MongoDB for our use case",
    timestamp=datetime.utcnow()
))

# Phase 4: Conclusion (Decision)
# We capture the final decision with full context
kernel.ingest(TranscriptInput(
    text="We will use PostgreSQL with JSONB for the application database",
    timestamp=datetime.utcnow()
))

# Query: Why did we make this decision?
# Mnemos returns the decision with full evolution chain
result = kernel.recall("Why did we choose PostgreSQL?")

# The result demonstrates Mnemos's unique value:
# It preserves not just what you decided, but WHY you decided it
# Complete with the evolution of your understanding from exploration to conclusion
```

### Why This Use Case Matters

The Research & Decision Tracking use case demonstrates Mnemos's unique value proposition:

1. **Preserves Context**: Every decision is linked to the research and evaluation that informed it
2. **Tracks Evolution**: Shows how understanding changed over time (initial hypothesis → evaluation → conclusion)
3. **Enables Auditing**: Future reviewers can trace the complete decision-making process
4. **Supports Learning**: Identifies patterns in how decisions were made (what worked, what didn't)
5. **Connects Related Decisions**: Similar research threads can be linked across time

This use case is particularly valuable for:
- **Technical Architects**: Tracking technology selection decisions
- **Product Managers**: Recording feature prioritization rationale
- **Consultants**: Documenting client recommendation journeys
- **Researchers**: Building literature review timelines
- **Legal Teams**: Maintaining decision audit trails

## Contributing

Mnemos is designed to be extended. Key extension points:

### Adding a New Intent Type

1. Add the intent to the `MemoryIntent` enum in `src/mnemos/kernel/memory_node.py`
2. Update the intent classifier in `src/mnemos/classifier/intent_classifier.py`
3. Add tests for the new intent
4. Update documentation

### Adding a New Evolution Link Type

1. Add the link type to the `LinkType` enum in `src/mnemos/evolution/linker.py`
2. Update the evolution comparator in `src/mnemos/evolution/comparator.py`
3. Add tests for the new link type behavior
4. Document when and why the link is created

### Adding a New Domain Constraint

1. Create a new validator class in `src/mnemos/constraints/validators.py`
2. Inherit from `BaseConstraint` and implement the `validate` method
3. Register the validator with the kernel configuration
4. Add tests for validation logic
5. Document the validator's purpose and configuration

## LLM Integration Guidelines

Mnemos supports optional LLM integration for enhanced capabilities:

- **Classification**: LLMs can improve intent classification accuracy
- **Summarization**: LLMs can generate more coherent temporal summaries
- **Query Understanding**: LLMs can parse complex natural language queries

### Best Practices

1. **Keep Deterministic Logic as Default**: Ensure the system works without LLMs
2. **Use LLMs for Understanding, Not Truth**: LLMs help interpret, but memories are facts
3. **Document Probabilistic Components**: Clearly mark which features use LLMs
4. **Provide Fallbacks**: Always have rule-based alternatives for LLM-dependent features

## Roadmap

Future enhancements planned for Mnemos:

- **Memory Reinforcement**: Increase importance when memories are recalled
- **Memory Decay**: Gradually reduce importance for long-unused memories
- **Epistemic States**: Distinguish between facts, beliefs, and decisions
- **Evolution Semantics**: Formalize link types (REFINES, CORRECTS, REINFORCES)

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
