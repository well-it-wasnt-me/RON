# Vector Memory

DeskBot's **vector memory** extends the base `Memory` system with
embedding-based semantic search. When an embedding function is available
(e.g. sentence-transformers), entries are embedded at add-time and can be
retrieved by cosine similarity - so the LLM can recall semantically
relevant past interactions even when exact keywords don't match.

---

## Semantic vs. keyword search

| Mode | How it works | When available |
|------|-------------|----------------|
| **Keyword search** | Substring match on entry content | Always |
| **Semantic search** | Cosine similarity between query and entry embeddings | When an `EmbeddingFn` is configured |

Keyword search (`search()`) works independently of embeddings and is
always available. Semantic search (`search_similar()`) requires an
embedding function and falls back to reverse-chronological order when
using `NoOpEmbedding`.

---

## Configuration

Vector memory is **opt-in** and disabled by default:

```bash
DESKBOT_VECTOR_MEMORY__ENABLED=true
```

When disabled, the base `Memory` keyword search is used instead.

To use sentence-transformers embeddings:

```bash
pip install sentence-transformers
```

---

## Core classes

### VectorMemory

[`VectorMemory`][robot.ai.vector_memory.VectorMemory] stores entries with
their pre-computed embedding vectors. It maintains a bounded deque of
entries (default capacity 1024) and supports both keyword and semantic
search:

```python
from robot.ai.vector_memory import VectorMemory, SentenceTransformerEmbedding

# With sentence-transformers
embedding_fn = SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2")
memory = VectorMemory(embedding_fn=embedding_fn)

# Add entries
memory.add("User asked about the weather", importance=0.3, tags=["weather"])
memory.add("Robot explained the forecast", importance=0.5, tags=["weather", "explanation"])

# Keyword search
results = memory.search("weather")

# Semantic search - returns (entry, similarity) tuples sorted by relevance
similar = memory.search_similar("What's the temperature outside?", limit=5)
for entry, score in similar:
    print(f"[{score:.2f}] {entry.content}")
```

### VectorMemoryEntry

Each entry is a [`VectorMemoryEntry`][robot.ai.vector_memory.VectorMemoryEntry]
with:

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | `datetime` | When the entry was created |
| `content` | `str` | The memory text |
| `importance` | `float` | Importance score (0.0–1.0) |
| `tags` | `tuple[str, ...]` | Optional tags for categorisation |
| `embedding` | `tuple[float, ...]` | Pre-computed embedding vector |

### Methods

| Method | Description |
|--------|-------------|
| `add(content, importance, tags)` | Add an entry, computing its embedding |
| `recall(limit)` | Return the N most recent entries |
| `search(query)` | Keyword substring search |
| `search_similar(query, limit, min_similarity)` | Semantic cosine-similarity search |
| `clear()` | Remove all entries |

---

## Embedding functions

### NoOpEmbedding

[`NoOpEmbedding`][robot.ai.vector_memory.NoOpEmbedding] is the default
fallback. It returns a zero vector of configurable dimension (default
128), meaning all entries have identical embeddings. `search_similar()`
returns entries in reverse-chronological order.

### SentenceTransformerEmbedding

[`SentenceTransformerEmbedding`][robot.ai.vector_memory.SentenceTransformerEmbedding]
wraps the `sentence-transformers` library and produces normalised
embeddings suitable for cosine similarity:

```python
from robot.ai.vector_memory import VectorMemory, SentenceTransformerEmbedding

fn = SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2")
memory = VectorMemory(embedding_fn=fn)

memory.add("The robot likes discussing philosophy")
memory.add("User prefers casual conversation style")

# Semantic search - finds entries by meaning, not just keywords
results = memory.search_similar("intellectual topics", limit=5)
```

Requires: `pip install sentence-transformers`

### Custom EmbeddingFn

You can implement the [`EmbeddingFn`][robot.ai.vector_memory.EmbeddingFn]
protocol to use any embedding backend (OpenAI embeddings, local models,
etc.):

```python
from robot.ai.vector_memory import EmbeddingFn

class MyEmbedding(EmbeddingFn):
    def embed(self, text: str) -> list[float]:
        # Call your embedding API or model here
        return my_model.encode(text)
```

---

## Cosine similarity

[`search_similar()`][robot.ai.vector_memory.VectorMemory.search_similar]
computes cosine similarity between the query embedding and each stored
entry's embedding:

```text
similarity(q, e) = (q · e) / (|q| × |e|)
```

Entries are returned sorted by descending similarity. The `min_similarity`
parameter (default 0.0) filters out entries below a threshold, allowing
you to exclude weakly related results.

---

## API reference

::: robot.ai.vector_memory.VectorMemory
    options:
      show_root_heading: true

::: robot.ai.vector_memory.VectorMemoryEntry
    options:
      show_root_heading: true

::: robot.ai.vector_memory.EmbeddingFn
    options:
      show_root_heading: true

::: robot.ai.vector_memory.NoOpEmbedding
    options:
      show_root_heading: true

::: robot.ai.vector_memory.SentenceTransformerEmbedding
    options:
      show_root_heading: true
