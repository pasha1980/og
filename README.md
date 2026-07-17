# OGrep (or "Oh my God, you broke it again!")

Local semantic code indexer and LLM context packer written in [Odin](https://odin-lang.org/).

OG implements a from-scratch RAG retrieval stack - chunking, indexing, hybrid BM25 + vector search, and token-budgeted context assembly - in a single static binary. No Python, LangChain, Chroma, or vector database required.

## Why OG?

Most "AI code search" are thin wrappers around cloud APIs and vector DBs. OG is light, portable and only uses retrieval primitives:

- **Code-aware chunking** at function/signature boundaries
- **Hybrid search** - BM25 for symbols + embeddings for intent
- **Reciprocal Rank Fusion** to merge keyword and semantic rankings
- **Context packing** with token budgets for agent workflows
- **Flat-file index** under `.og/` - portable, zero daemon

## Quick start

### Installation

#### One-liner

```
curl -Ls https://raw.githubusercontent.com/pasha1980/og/main/install/install.sh | bash
```

#### Build from source

Prerequisites

- [Odin compiler](https://odin-lang.org/docs/install/) (dev-2026-07 or newer)
- `clang` (for linking)


```bash
git clone https://github.com/pasha1980/og
cd og
make build
# or: odin build ./src -out:og
mv og ~/.local/bin/og
```

### Usage

#### Index a repository

```bash
# BM25-only (offline, no external services)
og index /path/to/repo

# With Ollama embeddings (recommended)
ollama pull nomic-embed-text
og index /path/to/repo --embed

# With local GGUF model (no Ollama)
og index /path/to/repo --gguf /path/to/model.gguf
```

#### Search

```bash
og search "where is JWT validated"
og search --json -k 5 "authentication middleware"
```

#### Pack context for an LLM

```bash
og context "fix refresh token expiry bug" --budget 4000 --format cursor
```

Copy the markdown output into Cursor, Claude, or any agent.

#### Incremental update

```bash
./og diff-index /path/to/repo
```

Re-indexes only files changed in git.

## Architecture

```
og index
  └─ walker      gitignore-aware file traversal
  └─ chunker     language-aware code splitting
  └─ embed       Ollama HTTP / GGUF token embeddings
  └─ index       binary persistence (.og/)

og search / context
  └─ search      BM25 + cosine similarity + RRF merge
  └─ packer      token-budgeted markdown output
```

### Index format

```
.og/
  manifest.json   repo metadata, embed model, dimensions
  chunks.bin      length-prefixed chunks + line metadata
  bm25.idx        inverted index (binary)
  vectors.bin     f32 embeddings (optional)
```

## Embedding backends

| Backend | Flag | Dependencies |
|---------|------|--------------|
| BM25 only | (default) | Odin `core` stdlib only |
| Ollama | `--embed` | Local Ollama on `127.0.0.1:11434` |
| GGUF | `--gguf PATH` | Minimal GGUF loader in pure Odin |

The GGUF backend loads `token_embd` weights and produces hash-token mean-pooled embeddings - useful offline and as a learning reference for quantized inference.

## Example queries

On this repository:

```bash
og index .
og search "reciprocal rank fusion"
og context "how does gitignore parsing work" --budget 2000
```

BM25 excels at symbol lookup (`ValidateJWT`, `rrf_merge`). With `--embed`, semantic queries like "where do we handle expired sessions" improve recall.

## Tests

```bash
make test
# or: odin test ./src/tests/ -file
```

## License

MIT - see [LICENSE](LICENSE).
