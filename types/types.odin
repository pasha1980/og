package types

Chunk :: struct {
    path:        string,
    start_line:  int,
    end_line:    int,
    language:    string,
    symbol_hint: string,
    text:        string,
}

SearchHit :: struct {
    chunk_id:    int,
    score:       f64,
    bm25_score:  f64,
    vector_score: f64,
    chunk:       Chunk,
}

Manifest :: struct {
    version:      string `json:"version"`,
    repo_path:    string `json:"repo_path"`,
    chunk_count:  int `json:"chunk_count"`,
    embed_model:  string `json:"embed_model"`,
    built_at:     string `json:"built_at"`,
    has_vectors:  bool `json:"has_vectors"`,
    vector_dim:   int `json:"vector_dim"`,
    repo_hash:    string `json:"repo_hash,omitempty"`,
}

OG_VERSION :: "0.1.0"
INDEX_DIR :: ".og"
CHUNKS_FILE :: "chunks.bin"
VECTORS_FILE :: "vectors.bin"
BM25_FILE :: "bm25.idx"
MANIFEST_FILE :: "manifest.json"

CHUNK_MAGIC :: "SGCH"
BM25_MAGIC :: "SGBM"
VECTOR_MAGIC :: "SGVT"

MAX_CHUNK_CHARS :: 2400
OVERLAP_CHARS :: 320
TARGET_CHUNK_CHARS :: 2000
