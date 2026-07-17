package search

import "core:math"
import "core:slice"
import "core:strings"
import "base:runtime"

import "../types"

BM25_K1 :: 1.5
BM25_B :: 0.75

TermPosting :: struct {
    chunk_id: int,
    tf:       int,
}

BM25Index :: struct {
    terms:        map[string][dynamic]TermPosting,
    doc_lengths:  [dynamic]int,
    avg_doc_len:  f64,
    chunk_count:  int,
}

bm25_init :: proc(idx: ^BM25Index) {
    idx.terms = make(map[string][dynamic]TermPosting)
    idx.doc_lengths = make([dynamic]int)
}

bm25_destroy :: proc(idx: ^BM25Index) {
    for k, postings in idx.terms {
        delete(k)
        delete(postings)
    }
    delete(idx.terms)
    delete(idx.doc_lengths)
}

tokenize :: proc(text: string, allocator := context.allocator) -> []string {
    tokens := make([dynamic]string, allocator = allocator)
    current: strings.Builder
    strings.builder_init(&current, allocator)

    flush :: proc(tokens: ^[dynamic]string, current: ^strings.Builder, alloc: runtime.Allocator) {
        if strings.builder_len(current^) > 0 {
            tok := strings.clone(strings.to_lower(string(current.buf[:])), alloc)
            if len(tok) >= 2 {
                append(tokens, tok)
            }
            strings.builder_reset(current)
        }
    }

    for c in text {
        if (c >= 'a' && c <= 'z') ||
           (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') ||
           c == '_' {
            strings.write_rune(&current, c)
        } else {
            flush(&tokens, &current, allocator)
        }
    }
    flush(&tokens, &current, allocator)
    return tokens[:]
}

bm25_build :: proc(idx: ^BM25Index, chunks: []types.Chunk) {
    bm25_destroy(idx)
    bm25_init(idx)
    idx.chunk_count = len(chunks)

    total_len := 0
    for chunk, chunk_id in chunks {
        tokens := tokenize(chunk.text)
        defer delete(tokens)
        tf_map := make(map[string]int, context.temp_allocator)

        for tok in tokens {
            tf_map[tok] += 1
        }

        doc_len := len(tokens)
        append(&idx.doc_lengths, doc_len)
        total_len += doc_len

        for tok, tf in tf_map {
            term := strings.clone(tok, context.allocator)
            postings, ok := idx.terms[term]
            if !ok {
                postings = make([dynamic]TermPosting)
            }
            append(&postings, TermPosting{chunk_id = chunk_id, tf = tf})
            idx.terms[term] = postings
        }
    }

    if len(chunks) > 0 {
        idx.avg_doc_len = f64(total_len) / f64(len(chunks))
    }
}

bm25_search :: proc(idx: ^BM25Index, query: string) -> []f64 {
    scores := make([]f64, idx.chunk_count)
    query_tokens := tokenize(query)

    for tok in query_tokens {
        postings, ok := idx.terms[tok]
        if !ok {
            continue
        }

        df := len(postings)
        idf := math.log_f64(
            (f64(idx.chunk_count) - f64(df) + 0.5) / (f64(df) + 0.5) + 1.0,
            math.E,
        )

        for posting in postings {
            dl := f64(idx.doc_lengths[posting.chunk_id])
            tf := f64(posting.tf)
            num := tf * (BM25_K1 + 1.0)
            den := tf + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / idx.avg_doc_len)
            scores[posting.chunk_id] += idf * (num / den)
        }
    }

    return scores
}

bm25_top_k :: proc(idx: ^BM25Index, query: string, top_k: int) -> (ids: []int, scores: []f64) {
    all_scores := bm25_search(idx, query)
    defer delete(all_scores)

    ranked := make([dynamic]int, 0, idx.chunk_count)
    for i in 0 ..< idx.chunk_count {
        append(&ranked, i)
    }

    for i in 0 ..< len(ranked) {
        for j in i + 1 ..< len(ranked) {
            if all_scores[ranked[j]] > all_scores[ranked[i]] {
                ranked[i], ranked[j] = ranked[j], ranked[i]
            }
        }
    }

    limit := min(top_k, len(ranked))
    ids = make([]int, limit)
    scores = make([]f64, limit)
    for i in 0 ..< limit {
        id := ranked[i]
        ids[i] = id
        scores[i] = all_scores[id]
    }
    delete(ranked)
    return ids, scores
}

rrf_merge :: proc(bm25_ids: []int, vector_ids: []int, k: int = 60) -> (merged_ids: []int, merged_scores: []f64) {
    score_map := make(map[int]f64)

    for id, rank in bm25_ids {
        entry := score_map[id]
        score_map[id] = entry + 1.0 / (f64(k) + f64(rank) + 1.0)
    }
    for id, rank in vector_ids {
        entry := score_map[id]
        score_map[id] = entry + 1.0 / (f64(k) + f64(rank) + 1.0)
    }

    ranked := make([dynamic]int, 0, len(score_map))
    for id in score_map {
        append(&ranked, id)
    }

    for i in 0 ..< len(ranked) {
        for j in i + 1 ..< len(ranked) {
            if score_map[ranked[j]] > score_map[ranked[i]] {
                ranked[i], ranked[j] = ranked[j], ranked[i]
            }
        }
    }

    merged_ids = make([]int, len(ranked))
    merged_scores = make([]f64, len(ranked))
    for id, i in ranked {
        merged_ids[i] = id
        merged_scores[i] = score_map[id]
    }
    delete(ranked)
    return merged_ids, merged_scores
}

cosine_similarity :: proc(a, b: []f32) -> f32 {
    if len(a) != len(b) || len(a) == 0 {
        return 0
    }
    dot: f32 = 0
    norm_a: f32 = 0
    norm_b: f32 = 0
    for i in 0 ..< len(a) {
        dot += a[i] * b[i]
        norm_a += a[i] * a[i]
        norm_b += b[i] * b[i]
    }
    if norm_a == 0 || norm_b == 0 {
        return 0
    }
    return dot / (math.sqrt_f32(norm_a) * math.sqrt_f32(norm_b))
}

vector_search :: proc(
    query_vec: []f32,
    vectors: []f32,
    dim: int,
    chunk_count: int,
    top_k: int,
) -> (
    ids: []int,
    scores: []f32,
) {
    if dim <= 0 || chunk_count <= 0 {
        return nil, nil
    }

    ranked := make([dynamic]int, 0, chunk_count)
    score_list := make([dynamic]f32, 0, chunk_count)

    for i in 0 ..< chunk_count {
        offset := i * dim
        vec := vectors[offset:offset + dim]
        score := cosine_similarity(query_vec, vec)
        append(&ranked, i)
        append(&score_list, score)
    }

    for i in 0 ..< len(ranked) {
        for j in i + 1 ..< len(ranked) {
            if score_list[j] > score_list[i] {
                ranked[i], ranked[j] = ranked[j], ranked[i]
                score_list[i], score_list[j] = score_list[j], score_list[i]
            }
        }
    }

    limit := min(top_k, len(ranked))
    ids = make([]int, limit)
    scores = make([]f32, limit)
    for i in 0 ..< limit {
        ids[i] = ranked[i]
        scores[i] = score_list[i]
    }
    delete(ranked)
    delete(score_list)
    return ids, scores
}

hybrid_search :: proc(
    bm25_idx: ^BM25Index,
    query: string,
    query_vec: []f32,
    vectors: []f32,
    vector_dim: int,
    chunks: []types.Chunk,
    top_k: int,
) -> (
    hits: [dynamic]types.SearchHit,
) {
    hits = make([dynamic]types.SearchHit)

    bm25_ids, bm25_scores := bm25_top_k(bm25_idx, query, top_k * 3)
    defer delete(bm25_ids)
    defer delete(bm25_scores)

    bm25_score_map := make(map[int]f64, context.temp_allocator)
    for id, i in bm25_ids {
        bm25_score_map[id] = bm25_scores[i]
    }

    vector_ids: []int
    vector_scores: []f32
    if len(query_vec) > 0 && len(vectors) > 0 {
        vector_ids, vector_scores = vector_search(
            query_vec,
            vectors,
            vector_dim,
            len(chunks),
            top_k * 3,
        )
        defer delete(vector_ids)
        defer delete(vector_scores)
    }

    vector_score_map := make(map[int]f32, context.temp_allocator)
    for id, i in vector_ids {
        vector_score_map[id] = vector_scores[i]
    }

    merged_ids, merged_scores := rrf_merge(bm25_ids, vector_ids)
    defer delete(merged_ids)
    defer delete(merged_scores)

    limit := min(top_k, len(merged_ids))
    for i in 0 ..< limit {
        id := merged_ids[i]
        append(&hits, types.SearchHit{
            chunk_id     = id,
            score        = merged_scores[i],
            bm25_score   = bm25_score_map[id],
            vector_score = f64(vector_score_map[id]),
            chunk        = chunks[id],
        })
    }

    return hits
}
