package og_test

import "core:testing"

import "../chunker"
import "../search"
import "../types"

@(test)
test_chunk_constants :: proc(t: ^testing.T) {
    testing.expect(t, types.TARGET_CHUNK_CHARS > types.OVERLAP_CHARS)
    testing.expect(t, len(chunker.SIGNATURE_PREFIXES) > 0)
}

@(test)
test_bm25_finds_keyword :: proc(t: ^testing.T) {
    chunks := []types.Chunk{
        {
            path = "auth.go",
            text = "func ValidateJWT(token string) error { return nil }",
        },
        {
            path = "db.go",
            text = "func ConnectDatabase(url string) {}",
        },
    }

    idx: search.BM25Index
    search.bm25_build(&idx, chunks)
    defer search.bm25_destroy(&idx)

    ids, scores := search.bm25_top_k(&idx, "ValidateJWT", 2)
    defer delete(ids)
    defer delete(scores)

    testing.expect(t, len(ids) > 0)
    testing.expect(t, ids[0] == 0)
    testing.expect(t, scores[0] > 0)
}

@(test)
test_rrf_merge :: proc(t: ^testing.T) {
    bm25_ids := []int{0, 1, 2}
    vector_ids := []int{2, 0, 3}

    merged, scores := search.rrf_merge(bm25_ids, vector_ids)
    defer delete(merged)
    defer delete(scores)

    testing.expect(t, len(merged) == 4)
    testing.expect(t, scores[0] > 0)
}

@(test)
test_cosine_similarity :: proc(t: ^testing.T) {
    a := []f32{1, 0, 0}
    b := []f32{1, 0, 0}
    c := []f32{0, 1, 0}

    testing.expect(t, search.cosine_similarity(a, b) > 0.99)
    testing.expect(t, search.cosine_similarity(a, c) < 0.01)
}
