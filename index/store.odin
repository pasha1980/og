package index

import "core:encoding/endian"
import "core:encoding/json"
import "core:fmt"
import "core:mem"
import "core:os"
import "core:path/filepath"
import "core:slice"
import "core:strings"
import "core:time"

import "../search"
import "../types"

Index_Error :: enum {
    None,
    Invalid_Data,
    Not_Found,
}

Store :: struct {
    root:       string,
    index_path: string,
    chunks:     [dynamic]types.Chunk,
    vectors:    []f32,
    bm25:       search.BM25Index,
    manifest:   types.Manifest,
}

store_init :: proc(s: ^Store, repo_root: string) {
    s.root = strings.clone(repo_root)
    index_sub, _ := filepath.join({repo_root, types.INDEX_DIR}, context.temp_allocator)
    s.index_path = strings.clone(index_sub)
    s.chunks = make([dynamic]types.Chunk)
    search.bm25_init(&s.bm25)
}

store_destroy :: proc(s: ^Store) {
    for c in s.chunks {
        delete(c.path)
        delete(c.language)
        delete(c.symbol_hint)
        delete(c.text)
    }
    delete(s.chunks)
    delete(s.vectors)
    search.bm25_destroy(&s.bm25)
    delete(s.root)
    delete(s.index_path)
}

index_dir :: proc(repo_root: string) -> string {
    dir, _ := filepath.join({repo_root, types.INDEX_DIR}, context.temp_allocator)
    return dir
}

ensure_index_dir :: proc(repo_root: string) -> os.Error {
    dir := index_dir(repo_root)
    if !os.is_dir(dir) {
        return os.make_directory(dir)
    }
    return nil
}

append_bytes :: proc(buf: ^[dynamic]u8, s: string) {
    append(buf, ..transmute([]u8)s)
}

write_u32 :: proc(data: ^[dynamic]u8, value: u32) {
    buf: [4]u8
    endian.put_u32(buf[:], .Little, value)
    append(data, ..buf[:])
}

write_u64 :: proc(data: ^[dynamic]u8, value: u64) {
    buf: [8]u8
    endian.put_u64(buf[:], .Little, value)
    append(data, ..buf[:])
}

write_string :: proc(data: ^[dynamic]u8, s: string) {
    write_u32(data, u32(len(s)))
    append_bytes(data, s)
}

serialize_chunks :: proc(chunks: []types.Chunk, allocator := context.allocator) -> ([]u8, os.Error) {
    buf := make([dynamic]u8, allocator = allocator)
    append_bytes(&buf, types.CHUNK_MAGIC)
    write_u32(&buf, 1)
    write_u64(&buf, u64(len(chunks)))

    for chunk in chunks {
        write_string(&buf, chunk.path)
        write_u32(&buf, u32(chunk.start_line))
        write_u32(&buf, u32(chunk.end_line))
        write_string(&buf, chunk.language)
        write_string(&buf, chunk.symbol_hint)
        write_string(&buf, chunk.text)
    }

    return buf[:], nil
}

read_u32 :: proc(data: []u8, offset: ^int) -> (u32, bool) {
    if offset^ + 4 > len(data) {
        return 0, false
    }
    val := endian.unchecked_get_u32le(data[offset^:])
    offset^ += 4
    return val, true
}

read_u64 :: proc(data: []u8, offset: ^int) -> (u64, bool) {
    if offset^ + 8 > len(data) {
        return 0, false
    }
    val := endian.unchecked_get_u64le(data[offset^:])
    offset^ += 8
    return val, true
}

read_string :: proc(data: []u8, offset: ^int, allocator := context.allocator) -> (string, bool) {
    n, ok := read_u32(data, offset)
    if !ok || offset^ + int(n) > len(data) {
        return "", false
    }
    s := strings.clone(string(data[offset^:][:n]), allocator)
    offset^ += int(n)
    return s, true
}

deserialize_chunks :: proc(
    data: []u8,
    allocator := context.allocator,
) -> (
    chunks: [dynamic]types.Chunk,
    err: Index_Error,
) {
    chunks = make([dynamic]types.Chunk, allocator = allocator)
    if len(data) < 12 {
        return chunks, .Invalid_Data
    }
    if string(data[:4]) != types.CHUNK_MAGIC {
        return chunks, .Invalid_Data
    }

    offset := 4
    version, ok := read_u32(data, &offset)
    if !ok || version != 1 {
        return chunks, .Invalid_Data
    }

    count_u64, ok2 := read_u64(data, &offset)
    if !ok2 {
        return chunks, .Invalid_Data
    }

    for _ in 0 ..< int(count_u64) {
        path, ok_p := read_string(data, &offset, allocator)
        start, ok_s := read_u32(data, &offset)
        end, ok_e := read_u32(data, &offset)
        language, ok_l := read_string(data, &offset, allocator)
        symbol, ok_sym := read_string(data, &offset, allocator)
        text, ok_t := read_string(data, &offset, allocator)
        if !ok_p || !ok_s || !ok_e || !ok_l || !ok_sym || !ok_t {
            return chunks, .Invalid_Data
        }
        append(&chunks, types.Chunk{
            path        = path,
            start_line  = int(start),
            end_line    = int(end),
            language    = language,
            symbol_hint = symbol,
            text        = text,
        })
    }

    return chunks, nil
}

serialize_bm25 :: proc(idx: ^search.BM25Index, allocator := context.allocator) -> ([]u8, os.Error) {
    buf := make([dynamic]u8, allocator = allocator)
    append_bytes(&buf, types.BM25_MAGIC)
    write_u32(&buf, 1)
    write_u64(&buf, u64(idx.chunk_count))
    write_u64(&buf, u64(len(idx.terms)))

    // avg doc len as f64
    avg_bits: u64
    mem.copy(&avg_bits, &idx.avg_doc_len, 8)
    write_u64(&buf, avg_bits)

    for dl in idx.doc_lengths {
        write_u32(&buf, u32(dl))
    }

    for term, postings in idx.terms {
        write_string(&buf, term)
        write_u32(&buf, u32(len(postings)))
        for p in postings {
            write_u32(&buf, u32(p.chunk_id))
            write_u32(&buf, u32(p.tf))
        }
    }

    return buf[:], nil
}

deserialize_bm25 :: proc(
    data: []u8,
    allocator := context.allocator,
) -> (
    idx: search.BM25Index,
    err: Index_Error,
) {
    search.bm25_init(&idx)
    if len(data) < 12 || string(data[:4]) != types.BM25_MAGIC {
        return idx, .Invalid_Data
    }

    offset := 4
    version, ok := read_u32(data, &offset)
    if !ok || version != 1 {
        return idx, .Invalid_Data
    }

    chunk_count_u64, ok2 := read_u64(data, &offset)
    if !ok2 {
        return idx, .Invalid_Data
    }
    idx.chunk_count = int(chunk_count_u64)

    term_count_u64, ok3 := read_u64(data, &offset)
    if !ok3 {
        return idx, .Invalid_Data
    }

    avg_bits, ok4 := read_u64(data, &offset)
    if !ok4 {
        return idx, .Invalid_Data
    }
    mem.copy(&idx.avg_doc_len, &avg_bits, 8)

    for _ in 0 ..< idx.chunk_count {
        dl, ok_dl := read_u32(data, &offset)
        if !ok_dl {
            return idx, .Invalid_Data
        }
        append(&idx.doc_lengths, int(dl))
    }

    for _ in 0 ..< int(term_count_u64) {
        term, ok_t := read_string(data, &offset, allocator)
        if !ok_t {
            return idx, .Invalid_Data
        }
        post_count, ok_pc := read_u32(data, &offset)
        if !ok_pc {
            return idx, .Invalid_Data
        }
        postings := make([dynamic]search.TermPosting)
        for _ in 0 ..< int(post_count) {
            cid, ok_c := read_u32(data, &offset)
            tf, ok_tf := read_u32(data, &offset)
            if !ok_c || !ok_tf {
                delete(postings)
                return idx, .Invalid_Data
            }
            append(&postings, search.TermPosting{chunk_id = int(cid), tf = int(tf)})
        }
        idx.terms[term] = postings
    }

    return idx, nil
}

serialize_vectors :: proc(
    vectors: []f32,
    dim: int,
    allocator := context.allocator,
) -> ([]u8, os.Error) {
    buf := make([dynamic]u8, allocator = allocator)
    append_bytes(&buf, types.VECTOR_MAGIC)
    write_u32(&buf, 1)
    write_u32(&buf, u32(dim))
    write_u64(&buf, u64(len(vectors)))

    for v in vectors {
        bits := transmute(u32)v
        write_u32(&buf, bits)
    }
    return buf[:], nil
}

deserialize_vectors :: proc(
    data: []u8,
    allocator := context.allocator,
) -> (
    vectors: []f32,
    dim: int,
    err: Index_Error,
) {
    if len(data) < 16 || string(data[:4]) != types.VECTOR_MAGIC {
        return nil, 0, .Invalid_Data
    }
    offset := 4
    version, ok := read_u32(data, &offset)
    if !ok || version != 1 {
        return nil, 0, .Invalid_Data
    }
    dim_u32, ok2 := read_u32(data, &offset)
    if !ok2 {
        return nil, 0, .Invalid_Data
    }
    dim = int(dim_u32)
    count_u64, ok3 := read_u64(data, &offset)
    if !ok3 {
        return nil, 0, .Invalid_Data
    }

    vectors = make([]f32, int(count_u64), allocator)
    for i in 0 ..< int(count_u64) {
        bits, ok_b := read_u32(data, &offset)
        if !ok_b {
            delete(vectors)
            return nil, 0, .Invalid_Data
        }
        mem.copy(&vectors[i], &bits, 4)
    }
    return vectors, dim, .None
}

write_manifest :: proc(path: string, manifest: types.Manifest) -> os.Error {
    data, err := json.marshal(manifest, {pretty = true})
    if err != nil {
        return os.ERROR_NONE // marshal failed
    }
    defer delete(data)
    return os.write_entire_file(path, data)
}

read_manifest :: proc(path: string) -> (types.Manifest, Index_Error) {
    data, err := os.read_entire_file(path, context.temp_allocator)
    if err != nil {
        return {}, .Not_Found
    }

    manifest: types.Manifest
    if json_err := json.unmarshal(data, &manifest); json_err != nil {
        return {}, .Invalid_Data
    }
    return manifest, .None
}

store_write :: proc(
    s: ^Store,
    chunks: []types.Chunk,
    vectors: []f32,
    vector_dim: int,
    embed_model: string,
) -> os.Error {
    if err := ensure_index_dir(s.root); err != nil {
        return err
    }

    chunk_data, err := serialize_chunks(chunks[:])
    if err != nil {
        return err
    }
    defer delete(chunk_data)

    search.bm25_build(&s.bm25, chunks)
    bm25_data, err2 := serialize_bm25(&s.bm25)
    if err2 != nil {
        return err2
    }
    defer delete(bm25_data)

    chunk_path, _ := filepath.join({s.index_path, types.CHUNKS_FILE}, context.temp_allocator)
    bm25_path, _ := filepath.join({s.index_path, types.BM25_FILE}, context.temp_allocator)
    manifest_path, _ := filepath.join({s.index_path, types.MANIFEST_FILE}, context.temp_allocator)

    if write_err := os.write_entire_file(chunk_path, chunk_data); write_err != nil {
        return write_err
    }
    if write_err := os.write_entire_file(bm25_path, bm25_data); write_err != nil {
        return write_err
    }

    if len(vectors) > 0 && vector_dim > 0 {
        vector_data, err3 := serialize_vectors(vectors, vector_dim)
        if err3 != nil {
            return err3
        }
        defer delete(vector_data)
        vector_path, _ := filepath.join({s.index_path, types.VECTORS_FILE}, context.temp_allocator)
        if write_err := os.write_entire_file(vector_path, vector_data); write_err != nil {
            return write_err
        }
    } else {
        vector_path, _ := filepath.join({s.index_path, types.VECTORS_FILE}, context.temp_allocator)
        if os.exists(vector_path) {
            os.remove(vector_path)
        }
    }

    now := time.now()
    time_buf: [64]u8
    built_at := time.to_string_yyyy_mm_dd(now, time_buf[:])
    s.manifest = types.Manifest{
        version     = types.OG_VERSION,
        repo_path   = s.root,
        chunk_count = len(chunks),
        embed_model = embed_model,
        built_at    = built_at,
        has_vectors = len(vectors) > 0,
        vector_dim  = vector_dim,
    }

    return write_manifest(manifest_path, s.manifest)
}

store_load :: proc(s: ^Store, repo_root: string) -> Index_Error {
    store_destroy(s)
    store_init(s, repo_root)

    if !os.is_dir(s.index_path) {
        return .Not_Found
    }

    manifest_path, _ := filepath.join({s.index_path, types.MANIFEST_FILE}, context.temp_allocator)
    manifest, manifest_err := read_manifest(manifest_path)
    if manifest_err != .None || manifest.version == "" {
        return .Not_Found
    }
    s.manifest = manifest

    chunk_path, _ := filepath.join({s.index_path, types.CHUNKS_FILE}, context.temp_allocator)
    chunk_data, chunk_err := os.read_entire_file(chunk_path, context.temp_allocator)
    if chunk_err != nil {
        return .Not_Found
    }

    chunks, derr := deserialize_chunks(chunk_data)
    if derr != .None {
        return derr
    }
    s.chunks = chunks

    bm25_path, _ := filepath.join({s.index_path, types.BM25_FILE}, context.temp_allocator)
    bm25_data, bm25_err := os.read_entire_file(bm25_path, context.temp_allocator)
    if bm25_err != nil {
        return .Not_Found
    }

    bm25_idx, berr := deserialize_bm25(bm25_data)
    if berr != .None {
        return berr
    }
    search.bm25_destroy(&s.bm25)
    s.bm25 = bm25_idx

    vector_path, _ := filepath.join({s.index_path, types.VECTORS_FILE}, context.temp_allocator)
    if os.exists(vector_path) {
        vector_data, vec_err := os.read_entire_file(vector_path, context.temp_allocator)
        if vec_err != nil {
            return .Not_Found
        }
        vectors, dim, verr := deserialize_vectors(vector_data)
        if verr != .None {
            return verr
        }
        s.vectors = vectors
        if s.manifest.vector_dim == 0 {
            s.manifest.vector_dim = dim
        }
    }

    return .None
}

estimate_tokens :: proc(text: string) -> int {
    return max(1, len(text) / 4)
}
