package cmd

import "core:encoding/json"
import "core:fmt"
import "core:os"
import "core:path/filepath"
import "core:slice"
import "core:strings"
import "core:strconv"
import "core:time"

import "../chunker"
import "../embed"
import "../index"
import "../packer"
import "../search"
import "../types"
import "../walker"

CLI_Error :: enum {
    None,
    Invalid_Args,
    Index_Not_Found,
    Index_Failed,
    Search_Failed,
}

resolve_repo :: proc(path: string) -> string {
    if path == "" || path == "." {
        cwd, err := os.getwd(context.temp_allocator)
        if err != nil {
            return "."
        }
        return cwd
    }
    abs, err := filepath.abs(path)
    if err != nil {
        return strings.clone(path)
    }
    return abs
}

parse_int_arg :: proc(s: string, default: int) -> int {
    v, ok := strconv.parse_int(s)
    if ok {
        return v
    }
    return default
}

run_index :: proc(args: []string) -> int {
    path := "."
    embed_flag := false
    ollama_host := embed.DEFAULT_OLLAMA_HOST
    ollama_model := embed.DEFAULT_MODEL
    use_gguf := false
    gguf_path := ""

    i := 0
    for i < len(args) {
        arg := args[i]
        switch arg {
        case "--embed":
            embed_flag = true
        case "--ollama-host":
            i += 1
            if i < len(args) {
                ollama_host = args[i]
            }
        case "--model":
            i += 1
            if i < len(args) {
                ollama_model = args[i]
            }
        case "--gguf":
            use_gguf = true
            i += 1
            if i < len(args) {
                gguf_path = args[i]
            }
        case "-h", "--help":
            print_index_help()
            return 0
        case:
            if !strings.has_prefix(arg, "-") {
                path = arg
            }
        }
        i += 1
    }

    repo := resolve_repo(path)
    fmt.println("Indexing:", repo)

    files, walk_err := walker.collect_files(repo)
    if walk_err != nil {
        fmt.eprintln("Failed to walk repository:", walk_err)
        return 1
    }
    defer {
        for f in files {
            delete(f)
        }
        delete(files)
    }

    chunks, chunk_err := chunker.chunk_files(files[:], repo)
    if chunk_err != nil {
        fmt.eprintln("Failed to chunk files:", chunk_err)
        return 1
    }
    defer {
        for c in chunks {
            delete(c.path)
            delete(c.language)
            delete(c.symbol_hint)
            delete(c.text)
        }
        delete(chunks)
    }

    fmt.printf("Found %d chunks from %d files\n", len(chunks), len(files))

    vectors: []f32
    vector_dim := 0
    embed_model := "bm25-only"

    if use_gguf && gguf_path != "" {
        texts := make([]string, len(chunks), context.temp_allocator)
        for c, i in chunks {
            texts[i] = c.text
        }
        vecs, dim, ok := embed.gguf_embed_batch(texts[:], gguf_path)
        if !ok {
            fmt.eprintln("GGUF embedding failed; index will be BM25-only")
        } else {
            vectors = vecs
            vector_dim = dim
            embed_model = fmt.tprintf("gguf:%s", filepath.base(gguf_path))
            fmt.printf("Embedded %d chunks with GGUF (dim=%d)\n", len(chunks), dim)
        }
    } else if embed_flag {
        config := embed.Ollama_Config{host = ollama_host, model = ollama_model}
        if !embed.ollama_available(config) {
            fmt.eprintln("Ollama not available at", ollama_host, "- indexing BM25 only")
        } else {
            texts := make([]string, len(chunks), context.temp_allocator)
            for c, i in chunks {
                texts[i] = c.text
            }
            vecs, dim, ok := embed.ollama_embed_batch(texts[:], config)
            if !ok {
                fmt.eprintln("Ollama embedding failed; index will be BM25-only")
            } else {
                embed.normalize_vectors(vecs, dim)
                vectors = vecs
                vector_dim = dim
                embed_model = ollama_model
                fmt.printf("Embedded %d chunks via Ollama (dim=%d)\n", len(chunks), dim)
            }
        }
    }

    store: index.Store
    index.store_init(&store, repo)
    defer index.store_destroy(&store)

    if err := index.store_write(&store, chunks[:], vectors, vector_dim, embed_model); err != nil {
        fmt.eprintln("Failed to write index:", err)
        delete(vectors)
        return 1
    }
    delete(vectors)

    index_path, _ := filepath.join({repo, types.INDEX_DIR}, context.temp_allocator)
    fmt.println("Index written to", index_path)
    return 0
}

run_search :: proc(args: []string) -> int {
    path := "."
    query := ""
    top_k := 10
    as_json := false
    ollama_host := embed.DEFAULT_OLLAMA_HOST
    ollama_model := ""
    use_gguf := false
    gguf_path := ""

    i := 0
    for i < len(args) {
        arg := args[i]
        switch arg {
        case "--json":
            as_json = true
        case "-k", "--top":
            i += 1
            if i < len(args) {
                top_k = parse_int_arg(args[i], top_k)
            }
        case "--ollama-host":
            i += 1
            if i < len(args) {
                ollama_host = args[i]
            }
        case "--gguf":
            use_gguf = true
            i += 1
            if i < len(args) {
                gguf_path = args[i]
            }
        case "-h", "--help":
            print_search_help()
            return 0
        case:
            if !strings.has_prefix(arg, "-") && query == "" {
                query = arg
            } else if !strings.has_prefix(arg, "-") && path == "." {
                path = arg
            }
        }
        i += 1
    }

    if query == "" {
        fmt.eprintln("Usage: og search [--json] [-k N] [path] \"query\"")
        return 1
    }

    repo := resolve_repo(path)
    store: index.Store
    if index.store_load(&store, repo) != .None {
        fmt.eprintln("No index found. Run: og index", repo)
        return 1
    }
    defer index.store_destroy(&store)

    query_vec: []f32
    defer delete(query_vec)

    if store.manifest.has_vectors {
        if use_gguf && gguf_path != "" {
            query_vec, _ = embed.gguf_embed(query, gguf_path)
        } else if store.manifest.embed_model != "bm25-only" &&
           !strings.has_prefix(store.manifest.embed_model, "gguf:") {
            model := store.manifest.embed_model
            if ollama_model != "" {
                model = ollama_model
            }
            config := embed.Ollama_Config{host = ollama_host, model = model}
            vec, ok := embed.ollama_embed(query, config)
            if ok {
                embed.normalize_vector(vec)
                query_vec = vec
            }
        } else if strings.has_prefix(store.manifest.embed_model, "gguf:") {
            // Re-use gguf path from manifest if user passed --gguf
            if gguf_path != "" {
                query_vec, _ = embed.gguf_embed(query, gguf_path)
            }
        }
    }

    hits := search.hybrid_search(
        &store.bm25,
        query,
        query_vec,
        store.vectors,
        store.manifest.vector_dim,
        store.chunks[:],
        top_k,
    )
    defer delete(hits)

    if as_json {
        print_search_json(hits[:])
    } else {
        for hit, rank in hits {
            fmt.printf(
                "%d. [%.4f] %s:%d-%d",
                rank + 1,
                hit.score,
                hit.chunk.path,
                hit.chunk.start_line,
                hit.chunk.end_line,
            )
            if hit.chunk.symbol_hint != "" {
                fmt.printf(" (%s)", hit.chunk.symbol_hint)
            }
            fmt.println()
            preview := hit.chunk.text
            if len(preview) > 120 {
                preview = preview[:120]
            }
            fmt.println("   ", strings.replace_all(preview, "\n", " ", context.temp_allocator))
        }
    }

    return 0
}

run_context :: proc(args: []string) -> int {
    path := "."
    query := ""
    budget := 4000
    format_str := "cursor"
    ollama_host := embed.DEFAULT_OLLAMA_HOST
    use_gguf := false
    gguf_path := ""

    i := 0
    for i < len(args) {
        arg := args[i]
        switch arg {
        case "--budget":
            i += 1
            if i < len(args) {
                budget = parse_int_arg(args[i], budget)
            }
        case "--format":
            i += 1
            if i < len(args) {
                format_str = args[i]
            }
        case "--ollama-host":
            i += 1
            if i < len(args) {
                ollama_host = args[i]
            }
        case "--gguf":
            use_gguf = true
            i += 1
            if i < len(args) {
                gguf_path = args[i]
            }
        case "-h", "--help":
            print_context_help()
            return 0
        case:
            if !strings.has_prefix(arg, "-") && query == "" {
                query = arg
            } else if !strings.has_prefix(arg, "-") {
                path = arg
            }
        }
        i += 1
    }

    if query == "" {
        fmt.eprintln("Usage: og context [--budget N] [--format cursor|claude|plain] [path] \"query\"")
        return 1
    }

    repo := resolve_repo(path)
    store: index.Store
    if index.store_load(&store, repo) != .None {
        fmt.eprintln("No index found. Run: og index", repo)
        return 1
    }
    defer index.store_destroy(&store)

    query_vec: []f32
    defer delete(query_vec)
    if store.manifest.has_vectors {
        if use_gguf && gguf_path != "" {
            query_vec, _ = embed.gguf_embed(query, gguf_path)
        } else if store.manifest.embed_model != "bm25-only" &&
           !strings.has_prefix(store.manifest.embed_model, "gguf:") {
            config := embed.Ollama_Config{
                host  = ollama_host,
                model = store.manifest.embed_model,
            }
            vec, ok := embed.ollama_embed(query, config)
            if ok {
                embed.normalize_vector(vec)
                query_vec = vec
            }
        } else if gguf_path != "" {
            query_vec, _ = embed.gguf_embed(query, gguf_path)
        }
    }

    hits := search.hybrid_search(
        &store.bm25,
        query,
        query_vec,
        store.vectors,
        store.manifest.vector_dim,
        store.chunks[:],
        20,
    )
    defer delete(hits)

    output := packer.pack_context(
        hits[:],
        budget,
        packer.format_from_string(format_str),
    )
    defer delete(output)
    fmt.println(output)
    return 0
}

run_diff_index :: proc(args: []string) -> int {
    path := "."
    embed_flag := true
    ollama_host := embed.DEFAULT_OLLAMA_HOST
    ollama_model := ""
    use_gguf := false
    gguf_path := ""

    i := 0
    for i < len(args) {
        arg := args[i]
        switch arg {
        case "--no-embed":
            embed_flag = false
        case "--ollama-host":
            i += 1
            if i < len(args) {
                ollama_host = args[i]
            }
        case "--model":
            i += 1
            if i < len(args) {
                ollama_model = args[i]
            }
        case "--gguf":
            use_gguf = true
            i += 1
            if i < len(args) {
                gguf_path = args[i]
            }
        case "-h", "--help":
            print_diff_help()
            return 0
        case:
            if !strings.has_prefix(arg, "-") {
                path = arg
            }
        }
        i += 1
    }

    repo := resolve_repo(path)
    store: index.Store
    if index.store_load(&store, repo) != .None {
        fmt.eprintln("No index found. Run: og index", repo)
        return 1
    }
    defer index.store_destroy(&store)

    changed := git_changed_files(repo)
    if len(changed) == 0 {
        fmt.println("No changed files detected.")
        return 0
    }

    fmt.printf("Re-indexing %d changed files\n", len(changed))

    // Remove chunks from changed files.
    new_chunks := make([dynamic]types.Chunk)
    for c in store.chunks {
        keep := true
        for ch in changed {
            if c.path == ch {
                keep = false
                delete(c.path)
                delete(c.language)
                delete(c.symbol_hint)
                delete(c.text)
                break
            }
        }
        if keep {
            append(&new_chunks, c)
        }
    }

    // Map old chunk paths to files for vector rebuild.
    changed_set := make(map[string]bool, context.temp_allocator)
    for ch in changed {
        changed_set[ch] = true
    }

    // Add fresh chunks for changed files.
    for rel in changed {
        full, _ := filepath.join({repo, rel}, context.temp_allocator)
        if !os.exists(full) {
            continue
        }
        file_chunks, err := chunker.chunk_file(full)
        if err != nil {
            continue
        }
        for &c in file_chunks {
            delete(c.path)
            c.path = strings.clone(rel)
        }
        append(&new_chunks, ..file_chunks[:])
        delete(file_chunks)
    }

    vectors: []f32
    vector_dim := store.manifest.vector_dim
    embed_model := store.manifest.embed_model

    if embed_flag && store.manifest.has_vectors {
        texts := make([]string, len(new_chunks), context.temp_allocator)
        for c, i in new_chunks {
            texts[i] = c.text
        }

        if use_gguf && gguf_path != "" {
            vecs, dim, ok := embed.gguf_embed_batch(texts[:], gguf_path)
            if ok {
                vectors = vecs
                vector_dim = dim
                embed_model = fmt.tprintf("gguf:%s", filepath.base(gguf_path))
            }
        } else if strings.has_prefix(embed_model, "gguf:") && gguf_path != "" {
            vecs, dim, ok := embed.gguf_embed_batch(texts[:], gguf_path)
            if ok {
                vectors = vecs
                vector_dim = dim
            }
        } else if embed_model != "bm25-only" {
            model := embed_model
            if ollama_model != "" {
                model = ollama_model
            }
            config := embed.Ollama_Config{host = ollama_host, model = model}
            vecs, dim, ok := embed.ollama_embed_batch(texts[:], config)
            if ok {
                embed.normalize_vectors(vecs, dim)
                vectors = vecs
                vector_dim = dim
            }
        }
    }

    if err := index.store_write(&store, new_chunks[:], vectors, vector_dim, embed_model); err != nil {
        fmt.eprintln("Failed to update index:", err)
        delete(vectors)
        for c in new_chunks {
            delete(c.path)
            delete(c.language)
            delete(c.symbol_hint)
            delete(c.text)
        }
        delete(new_chunks)
        return 1
    }
    delete(vectors)

    for c in new_chunks {
        delete(c.path)
        delete(c.language)
        delete(c.symbol_hint)
        delete(c.text)
    }
    delete(new_chunks)

    fmt.println("Index updated.")
    return 0
}

run_benchmark :: proc(args: []string) -> int {
    path := "."
    query := "authentication middleware token validation"
    top_k := 10

    i := 0
    for i < len(args) {
        arg := args[i]
        switch arg {
        case "-q", "--query":
            i += 1
            if i < len(args) {
                query = args[i]
            }
        case "-k":
            i += 1
            if i < len(args) {
                top_k = parse_int_arg(args[i], top_k)
            }
        case:
            if !strings.has_prefix(arg, "-") {
                path = arg
            }
        }
        i += 1
    }

    repo := resolve_repo(path)
    store: index.Store
    if index.store_load(&store, repo) != .None {
        fmt.eprintln("No index found. Run: og index", repo)
        return 1
    }
    defer index.store_destroy(&store)

    // Benchmark og BM25.
    start := time.now()
    hits := search.hybrid_search(&store.bm25, query, nil, nil, 0, store.chunks[:], top_k)
    elapsed_og := time.diff(start, time.now())
    delete(hits)

    // Benchmark ripgrep if available.
    rg_start := time.now()
    rg_count := run_rg_benchmark(repo, query)
    elapsed_rg := time.diff(rg_start, time.now())

    fmt.println("Benchmark:", query)
    fmt.printf("  og (BM25):  %v (%d hits)\n", elapsed_og, top_k)
    fmt.printf("  ripgrep:        %v (%d lines)\n", elapsed_rg, rg_count)
    return 0
}

@(private)
run_rg_benchmark :: proc(repo, query: string) -> int {
    cmd := fmt.tprintf("rg -l %s %s 2>/dev/null | wc -l", query, repo)
    output := execute_shell(cmd)
    defer delete(output)
    count, _ := strconv.parse_int(strings.trim_space(output))
    return count
}

@(private)
execute_shell :: proc(command: string) -> string {
    _, stdout, stderr, err := os.process_exec({
        command = {"sh", "-c", command},
    }, context.temp_allocator)
    if err != nil {
        return ""
    }
    defer delete(stdout)
    defer delete(stderr)
    return strings.clone(string(stdout))
}

@(private)
git_changed_files :: proc(repo: string) -> [dynamic]string {
    files := make([dynamic]string)

    cmd_out := execute_shell(fmt.tprintf("cd %s && git diff --name-only HEAD 2>/dev/null", repo))
    defer delete(cmd_out)
    for line in strings.split_lines(cmd_out) {
        trimmed := strings.trim_space(line)
        if trimmed != "" {
            append(&files, strings.clone(trimmed))
        }
    }

    untracked_out := execute_shell(fmt.tprintf("cd %s && git ls-files --others --exclude-standard 2>/dev/null", repo))
    defer delete(untracked_out)
    for line in strings.split_lines(untracked_out) {
        trimmed := strings.trim_space(line)
        if trimmed != "" {
            append(&files, strings.clone(trimmed))
        }
    }

    return files
}

@(private)
HitJSON :: struct {
    rank:         int `json:"rank"`,
    score:        f64 `json:"score"`,
    bm25_score:   f64 `json:"bm25_score"`,
    vector_score: f64 `json:"vector_score"`,
    path:         string `json:"path"`,
    start_line:   int `json:"start_line"`,
    end_line:     int `json:"end_line"`,
    symbol_hint:  string `json:"symbol_hint,omitempty"`,
    text:         string `json:"text"`,
}

@(private)
print_search_json :: proc(hits: []types.SearchHit) {
    json_hits := make([dynamic]HitJSON, context.temp_allocator)
    for hit, i in hits {
        append(&json_hits, HitJSON{
            rank         = i + 1,
            score        = hit.score,
            bm25_score   = hit.bm25_score,
            vector_score = hit.vector_score,
            path         = hit.chunk.path,
            start_line   = hit.chunk.start_line,
            end_line     = hit.chunk.end_line,
            symbol_hint  = hit.chunk.symbol_hint,
            text         = hit.chunk.text,
        })
    }

    data, err := json.marshal(json_hits[:], {pretty = true})
    if err != nil {
        return
    }
    defer delete(data)
    fmt.println(string(data))
}

print_index_help :: proc() {
    fmt.println("Usage: og index [options] [path]")
    fmt.println("  --embed           Embed chunks via Ollama")
    fmt.println("  --ollama-host H   Ollama host (default 127.0.0.1:11434)")
    fmt.println("  --model M         Embedding model (default nomic-embed-text)")
    fmt.println("  --gguf PATH       Embed using local GGUF model")
}

print_search_help :: proc() {
    fmt.println("Usage: og search [options] [path] \"query\"")
    fmt.println("  --json            Output JSON")
    fmt.println("  -k, --top N       Top results (default 10)")
}

print_context_help :: proc() {
    fmt.println("Usage: og context [options] [path] \"query\"")
    fmt.println("  --budget N        Token budget (default 4000)")
    fmt.println("  --format F        cursor|claude|plain")
}

print_diff_help :: proc() {
    fmt.println("Usage: og diff-index [options] [path]")
    fmt.println("  --no-embed        Skip re-embedding")
}

print_help :: proc() {
    fmt.println("Og - local semantic code indexer")
    fmt.println()
    fmt.println("Usage: og <command> [options]")
    fmt.println()
    fmt.println("Commands:")
    fmt.println("  index       Index a repository")
    fmt.println("  search      Search the index")
    fmt.println("  context     Pack search results for LLM context")
    fmt.println("  diff-index  Incrementally update index from git changes")
    fmt.println("  benchmark   Compare og vs ripgrep")
}
