package chunker

import "core:fmt"
import "core:os"
import "core:path/filepath"
import "core:slice"
import "core:strings"

import "../types"
import "../walker"

SIGNATURE_PREFIXES :: []string {
    "func ",
    "type ",
    "export ",
    "export default ",
    "function ",
    "class ",
    "interface ",
    "struct ",
    "enum ",
    "const ",
    "let ",
    "var ",
    "def ",
    "async ",
    "proc ",
    "impl ",
    "trait ",
    "package ",
    "import ",
}

detect_symbol_hint :: proc(line: string) -> string {
    trimmed := strings.trim_space(line)
    for prefix in SIGNATURE_PREFIXES {
        if strings.has_prefix(trimmed, prefix) {
            end := len(trimmed)
            for i in len(prefix) ..< len(trimmed) {
                c := trimmed[i]
                if c == '{' || c == '(' || c == ':' || c == '=' {
                    end = i
                    break
                }
            }
            return strings.clone(strings.trim_space(trimmed[:end]))
        }
    }
    return ""
}

is_signature_line :: proc(line: string) -> bool {
    trimmed := strings.trim_space(line)
    if len(trimmed) == 0 {
        return false
    }
    for prefix in SIGNATURE_PREFIXES {
        if strings.has_prefix(trimmed, prefix) {
            return true
        }
    }
    return false
}

estimate_chars :: proc(lines: []string) -> int {
    total := 0
    for line in lines {
        total += len(line) + 1
    }
    return total
}

chunk_file :: proc(
    path: string,
    allocator := context.allocator,
) -> (
    chunks: [dynamic]types.Chunk,
    err: os.Error,
) {
    chunks = make([dynamic]types.Chunk, allocator = allocator)

    data, read_err := os.read_entire_file(path, context.temp_allocator)
    if read_err != nil {
        return chunks, read_err
    }

    language, _ := walker.is_code_file(path)
    rel_path := filepath.base(path)

    content := string(data)
    lines := strings.split_lines(content, allocator)
    defer delete(lines)

    if len(lines) == 0 {
        return chunks, nil
    }

    // Small files become a single chunk.
    if estimate_chars(lines) <= types.MAX_CHUNK_CHARS {
        text, _ := strings.join(lines[:], "\n", allocator)
        append(&chunks, types.Chunk{
            path        = strings.clone(rel_path, allocator),
            start_line  = 1,
            end_line    = len(lines),
            language    = strings.clone(language, allocator),
            symbol_hint = detect_symbol_hint(lines[0]),
            text        = text,
        })
        return chunks, nil
    }

    start := 0
    for start < len(lines) {
        end := start
        char_count := 0
        symbol := ""

        for end < len(lines) {
            line := lines[end]
            if symbol == "" && is_signature_line(line) {
                symbol = detect_symbol_hint(line)
            }
            char_count += len(line) + 1
            end += 1

            if char_count >= types.TARGET_CHUNK_CHARS {
                // Prefer breaking at blank line or signature after target size.
                if end < len(lines) {
                    next := strings.trim_space(lines[end])
                    if len(next) == 0 || is_signature_line(lines[end]) {
                        break
                    }
                }
                if char_count >= types.MAX_CHUNK_CHARS {
                    break
                }
            }
        }

        if end == start {
            end = start + 1
        }

        chunk_lines := lines[start:end]
        text, _ := strings.join(chunk_lines[:], "\n", allocator)
        append(&chunks, types.Chunk{
            path        = strings.clone(rel_path, allocator),
            start_line  = start + 1,
            end_line    = end,
            language    = strings.clone(language, allocator),
            symbol_hint = symbol,
            text        = text,
        })

        if end >= len(lines) {
            break
        }

        // Overlap: walk back ~OVERLAP_CHARS from end.
        overlap_start := end
        overlap_chars := 0
        for overlap_start > start && overlap_chars < types.OVERLAP_CHARS {
            overlap_start -= 1
            overlap_chars += len(lines[overlap_start]) + 1
        }
        start = overlap_start
    }

    return chunks, nil
}

chunk_files :: proc(
    files: []string,
    root: string,
    allocator := context.allocator,
) -> (
    chunks: [dynamic]types.Chunk,
    err: os.Error,
) {
    chunks = make([dynamic]types.Chunk, allocator = allocator)

    for file_path in files {
        file_chunks, chunk_err := chunk_file(file_path, allocator)
        if chunk_err != nil {
            fmt.eprintfln("warning: failed to chunk %s: %v", file_path, chunk_err)
            continue
        }

        rel := strings.trim_prefix(file_path, root)
        rel = strings.trim_prefix(rel, "/")

        for &c in file_chunks {
            delete(c.path)
            c.path = strings.clone(rel, allocator)
        }

        append(&chunks, ..file_chunks[:])
        delete(file_chunks)
    }

    return chunks, nil
}
