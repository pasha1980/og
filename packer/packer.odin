package packer

import "core:fmt"
import "core:strings"

import "../index"
import "../types"

Format :: enum {
    Plain,
    Cursor,
    Claude,
}

format_from_string :: proc(s: string) -> Format {
    switch strings.to_lower(s) {
    case "cursor":
        return .Cursor
    case "claude":
        return .Claude
    case "plain":
        return .Plain
    }
    return .Plain
}

pack_context :: proc(
    hits: []types.SearchHit,
    budget: int,
    format: Format,
    allocator := context.allocator,
) -> string {
    builder: strings.Builder
    strings.builder_init(&builder, allocator)

    total_tokens := 0
    used := 0

    for hit in hits {
        tokens := index.estimate_tokens(hit.chunk.text)
        if total_tokens + tokens > budget {
            break
        }
        total_tokens += tokens
        used += 1
    }

    header := fmt.tprintf(
        "## Retrieved context (og %s, %d chunks, ~%d tokens)\n\n",
        types.OG_VERSION,
        used,
        total_tokens,
    )
    strings.write_string(&builder, header)

    for i in 0 ..< used {
        hit := hits[i]
        section := fmt.tprintf(
            "### %s:%d-%d",
            hit.chunk.path,
            hit.chunk.start_line,
            hit.chunk.end_line,
        )
        if hit.chunk.symbol_hint != "" {
            section = fmt.tprintf("%s (%s)", section, hit.chunk.symbol_hint)
        }
        strings.write_string(&builder, section)
        strings.write_string(&builder, "\n")

        lang := hit.chunk.language
        if lang == "" {
            lang = "text"
        }

        switch format {
        case .Plain:
            strings.write_string(&builder, hit.chunk.text)
            strings.write_string(&builder, "\n\n")
        case .Cursor, .Claude:
            strings.write_string(&builder, fmt.tprintf("```%s\n", lang))
            strings.write_string(&builder, hit.chunk.text)
            strings.write_string(&builder, "\n```\n\n")
        }
    }

    return strings.to_string(builder)
}
