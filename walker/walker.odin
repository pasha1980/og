#+feature dynamic-literals
package walker

import "core:fmt"
import "core:os"
import "core:path/filepath"
import "core:slice"
import "core:strings"

import "../types"

@(private)
Pattern :: struct {
    pattern:  string,
    negated:  bool,
    dir_only: bool,
    anchored: bool,
}

@(private)
Matcher :: struct {
    patterns: [dynamic]Pattern,
}

matcher_init :: proc(m: ^Matcher) {
    m.patterns = make([dynamic]Pattern)
}

matcher_destroy :: proc(m: ^Matcher) {
    for p in m.patterns {
        delete(p.pattern)
    }
    delete(m.patterns)
}

@(private)
load_ignore_file :: proc(m: ^Matcher, path: string) -> bool {
    data, err := os.read_entire_file(path, context.temp_allocator)
    if err != nil {
        return false
    }

    for line in strings.split_lines(string(data)) {
        trimmed := strings.trim_space(line)
        if len(trimmed) == 0 || strings.has_prefix(trimmed, "#") {
            continue
        }
        append_pattern(m, trimmed)
    }
    return true
}

@(private)
append_pattern :: proc(m: ^Matcher, raw: string) {
    p := Pattern{}
    pattern := raw
    if strings.has_prefix(pattern, "!") {
        p.negated = true
        pattern = pattern[1:]
    }
    if strings.has_prefix(pattern, "/") {
        p.anchored = true
        pattern = pattern[1:]
    }
    if strings.has_suffix(pattern, "/") {
        p.dir_only = true
        pattern = pattern[:len(pattern) - 1]
    }
    p.pattern = strings.clone(pattern)
    append(&m.patterns, p)
}

matcher_should_ignore :: proc(m: ^Matcher, rel_path: string, is_dir: bool) -> bool {
    normalized, _ := strings.replace_all(rel_path, "\\", "/", context.temp_allocator)
    if normalized == ".og" || strings.has_prefix(normalized, ".og/") {
        return true
    }

    matched := false
    for p in m.patterns {
        if p.dir_only && !is_dir {
            continue
        }
        if pattern_match(p.pattern, normalized, p.anchored) {
            matched = !p.negated
        }
    }
    return matched
}

@(private)
pattern_match :: proc(pattern, path: string, anchored: bool) -> bool {
    if pattern == "" {
        return false
    }

    if !strings.contains(pattern, "/") {
        base := filepath.base(path)
        if wildcard_match(pattern, base) {
            return true
        }
        parts := strings.split(path, "/")
        for part in parts {
            if wildcard_match(pattern, part) {
                return true
            }
        }
        return false
    }

    if anchored {
        return wildcard_match(pattern, path)
    }

    if strings.has_suffix(pattern, "/**") {
        prefix := pattern[:len(pattern) - 3]
        prefix_slash, _ := strings.concatenate({prefix, "/"}, context.temp_allocator)
        if path == prefix || strings.has_prefix(path, prefix_slash) {
            return true
        }
    }

    if strings.contains(pattern, "**") {
        pat, _ := strings.replace_all(pattern, "**", "*", context.temp_allocator)
        return wildcard_match(pat, path)
    }

    return strings.has_suffix(path, pattern) || wildcard_match(pattern, path)
}

@(private)
wildcard_match :: proc(pattern, text: string) -> bool {
    pi, ti := 0, 0
    star_pi, star_ti := -1, -1

    for ti <= len(text) {
        if pi < len(pattern) && ti < len(text) && (pattern[pi] == text[ti] || pattern[pi] == '?') {
            pi += 1
            ti += 1
            continue
        }
        if pi < len(pattern) && pattern[pi] == '*' {
            star_pi = pi
            star_ti = ti
            pi += 1
            continue
        }
        if star_pi != -1 {
            pi = star_pi + 1
            star_ti += 1
            ti = star_ti
            continue
        }
        if ti == len(text) && pi == len(pattern) {
            return true
        }
        return false
    }
    return pi == len(pattern)
}

DEFAULT_IGNORE_PATTERNS :: []string {
    ".git/",
    "node_modules/",
    "vendor/",
    "dist/",
    "build/",
    "target/",
    ".og/",
    "*.min.js",
    "*.map",
    "*.lock",
    "*.sum",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.ico",
    "*.svg",
    "*.woff",
    "*.woff2",
    "*.ttf",
    "*.eot",
    "*.pdf",
    "*.zip",
    "*.tar",
    "*.gz",
    "*.bin",
}

extension_language :: proc(ext: string) -> (language: string, ok: bool) {
    switch ext {
    case ".go":
        return "go", true
    case ".ts", ".tsx":
        return "typescript", true
    case ".js", ".jsx":
        return "javascript", true
    case ".vue":
        return "vue", true
    case ".odin":
        return "odin", true
    case ".py":
        return "python", true
    case ".rs":
        return "rust", true
    case ".c", ".h":
        return "c", true
    case ".cpp", ".hpp":
        return "cpp", true
    case ".md":
        return "markdown", true
    case ".json":
        return "json", true
    case ".yaml", ".yml":
        return "yaml", true
    case ".sql":
        return "sql", true
    case ".sh":
        return "shell", true
    case ".toml":
        return "toml", true
    }
    return "", false
}

is_code_file :: proc(path: string) -> (language: string, ok: bool) {
    ext := strings.to_lower(filepath.ext(path), context.temp_allocator)
    return extension_language(ext)
}

collect_files :: proc(
    root: string,
    allocator := context.allocator,
) -> (
    files: [dynamic]string,
    err: os.Error,
) {
    files = make([dynamic]string, allocator = allocator)

    matcher: Matcher
    defer matcher_destroy(&matcher)
    matcher_init(&matcher)

    for pattern in DEFAULT_IGNORE_PATTERNS {
        append_pattern(&matcher, pattern)
    }

    gitignore, _ := filepath.join({root, ".gitignore"}, allocator)
    if os.exists(gitignore) {
        load_ignore_file(&matcher, gitignore)
    }

    ogignore, _ := filepath.join({root, ".ogignore"}, allocator)
    if os.exists(ogignore) {
        load_ignore_file(&matcher, ogignore)
    }

    w := os.walker_create(root)
    defer os.walker_destroy(&w)

    for info in os.walker_walk(&w) {
        if path, walk_err := os.walker_error(&w); walk_err != nil {
            fmt.eprintfln("warning: walker error at %s: %v", path, walk_err)
            continue
        }

        if info.type == .Directory {
            rel := strings.trim_prefix(info.fullpath, root)
            rel = strings.trim_prefix(rel, "/")
            if matcher_should_ignore(&matcher, rel, true) {
                os.walker_skip_dir(&w)
            }
            continue
        }

        if info.type != .Regular {
            continue
        }

        rel := strings.trim_prefix(info.fullpath, root)
        rel = strings.trim_prefix(rel, "/")
        matcher_should_ignore(&matcher, rel, false) or_continue

        is_code_file(info.fullpath) or_continue

        full := strings.clone(info.fullpath, allocator)
        append(&files, full)
    }

    return files, nil
}
