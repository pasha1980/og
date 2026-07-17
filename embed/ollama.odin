package embed

import "core:encoding/json"
import "core:fmt"
import "core:math"
import "core:net"
import "core:slice"
import "core:strings"
import "core:strconv"

DEFAULT_OLLAMA_HOST :: "127.0.0.1:11434"
DEFAULT_MODEL :: "nomic-embed-text"

Ollama_Config :: struct {
    host:  string,
    model: string,
}

ollama_default_config :: proc() -> Ollama_Config {
    return Ollama_Config{
        host  = DEFAULT_OLLAMA_HOST,
        model = DEFAULT_MODEL,
    }
}

@(private)
EmbeddingRequest :: struct {
    model: string `json:"model"`,
    prompt: string `json:"prompt"`,
}

@(private)
EmbeddingResponse :: struct {
    embedding: []f32 `json:"embedding"`,
}

http_post :: proc(host: string, path: string, body: []u8) -> ([]u8, bool) {
    request := strings.builder_make(context.temp_allocator)
    defer strings.builder_destroy(&request)

    fmt.sbprintf(&request, "POST %s HTTP/1.1\r\n", path)
    fmt.sbprintf(&request, "Host: %s\r\n", host)
    fmt.sbprintf(&request, "Content-Type: application/json\r\n")
    fmt.sbprintf(&request, "Content-Length: %d\r\n", len(body))
    fmt.sbprintf(&request, "Connection: close\r\n\r\n")
    strings.write_bytes(&request, body)

    socket, err := net.dial_tcp_from_hostname_and_port_string(host)
    if err != nil {
        return nil, false
    }
    defer net.close(socket)

    req_bytes := transmute([]u8)strings.to_string(request)
    if _, send_err := net.send(socket, req_bytes); send_err != nil {
        return nil, false
    }

    response := make([dynamic]u8)
    buf: [4096]u8
    for {
        n, recv_err := net.recv(socket, buf[:])
        if n > 0 {
            append(&response, ..buf[:n])
        }
        if recv_err != nil || n == 0 {
            break
        }
    }

    raw := string(response[:])
    header_end := strings.index(raw, "\r\n\r\n")
    if header_end < 0 {
        return nil, false
    }

    body_start := header_end + 4
    headers := raw[:header_end]
    if !strings.contains(headers, "200") {
        return nil, false
    }

    // Handle chunked transfer encoding minimally.
    body_str := raw[body_start:]
    if strings.contains(strings.to_lower(headers), "transfer-encoding: chunked") {
        decoded := decode_chunked(body_str, context.temp_allocator)
        return transmute([]u8)decoded, true
    }

    return transmute([]u8)body_str, true
}

@(private)
decode_chunked :: proc(body: string, allocator := context.allocator) -> string {
    result: strings.Builder
    strings.builder_init(&result, allocator)
    rest := body
    for len(rest) > 0 {
        line_end := strings.index(rest, "\r\n")
        if line_end < 0 {
            break
        }
        size_line := rest[:line_end]
        size, _ := strconv.parse_u64_of_base(size_line, 16)
        rest = rest[line_end + 2:]
        if size == 0 {
            break
        }
        if len(rest) < int(size) + 2 {
            break
        }
        strings.write_string(&result, rest[:size])
        rest = rest[size + 2:]
    }
    return strings.to_string(result)
}

ollama_embed :: proc(
    text: string,
    config: Ollama_Config,
    allocator := context.allocator,
) -> (
    embedding: []f32,
    ok: bool,
) {
    req := EmbeddingRequest{
        model  = config.model,
        prompt = text,
    }
    body, err := json.marshal(req)
    if err != nil {
        return nil, false
    }
    defer delete(body)

    resp_body, ok_http := http_post(config.host, "/api/embeddings", body)
    if !ok_http {
        return nil, false
    }

    response: EmbeddingResponse
    if json_err := json.unmarshal(resp_body, &response); json_err != nil {
        return nil, false
    }
    if len(response.embedding) == 0 {
        return nil, false
    }

    embedding = make([]f32, len(response.embedding), allocator)
    copy(embedding, response.embedding)
    return embedding, true
}

ollama_embed_batch :: proc(
    texts: []string,
    config: Ollama_Config,
    allocator := context.allocator,
) -> (
    vectors: []f32,
    dim: int,
    ok: bool,
) {
    if len(texts) == 0 {
        return nil, 0, true
    }

    first, ok_first := ollama_embed(texts[0], config, allocator)
    if !ok_first {
        return nil, 0, false
    }
    dim = len(first)

    vectors = make([]f32, len(texts) * dim, allocator)
    copy(vectors[:dim], first)

    for i in 1 ..< len(texts) {
        vec, ok_vec := ollama_embed(texts[i], config, allocator)
        if !ok_vec || len(vec) != dim {
            delete(vectors)
            return nil, 0, false
        }
        offset := i * dim
        copy(vectors[offset:offset + dim], vec)
        delete(vec)
    }
    delete(first)

    return vectors, dim, true
}

ollama_available :: proc(config: Ollama_Config) -> bool {
    request := strings.builder_make(context.temp_allocator)
    defer strings.builder_destroy(&request)
    fmt.sbprintf(&request, "GET /api/tags HTTP/1.1\r\n")
    fmt.sbprintf(&request, "Host: %s\r\n", config.host)
    fmt.sbprintf(&request, "Connection: close\r\n\r\n")

    socket, err := net.dial_tcp_from_hostname_and_port_string(config.host)
    if err != nil {
        return false
    }
    defer net.close(socket)

    req_bytes := transmute([]u8)strings.to_string(request)
    if _, send_err := net.send(socket, req_bytes); send_err != nil {
        return false
    }

    buf: [1024]u8
    n, _ := net.recv(socket, buf[:])
    if n == 0 {
        return false
    }
    return strings.contains(string(buf[:n]), "200")
}

normalize_vector :: proc(vec: []f32) {
    if len(vec) == 0 {
        return
    }
    sum: f32 = 0
    for v in vec {
        sum += v * v
    }
    if sum == 0 {
        return
    }
    inv := 1.0 / math.sqrt_f32(sum)
    for &v in vec {
        v *= inv
    }
}

normalize_vectors :: proc(vectors: []f32, dim: int) {
    if dim <= 0 {
        return
    }
    count := len(vectors) / dim
    for i in 0 ..< count {
        offset := i * dim
        normalize_vector(vectors[offset:offset + dim])
    }
}
