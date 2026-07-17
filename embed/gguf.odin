package embed

import "core:encoding/endian"
import "core:fmt"
import "core:mem"
import "core:os"
import "core:strings"

GGUF_MAGIC :: 0x46554747 // "GGUF"
GGUF_VERSION :: 3

GGUF_Value_Type :: enum u32 {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
}

GGUF_Value :: struct {
    type:  GGUF_Value_Type,
    u64:   u64,
    i64:   i64,
    f64:   f64,
    str:   string,
    array: []GGUF_Value,
}

GGUF_Tensor_Info :: struct {
    name:       string,
    n_dims:     u32,
    dims:       [4]u64,
    type:       u32,
    offset:     u64,
}

GGUF_File :: struct {
    metadata:      map[string]GGUF_Value,
    tensors:       [dynamic]GGUF_Tensor_Info,
    data_offset:   u64,
    raw_data:      []u8,
    embedding_dim: int,
    vocab_size:    int,
}

gguf_destroy :: proc(g: ^GGUF_File) {
    for k, v in g.metadata {
        delete(k)
        gguf_value_destroy(v)
    }
    delete(g.metadata)
    for t in g.tensors {
        delete(t.name)
    }
    delete(g.tensors)
    delete(g.raw_data)
}

@(private)
gguf_value_destroy :: proc(v: GGUF_Value) {
    if v.type == .STRING {
        delete(v.str)
    }
    if v.type == .ARRAY {
        for item in v.array {
            gguf_value_destroy(item)
        }
        delete(v.array)
    }
}

read_string :: proc(data: []u8, offset: ^int) -> (string, bool) {
    if offset^ + 8 > len(data) {
        return "", false
    }
    n := int(endian.unchecked_get_u64le(data[offset^:]))
    offset^ += 8
    if offset^ + n > len(data) {
        return "", false
    }
    s := strings.clone(string(data[offset^:][:n]))
    offset^ += n
    return s, true
}

read_value :: proc(data: []u8, offset: ^int, allocator := context.allocator) -> (GGUF_Value, bool) {
    if offset^ + 4 > len(data) {
        return {}, false
    }
    vt := GGUF_Value_Type(endian.unchecked_get_u32le(data[offset^:]))
    offset^ += 4

    v: GGUF_Value
    v.type = vt

    switch vt {
    case .UINT8, .UINT16, .UINT32, .UINT64:
        if offset^ + 8 > len(data) {
            return {}, false
        }
        v.u64 = endian.unchecked_get_u64le(data[offset^:])
        offset^ += 8
    case .INT8, .INT16, .INT32, .INT64:
        if offset^ + 8 > len(data) {
            return {}, false
        }
        v.i64 = i64(endian.unchecked_get_u64le(data[offset^:]))
        offset^ += 8
    case .FLOAT32, .FLOAT64:
        if offset^ + 8 > len(data) {
            return {}, false
        }
        bits := endian.unchecked_get_u64le(data[offset^:])
        offset^ += 8
        mem.copy(&v.f64, &bits, 8)
    case .BOOL:
        if offset^ + 1 > len(data) {
            return {}, false
        }
        v.u64 = u64(data[offset^])
        offset^ += 1
    case .STRING:
        s, ok := read_string(data, offset)
        if !ok {
            return {}, false
        }
        v.str = s
    case .ARRAY:
        if offset^ + 4 > len(data) {
            return {}, false
        }
        elem_type := GGUF_Value_Type(endian.unchecked_get_u32le(data[offset^:]))
        offset^ += 4
        if offset^ + 8 > len(data) {
            return {}, false
        }
        count := int(endian.unchecked_get_u64le(data[offset^:]))
        offset^ += 8
        arr := make([]GGUF_Value, count, allocator)
        for i in 0 ..< count {
            // Array elements omit type tag in some versions; GGUF repeats type per element.
            _ = elem_type
            item, ok_item := read_value(data, offset, allocator)
            if !ok_item {
                for j in 0 ..< i {
                    gguf_value_destroy(arr[j])
                }
                delete(arr)
                return {}, false
            }
            arr[i] = item
        }
        v.array = arr
    }

    return v, true
}

gguf_load :: proc(path: string, allocator := context.allocator) -> (GGUF_File, bool) {
    data, err := os.read_entire_file(path, context.temp_allocator)
    if err != nil {
        return {}, false
    }

    g: GGUF_File
    g.raw_data = data
    g.metadata = make(map[string]GGUF_Value)

    offset := 0
    if len(data) < 16 {
        return {}, false
    }

    magic := endian.unchecked_get_u32le(data[:])
    if magic != GGUF_MAGIC {
        return {}, false
    }
    offset = 4

    version := endian.unchecked_get_u32le(data[offset:])
    offset += 4
    if version < 2 || version > 4 {
        return {}, false
    }

    tensor_count := int(endian.unchecked_get_u64le(data[offset:]))
    offset += 8
    metadata_count := int(endian.unchecked_get_u64le(data[offset:]))
    offset += 8

    for _ in 0 ..< metadata_count {
        key, ok_k := read_string(data, &offset)
        if !ok_k {
            gguf_destroy(&g)
            return {}, false
        }
        val, ok_v := read_value(data, &offset)
        if !ok_v {
            delete(key)
            gguf_destroy(&g)
            return {}, false
        }
        g.metadata[key] = val
    }

    g.tensors = make([dynamic]GGUF_Tensor_Info)
    for _ in 0 ..< tensor_count {
        name, ok_n := read_string(data, &offset)
        if !ok_n {
            gguf_destroy(&g)
            return {}, false
        }
        n_dims := endian.unchecked_get_u32le(data[offset:])
        offset += 4
        dims: [4]u64
        for d in 0 ..< int(n_dims) {
            dims[d] = endian.unchecked_get_u64le(data[offset:])
            offset += 8
        }
        tensor_type := endian.unchecked_get_u32le(data[offset:])
        offset += 4
        tensor_offset := endian.unchecked_get_u64le(data[offset:])
        offset += 8

        append(&g.tensors, GGUF_Tensor_Info{
            name   = name,
            n_dims = n_dims,
            dims   = dims,
            type   = tensor_type,
            offset = tensor_offset,
        })
    }

    g.data_offset = u64(offset)

    // Align to 32 bytes.
    align_pad := (32 - (offset % 32)) % 32
    offset += align_pad

    // Infer architecture metadata.
    if val, found := g.metadata["embedding_length"]; found && val.type == .UINT32 {
        g.embedding_dim = int(val.u64)
    } else if val, found := g.metadata["n_embd"]; found {
        g.embedding_dim = int(val.u64)
    }

    if val, found := g.metadata["n_vocab"]; found {
        g.vocab_size = int(val.u64)
    }

    // Find token embedding tensor.
    for tensor in g.tensors {
        if strings.contains(tensor.name, "token_embd") ||
           strings.contains(tensor.name, "embed_tokens") ||
           strings.contains(tensor.name, "wte") {
            if g.embedding_dim == 0 && tensor.n_dims >= 2 {
                g.embedding_dim = int(tensor.dims[0])
                g.vocab_size = int(tensor.dims[1])
            }
            break
        }
    }

    if g.embedding_dim == 0 {
        for tensor in g.tensors {
            if tensor.n_dims >= 2 {
                g.embedding_dim = int(tensor.dims[0])
                g.vocab_size = int(tensor.dims[1])
                break
            }
        }
    }

    return g, true
}

// GGML type IDs for quantization.
GGML_TYPE_F32 :: 0
GGML_TYPE_Q8_0 :: 8

@(private)
tensor_byte_size :: proc(info: GGUF_Tensor_Info) -> int {
    elements: u64 = 1
    for d in 0 ..< int(info.n_dims) {
        elements *= info.dims[d]
    }

    switch info.type {
    case GGML_TYPE_F32:
        return int(elements * 4)
    case GGML_TYPE_Q8_0:
        // 32 values per block, block = 32 bytes (1 scale f16 + 32 int8).
        blocks := (elements + 31) / 32
        return int(blocks * 34)
    case:
        return int(elements * 4)
    }
}

dequantize_q8_0 :: proc(data: []u8, count: int, allocator := context.allocator) -> []f32 {
    result := make([]f32, count, allocator)
    block_size :: 32
    offset := 0

    for i in 0 ..< count {
        block_idx := i / block_size
        in_block := i % block_size
        block_offset := block_idx * 34
        if block_offset + 2 > len(data) {
            break
        }
        scale_bits := endian.unchecked_get_u16le(data[block_offset:])
        scale: f16
        mem.copy(&scale, &scale_bits, 2)
        scale_f32 := f32(scale)
        val_offset := block_offset + 2 + in_block
        if val_offset >= len(data) {
            break
        }
        result[i] = scale_f32 * f32(i8(data[val_offset]))
        offset += 1
    }
    return result
}

load_tensor_f32 :: proc(g: ^GGUF_File, name_part: string, allocator := context.allocator) -> ([]f32, bool) {
    for tensor in g.tensors {
        if !strings.contains(tensor.name, name_part) {
            continue
        }

        start := int(g.data_offset + tensor.offset)
        size := tensor_byte_size(tensor)
        if start + size > len(g.raw_data) {
            return nil, false
        }
        raw := g.raw_data[start:start + size]

        elements: int = 1
        for d in 0 ..< int(tensor.n_dims) {
            elements *= int(tensor.dims[d])
        }

        switch tensor.type {
        case GGML_TYPE_F32:
            result := make([]f32, elements, allocator)
            for i in 0 ..< elements {
                bits := endian.unchecked_get_u32le(raw[i * 4:])
                mem.copy(&result[i], &bits, 4)
            }
            return result, true
        case GGML_TYPE_Q8_0:
            return dequantize_q8_0(raw, elements, allocator), true
        }
    }
    return nil, false
}

@(private)
hash_token :: proc(token: string, vocab: int) -> int {
    size := vocab
    if size <= 0 {
        size = 30522
    }
    h: u32 = 2166136261
    for c in token {
        h ~= u32(c)
        h *= 16777619
    }
    return int(h % u32(size))
}

gguf_tokenize :: proc(text: string, allocator := context.temp_allocator) -> []string {
    tokens := make([dynamic]string, allocator = allocator)
    current: strings.Builder
    strings.builder_init(&current, allocator)

    flush :: proc(tokens: ^[dynamic]string, current: ^strings.Builder) {
        if strings.builder_len(current^) > 0 {
            append(tokens, strings.clone(strings.to_lower(string(current.buf[:])), context.temp_allocator))
            strings.builder_reset(current)
        }
    }

    for c in text {
        if (c >= 'a' && c <= 'z') ||
           (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') {
            strings.write_rune(&current, c)
        } else {
            flush(&tokens, &current)
        }
    }
    flush(&tokens, &current)
    return tokens[:]
}

gguf_embed :: proc(
    text: string,
    model_path: string,
    allocator := context.allocator,
) -> (
    embedding: []f32,
    ok: bool,
) {
    g, loaded := gguf_load(model_path)
    if !loaded {
        return nil, false
    }
    defer gguf_destroy(&g)

    embd, found := load_tensor_f32(&g, "token_embd")
    if !found {
        embd, found = load_tensor_f32(&g, "embed_tokens")
    }
    if !found {
        embd, found = load_tensor_f32(&g, "wte")
    }
    if !found && len(g.tensors) > 0 {
        embd, found = load_tensor_f32(&g, g.tensors[0].name)
    }
    if !found || len(embd) == 0 {
        return nil, false
    }

    dim := g.embedding_dim
    vocab := g.vocab_size
    if dim <= 0 {
        dim = 384
    }
    if vocab <= 0 {
        vocab = len(embd) / dim
    }

    tokens := gguf_tokenize(text)
    defer delete(tokens)

    if len(tokens) == 0 {
        return make([]f32, dim, allocator), true
    }

    result := make([]f32, dim, allocator)
    for token in tokens {
        tid := hash_token(token, vocab)
        offset := tid * dim
        if offset + dim > len(embd) {
            tid = tid % max(1, len(embd) / dim)
            offset = tid * dim
        }
        for i in 0 ..< dim {
            result[i] += embd[offset + i]
        }
    }

    inv := 1.0 / f32(len(tokens))
    for &v in result {
        v *= inv
    }
    normalize_vector(result)
    return result, true
}

gguf_embed_batch :: proc(
    texts: []string,
    model_path: string,
    allocator := context.allocator,
) -> (
    vectors: []f32,
    dim: int,
    ok: bool,
) {
    if len(texts) == 0 {
        return nil, 0, true
    }

    first, ok_first := gguf_embed(texts[0], model_path, allocator)
    if !ok_first {
        return nil, 0, false
    }
    dim = len(first)

    vectors = make([]f32, len(texts) * dim, allocator)
    copy(vectors[:dim], first)
    delete(first)

    for i in 1 ..< len(texts) {
        vec, ok_vec := gguf_embed(texts[i], model_path, allocator)
        if !ok_vec || len(vec) != dim {
            delete(vectors)
            return nil, 0, false
        }
        copy(vectors[i * dim:(i + 1) * dim], vec)
        delete(vec)
    }

    return vectors, dim, true
}
