# WCF Format Specification

## Container

A `.wcf` file is a sequence of two block types:

1. **WCFH** — header block (metadata)
2. **WCFI** — tensor block (weights)

All integers are little-endian.

## WCFH header

| Field | Type | Description |
|-------|------|-------------|
| magic | 4 bytes | `"WCFH"` |
| version | u8 | format version (currently 1) |
| n_fields | u16 | number of metadata fields |
| fields | n_fields × field | metadata fields |

Each metadata field:

| Field | Type | Description |
|-------|------|-------------|
| key_len | u16 | key byte length |
| key | key_len bytes | key string (UTF-8) |
| vtype | u8 | value type |
| value | depends | value |

Value types (`vtype`):

| vtype | Type | Encoding |
|-------|------|----------|
| 0 | string | u32 len + bytes |
| 1 | int64 | i64 LE |
| 2 | double | f64 LE |
| 3 | bool | u8 (0/1) |
| 4 | string array | u32 n + (u32 len + bytes) × n |
| 5 | int32 array | u32 n + i32 × n |
| 6 | uint32 | u32 LE |
| 7 | float32 | f32 LE |
| 8 | int32 | i32 LE |

## WCFI tensor block

| Field | Type | Description |
|-------|------|-------------|
| magic | 4 bytes | `"WCFI"` |
| version | u8 | format version (currently 1) |
| n_tensors | u32 | number of tensors in this block |
| tensors | n_tensors × tensor | tensor records |

Each tensor record:

| Field | Type | Description |
|-------|------|-------------|
| name_len | u16 | name byte length |
| name | name_len bytes | tensor name (GGUF convention) |
| rows | u32 | ne[1] (rows) |
| cols | u32 | ne[0] (cols) |
| n_blocks | u32 | number of 32x32 tiles (0 = raw F32 payload) |
| ndim | u8 | number of dimensions |
| ne2 | u32 | product of dims [2..] (1 for 1D/2D) |

## Tensor data

### n_blocks == 0 (raw F32)

Used for 1D tensors, small tensors and quantization-sensitive tensors
(norms, SSM states, etc.). Followed by `rows * cols * ne2` f32 values
(native little-endian). Decoded as-is.

### n_blocks > 0 (compressed)

Followed by `n_blocks` tiles. Tiles are laid out in row-major order over the
(rows × cols) matrix: tile (r, c) covers rows `[32r, 32r+32)` and columns
`[32c, 32c+32)`. Partial edge tiles are padded to 32x32 (padding values are
zeros and ignored).

Each tile:

| Field | Type | Description |
|-------|------|-------------|
| scale | f32 | per-tile scale |
| ioff | i16 | per-tile integer offset |
| ibits | u8 | bits per value, 1-4 |
| n_packed | u32 | packed byte length |
| ipacked | n_packed bytes | bit-packed codes, MSB-first |

Value decoding: `value = (code + ioff) / scale`, where `code` is the
unsigned integer code (0 .. 2^ibits - 1).

Codes are packed MSB-first: the first value occupies the most significant
bits of the first byte.

## Quantization (encoder)

For each 32x32 tile:

1. Compute `fmin`, `fmax` of the tile (NaN/Inf replaced by 0).
2. If the tile is constant (`fmax == fmin`): store `ioff = round(fmin)`,
   `scale = 1.0`, `ibits = 0`, no packed data. Decodes to exactly `fmin`.
3. Otherwise: `scale = (fmax - fmin) / (2^ibits - 1)` with the code range
   centered so that `code = round((v * scale) - ioff)` stays in
   `[0, 2^ibits - 1]`.

The `ioff` integer offset is what distinguishes WCF from single-scale
formats: it removes the DC component per tile so the full code range is
used for the AC part.

## Native llama.cpp layout

At load time, llama.cpp repacks variable-bit file blocks into native
`GGML_TYPE_WCF` blocks: 32 values per block, fixed 4-bit nibbles (MSB-first),
one `scale` (f32), one `ioff` (i16) per block. The repack is exact: code
values and (scale, ioff) are preserved; only the bit-packing changes.

```
struct block_wcf {
    float     scale;   // f32 in native layout
    int16_t   ioff;
    uint8_t   ibits;   // always 4 in native layout
    uint8_t   qs[16];  // 32 x 4-bit codes, MSB-first
};
```

CPU `vec_dot`, CUDA `dequantize` and CUDA `mmvq` kernels all read this
layout directly; no re-quantization to Q4_0/Q8_0 happens.