# WCF — Weight Compression Format

WCF is a GGUF-compatible weight compression format for large language models.
It uses **per-block auto scale + integer offset + variable 1-4 bit codes** to
achieve better quality-per-bit than fixed layouts like Q4_0/Q8_0.

```
value = (code + ioff) / scale
```

Each 32x32 tile of a weight matrix is independently quantized:

- `scale` (f32): auto-computed per block (max range / code range)
- `ioff` (i16): integer offset, removes the DC component before coding
- `ibits` (u8): 1-4 bits per value, auto-selected per block
- `ipacked`: bit-packed codes, MSB-first

This per-block adaptivity is the whole point of WCF: blocks with small ranges
get fewer bits, blocks with large ranges get more, and the integer offset
centers the codes so the full code range is used.

## Repository layout

```
wcf_compress.py     GGUF -> .wcf encoder (reference implementation)
wcf_decompress.py   .wcf -> GGUF decoder (reference implementation)
tests/
  roundtrip_test.py GGUF -> WCF -> GGUF round-trip test
patches/
  llama-wcf-native.diff  llama.cpp patch: native GGML_TYPE_WCF support
vendor/gguf-py/     vendored gguf python library
```

## Usage

### Encode

```bash
python3 wcf_compress.py model.gguf model.wcf          # 4-bit (default)
python3 wcf_compress.py model.gguf model.wcf --bits 2 # 2-bit
python3 wcf_compress.py model.gguf model.wcf --no-compress  # header only (F32 passthrough)
```

### Decode (reference, to GGUF F32)

```bash
python3 wcf_decompress.py model.wcf model_decoded.gguf --source model.gguf
```

### llama.cpp (native)

Apply the patch to llama.cpp at commit `eaf937655`, then build as usual and
run with the `.wcf` file directly:

```bash
cd llama.cpp
git checkout eaf937655
git apply /path/to/patches/llama-wcf-native.diff
cmake -B build && cmake --build build --config Release -j
./build/bin/llama-server -m model.wcf -c 32768 -ngl 999
```

The patch adds:

- `GGML_TYPE_WCF` — native quantization type (per-block scale+offset+ibits)
- `gguf_init_from_wcf_file()` — WCF container parser (WCFH header + WCFI tensor blocks)
- `llama_model_loader` integration — WCF magic detection, no-mmap decode path,
  exact repack of variable-bit file blocks into native 4-bit blocks at load time
- CPU `vec_dot` and CUDA `dequantize`/`mmvq` kernels for GGML_TYPE_WCF

## Format

See `docs/format.md` for the byte-level layout.

## Quality

Per-block scale+offset+ibits gives higher SNR than fixed-scale formats at the
same bit budget. On a Qwen3.8-27B weight sample: WCF 4-bit reaches ~24 dB SNR
on 2D weights vs ~16 dB for Q4_0-style single-scale quantization.