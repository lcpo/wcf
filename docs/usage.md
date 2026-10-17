# Usage

## Python (reference tools)

### Encode GGUF -> WCF

```bash
python3 wcf_compress.py model.gguf model.wcf
```

Options:

- `--bits N` — bits per value (1-4, default 4). Lower bits = smaller file,
  lower SNR.
- `--no-compress` — write the header only; tensors are stored as raw F32
  (no compression). Useful for testing the container itself.

The encoder:

- keeps 1D tensors, small tensors (< 32x32) and quantization-sensitive
  tensors (norms, SSM states, biases) as raw F32
- splits 2D/3D weights into 32x32 tiles, each quantized independently
  (auto scale + integer offset + N-bit codes)
- pads edge tiles to 32x32 with the tile minimum

### Decode WCF -> GGUF (F32)

```bash
python3 wcf_decompress.py model.wcf model_f32.gguf --source model.gguf
```

`--source` copies metadata verbatim and carries over any tensor missing from
the `.wcf`. Without `--source`, metadata comes from the WCFH header.

### Round-trip test

```bash
python3 tests/roundtrip_test.py
```

Creates synthetic tensors (2D, 1D, 3D, small, constant, F16), encodes them,
decodes them, and compares. Reports per-tensor SNR and exactness.

## llama.cpp

See `docs/llama-cpp.md`. Short version:

```bash
cd llama.cpp && git checkout eaf937655
git apply /path/to/wcf-repo/patches/llama-wcf-native.diff
cmake -B build -DGGML_CUDA=ON && cmake --build build -j
./build/bin/llama-server -m model.wcf -c 32768 -ngl 999
```

## Header metadata

The encoder copies all GGUF metadata into the WCFH header and adds:

| Key | Type | Meaning |
|-----|------|---------|
| `wcf.encoding` | string | always `wcf_native` |
| `wcf_int.version` | int32 | encoder version |
| `wcf_int.bits` | int32 | bits per value used for compressed tiles |

## File size expectations

For a 27B dense model (BF16 source):

- BF16 GGUF: ~54 GB
- WCF 4-bit: ~15-16 GB (compressed weights) + F32 small tensors
- WCF 2-bit: ~8-9 GB

Actual ratios depend on the weight distribution; the encoder prints
`bits/weight` at the end.