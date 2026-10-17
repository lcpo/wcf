# llama.cpp integration

## Requirements

- llama.cpp at commit `eaf937655` (base)
- CMake >= 3.14
- CUDA toolkit (for GPU support) or CPU-only build

## Applying the patch

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout eaf937655
git apply /path/to/wcf-repo/patches/llama-wcf-native.diff
```

The patch touches 19 files and adds ~600 lines:

| Area | Files | What |
|------|-------|------|
| Type definition | `ggml/include/ggml.h`, `ggml/src/ggml-common.h` | `GGML_TYPE_WCF`, `block_wcf` (32 values, 4-bit nibbles MSB-first, f32 scale + i16 ioff per block) |
| Reference kernels | `ggml/src/ggml-quants.c/h`, `ggml/src/ggml.c` | `quantize_row_wcf_ref`, `dequantize_row_wcf`, type traits |
| CPU kernels | `ggml/src/ggml-cpu/ggml-cpu.c`, `ops.cpp` | fused `vec_dot_wcf_q8_1` (scale/offset folded into the dot product) |
| CUDA kernels | `ggml/src/ggml-cuda/{vecdotq,dequantize,convert,common,mmvq,mmq,ggml-cuda}*` | `dequantize_wcf`, `vec_dot_wcf_q8_1`, MMVQ dispatch (MMQ explicitly disabled for WCF) |
| Container parser | `ggml/src/gguf.cpp`, `ggml/include/gguf.h` | `gguf_init_from_wcf_file()` — reads WCFH + WCFI blocks, creates a `gguf_context` with `GGML_TYPE_WCF` tensors |
| Loader | `src/llama-model-loader.cpp/h` | WCF magic detection, mmap disabled for WCF, exact repack of variable-bit file blocks into native 4-bit blocks at load time |

## Building

```bash
cmake -B build -DGGML_CUDA=ON   # or omit for CPU-only
cmake --build build --config Release -j
```

## Running

```bash
./build/bin/llama-server -m model.wcf -c 32768 -ngl 999
```

WCF files are detected by the `WCFH`/`WCFI` magic; no flags needed.

## Design notes

- **Native type, no re-quantization.** The file stores variable-bit blocks
  (1-4 bits, per-tile). At load time these are repacked *exactly* (same codes,
  same scale/ioff) into the native 4-bit `block_wcf` layout. All kernels read
  the native layout directly. This preserves the per-block offset advantage
  (17.5 dB vs 16.1 dB for a Q4_0 re-quant on the same data).
- **MMQ disabled for WCF.** `ggml_cuda_should_use_mmq()` returns false for
  WCF; the type uses MMVQ (small batch) or cuBLAS dequantize+mul_mat paths.
- **mmap disabled for WCF.** Tensor data is decoded/repacked at load, so the
  loader reads through a host buffer instead of mmap.
- **Small/1D/quantization-sensitive tensors** (norms, SSM states, biases) are
  stored as raw F32 in the file (`n_blocks == 0`) and loaded as
  `GGML_TYPE_F32`.

## Validation

Tested on:

- AMD gfx1150 (ROCm/HIP) — Qwen3.8-27B 27B dense hybrid
- NVIDIA P102-100 (CUDA) — same model

Both platforms load the same `.wcf` file and produce identical logits
(determinism verified with `--temp 0 --seed 42`).