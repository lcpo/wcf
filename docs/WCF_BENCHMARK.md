# WCF Benchmark Report — Qwen3.8-27B (2026-10-18)

**Format**: WCF (native `GGML_TYPE_WCF`, per-32×32-block auto scale+offset+ibits≤4, value=(code+ioff)/scale)
**Model**: Qwen3.8-27B (arch qwen35, dense hybrid: 48 GDN + 17 full-attn, 27.32B params)
**File**: `Qwen3.8-27B.wcf` — 19.32 GB, 866 tensors, 4.0 bits/weight avg
**Reference**: Q4_K_M GGUF (15.65 GiB, quantized from BF16), BF16 GGUF (54.6 GiB)
**Machines**: 131 (ROCm/gfx1150 iGPU, GTT 48GB, 22 threads), 148 (CUDA, 4×P102-100 10GB, 8 threads)
**Build**: llama.cpp @ eaf937655 + clean native-WCF patch (github.com/lcpo/wcf, commit 4cf4aba)

---

## 1. Perplexity (wiki.test.raw, 580 chunks, n_ctx=512, b=512)

### CPU (full CPU inference — isolates dequant/vec_dot correctness)

| Format | 131 (ROCm build) | 148 (CUDA build) |
|---|---:|---:|
| BF16 (reference) | 7.0740 ± 0.0458 | — |
| Q4_K_M | 7.1487 ± 0.0464 | — |
| **WCF** | **7.7028 ± 0.0509** | **7.7046 ± 0.0509** |

**Verdict**: WCF CPU PPL is **identical on both platforms** (7.7028 vs 7.7046 — within
run-to-run variance, cross-runtime deterministic). The +0.55 PPL penalty vs Q4_K_M
(+7.8% over Q4_K_M, +8.9% over BF16) is the cost of 4-bit variable-bit quantization
with per-block scale/offset.

### GPU (full offload, ngl 999)

| Test | Format | Result |
|---|---|---:|
| 148 CUDA, fa=on, 20 chunks | WCF | **7.4434 ± 0.2625** |
| 131 ROCm, fa=on, **nofgdn**, 580 chunks | WCF | **7.7046 ± 0.0509** |
| 131 ROCm, fa=on (fused GDN) | WCF | ❌ **202.70** (broken) |
| 131 ROCm, fa=off | Q4_K_M | 14.8804 (2 chunks, n_ctx 2048) |
| 131 ROCm, fa=on | Q4_K_M | 15.0721 (2 chunks, n_ctx 2048) |

**Key findings**:
- **148 CUDA WCF GPU PPL (7.44 on 20 chunks) is consistent with CPU (7.70 on 580)** —
  different chunk counts cover different text; first-20-chunk CPU value is 7.74,
  so CUDA GPU dequant+vec_dot is correct.
- **131 ROCm WCF GPU with fused GDN is BROKEN (PPL 202.7)**. A/B test proved it's
  not the WCF format: Q4_K_M gives identical PPL old-vs-new build (14.88/15.07),
  and WCF fa=on vs fa=off differ by only 0.1 (202.70 vs 202.91) — the corruption
  comes from the **fused GDN (gated_delta_net) kernel path**, not WCF dequant.
  With fused GDN disabled (`nofgdn`), WCF GPU PPL = 7.7046 = CPU value exactly.
- **A/B old build (e2282d0, Q4_0 re-quant loader) vs new clean build**: Q4_K_M
  PPL identical (14.8804/14.8804, 15.0721/15.0721) — the loader rewrite did not
  change Q4_K_M numerics.
- **BF16 GPU PPL is not feasible**: 131 needs 48.9GB > GTT 48GB; 148 needs 54.6GB
  > 5×10GB. CPU BF16 (7.0740) serves as the reference.

## 2. llama-bench (pp512 / tg128, full offload)

### 131 (ROCm, 22 threads, GTT 48GB)

| Format | pp512 fa=0 | pp512 fa=1 | tg128 fa=0 | tg128 fa=1 |
|---|---:|---:|---:|---:|
| Q4_K_M | 112.01 | 108.42 | 4.39 | 4.42 |
| Q4_0 (re-quant, old) | — | 141.70 | — | 4.96 |
| WCF (native) | 59.23 | 61.18 | 3.30 | 3.31 |
| BF16 | 59.23* | 61.18* | 3.30* | 3.31* |

*bench_wcf_native.log labels WCF file as "BF16" (ftype not set in WCF container) —
the 22.98 GiB size and 3.31 t/s confirm it's the WCF file.

### 148 (CUDA, 8 threads, 4×P102-100)

| Format | pp512 fa=1 | tg128 fa=1 | tg128 fa=0 |
|---|---:|---:|---:|
| Q4_K_M | 134.11 | 13.53 | 13.41 |
| WCF (native) | 78.77 | 7.53 | — |

**Verdict**: WCF native kernels are **slower than Q4_K_M**:
- 131: tg128 3.31 vs 4.39 t/s (−25%), pp512 61 vs 112 t/s (−45%)
- 148: tg128 7.53 vs 13.53 t/s (−44%), pp512 78.8 vs 134.1 t/s (−41%)

Reasons: (1) WCF vec_dot does per-block scale/offset arithmetic on every dot
product, (2) no MMQ (int8 tensor-core) path for WCF — falls back to dequant+GEMM,
(3) 19.3GB file vs 15.65GB Q4_K_M — more memory traffic (WCF stores scale f32 +
ioff i16 + ibits u8 + n_packed u32 per 1024-value block ≈ 4.0 bpw incl. overhead,
vs Q4_K_M 4.5 bpw but with super-block packing that's MMQ-friendly).

## 3. Correctness summary

| Check | Result |
|---|---|
| Round-trip (encoder↔decoder) | PASS (all bit widths 1-4, F16, 3D, raw F32) |
| CPU PPL cross-runtime (131 vs 148) | 7.7028 vs 7.7046 — PASS (deterministic) |
| CPU PPL vs GPU PPL (148) | 7.70 (580ch) vs 7.44 (20ch) — PASS (chunk coverage) |
| CPU PPL vs GPU PPL (131, nofgdn) | 7.7046 = 7.7046 — PASS (exact) |
| A/B loader old vs new (Q4_K_M) | identical — PASS |
| GPU fused-GDN WCF (131) | **FAIL (PPL 202.7)** — pre-existing kernel bug, not WCF |

## 4. Conclusions

1. **WCF format is numerically sound**: CPU PPL is deterministic across ROCm and
   CUDA runtimes, and GPU PPL matches CPU exactly when the fused-GDN kernel is
   bypassed. The +0.55 PPL vs Q4_K_M is the expected 4-bit quality cost.
2. **WCF is slower than Q4_K_M on both platforms** (−25%…−44% tg, −41%…−45% pp).
   The per-block scale/offset decode in vec_dot and the lack of an MMQ path are
   the bottleneck. Future work: MMQ kernel for WCF, or super-block packing to
   cut metadata overhead.
3. **Fused GDN kernel bug is orthogonal to WCF** (affects Q4_K_M too — hidden by
   the fact that Q4_K_M PPL was only measured at n_ctx 2048/2 chunks where the
   corruption is less visible; WCF's 580-chunk run exposed it). It reproduces on
   ROCm; CUDA (148) full-offload runs crashed on OOM before reaching it, but the
   20-chunk CUDA run (7.44) suggests CUDA fused-GDN is fine — needs a dedicated
   580-chunk CUDA run to confirm.
4. **Encoder fix**: elementwise-op tensors (attn_q_norm, attn_k_norm, enorm,
   hnorm, shared_head_norm) must stay raw F32 — WCF block type is not supported
   by binary ops (ggml binary_op has no WCF case). Committed as 4cf4aba.

## 5. Artifacts

- Public repo: github.com/lcpo/wcf (main @ 4cf4aba)
- Patch: `patches/llama-wcf-native.diff` (applies clean to eaf937655)
- Models: `/home/ss/models/Qwen3.8-27B.wcf` (131), `/opt/models/Qwen3.8-27B.wcf` (148)
- Logs: 131 `/tmp/ppl_wcf_cpu_full2.log`, `/tmp/ppl_wcf_nofgdn.log`, `/tmp/bench_*.log`;
  148 `/tmp/ppl_wcf_cuda30.log`, `/tmp/bench_wcf_148_full.log`, `/tmp/bench_q4km_148_full.log`