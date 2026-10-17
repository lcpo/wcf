#!/usr/bin/env python3
"""Round-trip test: GGUF -> WCF -> GGUF, compare values."""
import sys, os
import numpy as np

sys.path.insert(0, "/Users/ss/ai/glm-pruning/wcf-repo/vendor/gguf-py")
from gguf import GGUFReader, GGUFWriter
from gguf.quants import dequantize
from gguf.constants import GGMLQuantizationType

T = "/tmp/wcf_test"
os.makedirs(T, exist_ok=True)
src = os.path.join(T, "src.gguf")
wcf = os.path.join(T, "test.wcf")
out = os.path.join(T, "out.gguf")

rng = np.random.default_rng(42)
w = GGUFWriter(src, "wcf")
w.add_uint32("general.file_type", 1)
w.add_uint32("general.quantization_version", 4)
w.add_uint32("qwen35.block_count", 4)
w.add_string("tokenizer.ggml.model", "gpt2")
t2d = rng.normal(0, 0.05, (48, 64)).astype(np.float32)
t1d = rng.normal(0, 1.0, 100).astype(np.float32)
t3d = rng.normal(0, 0.05, (4, 16, 32)).astype(np.float32)
t_small = rng.normal(0, 0.1, (8, 8)).astype(np.float32)
t_const = np.full((32, 32), -0.25, dtype=np.float32)
t_f16 = rng.normal(0, 0.05, (48, 64)).astype(np.float16)  # real F16 data
w.add_tensor("test.attn_q", t2d, raw_dtype=GGMLQuantizationType.F32)
w.add_tensor("test.f16", t_f16)  # writer auto-detects F16 from dtype
w.add_tensor("test.ssm_a", t1d, raw_dtype=GGMLQuantizationType.F32)
w.add_tensor("test.ffn_gate_exps", t3d, raw_dtype=GGMLQuantizationType.F32)
w.add_tensor("test.small", t_small, raw_dtype=GGMLQuantizationType.F32)
w.add_tensor("test.const", t_const, raw_dtype=GGMLQuantizationType.F32)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file()
w.close()
print("src.gguf written")

sys.path.insert(0, "/Users/ss/ai/glm-pruning/wcf-repo")
import wcf_compress as E
r = GGUFReader(src)
meta = E.read_gguf_metadata(src)
extra = [("wcf.encoding", "string", "wcf_native"), ("wcf_int.version", "int32", E.VERSION)]
hdr = E.encode_header(meta + extra)
with open(wcf, "wb") as f:
    f.write(hdr)
    for g in r.tensors:
        name = g.name[-64:]
        data, orig = E.compress_tensor(name, g)
        f.write(data)
print("test.wcf written")

import wcf_decompress as D
sys.argv = ["wcf_decompress.py", wcf, out, "--source", src]
D.main()
print("out.gguf written")

ro = GGUFReader(out)
out_tensors = dict((t.name, t) for t in ro.tensors)
rs = GGUFReader(src)
src_tensors = dict((t.name, t) for t in rs.tensors)

ok = True
for name, st in src_tensors.items():
    if name not in out_tensors:
        print("MISSING", name)
        ok = False
        continue
    sdata = dequantize(st.data, st.tensor_type)
    ot = out_tensors[name]
    odata = dequantize(ot.data, ot.tensor_type)
    if sdata.shape != odata.shape:
        print("SHAPE MISMATCH", name, sdata.shape, odata.shape)
        ok = False
        continue
    diff = np.abs(sdata - odata)
    snr = 10 * np.log10((np.var(sdata) + 1e-30) / (np.var(diff) + 1e-30))
    exact = bool(np.array_equal(sdata, odata))
    print(name, "shape", sdata.shape, "exact", exact, "SNR", round(float(snr), 1), "maxabs", round(float(diff.max()), 6))
    if not exact and snr < 20:
        print("  FAIL: SNR too low")
        ok = False
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
