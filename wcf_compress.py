#!/usr/bin/env python3
"""WCF encoder: convert a plain GGUF model to the WCF container format.

Reads ALL metadata from the source GGUF (arch, hparams, tokenizer) and encodes
every tensor in WCFI blocks: each 32x32 tile gets its own auto-computed scale
(f32), integer offset (i16) and variable bit width (1-4 bits, MSB-first).
value = (code + ioff) / scale. SSM/norm tensors are stored as raw F32.

Usage:
  python3 wcf_compress.py <source.gguf> <out.wcf> [--bits N] [--no-compress]

  --bits N       : bits per weight, 1-4 (default 4)
  --no-compress  : store every tensor as raw F32 (container test only)
"""
import sys, os, struct, time, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "gguf-py"))
from gguf import GGUFReader
from gguf.quants import dequantize

BLOCK_R, BLOCK_C = 32, 32
MAGIC = b"WCFI"
VERSION = 4
WCF_BITS = 4
NO_COMPRESS = False

# WCFH vtype codes (must match the reference decoder and gguf.cpp in the patch)
VT_STRING, VT_I64, VT_F64, VT_BOOL, VT_STRARR, VT_I32ARR, VT_U32, VT_F32, VT_I32 = 0, 1, 2, 3, 4, 5, 6, 7, 8


def pack_bits(values, n_bits):
    """Pack integers into bytes, MSB-first, n_bits per value."""
    arr = np.asarray(values, dtype=np.uint64)
    n = len(arr)
    if n == 0:
        return b""
    total_bits = n * n_bits
    n_bytes = (total_bits + 7) // 8
    bits = np.zeros(n_bytes * 8, dtype=np.uint8)
    for i in range(n_bits):
        bits[i::n_bits] = (arr >> (n_bits - 1 - i)) & 1
    powers = (1 << np.arange(7, -1, -1)).astype(np.uint8)
    bytes_arr = (bits.reshape(-1, 8).astype(np.uint8) @ powers).astype(np.uint8)
    return bytes(bytes_arr.tobytes())


def encode_int_block(block):
    """Quantize one 32x32 tile. Returns (scale, ioff, ibits, packed_bytes).

    Constant blocks (fspan == 0) store ioff = round(fmin), scale = 1.0,
    ibits = 0 so they decode back to exactly fmin.
    """
    flat = block.flatten()
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    fmin = float(flat.min())
    fmax = float(flat.max())
    fspan = fmax - fmin
    if fspan == 0 or not np.isfinite(fspan):
        # Constant block: ibits=0 (all codes 0), decode = ioff/scale.
        # Pick scale from the normal path so ioff/scale approximates fmin.
        ibits = WCF_BITS
        span = max(abs(fmin), 1e-12)
        scale = (float(1 << ibits) - 1.0) / span
        ioff = int(round(fmin * scale))
        ioff = max(-32768, min(32767, ioff))
        return scale, ioff, 0, b""
    ibits = WCF_BITS
    scale = (float(1 << ibits) - 1.0) / fspan
    ioff = int(round(fmin * scale))
    ioff = max(-32768, min(32767, ioff))
    ints = np.round(flat * scale).astype(np.int64) - ioff
    ints = np.clip(ints, 0, (1 << ibits) - 1)
    ipacked = pack_bits(ints, ibits)
    return scale, ioff, ibits, ipacked


def encode_int_tensor(name, tensor, ndim=2, ne2=1, rows_hdr=None):
    """Encode a 2D float tensor into WCFI blocks with edge padding."""
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, -1)
    rows, cols = tensor.shape
    hdr_rows = rows_hdr if rows_hdr is not None else rows
    n_br = (rows + BLOCK_R - 1) // BLOCK_R
    n_bc = (cols + BLOCK_C - 1) // BLOCK_C
    n_blocks = n_br * n_bc
    data = bytearray()
    data += MAGIC
    data += struct.pack('<B', VERSION)
    data += struct.pack('<I', 1)          # one tensor per WCFI block
    nb = name.encode('utf-8')
    data += struct.pack('<H', len(nb)) + nb
    data += struct.pack('<II', rows, cols)
    data += struct.pack('<I', n_blocks)
    data += struct.pack('<B', ndim)
    data += struct.pack('<I', ne2)
    if n_blocks == 0:
        data += tensor.astype(np.float32).tobytes()
    else:
        for br in range(n_br):
            for bc in range(n_bc):
                block = tensor[br*BLOCK_R:(br+1)*BLOCK_R, bc*BLOCK_C:(bc+1)*BLOCK_C]
                if block.shape != (BLOCK_R, BLOCK_C):
                    # pad with the block minimum so quantization stays optimal
                    fmin = float(np.nan_to_num(block, nan=0.0).min())
                    pad = np.full((BLOCK_R, BLOCK_C), fmin, dtype=np.float32)
                    pad[:block.shape[0], :block.shape[1]] = block
                    block = pad
                scale, ioff, ibits, ipacked = encode_int_block(block)
                data += struct.pack('<f', scale)
                data += struct.pack('<h', ioff)
                data += struct.pack('<B', ibits)
                data += struct.pack('<I', len(ipacked)) + ipacked
    return bytes(data)


def encode_raw_f32(name, tensor, ndim=2, ne2=1, rows_hdr=None, cols_hdr=None):
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, -1)
    rows, cols = tensor.shape
    hdr_rows = rows_hdr if rows_hdr is not None else rows
    hdr_cols = cols_hdr if cols_hdr is not None else cols
    data = bytearray()
    data += MAGIC
    data += struct.pack('<B', VERSION)
    data += struct.pack('<I', 1)
    nb = name.encode('utf-8')
    data += struct.pack('<H', len(nb)) + nb
    data += struct.pack('<II', hdr_rows, hdr_cols)
    data += struct.pack('<I', 0)          # n_blocks = 0 -> raw F32 payload
    data += struct.pack('<B', ndim)
    data += struct.pack('<I', ne2)
    data += tensor.astype(np.float32).tobytes()
    return bytes(data)


def gguf_tensor_to_wcf(name, g):
    """Convert a GGUF tensor to WCFI layout (ne0-fastest, llama.cpp order)."""
    data = dequantize(g.data, g.tensor_type)
    shape = g.shape                       # (ne0, ne1, ...) - logical dims
    ndim = len(shape)
    if ndim == 1:
        v = data.flatten().astype(np.float32)
        return v.reshape(1, -1), 1, 1, None, None
    elif ndim == 2:
        # g.data is (ne1, ne0) in numpy; store as-is, rows=ne1 cols=ne0
        v = data.astype(np.float32)
        return v, 2, 1, None, None
    elif ndim == 3:
        n_exp = shape[2]
        rows = shape[1]
        cols = shape[0]
        v2 = data.reshape(rows * n_exp, cols).astype(np.float32)
        return v2, 3, n_exp, rows, None
    else:
        raise ValueError(f"unsupported ndim {ndim} for {name}")


# Tensors stored as raw F32 (tiny / sensitive to quantization)
SSM_NAMES = ('ssm_alpha', 'ssm_beta', 'ssm_a', 'ssm_dt', 'ssm_out',
             'ssm_conv1d', 'ssm_norm', 'attn_norm', 'post_attention_norm',
             'output_norm', 'ffn_gate_inp')


def compress_tensor(name, g):
    v, ndim, ne2, rows_hdr, cols_hdr = gguf_tensor_to_wcf(name, g)
    is_ssm = any(s in name for s in SSM_NAMES)
    if not NO_COMPRESS and not is_ssm:
        data = encode_int_tensor(name, v, ndim=ndim, ne2=ne2, rows_hdr=rows_hdr)
        return (data, v.nbytes)
    data = encode_raw_f32(name, v, ndim=ndim, ne2=ne2, rows_hdr=rows_hdr, cols_hdr=cols_hdr)
    return (data, v.nbytes)


def read_gguf_metadata(gguf_path):
    """Read all GGUF metadata as (key, wcf_type, value) triples."""
    reader = GGUFReader(gguf_path)
    metadata = []
    for key, field in reader.fields.items():
        if key in ('GGUF.version', 'GGUF.tensor_count', 'GGUF.kv_count'):
            continue
        parts = field.parts
        types = field.types
        if len(parts) == 0:
            continue
        vtype = int(types[-1])
        is_array = (len(types) >= 2 and int(types[0]) == 9)
        if is_array:
            vtype = 9
        if vtype == 8:
            val = parts[-1]
            if isinstance(val, np.ndarray) and val.dtype == np.uint8:
                metadata.append((key, 'string', val.tobytes().decode('utf-8', errors='replace')))
            else:
                metadata.append((key, 'string', str(val)))
        elif vtype == 7:
            metadata.append((key, 'bool', bool(parts[-1][0])))
        elif vtype == 6:
            metadata.append((key, 'float32', float(parts[-1][0])))
        elif vtype == 5:
            metadata.append((key, 'int64', int(parts[-1][0])))
        elif vtype == 4:
            metadata.append((key, 'uint32', int(parts[-1][0])))
        elif vtype == 3:
            metadata.append((key, 'int32', int(parts[-1][0])))
        elif vtype == 2:
            metadata.append((key, 'int32', int(parts[-1][0])))
        elif vtype == 0:
            metadata.append((key, 'float32', float(parts[-1][0])))
        elif vtype == 1:
            metadata.append((key, 'float64', float(parts[-1][0])))
        elif vtype == 9:
            elem_type = int(types[1]) if len(types) > 1 else int(types[0])
            n_elements = int(parts[4][0]) if len(parts) > 4 else 0
            if elem_type == 8:
                arr = []; idx = 5
                for _ in range(n_elements):
                    if idx >= len(parts): break
                    slen = int(parts[idx][0]); idx += 1
                    if idx >= len(parts): break
                    sbytes = parts[idx]
                    if isinstance(sbytes, np.ndarray) and sbytes.dtype == np.uint8:
                        arr.append(sbytes.tobytes().decode('utf-8', errors='replace'))
                    idx += 1
                if arr:
                    metadata.append((key, 'string_array', arr))
            elif elem_type in (4, 5):
                arr = [int(parts[i][0]) for i in range(5, min(5 + n_elements, len(parts)))]
                if arr:
                    metadata.append((key, 'int32_array', arr))
        else:
            try:
                metadata.append((key, 'int32', int(parts[-1][0])))
            except Exception:
                pass
    return metadata


def encode_header(metadata):
    """Serialize the WCFH header from (key, wcf_type, value) triples."""
    data = bytearray()
    data += b'WCFH'
    data += struct.pack('<B', 1)
    data += struct.pack('<H', len(metadata))
    for key, vtype, val in metadata:
        kb = key.encode('utf-8')
        data += struct.pack('<H', len(kb)) + kb
        if vtype == 'string':
            vb = val.encode('utf-8')
            data += struct.pack('<B', 0)
            data += struct.pack('<I', len(vb)) + vb
        elif vtype == 'int64':
            data += struct.pack('<B', 1); data += struct.pack('<q', val)
        elif vtype == 'float64':
            data += struct.pack('<B', 2); data += struct.pack('<d', val)
        elif vtype == 'bool':
            data += struct.pack('<B', 3); data += struct.pack('<B', 1 if val else 0)
        elif vtype == 'string_array':
            data += struct.pack('<B', 4); data += struct.pack('<I', len(val))
            for s in val:
                sb = s.encode('utf-8'); data += struct.pack('<I', len(sb)) + sb
        elif vtype == 'int32_array':
            data += struct.pack('<B', 5); data += struct.pack('<I', len(val))
            for v in val: data += struct.pack('<i', v)
        elif vtype == 'uint32':
            data += struct.pack('<B', 6); data += struct.pack('<I', val)
        elif vtype == 'float32':
            data += struct.pack('<B', 7); data += struct.pack('<f', val)
        elif vtype == 'int32':
            data += struct.pack('<B', 8); data += struct.pack('<i', val)
        else:
            data += struct.pack('<B', 1); data += struct.pack('<q', int(val))
    return bytes(data)


def main():
    parser = argparse.ArgumentParser(description="Encode a GGUF model to the WCF format")
    parser.add_argument("gguf_file")
    parser.add_argument("out_file")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--no-compress", action="store_true")
    args = parser.parse_args()

    global WCF_BITS, NO_COMPRESS
    WCF_BITS = args.bits
    if not (1 <= WCF_BITS <= 4):
        print(f"ERROR: WCF_BITS must be 1-4 (decoder supports max 4 bits), got {WCF_BITS}.")
        sys.exit(1)
    NO_COMPRESS = args.no_compress
    print(f"bits/weight: {WCF_BITS}, no_compress={NO_COMPRESS}", flush=True)

    r = GGUFReader(args.gguf_file)
    tensors = list(r.tensors)
    print(f"Tensors: {len(tensors)}", flush=True)

    metadata = read_gguf_metadata(args.gguf_file)
    print(f"Metadata fields: {len(metadata)}", flush=True)
    # ggml expects u32 for these keys
    U32_KEYS = {'general.file_type', 'general.quantization_version'}
    new_meta = []
    for k, vt, v in metadata:
        if k in U32_KEYS:
            new_meta.append((k, 'uint32', int(v)))
        else:
            new_meta.append((k, vt, v))
    new_meta.append(('wcf.encoding', 'string', 'wcf_native'))
    new_meta.append(('wcf_int.version', 'int32', VERSION))
    new_meta.append(('wcf_int.bits', 'int32', WCF_BITS))
    header = encode_header(new_meta)
    print(f"Header: {len(header)} bytes, {len(new_meta)} fields", flush=True)

    t0 = time.time()
    total_size = len(header)
    total_orig = 0
    n_compressed = 0

    with open(args.out_file, "wb") as f:
        f.write(header)
        for g in tensors:
            name = g.name
            if len(name) > 64:
                name = name[-64:]
            data, orig = compress_tensor(name, g)
            f.write(data)
            total_size += len(data)
            total_orig += orig
            n_compressed += 1
            if n_compressed % 50 == 0:
                print(f"  {n_compressed}/{len(tensors)} tensors, {total_size/1e9:.2f} GB, {time.time()-t0:.0f}s", flush=True)

    print(f"Time: {time.time()-t0:.0f}s", flush=True)
    print(f"WCF size: {total_size/1e9:.2f} GB", flush=True)
    print(f"Tensors: {n_compressed}", flush=True)
    if total_orig > 0:
        print(f"bits/weight: {total_size*8/total_orig:.2f}", flush=True)
    print(f"Done: {args.out_file}", flush=True)


if __name__ == "__main__":
    main()