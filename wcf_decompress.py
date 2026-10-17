#!/usr/bin/env python3
"""WCF reference decoder: convert a .wcf file back to a plain F32 GGUF.

This is the ground-truth implementation of the WCF container format. It reads
the WCFH metadata header and every WCFI tensor block, decodes each 32x32 tile
(value = (code + ioff) / scale), and writes a standard F32 GGUF that any
llama.cpp build can load.

Usage:
  python3 wcf_decompress.py <model.wcf> <out.gguf> [--source <orig.gguf>]

  --source : optional original GGUF; its metadata is copied verbatim and any
             tensor missing from the .wcf is carried over with original bytes.
"""
import sys, os, struct, argparse, re
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "gguf-py"))
from gguf import GGUFReader, GGUFWriter
from gguf.constants import GGUFValueType, GGMLQuantizationType

BLOCK_R, BLOCK_C = 32, 32


def unpack_bits(data, count, n_bits):
    """Unpack `count` values of `n_bits` bits each from bytes (MSB-first)."""
    if n_bits == 0 or count == 0:
        return np.zeros(count, dtype=np.int64)
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr, bitorder='big')
    n_vals = min(count, len(bits) // n_bits)
    out = np.zeros(count, dtype=np.int64)
    if n_vals == 0:
        return out
    bit_matrix = bits[:n_vals * n_bits].reshape(n_vals, n_bits)
    powers = (1 << np.arange(n_bits - 1, -1, -1)).astype(np.uint32)
    out[:n_vals] = bit_matrix.astype(np.uint32) @ powers
    return out


def read_wcfh(f):
    """Read the WCFH metadata header. Returns dict or None if magic mismatch."""
    magic = f.read(4)
    if magic != b"WCFH":
        f.seek(0)
        return None
    struct.unpack("<B", f.read(1))[0]  # header version
    nf = struct.unpack("<H", f.read(2))[0]
    meta = {}
    for _ in range(nf):
        kl = struct.unpack("<H", f.read(2))[0]
        k = f.read(kl).decode('utf-8')
        vt = struct.unpack("<B", f.read(1))[0]
        if vt == 0:      # string
            vl = struct.unpack("<I", f.read(4))[0]
            meta[k] = f.read(vl).decode('utf-8', errors='replace')
        elif vt == 1:    # int64
            meta[k] = struct.unpack("<q", f.read(8))[0]
        elif vt == 2:    # float64
            meta[k] = struct.unpack("<d", f.read(8))[0]
        elif vt == 3:    # bool
            meta[k] = bool(struct.unpack("<B", f.read(1))[0])
        elif vt == 4:    # string array
            n = struct.unpack("<I", f.read(4))[0]
            meta[k] = [f.read(struct.unpack("<I", f.read(4))[0]).decode('utf-8', errors='replace') for _ in range(n)]
        elif vt == 5:    # int32 array
            n = struct.unpack("<I", f.read(4))[0]
            meta[k] = list(struct.unpack(f"<{n}i", f.read(4 * n)))
        elif vt == 6:    # uint32
            meta[k] = struct.unpack("<I", f.read(4))[0]
        elif vt == 7:    # float32
            meta[k] = struct.unpack("<f", f.read(4))[0]
        elif vt == 8:    # int32
            meta[k] = struct.unpack("<i", f.read(4))[0]
        else:
            raise ValueError(f"unknown WCFH value type {vt}")
    return meta


def iter_wcf_tensors(wcf_path):
    """Yield (name, rows, cols, ndim, ne2, float32_array) one tensor at a time."""
    with open(wcf_path, "rb") as f:
        if read_wcfh(f) is None:
            f.seek(0)
        while True:
            m = f.read(4)
            if not m or m != b"WCFI":
                break
            struct.unpack("<B", f.read(1))[0]  # block version
            nt = struct.unpack("<I", f.read(4))[0]
            for _ in range(nt):
                nl = struct.unpack("<H", f.read(2))[0]
                name = f.read(nl).decode('utf-8')
                rows, cols = struct.unpack("<II", f.read(8))
                n_blocks = struct.unpack("<I", f.read(4))[0]
                ndim = struct.unpack("<B", f.read(1))[0]
                ne2 = struct.unpack("<I", f.read(4))[0]
                if n_blocks == 0:
                    v = np.frombuffer(f.read(rows * cols * 4), dtype=np.float32).copy()
                    yield name, rows, cols, ndim, ne2, v
                    continue
                recon = np.zeros((rows, cols), np.float32)
                n_br = (rows + BLOCK_R - 1) // BLOCK_R
                n_bc = (cols + BLOCK_C - 1) // BLOCK_C
                for br in range(n_br):
                    for bc in range(n_bc):
                        scale = struct.unpack("<f", f.read(4))[0]
                        ioff = struct.unpack("<h", f.read(2))[0]
                        ibits = struct.unpack("<B", f.read(1))[0]
                        n_packed = struct.unpack("<I", f.read(4))[0]
                        ipacked = f.read(n_packed)
                        count = BLOCK_R * BLOCK_C
                        if ibits == 0:
                            block = np.full(count, float(ioff) / scale, dtype=np.float32).reshape(BLOCK_R, BLOCK_C)
                        else:
                            ints = unpack_bits(ipacked, count, ibits) + ioff
                            block = (ints.astype(np.float32) / scale).reshape(BLOCK_R, BLOCK_C)
                        r0, c0 = br * BLOCK_R, bc * BLOCK_C
                        rr, cc = min(BLOCK_R, rows - r0), min(BLOCK_C, cols - c0)
                        recon[r0:r0+rr, c0:c0+cc] = block[:rr, :cc]
                yield name, rows, cols, ndim, ne2, recon


def copy_metadata(writer, reader):
    """Copy metadata from the source GGUF into the writer."""
    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue
        try:
            p = field.parts[-1]
            if key in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges"):
                arr = []
                for i in range(5, len(field.parts), 2):
                    arr.append(bytes(field.parts[i + 1]).decode('utf-8', errors='replace'))
                writer.add_array(key, arr)
            elif key == "tokenizer.ggml.token_type":
                count = int(field.parts[4][0])
                writer.add_array(key, [int(field.parts[i][0]) for i in range(5, 5 + count)])
            elif key == "tokenizer.ggml.scores":
                count = int(field.parts[4][0])
                writer.add_array(key, [float(field.parts[i][0]) for i in range(5, 5 + count)])
            elif isinstance(p, np.ndarray) and p.dtype == np.uint8 and len(p) > 1:
                val = bytes(p).decode('utf-8', errors='replace')
                writer.add_key_value(key, val, GGUFValueType.STRING)
            else:
                val = p[0] if hasattr(p, '__len__') and len(p) > 0 else p
                if isinstance(val, np.generic):
                    val = val.item()
                dt = getattr(p, 'dtype', None)
                if dt == np.uint32:
                    writer.add_key_value(key, int(val), GGUFValueType.UINT32)
                elif dt == np.int32:
                    writer.add_key_value(key, int(val), GGUFValueType.INT32)
                elif dt == np.float32:
                    writer.add_key_value(key, float(val), GGUFValueType.FLOAT32)
                elif dt == np.float64:
                    writer.add_key_value(key, float(val), GGUFValueType.FLOAT64)
                elif dt == np.int64:
                    writer.add_key_value(key, int(val), GGUFValueType.INT64)
                elif dt == np.uint64:
                    writer.add_key_value(key, int(val), GGUFValueType.UINT64)
                elif dt == np.bool_:
                    writer.add_key_value(key, bool(val), GGUFValueType.BOOL)
                elif isinstance(val, bool):
                    writer.add_key_value(key, val, GGUFValueType.BOOL)
                elif isinstance(val, int):
                    writer.add_key_value(key, val, GGUFValueType.UINT32)
                elif isinstance(val, float):
                    writer.add_key_value(key, val, GGUFValueType.FLOAT32)
                elif isinstance(val, str):
                    writer.add_key_value(key, val, GGUFValueType.STRING)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Convert WCF back to F32 GGUF")
    parser.add_argument("wcf_file")
    parser.add_argument("out_file")
    parser.add_argument("--source", help="original GGUF for metadata + missing tensors")
    args = parser.parse_args()

    meta = None
    reader = None
    arch = "wcf"
    orig_shapes = {}
    if args.source:
        reader = GGUFReader(args.source)
        for key, field in reader.fields.items():
            if key == "general.architecture":
                arch = bytes(field.parts[-1]).decode('utf-8', errors='replace')
                break
        orig_shapes = {t.name: [int(x) for x in t.shape] for t in reader.tensors}
    else:
        with open(args.wcf_file, "rb") as f:
            meta = read_wcfh(f)
            if meta:
                arch = meta.get("general.architecture", "wcf")

    print(f"Architecture: {arch}", flush=True)

    writer = GGUFWriter(args.out_file, arch, use_temp_file=True)
    if reader:
        copy_metadata(writer, reader)
    else:
        # Minimal metadata from the WCFH header
        for k, v in (meta or {}).items():
            if isinstance(v, str):
                writer.add_key_value(k, v, GGUFValueType.STRING)
            elif isinstance(v, bool):
                writer.add_key_value(k, v, GGUFValueType.BOOL)
            elif isinstance(v, int):
                writer.add_key_value(k, v, GGUFValueType.UINT32)
            elif isinstance(v, float):
                writer.add_key_value(k, v, GGUFValueType.FLOAT32)
            elif isinstance(v, list):
                if v and isinstance(v[0], str):
                    writer.add_array(k, v)
                else:
                    writer.add_array(k, [int(x) for x in v])

    print("Decoding WCF tensors (streaming)...", flush=True)
    n_wcf = 0
    n_orig = 0
    written = set()

    for name, rows, cols, ndim, ne2, v in iter_wcf_tensors(args.wcf_file):
        if ndim == 3 and ne2 > 1:
            # encoder stored (rows*ne2, cols); rebuild (ne2, rows, cols)
            rows_hdr = rows // ne2
            v = v.reshape(ne2, rows_hdr, cols)
        elif ndim == 1:
            v = v.reshape(-1)
        writer.add_tensor(name, v, raw_dtype=GGMLQuantizationType.F32)
        written.add(name)
        n_wcf += 1
        if n_wcf % 1000 == 0:
            print(f"  {n_wcf} tensors...", flush=True)

    # Carry over tensors absent from the .wcf (original bytes preserved)
    if reader:
        for t in reader.tensors:
            if t.name not in written:
                writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
                n_orig += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    print(f"F32 GGUF written: {args.out_file} (decoded={n_wcf}, carried={n_orig})", flush=True)


if __name__ == "__main__":
    main()
