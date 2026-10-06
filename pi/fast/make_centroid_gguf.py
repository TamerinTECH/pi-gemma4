#!/usr/bin/env python3
"""Add the masked-embedding tensors to a Gemma 4 E2B/E4B assistant (MTP drafter) GGUF.

The reference converter (llama.cpp conversion/gemma.py, Gemma4AssistantModel.filter_tensors) drops
`masked_embedding.centroids.weight` and `masked_embedding.token_ordering`, so published drafter GGUFs compute
their draft logits over the whole vocabulary. This tool reads the two tensors from the original checkpoint on
Hugging Face with HTTP range requests (public repositories only, no token), and writes a copy of the given GGUF
with two extra tensors and three metadata keys:

    masked_embd_centroids.weight   F32 [n_embd, n_centroids]   (numpy shape (n_centroids, n_embd))
    masked_embd_ordering           I32 [n_vocab]               token ids in cluster order, equal cluster size
    gemma4-assistant.n_centroids / .centroid_top_k / .use_ordered_embeddings

Nothing else in the file changes (same quantised tensors, same metadata).

Usage:
  make_centroid_gguf.py --in mtp.gguf --out mtp-centroids.gguf \
      --repo google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant [--gguf-py third_party/llama.cpp/gguf-py]
"""
import argparse
import hashlib
import json
import struct
import sys
import urllib.request

import numpy as np


def fetch(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if len(data) != end - start + 1:
        raise SystemExit(f"range request returned {len(data)} bytes instead of {end - start + 1}")
    return data


def read_safetensors(repo, names, filename="model.safetensors"):
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    n = struct.unpack("<Q", fetch(url, 0, 7))[0]
    header = json.loads(fetch(url, 8, 8 + n - 1))
    out = {}
    for name in names:
        if name not in header:
            raise SystemExit(f"{repo}: tensor {name!r} not in {filename} (has masked tensors: "
                             f"{[k for k in header if 'masked' in k]})")
        meta = header[name]
        a, b = meta["data_offsets"]
        raw = fetch(url, 8 + n + a, 8 + n + b - 1)
        out[name] = (meta["dtype"], meta["shape"], raw)
    return out


def bf16_to_f32(raw):
    return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--top-k", type=int, default=32, help="centroid_intermediate_top_k of the checkpoint config")
    ap.add_argument("--gguf-py", default="third_party/llama.cpp/gguf-py")
    a = ap.parse_args()
    sys.path.insert(0, a.gguf_py)
    import gguf

    t = read_safetensors(a.repo, ["masked_embedding.centroids.weight", "masked_embedding.token_ordering"])
    dt_c, shape_c, raw_c = t["masked_embedding.centroids.weight"]
    dt_o, shape_o, raw_o = t["masked_embedding.token_ordering"]
    if dt_c == "BF16":
        cent = bf16_to_f32(raw_c)
    elif dt_c == "F32":
        cent = np.frombuffer(raw_c, dtype="<f4")
    elif dt_c == "F16":
        cent = np.frombuffer(raw_c, dtype="<f2").astype(np.float32)
    else:
        raise SystemExit(f"unsupported centroid dtype {dt_c}")
    cent = np.ascontiguousarray(cent.reshape(shape_c).astype(np.float32))       # (n_centroids, n_embd)
    order = np.frombuffer(raw_o, dtype={"I64": "<i8", "I32": "<i4"}[dt_o]).reshape(shape_o)
    n_vocab = int(order.shape[0])
    n_cent, n_embd = int(cent.shape[0]), int(cent.shape[1])
    if sorted(order.tolist()) != list(range(n_vocab)):
        raise SystemExit("token_ordering is not a permutation of the vocabulary")
    if n_vocab % n_cent != 0:
        raise SystemExit("vocabulary size is not a multiple of the number of centroids")
    order32 = np.ascontiguousarray(order.astype(np.int32))

    reader = gguf.GGUFReader(a.src)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    emb = [x for x in reader.tensors if x.name == "token_embd.weight"][0]
    if int(emb.shape[0]) != n_embd or int(emb.shape[1]) != n_vocab:
        raise SystemExit(f"shape mismatch: GGUF token_embd {list(map(int, emb.shape))}, centroids {cent.shape}")
    if any(x.name.startswith("masked_embd") for x in reader.tensors):
        raise SystemExit("the input GGUF already carries masked-embedding tensors")

    writer = gguf.GGUFWriter(a.out, arch=arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
    writer.add_uint32(f"{arch}.n_centroids", n_cent)
    writer.add_uint32(f"{arch}.centroid_top_k", a.top_k)
    writer.add_bool(f"{arch}.use_ordered_embeddings", True)
    for tensor in reader.tensors:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)
    writer.add_tensor_info("masked_embd_centroids.weight", cent.shape, cent.dtype, cent.nbytes, gguf.GGMLQuantizationType.F32)
    writer.add_tensor_info("masked_embd_ordering", order32.shape, order32.dtype, order32.nbytes, gguf.GGMLQuantizationType.I32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    writer.write_tensor_data(cent)
    writer.write_tensor_data(order32)
    writer.close()

    print(json.dumps({
        "out": a.out, "source_gguf": a.src, "checkpoint": a.repo,
        "n_centroids": n_cent, "n_embd": n_embd, "n_vocab": n_vocab, "tokens_per_cluster": n_vocab // n_cent,
        "centroid_dtype_in_checkpoint": dt_c, "ordering_dtype_in_checkpoint": dt_o,
        "centroids_sha256": hashlib.sha256(cent.tobytes()).hexdigest(),
        "ordering_sha256": hashlib.sha256(order32.tobytes()).hexdigest(),
        "candidates_at_top_k": a.top_k * (n_vocab // n_cent),
    }, indent=1))


if __name__ == "__main__":
    main()
