"""pigemma.expert_map - exact per-(layer, expert) byte ranges and page accounting for MoE GGUFs.

Why (brief 2026-10-05, sections 3 and 5): the pinning comparison must count *actual unique pages*
(incl. page-alignment overhead and boundary pages shared between neighbouring experts) instead of
nominal expert bytes, and the simulator must use the real layout of the files that run on the
device - including split GGUF parts and per-layer mixed quant types (unsloth "UD" files).

Built on :mod:`pigemma.gguf_map` (its header readers and ``classify_tensor``); nothing here
assumes uniform expert sizes or a single file.

Layout model
------------
A model is one or more GGUF *files* (a split model = its parts, ``split.no`` order). Every byte
range is file-relative, because the page cache indexes pages per file. Owners of byte ranges:

* ``header``: ``[0, data_offset)`` of each file (KV metadata incl. the vocabulary, tensor infos);
  read with ``fread`` at load time, never through the weight mapping.
* non-expert tensors: every tensor that is not a per-expert tensor (attention, dense FFN, norms,
  router, embeddings/tied head, rope_freqs).
* expert slices: for every tensor of MoE layer ``L`` whose name matches
  ``blk.L.ffn_<x>_exps.<y>`` and whose last ggml dim equals ``n_expert`` (``ffn_gate_up_exps``,
  ``ffn_gate_exps``/``ffn_up_exps``, ``ffn_down_exps`` and per-expert scale vectors such as
  ``ffn_down_exps.scale`` F32 [n_expert]), expert ``e`` owns the contiguous slice
  ``[offset + e*nb, offset + (e+1)*nb)`` with ``nb = n_bytes / n_expert`` (the expert dim is the
  outermost ggml dim, so slices are contiguous for every block-quantised type). The tensor's
  ggml type is recorded per slice; sizes are never assumed uniform across layers or roles.

Alignment padding between tensors (< ``general.alignment`` bytes) belongs to nobody.

Page accounting (per page size P; 4096 = desktop Docker VM, 16384 = Raspberry Pi 5 kernel)
--------------------------------------------------------------------------------------------
Page ``p`` of file ``f`` holds bytes ``[p*P, (p+1)*P)``. A range touches pages
``floor(start/P) .. floor((end-1)/P)``. Pages strictly inside one range belong to that range
alone; only the first/last page of a range can be shared, so ownership is resolved exactly from
those boundary pages. Per expert ``(L, e)``:

* ``pages``: unique pages of the union of all its slices = what must be resident (or mlocked) to
  run that expert without a fault. ``pages * P`` is its page-aligned footprint.
* ``exclusive``: pages touched by no other expert and no non-expert range/header.
* ``shared_expert``: pages also touched by another expert (neighbour in the same tensor, or every
  expert of the layer for the scale-vector page).
* ``shared_nonexpert``: pages also touched by a non-expert tensor or a header.
  (A page can be in both shared columns, so exclusive + shared_* may exceed pages.)

The unique page count of any *set* of owners is ``sum(interior pages) + |union of boundary
pages|`` (exact, see :meth:`ExpertMap.pages_for`), which is what a pinning budget must use.
``ExpertMap.ranges_for`` returns the page-aligned, merged, per-file ranges for such a set (what a
pinner locks).

Repack eligibility (static, not measured)
-----------------------------------------
``repack_eligible(type, shape, platform)`` reproduces ``ggml_repack_get_optimal_repack_type`` of
ggml/src/ggml-cpu/repack.cpp at llama.cpp a7b94df for two CPUs: ``x86_avx512`` (desktop Ryzen 5
8400F: AVX2 + AVX-512, no SVE) and ``arm_a76`` (Pi 5 Cortex-A76: NEON + dotprod, no i8mm/SVE).
Only MUL_MAT / MUL_MAT_ID weights (attention, dense FFN, experts and the tied output head, which
gemma4 creates as a TENSOR_DUPLICATED copy of token_embd) can be repacked. With repack on (the
llama.cpp default) eligible tensors are copied out of the file mapping into anonymous
CPU_REPACK memory; with ``-nr`` nothing is. This is a source-derived estimate: the real split
must be read from llama.cpp's load log (CPU_REPACK / CPU_Mapped buffer sizes).

CLI (``python -m pigemma.expert_map <cmd> --help``)::

    build <gguf> [<gguf> ...] --out map.json [--page-sizes 4096,16384]
          [--header-sizes N,N,...] [--labels a,b,...] [--embed-base-map] [--provenance FILE]
    summary --map map.json
    pages --map map.json --experts 0:5,3:7 [--page-size 16384] [--with-nonexpert]

``--header-sizes`` treats each input as a header-only prefix (e.g. streamed from the Pi with
``head -c``) of a file with that real size. ``--embed-base-map`` (single file only) writes the
pigemma.gguf_map map.json document with this module's result under the ``expert_map`` key, so
``gguf_map.load_map`` / ``plan_hotset`` / cache_sim keep working on the same file.

Exit codes: 0 ok; 1 error (malformed/inconsistent layout); 2 usage; 3 missing input file.

Public API: build_layout, build_expert_map, ExpertMap, load_expert_map, repack_eligible,
REPACK_PLATFORMS, ExpertMapError, main.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pigemma import __version__
from pigemma import gguf_map as gm

log = logging.getLogger("pigemma.expert_map")

FORMAT = "pigemma-expert-map"
FORMAT_VERSION = 1
GENERATOR = f"pigemma.expert_map/{__version__}"
DEFAULT_PAGE_SIZES: tuple[int, ...] = (4096, 16384)
LLAMA_CPP_COMMIT = "a7b94df"

# blk.<L>.ffn_<role>_exps.<suffix>  (weight, scale, bias, ...)
_RE_EXPERT_ANY = re.compile(r"^blk\.(\d+)\.ffn_([a-z0-9_]+?)_exps\.([a-z0-9_]+)$")

# ggml_repack_get_optimal_repack_type at a7b94df (ggml/src/ggml-cpu/repack.cpp ~4974-5165):
# platform -> {type: required ne[1] multiple}. x86: AVX2 rules + the AVX-512-only Q2_K rule
# (the Ryzen 5 8400F has both). arm_a76: NEON + dotprod rules (no i8mm, no SVE): the first
# matching branch for each type is the dotprod one, whose ne[1] multiple is listed.
REPACK_PLATFORMS: dict[str, dict[str, int]] = {
    "x86_avx512": {"Q4_0": 8, "Q4_K": 8, "Q2_K": 8, "IQ4_NL": 8, "MXFP4": 8},
    "arm_a76": {"Q4_0": 4, "Q4_K": 8, "Q5_K": 8, "Q6_K": 8, "IQ4_NL": 4, "MXFP4": 4, "Q8_0": 4,
                "Q1_0": 4},
}
_MATMUL_KINDS = ("attn", "ffn_dense", "expert", "output")


class ExpertMapError(ValueError):
    """Malformed or inconsistent GGUF layout (overlaps, bad split set, indivisible expert dim)."""


def repack_eligible(type_name: str, shape: Sequence[int], platform: str) -> bool:
    """True when llama.cpp a7b94df would pick a CPU_REPACK layout for this weight on ``platform``."""
    rules = REPACK_PLATFORMS[platform]
    mult = rules.get(str(type_name))
    if mult is None or len(shape) < 2:
        return False
    return int(shape[1]) % mult == 0


# --------------------------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------------------------
@dataclass
class FileLayout:
    index: int
    label: str
    size: int
    data_offset: int
    alignment: int
    tensors: list[dict[str, Any]]
    split_no: int | None = None


@dataclass
class Layout:
    arch: str
    n_layer: int
    n_expert: int
    n_expert_used: int
    files: list[FileLayout]
    fields: dict[str, Any] = field(default_factory=dict)


def _split_no(hdr: gm.GgufHeader) -> int | None:
    v = hdr.fields.get("split.no")
    return int(v) if isinstance(v, int) else None


def build_layout(headers: Sequence[gm.GgufHeader], labels: Sequence[str] | None = None) -> Layout:
    """Combine one GGUF header, or the headers of all parts of a split GGUF, into a Layout."""
    if not headers:
        raise ExpertMapError("no GGUF headers given")
    labels = list(labels) if labels is not None else [h.path for h in headers]
    if len(labels) != len(headers):
        raise ExpertMapError(f"{len(labels)} labels for {len(headers)} files")
    order = list(range(len(headers)))
    if len(headers) > 1:
        nos = [_split_no(h) for h in headers]
        counts = {h.fields.get("split.count") for h in headers}
        if any(n is None for n in nos):
            raise ExpertMapError("several files given but not all have split.no (not a split GGUF)")
        if len(counts) != 1 or next(iter(counts)) != len(headers):
            raise ExpertMapError(f"split.count {sorted(map(str, counts))} != {len(headers)} files given")
        if sorted(nos) != list(range(len(headers))):  # type: ignore[type-var]
            raise ExpertMapError(f"split.no values {nos} are not 0..{len(headers) - 1}")
        order = sorted(order, key=lambda i: nos[i])  # type: ignore[arg-type,return-value]
    main_hdr = next((headers[i] for i in order if "general.architecture" in headers[i].fields), None)
    if main_hdr is None:
        raise gm.MissingKeyError("general.architecture", labels[order[0]])
    f = main_hdr.fields
    arch = str(f["general.architecture"])
    n_layer = int(gm._kv(f, f"{arch}.block_count", main_hdr.path))
    n_expert = int(gm._kv(f, f"{arch}.expert_count", main_hdr.path, required=False, default=0) or 0)
    n_used = int(gm._kv(f, f"{arch}.expert_used_count", main_hdr.path, required=False, default=0) or 0)
    tcount = f.get("split.tensors.count")
    files: list[FileLayout] = []
    seen: set[str] = set()
    for fi, hi in enumerate(order):
        h = headers[hi]
        recs = sorted(h.tensors, key=lambda t: (t.offset, t.name))
        tens: list[dict[str, Any]] = []
        prev_end, prev = h.data_offset, None
        for t in recs:
            if t.name in seen:
                raise ExpertMapError(f"tensor {t.name!r} appears in more than one file")
            seen.add(t.name)
            if t.offset < prev_end:
                raise ExpertMapError(f"{labels[hi]}: tensor {t.name!r} overlaps {prev or 'the header'}")
            end = t.offset + t.n_bytes
            if end > h.file_size:
                raise ExpertMapError(f"{labels[hi]}: tensor {t.name!r} ends at {end} > file size {h.file_size}")
            prev_end, prev = end, t.name
            kind, layer = gm.classify_tensor(t.name)
            tens.append({"name": t.name, "type": t.type, "shape": list(t.shape), "n_bytes": t.n_bytes,
                         "offset": t.offset, "kind": kind, "layer": layer})
        files.append(FileLayout(fi, labels[hi], h.file_size, h.data_offset, h.alignment, tens, _split_no(h)))
    if isinstance(tcount, int) and tcount != len(seen):
        raise ExpertMapError(f"split.tensors.count {tcount} != {len(seen)} tensors found in the files given")
    return Layout(arch, n_layer, n_expert, n_used, files, dict(f))


def read_headers(paths_: Sequence[str | os.PathLike[str]], header_sizes: Sequence[int] | None = None,
                 backend: str = "stdlib") -> list[gm.GgufHeader]:
    """Read GGUF headers; with ``header_sizes`` each path is a header-only prefix of a file of that size."""
    out = []
    for i, p in enumerate(paths_):
        pp = Path(os.fspath(p))
        if not pp.is_file():
            raise FileNotFoundError(f"GGUF file not found: {pp}")
        if header_sizes is not None:
            with open(pp, "rb") as fh:
                out.append(gm.read_gguf_header(fh, "stdlib", file_size=int(header_sizes[i]), label=str(pp)))
        else:
            out.append(gm.read_gguf_header(pp, backend))
    return out


# --------------------------------------------------------------------------------------------
# Owners and page accounting
# --------------------------------------------------------------------------------------------
NONEXPERT = "N"   # owner id of every non-expert tensor
HEADER = "H"      # owner id of file headers ([0, data_offset))
_NON_EXPERT_IDS = (NONEXPERT, HEADER)


@dataclass
class _Range:
    file: int
    start: int
    end: int
    owner: Any           # (L, e) for an expert slice, NONEXPERT or HEADER otherwise
    tensor: str | None   # None for a header
    role: str | None = None
    type: str | None = None


def _expert_ranges(layout: Layout) -> tuple[list[_Range], dict[int, dict[str, Any]], list[_Range]]:
    """Split every file into owner ranges. Returns (all ranges, per-layer role info, non-expert ranges)."""
    ranges: list[_Range] = []
    nonexp: list[_Range] = []
    layers: dict[int, dict[str, Any]] = {}
    E = layout.n_expert
    for fl in layout.files:
        ranges.append(_Range(fl.index, 0, fl.data_offset, HEADER, None, "header"))
        for t in fl.tensors:
            m = _RE_EXPERT_ANY.match(t["name"])
            if m and E > 0 and t["shape"] and int(t["shape"][-1]) == E:
                il, base, suffix = int(m.group(1)), m.group(2), m.group(3)
                role = base if suffix == "weight" else f"{base}_{suffix}"
                if t["n_bytes"] % E:
                    raise ExpertMapError(f"{t['name']}: n_bytes {t['n_bytes']} not divisible by n_expert {E}")
                nb = t["n_bytes"] // E
                info = layers.setdefault(il, {"layer": il, "roles": {}})
                if role in info["roles"]:
                    raise ExpertMapError(f"duplicate expert role {role!r} in layer {il}")
                info["roles"][role] = {"tensor": t["name"], "type": t["type"], "shape": t["shape"],
                                       "file": fl.index, "offset": t["offset"], "n_bytes": t["n_bytes"],
                                       "nb_expert": nb}
                for e in range(E):
                    a = t["offset"] + e * nb
                    ranges.append(_Range(fl.index, a, a + nb, (il, e), t["name"], role, t["type"]))
            else:
                if t["n_bytes"] == 0:
                    continue
                r = _Range(fl.index, t["offset"], t["offset"] + t["n_bytes"], NONEXPERT, t["name"],
                           t["kind"], t["type"])
                ranges.append(r)
                nonexp.append(r)
    for info in layers.values():
        info["unit_bytes"] = sum(r["nb_expert"] for r in info["roles"].values())
    return ranges, layers, nonexp


@dataclass
class _PageIndex:
    """Exact page ownership for one page size: interior counts + owner sets of boundary pages."""

    page: int
    interior: dict[Any, int]                          # owner -> pages strictly inside its ranges
    boundary: dict[Any, set[tuple[int, int]]]         # owner -> its boundary pages (file, page)
    owners_at: dict[tuple[int, int], set[Any]]        # boundary page -> owners touching it

    def pages_for(self, owners: Iterable[Any]) -> int:
        owners = set(owners)
        inter = sum(self.interior.get(o, 0) for o in owners)
        b: set[tuple[int, int]] = set()
        for o in owners:
            b |= self.boundary.get(o, set())
        return inter + len(b)


def _page_index(ranges: Sequence[_Range], page: int) -> _PageIndex:
    interior: dict[Any, int] = {}
    boundary: dict[Any, set[tuple[int, int]]] = {}
    owners_at: dict[tuple[int, int], set[Any]] = {}
    for r in ranges:
        if r.end <= r.start:
            continue
        fp, lp = r.start // page, (r.end - 1) // page
        if lp - fp > 1:
            interior[r.owner] = interior.get(r.owner, 0) + (lp - fp - 1)
        bs = boundary.setdefault(r.owner, set())
        for p in {fp, lp}:
            key = (r.file, p)
            bs.add(key)
            owners_at.setdefault(key, set()).add(r.owner)
    return _PageIndex(page, interior, boundary, owners_at)


def _expert_page_stats(pi: _PageIndex, owner: tuple[int, int]) -> dict[str, int]:
    b = pi.boundary.get(owner, set())
    excl = sh_e = sh_n = 0
    for key in b:
        own = pi.owners_at[key]
        if own == {owner}:
            excl += 1
        if any(o != owner and o not in _NON_EXPERT_IDS for o in own):
            sh_e += 1
        if NONEXPERT in own or HEADER in own:
            sh_n += 1
    inter = pi.interior.get(owner, 0)
    return {"pages": inter + len(b), "exclusive": inter + excl, "shared_expert": sh_e, "shared_nonexpert": sh_n}


# --------------------------------------------------------------------------------------------
# ExpertMap document
# --------------------------------------------------------------------------------------------
class ExpertMap(Mapping[str, Any]):
    """The ``expert_map`` document (standalone file, or the ``expert_map`` key of a map.json)."""

    def __init__(self, data: Mapping[str, Any]) -> None:
        if data.get("format") != FORMAT:
            raise ExpertMapError(f"not a {FORMAT} document (format={data.get('format')!r})")
        if int(data.get("format_version", -1)) != FORMAT_VERSION:
            raise ExpertMapError(f"{FORMAT} format_version {data.get('format_version')} != {FORMAT_VERSION}")
        self._d = dict(data)
        self._by_key = {(int(x["layer"]), int(x["expert"])): x for x in self._d["experts"]}
        self._pi: dict[int, _PageIndex] = {}

    def __getitem__(self, k: str) -> Any:
        return self._d[k]

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._d)

    @property
    def files(self) -> list[dict[str, Any]]:
        return self._d["files"]

    def expert(self, layer: int, expert: int) -> dict[str, Any]:
        return self._by_key[(int(layer), int(expert))]

    def _ranges(self) -> list[_Range]:
        out: list[_Range] = []
        for f in self._d["files"]:
            out.append(_Range(int(f["index"]), 0, int(f["data_offset"]), HEADER, None, "header"))
        for r in self._d["nonexpert_ranges"]:
            # merged runs of consecutive non-expert tensors; their padding gaps hold no other owner
            out.append(_Range(int(r["file"]), int(r["offset"]), int(r["offset"]) + int(r["length"]), NONEXPERT, "run"))
        for x in self._d["experts"]:
            k = (int(x["layer"]), int(x["expert"]))
            for p in x["ranges"]:
                out.append(_Range(int(p["file"]), int(p["offset"]), int(p["offset"]) + int(p["n_bytes"]), k,
                                  p["tensor"], p["role"], p["type"]))
        return out

    def page_index(self, page: int) -> _PageIndex:
        if page not in self._pi:
            self._pi[page] = _page_index(self._ranges(), page)
        return self._pi[page]

    def pages_for(self, experts: Iterable[tuple[int, int]], page: int, with_nonexpert: bool = False) -> int:
        """Exact number of unique pages needed to hold ``experts`` (+ all non-expert data)."""
        owners: list[Any] = [(int(l), int(e)) for l, e in experts]
        for k in owners:
            if k not in self._by_key:
                raise KeyError(f"no expert {k}")
        if with_nonexpert:
            owners.append(NONEXPERT)
        return self.page_index(page).pages_for(owners)

    def ranges_for(self, experts: Iterable[tuple[int, int]], page: int,
                   with_nonexpert: bool = False) -> list[tuple[int, int, int]]:
        """Page-aligned, merged ``(file, offset, length)`` ranges covering ``experts`` (what a pinner locks)."""
        per_file: dict[int, list[tuple[int, int]]] = {}
        sizes = {int(f["index"]): int(f["size"]) for f in self._d["files"]}
        for l, e in experts:
            for p in self.expert(l, e)["ranges"]:
                fi = int(p["file"])
                per_file.setdefault(fi, []).append(gm.aligned(int(p["offset"]), int(p["n_bytes"]), page, sizes[fi]))
        if with_nonexpert:
            for r in self._d["nonexpert_ranges"]:
                fi = int(r["file"])
                per_file.setdefault(fi, []).append(gm.aligned(int(r["offset"]), int(r["length"]), page, sizes[fi]))
        out: list[tuple[int, int, int]] = []
        for fi in sorted(per_file):
            out.extend((fi, o, n) for o, n in gm.merge_ranges(per_file[fi]))
        return out


def load_expert_map(path: str | os.PathLike[str] | Mapping[str, Any]) -> ExpertMap:
    """Load a standalone expert-map JSON or a map.json that embeds one under ``expert_map``."""
    if isinstance(path, Mapping):
        data = path
    else:
        with open(os.fspath(path), encoding="utf-8") as fh:
            data = json.load(fh)
    if isinstance(data, Mapping) and data.get("format") != FORMAT and "expert_map" in data:
        data = data["expert_map"]
    return ExpertMap(data)


def _nonexpert_runs(layout: Layout, ranges: Sequence[_Range]) -> list[dict[str, Any]]:
    """Merge consecutive non-expert tensors of a file (padding between them owned by nobody)."""
    runs: list[dict[str, Any]] = []
    by_file: dict[int, list[_Range]] = {}
    for r in ranges:
        if r.tensor is not None:
            by_file.setdefault(r.file, []).append(r)
    for fi in sorted(by_file):
        cur: dict[str, Any] | None = None
        for r in sorted(by_file[fi], key=lambda x: x.start):
            if r.owner != NONEXPERT:
                if cur:
                    runs.append(cur)
                cur = None
                continue
            if cur is None:
                cur = {"file": fi, "offset": r.start, "length": 0, "n_tensors": 0, "first": r.tensor,
                       "last": r.tensor, "bytes_by_kind": {}}
            cur["length"] = r.end - cur["offset"]
            cur["n_tensors"] += 1
            cur["last"] = r.tensor
            cur["bytes_by_kind"][r.role] = cur["bytes_by_kind"].get(r.role, 0) + (r.end - r.start)
        if cur:
            runs.append(cur)
    return runs


def build_expert_map(layout: Layout, page_sizes: Sequence[int] = DEFAULT_PAGE_SIZES) -> ExpertMap:
    """Compute the per-expert ranges and page accounting for ``layout``."""
    if layout.n_expert <= 0:
        raise ExpertMapError("not a MoE model (expert_count is 0)")
    ranges, layers, _ = _expert_ranges(layout)
    if not layers:
        raise ExpertMapError("no per-expert tensors (blk.N.ffn_*_exps.*) found")
    # padding accounting: everything not owned is padding (sanity: < alignment per gap)
    total_size = sum(f.size for f in layout.files)
    owned = sum(r.end - r.start for r in ranges)
    pis = {P: _page_index(ranges, P) for P in page_sizes}
    E = layout.n_expert
    experts: list[dict[str, Any]] = []
    for il in sorted(layers):
        roles = layers[il]["roles"]
        for e in range(E):
            parts = []
            for role, ri in roles.items():
                parts.append({"file": ri["file"], "tensor": ri["tensor"], "role": role, "type": ri["type"],
                              "offset": ri["offset"] + e * ri["nb_expert"], "n_bytes": ri["nb_expert"]})
            parts.sort(key=lambda p: (p["file"], p["offset"]))
            experts.append({"layer": il, "expert": e, "bytes": sum(p["n_bytes"] for p in parts),
                            "ranges": parts,
                            "pages": {str(P): _expert_page_stats(pis[P], (il, e)) for P in page_sizes}})
    runs = _nonexpert_runs(layout, ranges)
    # ---- totals per page size
    totals: dict[str, Any] = {}
    exp_owners = [(x["layer"], x["expert"]) for x in experts]
    file_pages = {P: sum(-(-f.size // P) for f in layout.files) for P in page_sizes}
    for P in page_sizes:
        pi = pis[P]
        ue = pi.pages_for(exp_owners)
        un = pi.pages_for([NONEXPERT])
        hdr_pages = pi.pages_for([HEADER])
        mixed = sum(1 for own in pi.owners_at.values()
                    if (NONEXPERT in own or HEADER in own) and any(o not in _NON_EXPERT_IDS for o in own))
        multi_exp = sum(1 for own in pi.owners_at.values() if sum(1 for o in own if o not in _NON_EXPERT_IDS) > 1)
        per = [x["pages"][str(P)] for x in experts]
        nominal = sum(x["bytes"] for x in experts)
        totals[str(P)] = {
            "file_pages": file_pages[P],
            "expert_pages_union": ue,
            "expert_pages_sum_per_expert": sum(p["pages"] for p in per),
            "expert_pages_exclusive_sum": sum(p["exclusive"] for p in per),
            "pages_shared_by_2plus_experts": multi_exp,
            "pages_shared_expert_nonexpert": mixed,
            "nonexpert_pages_union": un,
            "header_pages": hdr_pages,
            "nonexpert_and_header_pages_union": pi.pages_for([NONEXPERT, HEADER]),
            "all_pages_union": pi.pages_for(exp_owners + [NONEXPERT, HEADER]),
            "per_expert_pages_mean": round(sum(p["pages"] for p in per) / len(per), 3),
            "per_expert_alignment_overhead_mean": round(
                sum(p["pages"] for p in per) * P / nominal - 1.0, 6) if nominal else None,
            "expert_bytes_nominal": nominal,
            "expert_bytes_aligned_union": ue * P,
            "expert_alignment_overhead": round(ue * P / nominal - 1.0, 6) if nominal else None,
            "nonexpert_bytes_aligned_union": un * P,
        }
    # ---- per-layer summary
    layer_out = []
    for il in sorted(layers):
        info = layers[il]
        ks = [(il, e) for e in range(E)]
        row: dict[str, Any] = {"layer": il, "unit_bytes": info["unit_bytes"], "roles": info["roles"], "pages": {}}
        for P in page_sizes:
            per = [pis[P].pages_for([k]) for k in ks]
            row["pages"][str(P)] = {"union": pis[P].pages_for(ks), "sum_per_expert": sum(per),
                                    "per_expert_min": min(per), "per_expert_max": max(per),
                                    "per_expert_mean": round(sum(per) / len(per), 3)}
        layer_out.append(row)
    # ---- non-expert summary + repack eligibility (static)
    kinds: dict[str, int] = {}
    types_nonexp: dict[str, int] = {}
    types_exp: dict[str, int] = {}
    for fl in layout.files:
        for t in fl.tensors:
            is_exp = bool(_RE_EXPERT_ANY.match(t["name"])) and int(t["shape"][-1]) == E
            if is_exp:
                types_exp[t["type"]] = types_exp.get(t["type"], 0) + t["n_bytes"]
            else:
                kinds[t["kind"]] = kinds.get(t["kind"], 0) + t["n_bytes"]
                types_nonexp[t["type"]] = types_nonexp.get(t["type"], 0) + t["n_bytes"]
    repack: dict[str, Any] = {}
    tied_head = None
    if not any(t["kind"] == "output" for fl in layout.files for t in fl.tensors):
        tied_head = next((t for fl in layout.files for t in fl.tensors if t["name"] == "token_embd.weight"), None)
    for plat in REPACK_PLATFORMS:
        by_kind: dict[str, int] = {}
        for fl in layout.files:
            for t in fl.tensors:
                if t["kind"] in _MATMUL_KINDS and repack_eligible(t["type"], t["shape"], plat):
                    by_kind[t["kind"]] = by_kind.get(t["kind"], 0) + t["n_bytes"]
        entry: dict[str, Any] = {"eligible_bytes_by_kind": by_kind,
                                 "eligible_bytes_total": sum(by_kind.values())}
        if tied_head is not None:
            entry["tied_output_head_duplicate_bytes"] = (
                tied_head["n_bytes"] if repack_eligible(tied_head["type"], tied_head["shape"], plat) else 0)
        repack[plat] = entry
    doc = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "generator": GENERATOR,
        "created_at": gm._utc_now(),
        "arch": layout.arch,
        "n_layer": layout.n_layer,
        "n_expert": E,
        "n_expert_used": layout.n_expert_used,
        "page_sizes": [int(P) for P in page_sizes],
        "files": [{"index": f.index, "label": f.label, "name": Path(str(f.label).replace("\\", "/")).name,
                   "size": f.size, "data_offset": f.data_offset, "alignment": f.alignment,
                   "split_no": f.split_no, "n_tensors": len(f.tensors),
                   "first_tensor": f.tensors[0]["name"] if f.tensors else None,
                   "last_tensor": f.tensors[-1]["name"] if f.tensors else None} for f in layout.files],
        "conventions": {
            "offsets": "file-relative byte offsets (files[].index); pages are per file",
            "expert_slice": "offset + e*nb_expert, nb_expert = n_bytes / n_expert (expert = outermost ggml dim)",
            "pages": "unique pages of the union of the expert's slices",
            "exclusive": "pages touched by no other expert, non-expert tensor or header",
            "shared_expert": "pages also touched by another expert",
            "shared_nonexpert": "pages also touched by a non-expert tensor or a header",
            "expert_scale": "per-expert scale vectors (ffn_*_exps.scale) are sliced per expert here; "
                            "pigemma.gguf_map counts them as non-expert kind 'expert_scale'",
            "repack": f"static eligibility from ggml/src/ggml-cpu/repack.cpp @ {LLAMA_CPP_COMMIT}; "
                      "applies only with repack on (llama.cpp default), never with -nr; not measured",
        },
        "summary": {
            "file_bytes_total": total_size,
            "owned_bytes": owned,
            "padding_bytes": total_size - owned,
            "header_bytes": sum(f.data_offset for f in layout.files),
            "expert_bytes": sum(x["bytes"] for x in experts),
            "nonexpert_tensor_bytes": sum(kinds.values()),
            "nonexpert_bytes_by_kind": kinds,
            "expert_bytes_by_type": types_exp,
            "nonexpert_bytes_by_type": types_nonexp,
            "n_moe_layers": len(layers),
            "expert_unit_bytes_min": min(x["bytes"] for x in experts),
            "expert_unit_bytes_max": max(x["bytes"] for x in experts),
            "expert_bytes_per_token_worst": layout.n_expert_used * sum(i["unit_bytes"] for i in layers.values()),
            "n_nonexpert_runs": len(runs),
        },
        "page_totals": totals,
        "repack_eligibility": repack,
        "layers": layer_out,
        "nonexpert_ranges": runs,
        "experts": experts,
    }
    em = ExpertMap(doc)
    em._pi.update(pis)
    return em


# --------------------------------------------------------------------------------------------
# Serialisation (one expert / one range per line keeps a 2 MB file diffable and greppable)
# --------------------------------------------------------------------------------------------
def dumps(doc: Mapping[str, Any]) -> str:
    """JSON with indent 1, but every entry of the ``experts`` / ``nonexpert_ranges`` lists on one line."""
    compact_keys = ("experts", "nonexpert_ranges")
    holders: dict[str, str] = {}

    def subst(obj: Any) -> Any:
        if isinstance(obj, Mapping):
            out = {}
            for k, v in obj.items():
                if k in compact_keys and isinstance(v, list):
                    lst = []
                    for item in v:
                        tok = f"@@C{len(holders)}@@"
                        holders[tok] = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                        lst.append(tok)
                    out[k] = lst
                else:
                    out[k] = subst(v)
            return out
        if isinstance(obj, list):
            return [subst(x) for x in obj]
        return obj

    text = json.dumps(subst(doc), indent=1, ensure_ascii=False)
    return re.sub(r'"(@@C\d+@@)"', lambda m: holders[m.group(1)], text) + "\n"


def write_json(doc: Mapping[str, Any], path: str | os.PathLike[str]) -> Path:
    p = Path(os.fspath(path))
    gm._atomic_write_text(p, dumps(doc))
    return p


def format_summary(em: ExpertMap) -> str:
    s = em["summary"]
    lines = [f"files        {len(em.files)}: " + ", ".join(f"{f['name']} ({f['size']:,} B)" for f in em.files),
             f"experts      {em['n_expert']} x {s['n_moe_layers']} MoE layers, top-{em['n_expert_used']}; "
             f"unit {s['expert_unit_bytes_min']:,}..{s['expert_unit_bytes_max']:,} B; "
             f"worst/token {s['expert_bytes_per_token_worst']:,} B",
             f"bytes        expert {s['expert_bytes']:,}  non-expert tensors {s['nonexpert_tensor_bytes']:,}  "
             f"header {s['header_bytes']:,}  padding {s['padding_bytes']:,}",
             f"expert types {s['expert_bytes_by_type']}"]
    for P, t in em["page_totals"].items():
        lines.append(f"page {P:>5}  expert union {t['expert_pages_union']:,} pages "
                     f"({t['expert_bytes_aligned_union']:,} B, +{100 * (t['expert_alignment_overhead'] or 0):.3f} %), "
                     f"sum per expert {t['expert_pages_sum_per_expert']:,} (per-expert mean "
                     f"{t['per_expert_pages_mean']} pages, +{100 * (t['per_expert_alignment_overhead_mean'] or 0):.2f} %), "
                     f"shared by >=2 experts "
                     f"{t['pages_shared_by_2plus_experts']:,}, expert/non-expert mixed {t['pages_shared_expert_nonexpert']:,}, "
                     f"non-expert union {t['nonexpert_pages_union']:,} ({t['nonexpert_bytes_aligned_union']:,} B)")
    for plat, r in em["repack_eligibility"].items():
        lines.append(f"repack {plat:<10} eligible {r['eligible_bytes_total']:,} B {r['eligible_bytes_by_kind']}"
                     + (f", tied-head duplicate {r['tied_output_head_duplicate_bytes']:,} B"
                        if "tied_output_head_duplicate_bytes" in r else ""))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def _int_list(s: str) -> list[int]:
    try:
        return [int(x) for x in s.split(",") if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {s!r}") from None


def _expert_list(s: str) -> list[tuple[int, int]]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        m = re.fullmatch(r"(\d+):(\d+)", tok)
        if not m:
            raise argparse.ArgumentTypeError(f"expert must be LAYER:EXPERT, got {tok!r}")
        out.append((int(m.group(1)), int(m.group(2))))
    return out


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m pigemma.expert_map", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="map one GGUF or all parts of a split GGUF")
    b.add_argument("gguf", nargs="+")
    b.add_argument("--out", required=True)
    b.add_argument("--page-sizes", type=_int_list, default=list(DEFAULT_PAGE_SIZES))
    b.add_argument("--header-sizes", type=_int_list, default=None,
                   help="inputs are header-only prefixes of files with these real sizes")
    b.add_argument("--labels", default=None, help="comma-separated labels (real paths) for the inputs")
    b.add_argument("--embed-base-map", action="store_true",
                   help="single file: write the gguf_map map.json with this result under 'expert_map'")
    b.add_argument("--provenance", default=None, help="JSON file merged into the output's 'provenance'")
    s = sub.add_parser("summary", help="print a map's summary")
    s.add_argument("--map", required=True)
    p = sub.add_parser("pages", help="unique pages of a set of experts")
    p.add_argument("--map", required=True)
    p.add_argument("--experts", type=_expert_list, required=True, help="LAYER:EXPERT,...")
    p.add_argument("--page-size", type=int, default=16384)
    p.add_argument("--with-nonexpert", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        args = _parser().parse_args(argv)
    except SystemExit as e:
        return int(e.code or 0)
    try:
        if args.cmd == "build":
            labels = args.labels.split(",") if args.labels else None
            if args.header_sizes is not None and len(args.header_sizes) != len(args.gguf):
                print("--header-sizes needs one size per input", file=sys.stderr)
                return 2
            hdrs = read_headers(args.gguf, args.header_sizes)
            layout = build_layout(hdrs, labels)
            em = build_expert_map(layout, args.page_sizes)
            prov = None
            if args.provenance:
                with open(args.provenance, encoding="utf-8") as fh:
                    prov = json.load(fh)
            if args.embed_base_map:
                if len(args.gguf) != 1 or args.header_sizes is not None:
                    print("--embed-base-map needs exactly one complete GGUF file", file=sys.stderr)
                    return 2
                base = gm.build_map(args.gguf[0], max(args.page_sizes), backend="stdlib").to_dict()
                doc: dict[str, Any] = dict(base)
                doc["expert_map"] = em.to_dict()
            else:
                doc = em.to_dict()
            if prov is not None:
                doc["provenance"] = prov
            write_json(doc, args.out)
            print(format_summary(em))
            print(f"wrote {args.out}")
            return 0
        if args.cmd == "summary":
            print(format_summary(load_expert_map(args.map)))
            return 0
        if args.cmd == "pages":
            em = load_expert_map(args.map)
            n = em.pages_for(args.experts, args.page_size, args.with_nonexpert)
            print(json.dumps({"experts": len(args.experts), "page_size": args.page_size,
                              "with_nonexpert": args.with_nonexpert, "pages": n, "bytes": n * args.page_size}))
            return 0
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    except (ExpertMapError, gm.GgufMapError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
