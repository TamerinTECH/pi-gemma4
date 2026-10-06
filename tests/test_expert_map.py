"""tests/test_expert_map.py - pigemma.expert_map: per-expert byte ranges, exact page accounting,
split GGUF parts, header-only prefixes, repack eligibility rules, JSON/CLI.

Every page statistic is checked against an independent brute force (explicit per-page owner sets
built from the GGUF writer's own offsets), at page sizes small enough that experts share pages in
every possible way. No model files needed except the integration test at the end.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import synth
from pigemma import expert_map as xm
from pigemma import gguf_map as gm

import gguf  # noqa: E402  (pigemma puts gguf-py on sys.path)

REPO = Path(__file__).resolve().parents[1]
PAGES = (32, 64, 96, 256, 1024, 4096)


# ------------------------------------------------------------------------------------- helpers
def _tiny(tmp_path: Path, name: str = "tiny.gguf", **kw: Any) -> synth.TinyGguf:
    kw.setdefault("n_expert", 8)
    kw.setdefault("n_expert_used", 2)
    kw.setdefault("n_ff", 64)
    return synth.write_tiny_gguf(tmp_path / name, **kw)


def _single_map(g: synth.TinyGguf, pages=PAGES) -> xm.ExpertMap:
    hdr = gm.read_gguf_header(Path(g), "stdlib")
    return xm.build_expert_map(xm.build_layout([hdr]), pages)


def _writer_owner_intervals(g: synth.TinyGguf) -> list[tuple[int, int, int, Any]]:
    """(file, start, end, owner) derived only from the writer's view (independent of expert_map)."""
    out: list[tuple[int, int, int, Any]] = [(0, 0, g.data_offset, "H")]
    E = g.n_expert
    for t in g.tensors.values():
        if "_exps." in t.name and t.shape[-1] == E:
            il = int(t.name.split(".")[1])
            nb = t.n_bytes // E
            for e in range(E):
                out.append((0, t.offset + e * nb, t.offset + (e + 1) * nb, (il, e)))
        else:
            out.append((0, t.offset, t.offset + t.n_bytes, "N"))
    return out


def _brute(intervals: list[tuple[int, int, int, Any]], P: int) -> dict[tuple[int, int], set[Any]]:
    owners: dict[tuple[int, int], set[Any]] = {}
    for f, a, b, o in intervals:
        for p in range(a // P, (b - 1) // P + 1):
            owners.setdefault((f, p), set()).add(o)
    return owners


def _brute_stats(owners: dict[tuple[int, int], set[Any]], k: tuple[int, int]) -> dict[str, int]:
    mine = [own for own in owners.values() if k in own]
    return {
        "pages": len(mine),
        "exclusive": sum(1 for own in mine if own == {k}),
        "shared_expert": sum(1 for own in mine if any(o != k and o not in ("N", "H") for o in own)),
        "shared_nonexpert": sum(1 for own in mine if "N" in own or "H" in own),
    }


# ------------------------------------------------------------------------------------- ranges
def test_ranges_match_writer_and_bytes(tmp_path: Path) -> None:
    g = _tiny(tmp_path)
    em = _single_map(g)
    E = g.n_expert
    data = Path(g).read_bytes()
    assert em["n_expert"] == E and len(em["experts"]) == g.n_layer * E
    for il in range(g.n_layer):
        gu = g.tensor(f"blk.{il}.ffn_gate_up_exps.weight")
        dn = g.tensor(f"blk.{il}.ffn_down_exps.weight")
        sc = g.tensor(f"blk.{il}.ffn_down_exps.scale")
        for e in range(E):
            x = em.expert(il, e)
            by_role = {r["role"]: r for r in x["ranges"]}
            assert set(by_role) == {"gate_up", "down", "down_scale"}
            for role, t in (("gate_up", gu), ("down", dn), ("down_scale", sc)):
                r = by_role[role]
                nb = t.n_bytes // E
                assert (r["offset"], r["n_bytes"], r["type"], r["tensor"]) == (t.offset + e * nb, nb, t.type, t.name)
                # the slice in the file is exactly the writer's stored data of expert e
                assert data[r["offset"]:r["offset"] + nb] == np.ascontiguousarray(t.stored[e]).tobytes()
            assert x["bytes"] == sum(r["n_bytes"] for r in x["ranges"])
            assert [r["offset"] for r in x["ranges"]] == sorted(r["offset"] for r in x["ranges"])
    s = em["summary"]
    assert s["file_bytes_total"] == g.file_size
    assert s["expert_bytes"] + s["nonexpert_tensor_bytes"] + s["header_bytes"] + s["padding_bytes"] == g.file_size
    assert 0 <= s["padding_bytes"] < 32 * len(g.tensors)
    # gguf_map counts the scale vectors as non-expert 'expert_scale'; expert_map slices them per expert
    base = gm.build_map(Path(g), 4096, backend="stdlib", with_sha=False).summary
    assert s["expert_bytes"] == base["expert_bytes"] + base["bytes_per_kind"]["expert_scale"]


def test_nonexpert_runs_cover_all_nonexpert_tensors(tmp_path: Path) -> None:
    g = _tiny(tmp_path)
    em = _single_map(g)
    runs = em["nonexpert_ranges"]
    for t in g.tensors.values():
        if "_exps." in t.name:
            continue
        assert any(r["offset"] <= t.offset and t.offset + t.n_bytes <= r["offset"] + r["length"] for r in runs), t.name
    for x in em["experts"]:  # no run overlaps an expert slice
        for p in x["ranges"]:
            assert all(p["offset"] + p["n_bytes"] <= r["offset"] or r["offset"] + r["length"] <= p["offset"]
                       for r in runs)
    assert sum(sum(r["bytes_by_kind"].values()) for r in runs) == em["summary"]["nonexpert_tensor_bytes"]


# ------------------------------------------------------------------------------------- pages
@pytest.mark.parametrize("alignment", [None, 64])
def test_page_stats_match_bruteforce(tmp_path: Path, alignment: int | None) -> None:
    g = _tiny(tmp_path, alignment=alignment)
    em = _single_map(g)
    iv = _writer_owner_intervals(g)
    for P in PAGES:
        owners = _brute(iv, P)
        for x in em["experts"]:
            k = (x["layer"], x["expert"])
            assert x["pages"][str(P)] == _brute_stats(owners, k), (P, k)
        t = em["page_totals"][str(P)]
        exp_pages = [own for own in owners.values() if any(o not in ("N", "H") for o in own)]
        assert t["expert_pages_union"] == len(exp_pages)
        assert t["nonexpert_pages_union"] == sum(1 for own in owners.values() if "N" in own)
        assert t["header_pages"] == sum(1 for own in owners.values() if "H" in own)
        assert t["all_pages_union"] == len(owners)
        assert t["pages_shared_by_2plus_experts"] == sum(
            1 for own in owners.values() if sum(1 for o in own if o not in ("N", "H")) > 1)
        assert t["pages_shared_expert_nonexpert"] == sum(
            1 for own in exp_pages if "N" in own or "H" in own)
        assert t["expert_pages_sum_per_expert"] == sum(x["pages"][str(P)]["pages"] for x in em["experts"])
        assert t["file_pages"] == -(-g.file_size // P)
        for row in em["layers"]:
            ks = {(row["layer"], e) for e in range(g.n_expert)}
            assert row["pages"][str(P)]["union"] == sum(1 for own in owners.values() if own & ks)


def test_pages_for_sets_and_reload_agree(tmp_path: Path) -> None:
    g = _tiny(tmp_path)
    em = _single_map(g)
    iv = _writer_owner_intervals(g)
    out = tmp_path / "em.json"
    xm.write_json(em.to_dict(), out)
    em2 = xm.load_expert_map(out)  # rebuilt from the JSON (merged non-expert runs, not tensors)
    keys = [(x["layer"], x["expert"]) for x in em["experts"]]
    rng = random.Random(7)
    for P in PAGES:
        owners = _brute(iv, P)
        for _ in range(25):
            S = set(rng.sample(keys, rng.randint(1, len(keys))))
            want = sum(1 for own in owners.values() if own & S)
            want_n = sum(1 for own in owners.values() if own & S or "N" in own)
            assert em.pages_for(S, P) == em2.pages_for(S, P) == want
            assert em.pages_for(S, P, with_nonexpert=True) == em2.pages_for(S, P, with_nonexpert=True) == want_n
    with pytest.raises(KeyError):
        em.pages_for([(0, 99)], 4096)


def test_ranges_for_is_aligned_merged_and_covers(tmp_path: Path) -> None:
    g = _tiny(tmp_path)
    em = _single_map(g)
    for P in (64, 4096):
        S = [(0, 1), (0, 2), (1, 7)]
        rr = em.ranges_for(S, P)
        assert all(o % P == 0 and (n % P == 0 or o + n == g.file_size) for _, o, n in rr)
        assert all(a[1] + a[2] < b[1] for a, b in zip(rr, rr[1:]))  # sorted, merged (non-touching)
        for k in S:
            for p in em.expert(*k)["ranges"]:
                assert any(o <= p["offset"] and p["offset"] + p["n_bytes"] <= o + n for _, o, n in rr)
        assert sum(n for _, _, n in rr) <= em.pages_for(S, P) * P


# ------------------------------------------------------------------------------------- split parts
def _write_part(path: Path, g: synth.TinyGguf, names: list[str], split_no: int, split_count: int,
                n_total: int, with_kv: bool) -> None:
    w = gguf.GGUFWriter(str(path), g.arch if with_kv else None)  # type: ignore[arg-type]
    if with_kv:
        for k, v in g.kv.items():
            if k == "general.architecture":
                continue
            if isinstance(v, bool):
                w.add_bool(k, v)
            elif isinstance(v, int):
                w.add_uint32(k, v)
            elif isinstance(v, float):
                w.add_float32(k, v)
            elif isinstance(v, str):
                w.add_string(k, v)
            else:
                w.add_array(k, list(v))
    w.add_uint16("split.no", split_no)
    w.add_uint16("split.count", split_count)
    w.add_int32("split.tensors.count", n_total)
    for n in names:
        t = g.tensor(n)
        if t.type in ("F32", "F16"):
            w.add_tensor(n, t.stored)
        else:
            w.add_tensor(n, t.stored, raw_dtype=gguf.GGMLQuantizationType[t.type])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _split(tmp_path: Path, g: synth.TinyGguf, cut: int) -> list[Path]:
    names = [t.name for t in sorted(g.tensors.values(), key=lambda t: t.offset)]
    parts = [names[:cut], names[cut:]]
    out = []
    for i, ns in enumerate(parts):
        p = tmp_path / f"tiny-0000{i + 1}-of-00002.gguf"
        _write_part(p, g, ns, i, 2, len(names), with_kv=(i == 0))
        out.append(p)
    return out


def test_split_parts_map_to_identical_bytes(tmp_path: Path) -> None:
    g = _tiny(tmp_path)
    single = _single_map(g, (64, 4096))
    names = [t.name for t in sorted(g.tensors.values(), key=lambda t: t.offset)]
    cut = names.index("blk.1.ffn_gate_up_exps.weight") + 1  # layer 1's expert tensors end up in both files
    parts = _split(tmp_path, g, cut)
    # give the parts in reverse order: build_layout must sort by split.no
    hdrs = xm.read_headers(list(reversed(parts)))
    lay = xm.build_layout(hdrs, [str(p) for p in reversed(parts)])
    assert [f.split_no for f in lay.files] == [0, 1] and lay.n_expert == g.n_expert
    em = xm.build_expert_map(lay, (64, 4096))
    blobs = [p.read_bytes() for p in parts]
    src = Path(g).read_bytes()
    for x in em["experts"]:
        y = single.expert(x["layer"], x["expert"])
        assert [(r["role"], r["n_bytes"]) for r in sorted(x["ranges"], key=lambda r: r["role"])] == \
               [(r["role"], r["n_bytes"]) for r in sorted(y["ranges"], key=lambda r: r["role"])]
        for r in x["ranges"]:
            ry = next(q for q in y["ranges"] if q["role"] == r["role"])
            assert blobs[r["file"]][r["offset"]:r["offset"] + r["n_bytes"]] == src[ry["offset"]:ry["offset"] + ry["n_bytes"]]
    files_used = {(x["layer"], r["role"]): r["file"] for x in em["experts"] for r in x["ranges"]}
    for (il, role), fi in files_used.items():
        tname = next(r["tensor"] for r in em.expert(il, 0)["ranges"] if r["role"] == role)
        assert fi == (0 if names.index(tname) < cut else 1), (il, role)
    assert len({fi for (il, _), fi in files_used.items() if il == 1}) == 2  # layer 1 really is split
    # page stats are per file: brute force on both parts
    iv: list[tuple[int, int, int, Any]] = []
    for fl in lay.files:
        iv.append((fl.index, 0, fl.data_offset, "H"))
        for t in fl.tensors:
            if "_exps." in t["name"] and t["shape"][-1] == g.n_expert:
                il, nb = int(t["name"].split(".")[1]), t["n_bytes"] // g.n_expert
                iv += [(fl.index, t["offset"] + e * nb, t["offset"] + (e + 1) * nb, (il, e)) for e in range(g.n_expert)]
            else:
                iv.append((fl.index, t["offset"], t["offset"] + t["n_bytes"], "N"))
    for P in (64, 4096):
        owners = _brute(iv, P)
        for x in em["experts"]:
            assert x["pages"][str(P)] == _brute_stats(owners, (x["layer"], x["expert"]))
    assert em["summary"]["file_bytes_total"] == sum(len(b) for b in blobs)


def test_split_set_errors(tmp_path: Path) -> None:
    g = _tiny(tmp_path)
    parts = _split(tmp_path, g, 20)
    with pytest.raises(xm.ExpertMapError, match="split.no values"):  # the same part twice
        xm.build_layout(xm.read_headers([parts[0], parts[0]]))
    with pytest.raises(xm.ExpertMapError, match="split.tensors.count"):  # a missing part
        xm.build_layout(xm.read_headers(parts[:1]))
    with pytest.raises(xm.ExpertMapError, match="split.count"):  # more files than split.count
        xm.build_layout(xm.read_headers([parts[0], parts[1], parts[1]]))
    with pytest.raises(xm.ExpertMapError, match="not a split"):
        xm.build_layout(xm.read_headers([Path(g), Path(g)]))
    hdrs = xm.read_headers(parts)
    with pytest.raises(xm.ExpertMapError):  # tensor from part 0 duplicated into part 1
        hdrs[1].tensors.append(hdrs[0].tensors[0])
        xm.build_layout(hdrs)
    with pytest.raises(gm.MissingKeyError):  # only the second part (no general.architecture)
        xm.build_layout(xm.read_headers(parts[1:]))


def test_header_only_prefix_equals_full_file(tmp_path: Path) -> None:
    g = _tiny(tmp_path)
    full = _single_map(g, (64, 4096)).to_dict()
    prefix = tmp_path / "prefix.hdr"
    prefix.write_bytes(Path(g).read_bytes()[:g.data_offset + 7])  # header + a few data bytes
    hdrs = xm.read_headers([prefix], [g.file_size])
    pre = xm.build_expert_map(xm.build_layout(hdrs, [str(Path(g))]), (64, 4096)).to_dict()
    for d in (full, pre):
        d.pop("created_at")
        for f in d["files"]:
            f.pop("label"), f.pop("name")
    assert pre == full


def test_dense_model_is_rejected(tmp_path: Path) -> None:
    g = synth.write_tiny_dense_gguf(tmp_path / "dense.gguf")
    with pytest.raises(xm.ExpertMapError, match="not a MoE"):
        xm.build_expert_map(xm.build_layout(xm.read_headers([Path(g)])))


# ------------------------------------------------------------------------------------- repack rules
@pytest.mark.parametrize("typ,shape,plat,want", [
    ("IQ4_NL", [704, 2816, 128], "x86_avx512", True),
    ("IQ4_NL", [704, 2812, 128], "x86_avx512", False),    # ne[1] % 8 != 0
    ("IQ4_NL", [704, 2812, 128], "arm_a76", True),        # dotprod rule needs ne[1] % 4
    ("IQ3_XXS", [2816, 1408, 128], "x86_avx512", False),
    ("IQ3_XXS", [2816, 1408, 128], "arm_a76", False),
    ("Q8_0", [2816, 2048], "x86_avx512", False),
    ("Q8_0", [2816, 2048], "arm_a76", True),
    ("Q6_K", [2816, 262144], "x86_avx512", False),
    ("Q6_K", [2816, 262144], "arm_a76", True),
    ("Q4_0", [2816, 1408, 128], "x86_avx512", True),
    ("Q4_0", [2816, 1412, 128], "x86_avx512", False),
    ("Q4_0", [2816, 1412, 128], "arm_a76", True),
    ("Q3_K", [2816, 2048], "arm_a76", False),
    ("Q2_K", [2816, 2048], "x86_avx512", True),
    ("Q2_K", [2816, 2048], "arm_a76", False),
    ("IQ4_XS", [2816, 1408, 128], "arm_a76", False),
    ("F32", [2816, 128], "arm_a76", False),
])
def test_repack_rules(typ: str, shape: list[int], plat: str, want: bool) -> None:
    assert xm.repack_eligible(typ, shape, plat) is want


def test_repack_summary_counts_matmul_weights_only(tmp_path: Path) -> None:
    g = _tiny(tmp_path, qtype="Q4_0")
    em = _single_map(g, (4096,))
    r = em["repack_eligibility"]["x86_avx512"]
    want = sum(t.n_bytes for t in g.tensors.values()
               if t.kind in ("attn", "ffn_dense", "expert") and xm.repack_eligible(t.type, t.shape, "x86_avx512"))
    assert r["eligible_bytes_total"] == want > 0
    # tied head (no output.weight): F16 token_embd is never repacked
    assert r["tied_output_head_duplicate_bytes"] == 0


# ------------------------------------------------------------------------------------- JSON / CLI
def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "pigemma.expert_map", *args], cwd=REPO,
                          capture_output=True, text=True, timeout=120)


def test_cli_build_embed_summary_pages(tmp_path: Path) -> None:
    g = _tiny(tmp_path)
    prov = tmp_path / "prov.json"
    prov.write_text(json.dumps({"sha256": "x" * 64}))
    out = tmp_path / "map.json"
    p = _cli("build", str(Path(g)), "--out", str(out), "--page-sizes", "64,4096", "--embed-base-map",
             "--provenance", str(prov))
    assert p.returncode == 0, p.stderr
    base = gm.load_map(out)  # still a valid gguf_map map.json
    assert base.n_expert == g.n_expert and base.page_size == 4096
    doc = json.loads(out.read_text())
    assert doc["provenance"] == {"sha256": "x" * 64}
    em = xm.load_expert_map(out)
    assert em.pages_for([(0, 0)], 64) == em.expert(0, 0)["pages"]["64"]["pages"]
    # one expert per line
    lines = out.read_text().splitlines()
    assert sum(1 for ln in lines if ln.lstrip().startswith('{"layer":')) == len(em["experts"])
    p = _cli("summary", "--map", str(out))
    assert p.returncode == 0 and "expert union" in p.stdout
    p = _cli("pages", "--map", str(out), "--experts", "0:0,0:1", "--page-size", "64")
    assert p.returncode == 0, p.stderr
    r = json.loads(p.stdout.strip().splitlines()[-1])
    assert r["pages"] == em.pages_for([(0, 0), (0, 1)], 64) and r["bytes"] == r["pages"] * 64


def test_cli_errors(tmp_path: Path) -> None:
    assert _cli("build", str(tmp_path / "missing.gguf"), "--out", str(tmp_path / "x.json")).returncode == 3
    g = _tiny(tmp_path)
    assert _cli("build", str(Path(g)), str(Path(g)), "--out", str(tmp_path / "x.json"),
                "--embed-base-map").returncode == 1  # two non-split files
    assert _cli("build", str(Path(g)), "--out", str(tmp_path / "x.json"), "--header-sizes", "1,2").returncode == 2
    assert _cli("pages", "--map", str(tmp_path / "x.json"), "--experts", "bad").returncode == 2
    with pytest.raises(xm.ExpertMapError):
        xm.load_expert_map({"format": "something-else"})


# ------------------------------------------------------------------------------------- real file
UD = REPO / "models/unsloth__gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf"


@pytest.mark.integration
def test_real_ud_q3_k_m_layout() -> None:
    if not UD.is_file() or UD.stat().st_size != 12728497888:
        pytest.skip("UD-Q3_K_M not present")
    em = xm.build_expert_map(xm.build_layout(xm.read_headers([UD])), (4096, 16384))
    s = em["summary"]
    assert (em["n_expert"], em["n_expert_used"], s["n_moe_layers"]) == (128, 8, 30)
    assert s["expert_bytes_by_type"] == {"F32": 15360, "IQ4_NL": 4139384832, "IQ3_XXS": 5634162688,
                                         "Q8_0": 269615104, "IQ4_XS": 269615104}
    assert em.expert(0, 0)["bytes"] == 2632964 and em.expert(29, 0)["bytes"] == 4212740
    assert s["expert_bytes"] + s["nonexpert_tensor_bytes"] + s["header_bytes"] + s["padding_bytes"] == 12728497888
    x = em.expert(0, 0)
    assert x["pages"]["4096"] == {"pages": 644, "exclusive": 640, "shared_expert": 3, "shared_nonexpert": 2}
