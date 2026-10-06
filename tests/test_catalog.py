"""Unit tests for pigemma/data/catalog.json (the data-driven model catalog + device profiles).

Covers the manifest test field for 'pigemma/data/catalog.json':
  * schema: required keys, unique ids, draft cross-references, strict JSON;
  * sources enumerate: every source string starts with a kind from catalog["source_kinds"], every number has
    a source, and every source that is not 'measured' contains the word 'assumed';
  * model bytes for the 5 local files equal pigemma.paths.MODELS (skipped until pigemma/paths.py exists) and the
    on-disk file sizes (skipped per file when the model is not downloaded);
  * merge of a measured io_mem fixture through pigemma.feasibility.load_catalog() overrides the device values and
    sets source 'measured' (skipped until pigemma/feasibility.py exists).
Plus byte-arithmetic invariants (header + tensors <= file, expert totals vs per-token worst case) and the
spec-pinned constants that docs and projections quote.

Runs on Windows and on the Pi without models: stdlib + pytest only.
"""
from __future__ import annotations

import copy
import fnmatch
import json
import math
import re
from pathlib import Path
from typing import Any, Iterator

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "pigemma" / "data" / "catalog.json"

# Exact sizes of the five files downloaded on the desktop (manifest: pigemma/paths.py MODELS).
LOCAL_BYTES = {
    "26b-q4": 14439363584,
    "31b-q4": 17651001568,
    "26b-mtp": 251939520,
    "26b-dflash": 472433824,
    "31b-mtp": 279955968,
}

# Spec-pinned device profile values (manifest files[] 'pigemma/data/catalog.json').
# name -> (ram_bytes, os_reserve_bytes, compute_buffers_bytes, n_ctx, mem_bw_gbps, storage type, seq, rand_3mb, access_ms)
PI_DEVICE_SPEC = {
    "pi5-4gb-microsd": (4.0e9, 0.7e9, 0.3e9, 2048, 10.0, "microsd", 85, 60, 2.0),
    "pi5-8gb-usb3": (8.0e9, 0.8e9, 0.3e9, 4096, 10.0, "usb-ssd", 380, 350, 0.5),
    "pi5-8gb-nvme-gen3": (8.0e9, 0.8e9, 0.3e9, 4096, 10.0, "nvme", 850, 700, 0.2),
    "pi5-16gb-nvme-gen3": (16.0e9, 0.9e9, 0.4e9, 4096, 10.0, "nvme", 850, 700, 0.2),
}
REQUIRED_DEVICES = set(PI_DEVICE_SPEC) | {"desktop-8400f"}

REQUIRED_TOP = {"schema_version", "source_kinds", "models", "devices", "selection", "protected_hosts",
                "protected_service_globs", "storage_classes"}
REQUIRED_MODEL = {"id", "family", "model", "kind", "quant", "repo", "file", "bytes", "bytes_exact", "estimated",
                  "nonexpert_bytes", "expert_bytes", "expert_unit_bytes", "expert_bytes_per_token_worst",
                  "nonexpert_bytes_per_token", "n_moe_layers", "n_expert", "n_expert_used", "drafts", "sources"}
REQUIRED_DEVICE = {"name", "ram_bytes", "os_reserve_bytes", "compute_buffers_bytes", "n_ctx", "mem_bw_gbps",
                   "storage", "sources"}
STORAGE_METRICS = ("seq_read_mbps", "rand_read_3mb_mbps", "access_ms")
STATIC_SOURCE_KINDS = {"measured", "assumed"}  # 'simulated' only appears after runtime merges of sim.json
GGUF_ALIGNMENT = 32
PLACEHOLDER = "<fill from HF listing>"
KIND_RE = re.compile(r"^([a-z]+)\b")


# ----------------------------------------------------------------------------------------------- helpers
def _strict_load(path: Path) -> Any:
    """Load JSON as UTF-8, rejecting NaN/Infinity and duplicate keys."""

    def no_dupes(pairs: list[tuple[str, Any]]) -> dict:
        keys = [k for k, _ in pairs]
        dupes = {k for k in keys if keys.count(k) > 1}
        if dupes:
            raise ValueError(f"duplicate keys: {sorted(dupes)}")
        return dict(pairs)

    def bad_constant(name: str) -> Any:
        raise ValueError(f"non-finite constant {name} in catalog")

    text = path.read_bytes().decode("utf-8")
    return json.loads(text, object_pairs_hook=no_dupes, parse_constant=bad_constant)


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _iter_sources(node: Any, where: str = "$") -> Iterator[tuple[str, str]]:
    """Yield (json-path, source string) for every 'source' string and every value of a 'sources' dict."""
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{where}.{k}"
            if k == "source":
                assert isinstance(v, str) and v.strip(), f"{p}: source must be a non-empty string"
                yield p, v
            elif k == "sources":
                assert isinstance(v, dict), f"{p}: 'sources' must be an object"
                for fk, fv in v.items():
                    assert isinstance(fv, str) and fv.strip(), f"{p}.{fk}: source must be a non-empty string"
                    yield f"{p}.{fk}", fv
            else:
                yield from _iter_sources(v, p)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _iter_sources(v, f"{where}[{i}]")


def _unsourced_numbers(node: Any, where: str = "$", top: bool = True) -> list[str]:
    """Return json-paths of numbers (or numeric lists) that carry no provenance.

    A number is sourced when its dict has an own 'source' string (dict-level or the {value, source} form) or when
    the dict's sibling 'sources' object has an entry for the field.
    """
    missing: list[str] = []
    if isinstance(node, dict):
        own = isinstance(node.get("source"), str)
        srcs = node.get("sources") if isinstance(node.get("sources"), dict) else {}
        for k, v in node.items():
            p = f"{where}.{k}"
            if k in ("source", "sources"):
                continue
            if top and k == "schema_version":
                continue  # file-format metadata, not a quantity
            numeric = _is_num(v) or (isinstance(v, list) and v and all(
                _is_num(x) or (isinstance(x, list) and all(_is_num(y) for y in x)) for x in v))
            if numeric:
                if not (own or k in srcs):
                    missing.append(p)
            elif isinstance(v, (dict, list)):
                missing.extend(_unsourced_numbers(v, p, top=False))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            if isinstance(v, (dict, list)):
                missing.extend(_unsourced_numbers(v, f"{where}[{i}]", top=False))
    return missing


def _devices(cat: Any) -> list[dict]:
    if isinstance(cat, dict):
        return cat["devices"]
    return list(getattr(cat, "devices"))


def _device(cat: Any, name: str) -> dict:
    for d in _devices(cat):
        if d["name"] == name:
            return d
    raise KeyError(name)


@pytest.fixture(scope="module")
def catalog() -> dict:
    assert CATALOG_PATH.is_file(), f"missing {CATALOG_PATH}"
    return _strict_load(CATALOG_PATH)


@pytest.fixture(scope="module")
def models(catalog: dict) -> dict[str, dict]:
    return {m["id"]: m for m in catalog["models"]}


# ----------------------------------------------------------------------------------------------- schema
def test_top_level_schema(catalog: dict) -> None:
    missing = REQUIRED_TOP - set(catalog)
    assert not missing, f"missing top-level keys: {sorted(missing)}"
    assert isinstance(catalog["schema_version"], int) and catalog["schema_version"] >= 1
    assert set(catalog["source_kinds"]) >= STATIC_SOURCE_KINDS | {"simulated"}


def test_model_schema_and_ids(catalog: dict, models: dict[str, dict]) -> None:
    ids = [m["id"] for m in catalog["models"]]
    assert len(ids) == len(set(ids)), "duplicate model ids"
    for m in catalog["models"]:
        missing = REQUIRED_MODEL - set(m)
        assert not missing, f"{m.get('id')}: missing keys {sorted(missing)}"
        assert m["kind"] in {"moe", "dense", "draft"}, m["id"]
        assert m["family"] in {"26b", "31b"}, m["id"]
        assert re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", m["repo"]), f"{m['id']}: bad repo {m['repo']!r}"
        for k in ("bytes", "nonexpert_bytes", "expert_bytes", "expert_unit_bytes", "expert_bytes_per_token_worst",
                  "nonexpert_bytes_per_token", "n_moe_layers", "n_expert", "n_expert_used"):
            assert isinstance(m[k], int) and not isinstance(m[k], bool) and m[k] >= 0, f"{m['id']}.{k} must be int>=0"
        assert m["bytes"] > 0
        assert isinstance(m["bytes_exact"], bool) and isinstance(m["estimated"], bool)
        if m["bytes_exact"]:
            # exact entries must be downloadable as-is: real file name and a sha256
            assert PLACEHOLDER not in m["file"] and "<" not in m["file"], f"{m['id']}: placeholder file"
            assert re.fullmatch(r"[A-Za-z0-9_./-]+\.gguf", m["file"]), f"{m['id']}: bad file {m['file']!r}"
            assert re.fullmatch(r"[0-9a-f]{64}", m.get("sha256") or ""), f"{m['id']}: sha256 missing/invalid"
        else:
            assert "bytes" in m["sources"] and "assumed" in m["sources"]["bytes"], m["id"]
        if m.get("sha256") is not None:
            assert re.fullmatch(r"[0-9a-f]{64}", m["sha256"]), m["id"]
        url = f"https://huggingface.co/{m['repo']}/resolve/main/{m['file']}"
        assert " " not in url and url.count("//") == 1


def test_drafts_cross_reference(models: dict[str, dict]) -> None:
    for mid, m in models.items():
        for d in m["drafts"]:
            assert d in models, f"{mid}: unknown draft {d}"
            assert models[d]["kind"] == "draft", f"{mid}: {d} is not a draft"
            assert models[d]["family"] == m["family"], f"{mid}: draft {d} from another family"
        if m["kind"] == "draft":
            assert m["drafts"] == [], f"{mid}: a draft has no drafts"
            assert m.get("draft_of") in models and models[m["draft_of"]]["kind"] != "draft", mid
            assert m.get("spec_type") in {"draft-mtp", "draft-dflash"}, mid
            assert m["expert_bytes"] == 0
    assert set(models["26b-q4"]["drafts"]) == {"26b-mtp", "26b-dflash"}
    assert models["31b-q4"]["drafts"] == ["31b-mtp"]


def test_device_schema(catalog: dict) -> None:
    names = [d["name"] for d in catalog["devices"]]
    assert len(names) == len(set(names)), "duplicate device names"
    assert REQUIRED_DEVICES <= set(names), f"missing devices {sorted(REQUIRED_DEVICES - set(names))}"
    for d in catalog["devices"]:
        missing = REQUIRED_DEVICE - set(d)
        assert not missing, f"{d['name']}: missing keys {sorted(missing)}"
        assert set(d["mem_bw_gbps"]) >= {"value", "source"} and d["mem_bw_gbps"]["value"] > 0
        st = d["storage"]
        assert st["type"] in {"microsd", "usb-ssd", "nvme", "unknown"}, d["name"]
        for k in STORAGE_METRICS:
            assert set(st[k]) >= {"value", "source"}, f"{d['name']}.storage.{k}"
            assert _is_num(st[k]["value"]) and st[k]["value"] > 0
        usable = d["ram_bytes"] - d["os_reserve_bytes"] - d["compute_buffers_bytes"]
        assert usable > 0, d["name"]


def test_device_spec_values(catalog: dict) -> None:
    for name, (ram, osr, cb, nctx, bw, typ, seq, rnd, acc) in PI_DEVICE_SPEC.items():
        d = _device(catalog, name)
        assert d["ram_bytes"] == ram and d["os_reserve_bytes"] == osr and d["compute_buffers_bytes"] == cb, name
        assert d["n_ctx"] == nctx and d["mem_bw_gbps"]["value"] == bw, name
        st = d["storage"]
        assert st["type"] == typ, name
        assert (st["seq_read_mbps"]["value"], st["rand_read_3mb_mbps"]["value"], st["access_ms"]["value"]) == \
            (seq, rnd, acc), name
    desk = _device(catalog, "desktop-8400f")
    assert desk["ram_bytes"] >= 64e9 and desk["n_ctx"] == 4096 and desk["storage"]["type"] == "nvme"


def test_storage_classes(catalog: dict) -> None:
    classes = catalog["storage_classes"]
    assert {"microsd", "usb-ssd", "nvme-gen2", "nvme-gen3", "unknown"} <= set(classes)
    for name, c in classes.items():
        for k in STORAGE_METRICS:
            assert set(c[k]) >= {"value", "source"} and c[k]["value"] > 0, f"{name}.{k}"
    unk = classes["unknown"]
    for name, c in classes.items():  # 'unknown' must be the most conservative class
        assert unk["seq_read_mbps"]["value"] <= c["seq_read_mbps"]["value"], name
        assert unk["rand_read_3mb_mbps"]["value"] <= c["rand_read_3mb_mbps"]["value"], name
        assert unk["access_ms"]["value"] >= c["access_ms"]["value"], name
    mapping = catalog.get("storage_type_default_class", {})
    for typ in ("microsd", "usb-ssd", "nvme", "unknown"):
        assert mapping.get(typ) in classes, f"no default class for storage type {typ}"
    for d in catalog["devices"]:  # Pi profiles reuse the class numbers verbatim
        cls = d["storage"].get("class")
        if cls in classes:
            for k in STORAGE_METRICS:
                assert d["storage"][k]["value"] == classes[cls][k]["value"], f"{d['name']}.{k}"


def test_selection_and_protection(catalog: dict) -> None:
    sel = catalog["selection"]
    curve = sel["stream_hit_rate_curve_assumed"]
    assert curve == [[0.15, 0.60], [0.25, 0.74], [0.35, 0.84], [0.45, 0.88]]
    xs, ys = [p[0] for p in curve], [p[1] for p in curve]
    assert all(0 < v < 1 for v in xs + ys)
    assert xs == sorted(xs) and ys == sorted(ys) and len(set(xs)) == len(xs)
    assert 0 < sel["viable_stream_tok_s"] < sel["good_stream_tok_s"]
    assert sel["viable_stream_tok_s"] == 1.0 and sel["good_stream_tok_s"] == 2.0
    assert "assumed" in sel["sources"]["stream_hit_rate_curve_assumed"]
    assert "context-friend-1" in catalog["protected_hosts"]
    globs = catalog["protected_service_globs"]
    assert {"contextfriend*", "context-friend*"} <= set(globs)
    for svc in ("contextfriend-api.service", "context-friend-audio.service"):
        assert any(fnmatch.fnmatch(svc, g) for g in globs), svc
    assert not any(fnmatch.fnmatch("ssh.service", g) for g in globs)


# ----------------------------------------------------------------------------------------------- provenance
def test_sources_enumerate(catalog: dict) -> None:
    allowed = set(catalog["source_kinds"])
    seen = 0
    for path, src in _iter_sources(catalog):
        seen += 1
        m = KIND_RE.match(src)
        assert m and m.group(1) in allowed, f"{path}: source {src[:60]!r} does not start with a kind in {allowed}"
        assert m.group(1) in STATIC_SOURCE_KINDS, f"{path}: static catalog may only hold measured/assumed sources"
    assert seen > 50


def test_every_unmeasured_source_says_assumed(catalog: dict) -> None:
    for path, src in _iter_sources(catalog):
        if not src.startswith("measured"):
            assert "assumed" in src, f"{path}: non-measured source lacks 'assumed': {src[:80]!r}"


def test_every_number_has_a_source(catalog: dict) -> None:
    missing = _unsourced_numbers(catalog)
    assert not missing, f"numbers without provenance: {missing[:20]}"


# ----------------------------------------------------------------------------------------------- byte arithmetic
def test_verified_constants(models: dict[str, dict]) -> None:
    m = models["26b-q4"]
    assert (m["bytes"], m["nonexpert_bytes"], m["expert_bytes"], m["expert_unit_bytes"]) == \
        (14439363584, 1577172088, 12846366720, 3345408)
    assert (m["n_moe_layers"], m["n_expert"], m["n_expert_used"]) == (30, 128, 8)
    assert m["expert_bytes_per_token_worst"] == 802897920 == 30 * 8 * 3345408
    assert m["kind"] == "moe" and m["quant"] == "Q4_0" and m["bytes_exact"] is True
    assert m["repo"] == "google/gemma-4-26B-A4B-it-qat-q4_0-gguf" and m["file"] == "gemma-4-26B_q4_0-it.gguf"
    d = models["31b-q4"]
    assert (d["bytes"], d["nonexpert_bytes"], d["expert_bytes"]) == (17651001568, 17635166448, 0)
    assert d["kind"] == "dense" and d["quant"] == "Q4_0" and d["bytes_exact"] is True
    assert d["repo"] == "google/gemma-4-31B-it-qat-q4_0-gguf" and d["file"] == "gemma-4-31B_q4_0-it.gguf"
    assert models["26b-dflash"]["quant"] == "Q8_0"


def test_byte_invariants(catalog: dict) -> None:
    for m in catalog["models"]:
        mid = m["id"]
        tensors = m["nonexpert_bytes"] + m["expert_bytes"]
        assert tensors <= m["bytes"], mid
        assert m["nonexpert_bytes_per_token"] == m["nonexpert_bytes"], mid
        if m["bytes_exact"] and "data_offset" in m and "n_tensors" in m and not m["estimated"]:
            slack = m["bytes"] - m["data_offset"] - tensors
            assert 0 <= slack < GGUF_ALIGNMENT * m["n_tensors"], f"{mid}: header+tensors != file (slack {slack})"
        if m["kind"] == "moe":
            assert m["n_moe_layers"] > 0 and 0 < m["n_expert_used"] < m["n_expert"], mid
            worst = m["expert_bytes_per_token_worst"]
            assert worst * m["n_expert"] == m["expert_bytes"] * m["n_expert_used"], mid
            lo = m.get("expert_unit_bytes_min", m["expert_unit_bytes"])
            hi = m.get("expert_unit_bytes_max", m["expert_unit_bytes"])
            assert 0 < lo <= m["expert_unit_bytes"] <= hi, mid
            assert lo * m["n_moe_layers"] * m["n_expert_used"] <= worst <= hi * m["n_moe_layers"] * m["n_expert_used"]
            if lo == hi:
                assert m["expert_unit_bytes"] * m["n_moe_layers"] * m["n_expert"] == m["expert_bytes"], mid
                assert m["expert_unit_bytes"] * m["n_moe_layers"] * m["n_expert_used"] == worst, mid
        else:
            assert m["expert_bytes"] == 0 and m["expert_bytes_per_token_worst"] == 0 and m["n_expert"] == 0, mid
    bd = {m["id"]: m.get("breakdown_bytes") for m in catalog["models"]}
    b26, b31 = bd["26b-q4"], bd["31b-q4"]
    assert b26["attn"] + b26["head_tied_token_embd_q6_k"] + b26["dense_ffn_shared_expert"] + \
        b26["router_norms_scales"] == 1577172088
    assert b26["expert_gate_up_per_expert_layer"] + b26["expert_down_per_expert_layer"] == 3345408
    assert b31["ffn"] + b31["attn"] + b31["head_tied_token_embd_q6_k"] + b31["norms_other"] == 17635166448


def test_alternatives_present(models: dict[str, dict]) -> None:
    want = {("26b", q) for q in ("UD-IQ2_XXS", "UD-Q2_K_XL", "UD-IQ3_XXS", "UD-Q3_K_M", "UD-IQ4_XS")} | \
           {("31b", q) for q in ("UD-IQ2_XXS", "UD-Q2_K_XL", "UD-IQ3_XXS", "Q3_K_S", "IQ4_XS")}
    have = {(m["family"], m["quant"]): m for m in models.values() if m["kind"] != "draft"}
    assert want <= set(have), f"missing alternatives {sorted(want - set(have))}"
    approx_gb = {("26b", "UD-IQ2_XXS"): 9.9, ("26b", "UD-Q2_K_XL"): 10.6, ("26b", "UD-IQ3_XXS"): 11.4,
                 ("26b", "UD-Q3_K_M"): 12.7, ("26b", "UD-IQ4_XS"): 13.6, ("31b", "UD-IQ2_XXS"): 8.5,
                 ("31b", "UD-Q2_K_XL"): 11.8, ("31b", "UD-IQ3_XXS"): 11.8, ("31b", "Q3_K_S"): 13.2,
                 ("31b", "IQ4_XS"): 16.4}
    for key, gb in approx_gb.items():
        m = have[key]
        assert m["repo"] == ("unsloth/gemma-4-26B-A4B-it-GGUF" if key[0] == "26b" else "unsloth/gemma-4-31B-it-GGUF")
        assert math.isclose(m["bytes"] / 1e9, gb, abs_tol=0.06), key
        if m["estimated"]:
            for k in ("nonexpert_bytes", "expert_bytes"):
                assert "assumed" in m["sources"][k], f"{key}: estimated split must be sourced 'assumed'"


# ----------------------------------------------------------------------------------------------- local files
def test_local_model_bytes_match_registry_constants(models: dict[str, dict]) -> None:
    for mid, size in LOCAL_BYTES.items():
        assert models[mid]["bytes"] == size, mid
        assert models[mid]["local_path"], mid


@pytest.mark.parametrize("mid", sorted(LOCAL_BYTES))
def test_local_files_on_disk(models: dict[str, dict], mid: str) -> None:
    p = REPO_ROOT / models[mid]["local_path"]
    if not p.is_file():
        pytest.skip(f"{p} not downloaded")
    assert p.stat().st_size == models[mid]["bytes"]


def test_local_model_bytes_equal_paths_models(models: dict[str, dict]) -> None:
    paths = pytest.importorskip("pigemma.paths")
    registry = getattr(paths, "MODELS")
    for mid in LOCAL_BYTES:
        assert mid in registry, f"pigemma.paths.MODELS lacks {mid}"
        entry = registry[mid]
        reg_bytes = entry["bytes"] if isinstance(entry, dict) else getattr(entry, "bytes")
        assert reg_bytes == models[mid]["bytes"], mid
        reg_path = entry["path"] if isinstance(entry, dict) else getattr(entry, "path")
        assert Path(reg_path).as_posix().endswith(models[mid]["local_path"]), mid
        if isinstance(entry, dict):  # optional registry fields must agree with the catalog when present
            for k in ("repo", "file", "kind", "quant", "family", "draft_of", "spec_type"):
                if entry.get(k) is not None:
                    assert entry[k] == models[mid].get(k), f"{mid}.{k}: paths.MODELS {entry[k]!r} != catalog"


# ----------------------------------------------------------------------------------------------- io_mem merge
IO_MEM_FIXTURE = {
    "host": "pi-under-test",
    "device": "pi5-8gb-nvme-gen3",
    "collected_at": "2026-10-05T12:00:00Z",
    "storage": {
        "seq_read_mbps_qd1": {"value": 801.0, "source": "measured", "cmd": "fio --name=seq --rw=read --bs=1M --iodepth=1"},
        "seq_read_mbps_qd8": {"value": 802.0, "source": "measured", "cmd": "fio --name=seq --rw=read --bs=1M --iodepth=8"},
        "rand_read_3mb_mbps_qd1": {"value": 611.0, "source": "measured", "cmd": "fio --name=r3m --rw=randread --bs=3M --iodepth=1"},
        "rand_read_3mb_mbps_qd4": {"value": 612.0, "source": "measured", "cmd": "fio --name=r3m --rw=randread --bs=3M --iodepth=4"},
        "rand_read_16k_iops": {"value": 12345.0, "source": "measured", "cmd": "fio --name=r16k --rw=randread --bs=16k"},
        "access_ms": {"value": 0.123, "source": "measured", "cmd": "derived from the 16k QD1 latency"},
    },
    "memory": {"read_gbps_by_threads": {"1": 5.1, "2": 7.2, "4": 9.4}, "copy_gbps": 4.5, "triad_gbps": 6.6,
               "source": "measured"},
}


def _load_with_io_mem(load_catalog: Any, io_mem_path: Path) -> Any:
    """Call load_catalog(path, io_mem=...) accepting either a path or an already-parsed dict."""
    try:
        return load_catalog(CATALOG_PATH, io_mem=io_mem_path)
    except (TypeError, AttributeError):
        return load_catalog(CATALOG_PATH, io_mem=copy.deepcopy(IO_MEM_FIXTURE))


def test_io_mem_merge_overrides_and_marks_measured(tmp_path: Path, catalog: dict) -> None:
    feas = pytest.importorskip("pigemma.feasibility")
    load_catalog = getattr(feas, "load_catalog")
    io_path = tmp_path / "io_mem.json"
    io_path.write_text(json.dumps(IO_MEM_FIXTURE), encoding="utf-8")
    before = CATALOG_PATH.read_bytes()

    plain = load_catalog(CATALOG_PATH)
    for d in _devices(plain):  # without io_mem nothing changes
        raw = _device(catalog, d["name"])
        assert d["mem_bw_gbps"]["value"] == raw["mem_bw_gbps"]["value"]
        for k in STORAGE_METRICS:
            assert d["storage"][k]["value"] == raw["storage"][k]["value"]

    merged = _load_with_io_mem(load_catalog, io_path)
    dev = _device(merged, "pi5-8gb-nvme-gen3")
    st = dev["storage"]
    assert st["seq_read_mbps"]["value"] in (801.0, 802.0)
    assert st["rand_read_3mb_mbps"]["value"] in (611.0, 612.0)
    assert st["access_ms"]["value"] == pytest.approx(0.123)
    mem = IO_MEM_FIXTURE["memory"]
    assert dev["mem_bw_gbps"]["value"] in (*mem["read_gbps_by_threads"].values(), mem["copy_gbps"], mem["triad_gbps"])
    for field in (st["seq_read_mbps"], st["rand_read_3mb_mbps"], st["access_ms"], dev["mem_bw_gbps"]):
        assert str(field["source"]).startswith("measured"), field
        assert "assumed" not in str(field["source"])

    for d in _devices(merged):  # other profiles untouched
        if d["name"] == "pi5-8gb-nvme-gen3":
            continue
        raw = _device(catalog, d["name"])
        assert d["mem_bw_gbps"] == raw["mem_bw_gbps"], d["name"]
        for k in STORAGE_METRICS:
            assert d["storage"][k] == raw["storage"][k], f"{d['name']}.{k}"

    assert CATALOG_PATH.read_bytes() == before, "load_catalog must not rewrite the catalog file"
