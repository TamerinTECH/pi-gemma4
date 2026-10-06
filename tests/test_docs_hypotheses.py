"""Checks for docs/HYPOTHESES.md (manifest files[] 'docs/HYPOTHESES.md', test field).

The manifest test field asks that tests/test_docs.py greps the catalog's assumed constants (storage
MB/s, mem_bw, os_reserve) and asserts they appear in docs/HYPOTHESES.md. tests/test_docs.py is shared
by five doc entries and other agents may be writing it, so the HYPOTHESES-specific checks live in
this separate module. pytest collects it the same way, and test_docs.py may import from it.

What is checked:

* The machine-readable block between ``<!-- pigemma-catalog-constants:begin -->`` and ``...:end -->``
  equals pigemma/data/catalog.json value by value. Paths are ``devices.<name>.``,
  ``models.<id>.``, ``storage_classes.<class>.`` and ``selection.``.
* The block covers every device and storage-class constant (completeness).
* Every assumed device and storage-class constant (storage MB/s, access ms, mem_bw GB/s, os_reserve
  GB) also appears in human-readable form in the prose above the block.
* The headline projections quoted in the prose are what pigemma.feasibility.verdict() computes from
  the catalog today: expert cache, resident fraction, prior hit and tok/s per catalog Pi profile,
  plus the in-RAM ceiling.
* The feasibility module constants quoted in section 8 equal the code.
* Every relative Markdown link resolves to an existing file, or to a manifest files[] path that a
  later dependency level writes.
* The pre-registered thresholds are present.

stdlib + pytest only. pigemma.feasibility is stdlib-only.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest

from pigemma import feasibility as F

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "HYPOTHESES.md"
CATALOG = REPO / "pigemma" / "data" / "catalog.json"
MANIFEST = REPO / "docs" / "design" / "manifest.json"

BEGIN = "<!-- pigemma-catalog-constants:begin -->"
END = "<!-- pigemma-catalog-constants:end -->"
LINE_RE = re.compile(r"^([A-Za-z0-9_.\-]+) = (.+)$")
LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")

DEVICE_KEYS = (
    "ram_bytes", "os_reserve_bytes", "compute_buffers_bytes", "n_ctx", "mem_bw_gbps.value",
    "storage.seq_read_mbps.value", "storage.rand_read_3mb_mbps.value", "storage.access_ms.value",
)
CLASS_KEYS = ("seq_read_mbps.value", "rand_read_3mb_mbps.value", "access_ms.value")
SELECTION_KEYS = ("stream_hit_rate_curve_assumed", "viable_stream_tok_s", "good_stream_tok_s")


# ------------------------------------------------------------------------------------------ helpers
def _doc() -> str:
    return DOC.read_text(encoding="utf-8")


def _catalog() -> dict:
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def _split(doc: str) -> tuple[str, str]:
    """(prose before the constants block, block body)."""
    assert doc.count(BEGIN) == 1 and doc.count(END) == 1, "constants block markers missing or repeated"
    head, rest = doc.split(BEGIN, 1)
    body, _tail = rest.split(END, 1)
    return head, body


def _block(doc: str) -> dict[str, Any]:
    _, body = _split(doc)
    out: dict[str, Any] = {}
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        m = LINE_RE.match(line)
        assert m, f"malformed constants line: {raw!r}"
        key, val = m.group(1), m.group(2).strip()
        assert key not in out, f"duplicate constants key {key}"
        out[key] = json.loads(val)
    return out


def _resolve(cat: dict, path: str) -> Any:
    """Catalog value for a dotted path; devices/models are addressed by name/id."""
    head, _, rest = path.partition(".")
    if head in ("devices", "models"):
        ident_key = "name" if head == "devices" else "id"
        items = {it[ident_key]: it for it in cat[head]}
        # ids may contain '-' and '_' but never '.', so the next segment is the id
        ident, _, rest = rest.partition(".")
        node: Any = items[ident]
    else:
        node = cat[head]
    for part in rest.split(".") if rest else ():
        node = node[part]
    return node


def _facts(ram_total_kb: int, stype: str, speed: str | None) -> dict:
    return {
        "hostname": "pi-lab", "kernel": "6.18.0-v8-16k+", "page_size": 16384, "cpu_model": "Cortex-A76",
        "cpu_flags": ["fp", "asimd", "fphp", "asimdhp", "asimddp"], "nproc": 4,
        "ram_total_kb": ram_total_kb, "ram_available_kb": ram_total_kb - 500_000, "swap_total_kb": 0,
        "loadavg1": 0.2, "model_dir": "/nonexistent/pigemma-models", "free_disk_bytes": 400_000_000_000,
        "storage": {"dev": "/dev/x", "type": stype, "rotational": False, "pcie_link_speed": speed,
                    "pcie_link_width": 1 if speed else None, "read_ahead_kb": 128},
        "running_services": [], "hat_product": None, "pciex1_gen_cfg": None, "governor": "performance",
        "memlock_limit": "unlimited", "tools": {}, "override": None, "collected_at": "2026-10-05T00:00:00Z",
    }


PI_PROFILES = {
    "pi5-4gb-microsd": (3_906_250, "microsd", None),
    "pi5-8gb-usb3": (7_812_500, "usb-ssd", None),
    "pi5-8gb-nvme-gen3": (7_812_500, "nvme", "8.0 GT/s PCIe"),
    "pi5-16gb-nvme-gen3": (15_625_000, "nvme", "8.0 GT/s PCIe"),
}


def _row(device: str, model_id: str = "26b-q4") -> dict:
    kb, stype, speed = PI_PROFILES[device]
    v = F.verdict(_facts(kb, stype, speed), copy.deepcopy(F.load_catalog(CATALOG)), device=device,
                  check_present=False)
    for r in v.rows:
        if r["model_id"] == model_id:
            return r
    raise AssertionError(f"no row for {model_id} on {device}")


# -------------------------------------------------------------------------------------------- tests
def test_doc_exists_and_is_utf8() -> None:
    assert DOC.is_file(), "docs/HYPOTHESES.md missing"
    raw = DOC.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "no BOM"
    raw.decode("utf-8")
    assert b"\r\n" not in raw, "LF line endings"


def test_constants_block_matches_catalog() -> None:
    cat, block = _catalog(), _block(_doc())
    assert block, "constants block is empty"
    mismatches = []
    for key, val in block.items():
        try:
            want = _resolve(cat, key)
        except (KeyError, StopIteration) as exc:
            mismatches.append(f"{key}: not in catalog ({exc!r})")
            continue
        if want != val or type(want) is not type(val):
            mismatches.append(f"{key}: doc {val!r} != catalog {want!r}")
    assert not mismatches, "\n".join(mismatches)


def test_constants_block_covers_assumed_constants() -> None:
    cat, block = _catalog(), _block(_doc())
    missing = [f"devices.{d['name']}.{k}" for d in cat["devices"] for k in DEVICE_KEYS
               if f"devices.{d['name']}.{k}" not in block]
    missing += [f"storage_classes.{c}.{k}" for c in cat["storage_classes"] for k in CLASS_KEYS
                if f"storage_classes.{c}.{k}" not in block]
    missing += [f"selection.{k}" for k in SELECTION_KEYS if f"selection.{k}" not in block]
    assert not missing, "constants block lacks: " + ", ".join(missing)


def _human_forms(cat: dict) -> dict[str, str]:
    """Human-readable spelling of every assumed storage / mem_bw / os_reserve constant."""
    forms: dict[str, str] = {}
    sources = [(f"devices.{d['name']}", d["storage"], d) for d in cat["devices"]]
    sources += [(f"storage_classes.{c}", sc, None) for c, sc in cat["storage_classes"].items()]
    for label, st, dev in sources:
        forms[f"{label}.seq_read_mbps"] = f"{st['seq_read_mbps']['value']:g} MB/s"
        forms[f"{label}.rand_read_3mb_mbps"] = f"{st['rand_read_3mb_mbps']['value']:g} MB/s"
        forms[f"{label}.access_ms"] = f"{st['access_ms']['value']:.1f} ms"
        if dev is not None:
            forms[f"{label}.mem_bw_gbps"] = f"{dev['mem_bw_gbps']['value']:.1f} GB/s"
            forms[f"{label}.os_reserve_bytes"] = f"{dev['os_reserve_bytes'] / 1e9:g} GB"
    return forms


def test_assumed_constants_appear_in_prose() -> None:
    prose, _ = _split(_doc())
    cat = _catalog()
    absent = [f"{k} ({form!r})" for k, form in _human_forms(cat).items() if form not in prose]
    assert not absent, "constants missing from the prose: " + "; ".join(absent)
    # every catalog source of these constants is 'assumed' (or measured, after an io_mem merge), and
    # the doc must say which: an assumed storage number is labelled ASSUMED somewhere in the prose
    assert "ASSUMED" in prose and "MEASURED" in prose and "PROJECTED" in prose and "PRIOR" in prose


@pytest.mark.parametrize("device", sorted(PI_PROFILES))
def test_headline_projections_match_feasibility(device: str) -> None:
    prose, _ = _split(_doc())
    r = _row(device)
    tok = f"{r['tok_s']:.2f}"
    assert tok in prose, f"{device}: tok/s {tok} not quoted"
    assert r["verdict"] in prose
    if r["mode"] == "stream_moe":
        cache = f"{r['expert_cache_bytes'] / 1e9:.2f} GB"
        frac = f"{r['resident_fraction'] * 100:.1f}%"
        hit = f"{r['hit_rate']:.3f}"
        for s in (cache, frac, hit):
            assert s in prose, f"{device}: {s} not quoted"


def test_in_ram_ceiling_and_byte_budgets_match_catalog() -> None:
    prose, _ = _split(_doc())
    cat = _catalog()
    models = {m["id"]: m for m in cat["models"]}
    m26, m31 = models["26b-q4"], models["31b-q4"]
    per_tok_26 = m26["nonexpert_bytes_per_token"] + m26["expert_bytes_per_token_worst"]
    assert f"{per_tok_26:,}" in prose
    assert f"{m26['nonexpert_bytes_per_token']:,}" in prose
    assert f"{m26['expert_bytes_per_token_worst']:,}" in prose
    assert f"{m26['expert_bytes']:,}" in prose
    assert f"{m31['nonexpert_bytes_per_token']:,}" in prose
    pi = next(d for d in cat["devices"] if d["name"] == "pi5-16gb-nvme-gen3")
    ceiling = pi["mem_bw_gbps"]["value"] * 1e9 * F.LLAMA_EFFICIENCY / per_tok_26
    assert f"{ceiling:.2f}" in prose
    # worst case = n_moe_layers x n_expert_used x expert_unit_bytes for the uniform Q4_0 file
    assert m26["expert_bytes_per_token_worst"] == (
        m26["n_moe_layers"] * m26["n_expert_used"] * m26["expert_unit_bytes"])
    # H2 arithmetic: attention + head of the 31B (the non-promise)
    bd = m31["breakdown_bytes"]
    assert f"{bd['attn'] + bd['head_tied_token_embd_q6_k']:,}" in prose


def test_feasibility_constants_quoted_correctly() -> None:
    _, tail = _doc().split(END, 1)
    assert f"`LLAMA_EFFICIENCY` = {F.LLAMA_EFFICIENCY}" in tail
    assert f'`KV_Q8_4K["26b"]` = {F.KV_Q8_4K["26b"]}' in tail
    assert f'`KV_Q8_4K["31b"]` = {F.KV_Q8_4K["31b"]}' in tail
    assert f"`COMPUTE_BUFFERS` = {F.COMPUTE_BUFFERS}" in tail
    assert f"`MIN_STREAM_CACHE` = {F.MIN_STREAM_CACHE}" in tail
    assert f"`SLOW_TOK_S` = {F.SLOW_TOK_S}" in tail


def test_relative_links_resolve() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    planned = {f["path"] for f in manifest["files"]}
    bad = []
    for target in LINK_RE.findall(_doc()):
        if re.match(r"^[a-z][a-z0-9+.\-]*:", target) or target.startswith("#"):
            continue
        rel = target.split("#", 1)[0]
        path = (DOC.parent / rel).resolve()
        if path.exists():
            continue
        try:
            repo_rel = path.relative_to(REPO).as_posix()
        except ValueError:
            bad.append(f"{target} (outside the repo)")
            continue
        if repo_rel not in planned:
            bad.append(f"{target} (missing and not a manifest file)")
    assert not bad, "broken links: " + ", ".join(bad)


@pytest.mark.parametrize("needle", [
    "coverage@32 ≥ 0.70", "adjacent reuse ≥ 0.40", "tg_overlap ≥ 2.0", "tg_serial ≥ 1.0",
    "≤ 0.30", "≤ 0.35", "1.15 × unpinned", "≥ 20%", "**≤ 25%**", "**≥ 50%**", "**< 8.5 GB**",
    "**< 0.03**", "**== 0**", "precision@16", "KLD mean < 0.001", "5.93 GB",
])
def test_preregistered_thresholds_present(needle: str) -> None:
    assert needle in _doc(), f"threshold text {needle!r} missing"


def test_hit_curve_prior_values_quoted() -> None:
    prose, _ = _split(_doc())
    curve = _catalog()["selection"]["stream_hit_rate_curve_assumed"]
    for x, y in curve:
        assert f"{x * 100:.0f}% → {y * 100:.0f}%" in prose, f"prior point ({x}, {y}) not quoted"
    assert "0.444" in prose  # Qwen3-30B-A3B next-token reuse prior
