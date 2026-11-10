"""tests/test_docs.py: checks for README.md and a link check over every Markdown document.

Owner: docs/design/manifest.json files[] entry "README.md". Its test field reads "tests/test_docs.py:
all relative links resolve; the numbers quoted for the byte budgets match pigemma/data/catalog.json".
The docs/TRACE_FORMAT.md entry also names this file for "every relative link in docs/ resolves", so
the link check runs over README.md and docs/**/*.md. The other documents have their own content tests
(test_docs_hypotheses.py, test_experiment_plan.py, test_pi_runbook.py); this module does not import them.

What is checked in README.md:
- UTF-8, LF line endings, no BOM; the sections the manifest spec asks for exist.
- Relative links and #anchors resolve. A link to a file that does not exist yet passes only when the
  manifest plans that path.
- Byte budgets: every table row that names a catalog field (`models.<id>.<key>`, or a sum `a + b`)
  quotes exactly that byte count, and its human-readable size rounds from it.
- The llama.cpp line references (`third_party/llama.cpp/<file>:<a>-<b>`) still point at the code
  they describe. Skipped when the llama.cpp sources are absent.
- The hardware-options grid and the other PROJECTED numbers are what pigemma.feasibility computes
  from the catalog. The ASSUMED constants quoted are the catalog's.
- The status table: every MEASURED cell that is not pending links a source file, and every decimal
  number in it occurs in a linked file. This keeps measured numbers traceable.
- The pre-registered thresholds quoted are the ones in docs/HYPOTHESES.md.
- Commands: `python -m pigemma.<module> [sub]` flags appear in that module's --help; pc/ and pi/
  script flags appear in the script text; .ps1 parameters exist. Model ids exist in the catalog;
  experiment ids are E0..E8.

stdlib + pytest only. It imports pigemma.feasibility (stdlib only) and runs `python -m pigemma.<module>
--help` as subprocesses, which takes about 10 s.
"""
from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
DOCS_DIR = REPO_ROOT / "docs"
CATALOG = REPO_ROOT / "pigemma" / "data" / "catalog.json"
MANIFEST = REPO_ROOT / "docs" / "design" / "manifest.json"
HYPOTHESES = REPO_ROOT / "docs" / "HYPOTHESES.md"
LLAMA_SRC = REPO_ROOT / "third_party" / "llama.cpp"

# Sections the manifest spec asks for (the headings may say more; matched as prefixes).
REQUIRED_SECTIONS = (
    "The two models and their byte budgets",
    "The one llama.cpp fact everything depends on",
    "Honest status",
    "Hardware options",
    "Hypotheses H1 and H2",
    "Plain statements",
    "Quick start: desktop",
    "Quick start: Raspberry Pi",
    "Safety",
    "Repository map",
    "Documents",
)
# Catalog fields the spec names explicitly for the byte-budget table.
REQUIRED_BYTE_FIELDS = (
    "models.26b-q4.nonexpert_bytes",
    "models.26b-q4.breakdown_bytes.head_tied_token_embd_q6_k",
    "models.26b-q4.expert_bytes_per_token_worst",
    "models.31b-q4.nonexpert_bytes_per_token",
)
# Hardware-options table: terms the operator's request and FACTS.md require.
HARDWARE_TERMS = ("4 GB", "8 GB", "16 GB", "microSD", "USB 3.0 SSD", "NVMe Gen2", "NVMe Gen3",
                  "AI HAT", "Hailo", "Active Cooler", "dual-slot", "pciex1_gen=3", "recommended",
                  "llama.cpp RPC")
# What each cited llama.cpp line range must still contain (source facts the README relies on).
LLAMA_LINE_KEYWORDS = {
    "src/llama-model.cpp": "is_default_buft",
    "src/llama-model-loader.cpp": "buft_list_cpu",
    "common/arg.cpp": "ggml_backend_dev_buffer_type",
    "src/llama-graph.cpp": "warmup",
}
# Grid profiles, as in tests/test_docs_hypotheses.py: (MemTotal kB, storage type, PCIe link speed).
RAM_TIERS_KB = (3_906_250, 7_812_500, 15_625_000)          # 4, 8, 16 GB boards (decimal GB)
STORAGES = (("microsd", None), ("usb-ssd", None), ("nvme", "5.0 GT/s PCIe"), ("nvme", "8.0 GT/s PCIe"))
CF1_RAM_KB = 16_607_552                                    # context-friend-1 MemTotal (FACTS.md)

_FENCE = re.compile(r"^(```|~~~)")
_LINK = re.compile(r"!?\[[^\]\n]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


# ------------------------------------------------------------------------------------------ helpers
def strip_fenced(text: str) -> str:
    """The text with fenced code blocks blanked out (line numbers kept)."""
    lines, inside = [], False
    for line in text.splitlines():
        if _FENCE.match(line.strip()):
            inside = not inside
            lines.append("")
            continue
        lines.append("" if inside else line)
    return "\n".join(lines)


def fenced_blocks(text: str) -> list[tuple[str, list[str]]]:
    """[(info string, lines)] for every fenced code block."""
    out: list[tuple[str, list[str]]] = []
    cur: list[str] | None = None
    info = ""
    for line in text.splitlines():
        s = line.strip()
        if cur is None:
            if _FENCE.match(s):
                cur, info = [], s[3:].strip()
        elif _FENCE.match(s):
            out.append((info, cur))
            cur = None
        else:
            cur.append(line)
    return out


def inline_code(text: str) -> list[str]:
    """Contents of single-backtick code spans outside fenced blocks."""
    return re.findall(r"(?<!`)`([^`\n]+)`(?!`)", strip_fenced(text))


def markdown_links(text: str) -> list[str]:
    """Targets of [text](target) outside code blocks and code spans."""
    body = re.sub(r"`[^`\n]*`", "", strip_fenced(text))
    return _LINK.findall(body)


def github_slug(heading: str) -> str:
    h = heading.strip().lower().replace("`", "")
    h = re.sub(r"[^\w\- ]", "", h)
    return h.replace(" ", "-")


def heading_slugs(text: str) -> set[str]:
    seen: dict[str, int] = {}
    slugs: set[str] = set()
    for line in strip_fenced(text).splitlines():
        m = re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)
        if not m:
            continue
        base = github_slug(m.group(1))
        n = seen.get(base, 0)
        slugs.add(base if n == 0 else f"{base}-{n}")
        seen[base] = n + 1
    return slugs


def planned_paths(manifest: dict) -> set[str]:
    """Repo-relative paths the manifest plans, including its '(+ a, b)' suffix files."""
    out: set[str] = set()
    for f in manifest.get("files", []):
        path = f["path"]
        out.add(path.split(" (+")[0].strip())
        if "(+" in path:
            inner = path.split("(+", 1)[1].rsplit(")", 1)[0]
            for part in inner.split(","):
                tok = part.strip().split()[0] if part.strip() else ""
                if tok and "*" not in tok and ("/" in tok or "." in tok):
                    out.add(tok)
    return out


def split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    return [c.strip() for c in re.split(r"(?<!\\)\|", s)]


def md_tables(text: str) -> list[list[list[str]]]:
    """Every GFM table outside fenced blocks as a list of rows (header first, separator dropped)."""
    tables: list[list[list[str]]] = []
    cur: list[list[str]] = []
    for line in strip_fenced(text).splitlines() + [""]:
        if line.lstrip().startswith("|"):
            cells = split_row(line)
            if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                cur.append(cells)
        elif cur:
            tables.append(cur)
            cur = []
    return tables


def section(text: str, title_prefix: str) -> str:
    """Body of the '## <title_prefix>...' section, up to the next '## ' heading."""
    out, inside, fenced = [], False, False
    for line in text.splitlines():
        if _FENCE.match(line.strip()):
            fenced = not fenced
        if not fenced and line.startswith("## "):
            if inside:
                break
            inside = line[3:].strip().startswith(title_prefix)
            continue
        if inside:
            out.append(line)
    assert out, f"README has no '## {title_prefix}' section"
    return "\n".join(out)


def resolve_catalog(catalog: dict, dotted: str):
    """Value at a dotted catalog path; models/devices are addressed by id/name."""
    parts = dotted.split(".")
    if parts[0] == "models":
        node = next(m for m in catalog["models"] if m["id"] == parts[1])
        rest = parts[2:]
    elif parts[0] == "devices":
        node = next(d for d in catalog["devices"] if d["name"] == parts[1])
        rest = parts[2:]
    else:
        node, rest = catalog, parts
    for key in rest:
        node = node[key]
    return node


def facts(ram_total_kb: int, stype: str, speed: str | None) -> dict:
    """A neutral, unprotected Pi 5 facts dict (same shape as tests/test_docs_hypotheses.py)."""
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


# ------------------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def text() -> str:
    assert README.is_file(), "README.md is missing"
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def catalog() -> dict:
    return json.loads(CATALOG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def feas():
    from pigemma import feasibility
    return feasibility


@pytest.fixture(scope="module")
def grid(feas) -> dict:
    """{(ram_kb, stype, speed): {model_id: row}} from pigemma.feasibility with the plain catalog."""
    cat = feas.load_catalog(CATALOG)
    out = {}
    for kb in RAM_TIERS_KB:
        for stype, speed in STORAGES:
            v = feas.verdict(facts(kb, stype, speed), copy.deepcopy(cat), check_present=False)
            out[(kb, stype, speed)] = {r["model_id"]: r for r in v.rows}
    return out


def _all_docs() -> list[Path]:
    agents = REPO_ROOT / "AGENTS.md"
    docs = sorted(p for p in DOCS_DIR.rglob("*.md") if p.is_file())
    return [README, agents] + docs


# ------------------------------------------------------------------------------------------ tests
def test_readme_is_utf8_lf_no_bom() -> None:
    raw = README.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "README.md starts with a BOM"
    assert b"\r\n" not in raw, "README.md must use LF line endings"
    raw.decode("utf-8")


def test_readme_has_required_sections(text: str) -> None:
    headings = [ln[3:].strip() for ln in strip_fenced(text).splitlines() if ln.startswith("## ")]
    for want in REQUIRED_SECTIONS:
        assert any(h.startswith(want) for h in headings), f"missing section '## {want}'"
    assert text.startswith("# "), "README.md must start with a title"


def test_readme_tables_are_rectangular(text: str) -> None:
    for table in md_tables(text):
        widths = {len(r) for r in table}
        assert len(widths) == 1, f"table starting {table[0][:3]} has rows of widths {sorted(widths)}"


@pytest.mark.parametrize("doc", _all_docs(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix())
def test_relative_links_resolve(doc: Path, manifest: dict) -> None:
    planned = planned_paths(manifest)
    missing, pending = [], []
    for target in markdown_links(doc.read_text(encoding="utf-8")):
        if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
            continue
        path_part = target.split("#", 1)[0]
        resolved = (doc.parent / path_part).resolve()
        try:
            rel = resolved.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            missing.append(f"{target} (outside the repo)")
            continue
        if resolved.exists():
            continue
        if rel in planned:
            pending.append(rel)
            continue
        missing.append(target)
    if pending:
        print(f"{doc.name}: planned, not written yet: {sorted(set(pending))}")
    assert not missing, f"{doc.name}: links to files that neither exist nor are planned: {missing}"


def test_readme_anchor_links_resolve(text: str) -> None:
    own = heading_slugs(text)
    bad = []
    for target in markdown_links(text):
        if "#" not in target or re.match(r"^[a-z][a-z0-9+.-]*:", target):
            continue
        path_part, anchor = target.split("#", 1)
        if not path_part:
            if anchor not in own:
                bad.append(target)
            continue
        doc = (README.parent / path_part).resolve()
        if doc.suffix == ".md" and doc.is_file():
            if anchor not in heading_slugs(doc.read_text(encoding="utf-8")):
                bad.append(target)
    assert not bad, f"anchors that do not exist: {bad}"


def _byte_rows(text: str) -> list[tuple[str, int, str]]:
    """(catalog expression, quoted bytes, quoted size) for every byte-budget row."""
    rows = []
    for table in md_tables(text):
        header = table[0]
        if "Catalog field" not in header or "Bytes" not in header:
            continue
        i_field, i_bytes, i_size = header.index("Catalog field"), header.index("Bytes"), header.index("Size")
        for cells in table[1:]:
            expr = cells[i_field].strip("` ")
            quoted = int(cells[i_bytes].replace(",", "").replace("*", "").strip())
            rows.append((expr, quoted, cells[i_size].replace("*", "").strip()))
    return rows


def test_byte_budgets_match_catalog(text: str, catalog: dict) -> None:
    rows = _byte_rows(text)
    assert len(rows) >= 15, f"expected the 26B and 31B byte-budget tables, found {len(rows)} rows"
    problems = []
    for expr, quoted, size in rows:
        terms = [t.strip().strip("`") for t in expr.split("+")]
        value = 0
        for term in terms:
            v = resolve_catalog(catalog, term)
            assert isinstance(v, int) and not isinstance(v, bool), f"{term} is not an int byte count"
            value += v
        if value != quoted:
            problems.append(f"{expr}: README quotes {quoted:,}, catalog has {value:,}")
        m = re.fullmatch(r"([\d.]+) (GB|MB)", size)
        assert m, f"{expr}: size cell {size!r} is not '<number> GB|MB'"
        shown, unit = m.group(1), {"GB": 1e9, "MB": 1e6}[m.group(2)]
        nd = len(shown.split(".")[1]) if "." in shown else 0
        if f"{value / unit:.{nd}f}" != shown:
            problems.append(f"{expr}: size {size} does not round from {value:,} B")
    assert not problems, "\n".join(problems)


def test_byte_budgets_cover_the_spec_figures(text: str, catalog: dict) -> None:
    exprs = {e for e, _, _ in _byte_rows(text)}
    for field in REQUIRED_BYTE_FIELDS:
        assert field in exprs, f"byte-budget table lacks {field}"
    # The products quoted in the row labels.
    q4 = resolve_catalog(catalog, "models.26b-q4")
    assert q4["n_moe_layers"] * q4["n_expert_used"] * q4["expert_unit_bytes"] == q4["expert_bytes_per_token_worst"]
    assert f"{q4['n_moe_layers']} × {q4['n_expert_used']} × {q4['expert_unit_bytes']:,}" in text
    d = resolve_catalog(catalog, "models.31b-q4.breakdown_bytes")
    assert 60 * 3 * d["ffn_per_matrix"] == d["ffn"]
    assert f"{d['ffn_per_matrix']:,}" in text


def test_llama_line_references_still_hold(text: str) -> None:
    refs = re.findall(r"`third_party/llama\.cpp/([\w./-]+):(\d+)(?:-(\d+))?`", text)
    assert refs, "README cites no llama.cpp line references"
    assert {r[0] for r in refs} >= set(LLAMA_LINE_KEYWORDS), "a cited llama.cpp file lost its reference"
    if not (LLAMA_SRC / "src" / "llama-model.cpp").is_file():
        pytest.skip("third_party/llama.cpp sources not present")
    for rel, a, b in refs:
        lines = (LLAMA_SRC / rel).read_text(encoding="utf-8", errors="replace").splitlines()
        lo, hi = int(a), int(b or a)
        chunk = "\n".join(lines[lo - 1:hi])
        assert LLAMA_LINE_KEYWORDS[rel] in chunk, f"{rel}:{lo}-{hi} no longer contains {LLAMA_LINE_KEYWORDS[rel]!r}"
    m = re.search(r"`is_default_buft`\s+test is at line (\d+)", text)
    assert m, "README should name the line of the is_default_buft test"
    line = (LLAMA_SRC / "src" / "llama-model.cpp").read_text(encoding="utf-8").splitlines()[int(m.group(1)) - 1]
    assert "is_default_buft" in line and "use_mmap" in line, line


def test_hardware_options_match_feasibility(text: str, grid: dict) -> None:
    sec = section(text, "Hardware options")
    for term in HARDWARE_TERMS:
        assert term in sec, f"'Hardware options' lacks {term!r}"
    problems = []
    for (kb, stype, speed), rows in grid.items():
        r = rows["26b-q4"]
        tok = f"{r['tok_s']:.2f}"
        if tok not in sec or r["verdict"] not in sec:
            problems.append(f"{kb} kB {stype} {speed}: 26b-q4 {tok} {r['verdict']} not quoted")
        d = rows["31b-q4"]
        spt = f"{d['s_per_token']:.1f} s/token"
        if f"{d['s_per_token']:.1f}" not in sec:
            problems.append(f"{kb} kB {stype} {speed}: 31b-q4 {spt} not quoted")
    assert not problems, "\n".join(problems)
    # The recommendation is the 16 GB + NVMe Gen3 row, and the AI HAT row says it does not help.
    rec = next(ln for ln in sec.splitlines() if ln.startswith("| 16 GB") and "NVMe Gen3" in ln)
    assert "recommended" in rec and "FITS_MAPPED" in rec, rec
    hat = next(ln for ln in sec.splitlines() if "AI HAT" in ln and ln.startswith("|"))
    assert "does not help" in hat, hat


def test_other_projections_match_feasibility(text: str, feas, catalog: dict) -> None:
    cat = feas.load_catalog(CATALOG)
    v = feas.verdict(facts(CF1_RAM_KB, "microsd", None), copy.deepcopy(cat), check_present=False)
    rows = {r["model_id"]: r for r in v.rows}
    for mid in ("26b-ud-q3_k_m", "31b-ud-iq3_xxs"):
        tok = f"{rows[mid]['tok_s']:.2f}"
        assert f"{tok} tok/s" in text or f"{tok} projected" in text, f"{mid}: projected {tok} tok/s not quoted"
    # The in-RAM ceilings: mem_bw x efficiency / bytes per token.
    bw = resolve_catalog(catalog, "devices.pi5-16gb-nvme-gen3.mem_bw_gbps.value") * 1e9 * feas.LLAMA_EFFICIENCY
    q4 = resolve_catalog(catalog, "models.26b-q4")
    dense = resolve_catalog(catalog, "models.31b-q4.nonexpert_bytes_per_token")
    ceiling_26b = bw / (q4["nonexpert_bytes_per_token"] + q4["expert_bytes_per_token_worst"])
    assert f"= {ceiling_26b:.2f} tok/s" in text
    assert f"= {bw / dense:.2f} tok/s" in text
    assert f"{bw / 1e9:.1f} GB/s" in text  # 10.0 GB/s x 0.65 = 6.5 GB/s


def test_assumed_constants_match_catalog(text: str, catalog: dict, feas) -> None:
    status = section(text, "Honest status")
    storage_row = next(ln for ln in status.splitlines() if ln.startswith("| Pi storage"))
    sc = catalog["storage_classes"]
    for cls, label in (("microsd", "microSD"), ("usb-ssd", "USB 3.0 SSD"), ("nvme-gen2", "NVMe Gen2"),
                       ("nvme-gen3", "NVMe Gen3")):
        seq = sc[cls]["seq_read_mbps"]["value"]
        rnd = sc[cls]["rand_read_3mb_mbps"]["value"]
        assert f"{label} {seq:g} / {rnd:g}" in storage_row, f"{cls}: {seq} / {rnd} MB/s not quoted"
    bw = {d["mem_bw_gbps"]["value"] for d in catalog["devices"] if d["name"].startswith("pi5-")}
    assert len(bw) == 1
    assert f"{bw.pop():.1f} GB/s" in status
    assert f"{feas.LLAMA_EFFICIENCY:g}" in status
    curve = catalog["selection"]["stream_hit_rate_curve_assumed"]
    assert ", ".join(f"{x:.2f} → {y:.2f}" for x, y in curve) in status
    sel = catalog["selection"]
    assert f"≥ {sel['good_stream_tok_s']:.1f} tok/s is good" in text
    assert f"≥ {sel['viable_stream_tok_s']:.1f} tok/s" in text


def test_status_table_measured_cells_are_sourced(text: str) -> None:
    status = section(text, "Honest status")
    tables = [t for t in md_tables(status) if t[0][:2] == ["Quantity", "MEASURED (host, date, file)"]]
    assert len(tables) == 1, "expected one status table with MEASURED / PROJECTED / ASSUMED columns"
    table = tables[0]
    assert table[0][2].startswith("PROJECTED") and table[0][3].startswith("ASSUMED"), table[0]
    assert len(table) >= 15, "status table looks truncated"
    problems = []
    for row in table[1:]:
        cell = row[1]
        if cell.startswith(("_pending_", "—")):
            continue
        links = _LINK.findall(cell)
        if not links:
            problems.append(f"{row[0]}: MEASURED cell has no source link")
            continue
        sources = ""
        for target in links:
            p = (README.parent / target.split("#", 1)[0]).resolve()
            if p.is_file():
                sources += p.read_text(encoding="utf-8", errors="replace")
        plain = _LINK.sub("", cell)
        for num in re.findall(r"\d+\.\d+", plain):
            if num not in sources:
                problems.append(f"{row[0]}: {num} not found in {links}")
    assert not problems, "\n".join(problems)


def test_thresholds_match_hypotheses(text: str, catalog: dict) -> None:
    hyp = HYPOTHESES.read_text(encoding="utf-8")
    for quote in ("hit ≥ 0.582", "below 0.408", "≤ 25%", "≥ 50%", "< 8.5 GB/token", "KLD mean < 0.001",
                  "coverage@32 ≥ 0.70", "≥ 1.15 ×", "≥ 20% fewer bytes", "±30%", "±35%", "1.7–3.0"):
        assert quote in text, f"README no longer quotes the threshold {quote!r}"
    for number in ("0.582", "0.408", "25%", "50%", "8.5 GB", "0.001", "0.70", "1.15", "20%"):
        assert number in hyp, f"HYPOTHESES.md does not contain {number!r}; the README quote is unsourced"
    assert "NO_ADVANTAGE" in hyp and "NOT_SUPPORTED" in hyp


def test_plain_statements(text: str) -> None:
    sec = section(text, "Plain statements")
    assert "cannot reach good speed" in sec and "31B" in sec
    assert "H2 is a measurement, not a promise" in sec
    assert "MoE is the realistic tier" in sec
    assert "16 GB + NVMe Gen3" in sec


def test_safety_names_the_protected_host(text: str, catalog: dict) -> None:
    sec = section(text, "Safety")
    for host in catalog["protected_hosts"]:
        assert host in sec, host
    for needle in ("PIGEMMA_I_KNOW=1", "nice -n 10", "2 GB free", "no mlock"):
        assert needle in sec, needle


def test_only_known_experiment_ids(text: str) -> None:
    from pigemma import results
    ids = {int(n) for n in re.findall(r"\bE(\d+)\b", text)}
    assert ids and max(ids) <= 8, sorted(ids)
    for value in re.findall(r"--experiment\s+([^\s`]+)", text):
        for part in value.split(","):
            assert results.normalize_experiment(part) in results.EXPERIMENTS


def test_model_ids_exist_in_catalog(text: str, catalog: dict) -> None:
    known = {m["id"] for m in catalog["models"]} | {"26b", "31b"}
    commands = [ln for _, lines in fenced_blocks(text) for ln in lines] + inline_code(text)
    bad = []
    for cmd in commands:
        for val in re.findall(r"(?:--model|--want|pigemma\.gguf_map build|pigemma\.paths model)\s+(\S+)", cmd):
            if any(c in val for c in "/\\<.") or val.startswith("$"):
                continue
            if val not in known:
                bad.append(val)
    assert not bad, f"unknown model ids: {sorted(set(bad))}"


# ---------------------------------------------------------------- command flags vs the real tools
_SHELL_STOP = {"|", "||", "&&", ";", ">", ">>", "2>", "&", "#"}


def _long_flags(tokens: list[str]) -> list[str]:
    out = []
    for t in tokens:
        if t == "--" or t in _SHELL_STOP or t.startswith("#"):
            break
        m = re.match(r"^(--[a-z][a-z0-9-]*)(?:=.*)?$", t)
        if m:
            out.append(m.group(1))
    return out


@lru_cache(maxsize=None)
def _help_text(module: str, sub: str | None) -> tuple[int, str]:
    args = [sys.executable, "-m", f"pigemma.{module}"] + ([sub] if sub else []) + ["--help"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    p = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=120, env=env)
    return p.returncode, p.stdout + p.stderr


def _check_python_module(cmd: str, problems: list[str]) -> None:
    for m in re.finditer(r"python3?\s+-m\s+pigemma\.([\w.]+)((?:\s+\S+)*)", cmd):
        module, rest = m.group(1), m.group(2).split()
        src = REPO_ROOT / "pigemma" / (module.replace(".", "/") + ".py")
        assert src.is_file(), f"README runs pigemma.{module}, which does not exist"
        sub = rest[0] if rest and re.fullmatch(r"[a-z][a-z0-9-]*", rest[0]) else None
        code, help_text = _help_text(module, sub)
        if code != 0:
            problems.append(f"`python -m pigemma.{module} {sub or ''} --help` exits {code}")
            continue
        for flag in _long_flags(rest[1:] if sub else rest):
            if flag not in help_text:
                problems.append(f"pigemma.{module} {sub or ''}: unknown flag {flag}")


def _check_scripts(cmd: str, problems: list[str]) -> None:
    for m in re.finditer(r"(?:bash\s+)((?:pc|pi)/[\w./-]+\.sh)((?:\s+\S+)*)", cmd):
        script, rest = REPO_ROOT / m.group(1), m.group(2).split()
        if "bash" in rest:
            rest = rest[:rest.index("bash")]
        assert script.is_file(), f"README runs {m.group(1)}, which does not exist"
        body = script.read_text(encoding="utf-8", errors="replace")
        for flag in _long_flags(rest):
            if flag not in body:
                problems.append(f"{m.group(1)}: unknown flag {flag}")
    for m in re.finditer(r"\.\\(pc\\[\w-]+\.ps1)((?:\s+\S+)*)", cmd):
        rel = m.group(1).replace("\\", "/")
        script = REPO_ROOT / rel
        assert script.is_file(), f"README runs {rel}, which does not exist"
        body = script.read_text(encoding="utf-8", errors="replace")
        for tok in m.group(2).split():
            if tok.startswith("#"):
                break
            if re.fullmatch(r"-[A-Z][A-Za-z]+", tok) and not re.search(r"\$" + tok[1:] + r"\b", body, re.I):
                problems.append(f"{rel}: unknown parameter {tok}")


def test_command_flags_exist_in_the_tools(text: str) -> None:
    problems: list[str] = []
    code_lines = [ln for _, lines in fenced_blocks(text) for ln in lines]
    assert any("pc/docker_run.sh build-tools" in ln for ln in code_lines)
    assert any("desktop_bench.ps1 -Stage bench,spec,quality" in ln for ln in code_lines)
    for cmd in code_lines + inline_code(text):
        _check_python_module(cmd, problems)
        _check_scripts(cmd, problems)
    assert not problems, "\n".join(sorted(set(problems)))


def test_pi_quick_start_runs_the_steps_in_order(text: str) -> None:
    sec = section(text, "Quick start: Raspberry Pi")
    block = next(lines for _, lines in fenced_blocks(sec))
    steps = [int(m.group(1)) for ln in block for m in [re.search(r"pi/0(\d)_", ln)] if m]
    first = [s for i, s in enumerate(steps) if s not in steps[:i]]
    assert first == sorted(first) == list(range(7)), f"pi steps out of order: {steps}"
    for n in range(7):
        assert list((REPO_ROOT / "pi").glob(f"0{n}_*.sh")), f"pi/0{n}_*.sh does not exist"
