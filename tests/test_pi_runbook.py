"""Checks for docs/PI_RUNBOOK.md (stdlib + pytest + pigemma.feasibility; no models, no network, ~10 s).

Manifest entry docs/PI_RUNBOOK.md, test field: "tests/test_docs.py link check; commands in the runbook are
the same strings as the scripts' usage lines (grep)". tests/test_docs.py is shared by the four documents
and owned by whoever writes it; this module holds the PI_RUNBOOK-specific checks so the files never
overwrite each other. The helpers are module-level and self-contained (no import of other test modules).

What is asserted:
  * UTF-8 without BOM, LF line endings, newline-terminated;
  * usage lines: every `Usage:` synopsis that a pi script prints with --help (pi/00..pi/06 and
    pi/lib/common.sh, read statically from the script source, with $VAR expanded from the script's own
    assignments) appears verbatim (whitespace-normalised) in a fenced block of the runbook;
  * every pi script invocation in the runbook's bash blocks and code spans names an existing script,
    a valid subcommand (pi/05: bench|stream-bench|chat|server, pi/06: start|stop|status|ab) and only
    long options that exist in that script's source;
  * every `python -m pigemma.<mod> [sub]` command uses only flags listed by its --help;
  * every long flag quoted on its own in a code span exists in at least one source of the project
    (pi/pc scripts, pigemma, tools, llama.cpp arg.cpp / llama-bench.cpp);
  * the flag strings per verdict equal pigemma.feasibility's FLAGS_* at THREADS=4, CTX=4096;
  * the "Hardware options" table names every RAM size, storage class, the AI HAT and the Active Cooler,
    quotes the catalog storage speeds, and its projected 26b-q4 tok/s and 31b-q4 s/token values equal
    what pigemma.feasibility computes today for each RAM x storage cell;
  * the context-friend-1 verdict rows in step 00 equal pigemma.feasibility on the real-Pi fixture
    (facts_pi5-16gb-microsd-real.json with the PIGEMMA_I_KNOW override), including pi/04's disk need;
  * MEASURED numbers quoted from results/pi-real/README.md and docs/design/FACTS.md occur in those files;
  * steps 00..06 appear in order and each lists its exit codes; the refusal rules name the catalog's
    protected hosts and service globs; every GFM table is rectangular;
  * relative links resolve (or point at a file planned in manifest.json files[]); '#anchor' links
    match a heading slug of the target document; model ids in code spans exist in the catalog.
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
RUNBOOK = REPO_ROOT / "docs" / "PI_RUNBOOK.md"
MANIFEST = REPO_ROOT / "docs" / "design" / "manifest.json"
CATALOG = REPO_ROOT / "pigemma" / "data" / "catalog.json"
REAL_PI_FACTS = REPO_ROOT / "tests" / "fixtures" / "facts_pi5-16gb-microsd-real.json"

USAGE_SCRIPTS = (
    "pi/00_feasibility.sh", "pi/01_tune_os.sh", "pi/02_build_llama.sh", "pi/03_bench_io_mem.sh",
    "pi/04_fetch_model.sh", "pi/05_run.sh", "pi/06_pin_hotset.sh", "pi/lib/common.sh",
)
SUBCOMMANDS = {
    "pi/05_run.sh": {"bench", "stream-bench", "chat", "server"},
    "pi/06_pin_hotset.sh": {"start", "stop", "status", "ab"},
}
# (quoted string, file it must also occur in): MEASURED numbers the runbook cites.
MEASURED_QUOTES = (
    ("91.6 MB/s", "results/pi-real/README.md"),
    ("410 s", "results/pi-real/README.md"),
    ("355 MB", "results/pi-real/README.md"),
    ("138 s", "results/pi-real/README.md"),
    ("148 s", "results/pi-real/README.md"),
    ("125 s", "results/pi-real/README.md"),
    ("1692 s", "results/pi-real/README.md"),
    ("0.53 tok/s", "results/pi-real/README.md"),
    ("3.17–3.27 tok/s", "results/pi-real/README.md"),
    ("9.1–9.3 tok/s", "results/pi-real/README.md"),
    ("2.50 tok/s", "results/pi-real/README.md"),
    ("0.55–0.79", "results/pi-real/README.md"),
    ("2.36", "results/pi-real/README.md"),
    ("4.14", "results/pi-real/README.md"),
    ("72.5", "results/pi-real/README.md"),
    ("35/36", "results/pi-real/README.md"),
    ("34/36", "results/pi-real/README.md"),
    ("+34 %", "results/pi-real/README.md"),
    ("−29 %", "results/pi-real/README.md"),
    ("−6 %", "results/pi-real/README.md"),
    ("0.44 GB", "results/pi-real/README.md"),
    ("0.99 GB", "results/pi-real/README.md"),
    ("8192 kB", "docs/design/FACTS.md"),
    ("15,501,496,320", "docs/design/FACTS.md"),
    ("66 %", "docs/design/FACTS.md"),
)
HARDWARE_TERMS = ("4 GB", "8 GB", "16 GB", "microSD", "USB 3.0 SSD", "NVMe Gen2", "NVMe Gen3",
                  "AI HAT", "Hailo", "Active Cooler", "dual-slot", "pciex1_gen=3", "recommended")
CF1_ROWS = ("26b-q4", "26b-ud-iq4_xs", "26b-ud-q3_k_m", "31b-ud-iq3_xxs", "31b-q4")

# ------------------------------------------------------------------------------------------ helpers
_FENCE = re.compile(r"^(```|~~~)")


def norm_ws(text: str) -> str:
    return " ".join(text.split())


def fenced_blocks(text: str) -> list[tuple[str, list[str]]]:
    """[(info string, lines)] for every fenced code block (indented fences included)."""
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


def strip_fenced(text: str) -> str:
    lines, inside = [], False
    for line in text.splitlines():
        if _FENCE.match(line.strip()):
            inside = not inside
            lines.append("")
            continue
        lines.append("" if inside else line)
    return "\n".join(lines)


def inline_code(text: str) -> list[str]:
    return re.findall(r"(?<!`)`([^`\n]+)`(?!`)", strip_fenced(text))


def markdown_links(text: str) -> list[str]:
    body = re.sub(r"`[^`\n]*`", "", strip_fenced(text))
    return re.findall(r"!?\[[^\]\n]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", body)


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


def md_tables(text: str) -> list[tuple[int, list[str]]]:
    tables: list[tuple[int, list[str]]] = []
    cur: list[str] = []
    start = 0
    for i, line in enumerate(strip_fenced(text).splitlines() + [""], 1):
        if line.lstrip().startswith("|"):
            if not cur:
                start = i
            cur.append(line)
        elif cur:
            tables.append((start, cur))
            cur = []
    return tables


def section(text: str, heading_re: str) -> str:
    """Body of the first heading matching heading_re, up to the next heading of the same or a higher
    level (fenced blocks are skipped when looking for headings)."""
    lines = text.splitlines()
    inside = False
    start = level = None
    for i, line in enumerate(lines):
        if _FENCE.match(line.strip()):
            inside = not inside
            continue
        if inside:
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if not m:
            continue
        if start is None:
            if re.search(heading_re, m.group(2)):
                start, level = i + 1, len(m.group(1))
        elif len(m.group(1)) <= level:
            return "\n".join(lines[start:i])
    assert start is not None, f"no heading matches {heading_re!r}"
    return "\n".join(lines[start:])


def usage_entries(rel: str) -> list[str]:
    """The synopsis lines of the script's --help 'Usage:' block, whitespace-normalised.

    Starts at the first line that begins with 'Usage:' (the usage() heredoc; header comments start
    with '#'). An entry starts with 'pi/..sh' or 'bash pi/..sh'; following lines that start with '['
    continue it; a description after 3+ spaces is dropped; anything else ends the block. $VAR and
    ${VAR} are expanded from VAR="..." assignments in the same file (PI05_USAGE_LINE, PI06_USAGE_LINE).
    """
    text = (REPO_ROOT / rel).read_text(encoding="utf-8")
    lines = text.splitlines()
    var = dict(re.findall(r'^([A-Z][A-Z0-9_]*)="([^"]*)"\s*$', text, re.M))

    def expand(s: str) -> str:
        return re.sub(r"\$\{?([A-Z][A-Z0-9_]*)\}?", lambda m: var.get(m.group(1), m.group(0)), s)

    start = next((i for i, ln in enumerate(lines) if re.match(r"^Usage:", ln)), None)
    if start is None:
        return []
    first = lines[start][len("Usage:"):]
    seq = ([first] if first.strip() else []) + lines[start + 1:]
    entries: list[str] = []
    cur: str | None = None
    for raw in seq:
        s = expand(raw).strip()
        if not s:
            if cur is not None or entries:
                break
            continue
        s = re.split(r"\s{3,}", s)[0]
        if re.match(r"^(bash\s+)?pi/(?:\d\d_\w+|lib/common)\.sh\b", s):
            if cur:
                entries.append(cur)
            cur = s
        elif s.startswith("[") and cur is not None:
            cur += " " + s
        else:
            break
    if cur:
        entries.append(cur)
    return [norm_ws(e) for e in entries]


_SCRIPT_RE = re.compile(r"(?<![\w/.])(pi/(?:\d\d_\w+|lib/common)\.sh)\b")
_STOP_TOKENS = {"|", "||", "&&", ";", "&", "--"}


def _strip_comment(line: str) -> str:
    return re.sub(r"(^|\s)#.*$", "", line)


def script_invocations(line: str) -> list[tuple[str, list[str]]]:
    """[(script, argument tokens)] for every pi script named in a shell line."""
    out = []
    line = _strip_comment(line)
    for m in _SCRIPT_RE.finditer(line):
        args = []
        for tok in line[m.end():].split():
            tok = tok.strip("\"'")
            if tok in _STOP_TOKENS or re.match(r"^(\d?>|\||&|\))", tok):
                break
            tok = tok.rstrip(")")
            args.append(tok)
            if _SCRIPT_RE.search(tok):
                break
        out.append((m.group(1), args))
    return out


def long_flags(tokens: list[str]) -> list[str]:
    flags = []
    for t in tokens:
        m = re.match(r"^(--[a-z][a-z0-9-]*)(?:=.*)?$", t)
        if m:
            flags.append(m.group(1))
    return flags


@lru_cache(maxsize=None)
def _help_text(module: str, sub: str | None) -> tuple[int, str]:
    args = [sys.executable, "-m", f"pigemma.{module}"] + ([sub] if sub else []) + ["--help"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    p = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=120, env=env)
    return p.returncode, p.stdout + p.stderr


@lru_cache(maxsize=None)
def _source_union() -> str | None:
    llama = [REPO_ROOT / "third_party/llama.cpp/common/arg.cpp",
             REPO_ROOT / "third_party/llama.cpp/tools/llama-bench/llama-bench.cpp"]
    if not all(p.is_file() for p in llama):
        return None
    files = list(llama)
    for pattern in ("pi/*.sh", "pi/lib/*.sh", "pc/*.sh", "pc/*.ps1", "pigemma/*.py", "pigemma/h2/*.py",
                    "tools/*/*.c", "tools/*/*.cpp", "pi/adhoc/*"):
        files += [p for p in REPO_ROOT.glob(pattern) if p.is_file()]
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in files)


def bash_lines(text: str) -> list[str]:
    return [ln for info, lines in fenced_blocks(text) if info in ("bash", "sh", "shell") for ln in lines]


def feasibility_module():
    from pigemma import feasibility
    return feasibility


def _grid_facts(kb: int, stype: str, speed: str | None) -> dict:
    """The synthetic Pi facts used by docs/HYPOTHESES.md for its RAM x storage grid."""
    return {
        "hostname": "pi-lab", "kernel": "6.18.0-v8-16k+", "page_size": 16384, "cpu_model": "Cortex-A76",
        "cpu_flags": ["fp", "asimd", "fphp", "asimdhp", "asimddp"], "nproc": 4,
        "ram_total_kb": kb, "ram_available_kb": kb - 500_000, "swap_total_kb": 0,
        "loadavg1": 0.2, "model_dir": "/nonexistent/pigemma-models", "free_disk_bytes": 400_000_000_000,
        "storage": {"dev": "/dev/x", "type": stype, "rotational": False, "pcie_link_speed": speed,
                    "pcie_link_width": 1 if speed else None, "read_ahead_kb": 128},
        "running_services": [], "hat_product": None, "pciex1_gen_cfg": None, "governor": "performance",
        "memlock_limit": "unlimited", "tools": {}, "override": None, "collected_at": "2026-10-05T00:00:00Z",
    }


def _numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", text.replace(",", ""))]


# ------------------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def text() -> str:
    assert RUNBOOK.is_file(), f"{RUNBOOK} is missing"
    return RUNBOOK.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def catalog() -> dict:
    return json.loads(CATALOG.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------ tests
def test_runbook_is_utf8_lf_no_bom() -> None:
    raw = RUNBOOK.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "BOM"
    assert b"\r" not in raw, "CR line endings"
    assert raw.endswith(b"\n")
    raw.decode("utf-8")


@pytest.mark.parametrize("script", USAGE_SCRIPTS)
def test_usage_lines_match_scripts(text: str, script: str) -> None:
    entries = usage_entries(script)
    assert entries, f"{script}: no 'Usage:' synopsis found in the script (extraction broke?)"
    code = norm_ws("\n".join(ln for _, lines in fenced_blocks(text) for ln in lines))
    missing = [e for e in entries if e not in code]
    assert not missing, f"{script}: usage line(s) not copied verbatim into the runbook: {missing}"


def test_pi_commands_use_real_options(text: str) -> None:
    problems: list[str] = []
    seen = 0
    for line in bash_lines(text) + inline_code(text):
        for script, args in script_invocations(line):
            seen += 1
            path = REPO_ROOT / script
            if not path.is_file():
                problems.append(f"{script}: no such script ({line.strip()})")
                continue
            body = path.read_text(encoding="utf-8", errors="replace")
            if script in SUBCOMMANDS:
                pos = [a for a in args if not a.startswith("-")]
                if args and not args[0].startswith("-") and args[0] not in SUBCOMMANDS[script]:
                    problems.append(f"{script}: unknown subcommand {args[0]!r} ({line.strip()})")
                elif not args and not pos and "--help" not in line:
                    pass  # a bare mention such as `pi/05_run.sh` in prose
            for flag in long_flags(args):
                if flag not in body:
                    problems.append(f"{script}: unknown option {flag} ({line.strip()})")
    assert seen >= 30, f"only {seen} pi script invocations found: the parser or the runbook changed"
    assert not problems, "\n".join(sorted(set(problems)))


def test_python_commands_use_real_options(text: str) -> None:
    problems: list[str] = []
    seen = 0
    for cmd in bash_lines(text) + inline_code(text):
        cmd = _strip_comment(cmd)
        for m in re.finditer(r"python3?\s+-m\s+pigemma\.([\w.]+)((?:\s+[^\s|;&>]+)*)", cmd):
            module, rest = m.group(1), m.group(2).split()
            if not (REPO_ROOT / "pigemma" / (module.replace(".", "/") + ".py")).is_file():
                problems.append(f"pigemma.{module}: no such module")
                continue
            seen += 1
            sub = rest[0] if rest and re.fullmatch(r"[a-z][a-z0-9-]*", rest[0]) else None
            code, help_text = _help_text(module, sub)
            if code != 0:
                problems.append(f"`python -m pigemma.{module} {sub or ''} --help` exits {code}")
                continue
            for flag in long_flags(rest[1:] if sub else rest):
                if flag not in help_text:
                    problems.append(f"pigemma.{module} {sub or ''}: unknown flag {flag}")
    assert seen >= 5, f"only {seen} python -m pigemma commands found"
    assert not problems, "\n".join(sorted(set(problems)))


def test_quoted_flags_exist_somewhere(text: str) -> None:
    union = _source_union()
    if union is None:
        pytest.skip("third_party/llama.cpp sources not present")
    bad = []
    for span in inline_code(text):
        if not span.startswith("-"):
            continue
        for flag in re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]+)", span):
            if flag not in union:
                bad.append(flag)
    assert not bad, f"flags quoted in the runbook that no project or llama.cpp source knows: {sorted(set(bad))}"


def test_flags_per_verdict_match_feasibility(text: str) -> None:
    F = feasibility_module()
    sec = section(text, r"^8\. llama\.cpp flags per verdict")
    for name in ("FLAGS_FITS_REPACKED", "FLAGS_FITS_MAPPED", "FLAGS_STREAM"):
        flags = getattr(F, name).format(t=4, c=4096)
        assert f"`{flags}`" in sec, f"{name} ({flags}) is not quoted in section 8"
    assert F.flags_for(F.FITS_MAPPED, fits_repacked=False, threads=4, n_ctx=4096) in text
    assert F.flags_for(F.STREAM_OK, threads=4, n_ctx=4096) in text


def test_hardware_options_table(text: str, catalog: dict) -> None:
    F = feasibility_module()
    sec = section(text, r"^3\. Hardware options")
    for term in HARDWARE_TERMS:
        assert term in sec, f"'Hardware options' lacks {term!r}"
    sc = catalog["storage_classes"]
    for cls in ("microsd", "usb-ssd", "nvme-gen2", "nvme-gen3"):
        v = sc[cls]["seq_read_mbps"]["value"]
        assert re.search(rf"(?<![\d.]){v:g}(?: / \d+)? MB/s", sec), f"catalog {cls} seq read {v} MB/s not quoted"
    numbers = _numbers(sec)
    cat = F.load_catalog(CATALOG)
    q4 = next(m for m in catalog["models"] if m["id"] == "31b-q4")
    storages = (("microsd", None), ("usb-ssd", None), ("nvme", "5.0 GT/s PCIe"), ("nvme", "8.0 GT/s PCIe"))
    for kb in (3_906_250, 7_812_500, 15_625_000):
        for stype, speed in storages:
            v = F.verdict(_grid_facts(kb, stype, speed), copy.deepcopy(cat), check_present=False)
            row = next(r for r in v.rows if r["model_id"] == "26b-q4")
            tok = row["tok_s"]
            assert any(abs(n - tok) <= 0.0051 for n in numbers), \
                f"26b-q4 at {kb} kB / {stype} {speed}: projected {tok:.3f} tok/s not in the table"
            assert row["verdict"] in sec, row["verdict"]
    for cls in ("microsd", "usb-ssd", "nvme-gen2", "nvme-gen3"):
        s = q4["bytes"] / (sc[cls]["seq_read_mbps"]["value"] * 1e6)
        assert any(abs(n - s) <= 0.051 for n in numbers), f"31b-q4 {cls}: {s:.1f} s/token not in the table"


def test_context_friend_rows_match_feasibility(text: str, catalog: dict) -> None:
    F = feasibility_module()
    facts = json.loads(REAL_PI_FACTS.read_text(encoding="utf-8"))
    facts["override"] = "PIGEMMA_I_KNOW"
    v = F.verdict(facts, copy.deepcopy(F.load_catalog(CATALOG)), check_present=False)
    assert v.exit_code == 0 and v.recommended and v.recommended["model_id"] == "26b-q4"
    rows = {r["model_id"]: r for r in v.rows}
    sec = section(text, r"^Step 00")
    free = facts["free_disk_bytes"]
    for mid in CF1_ROWS:
        line = next((ln for ln in sec.splitlines() if ln.startswith(f"| `{mid}`")), None)
        assert line is not None, f"step 00 table has no row for {mid}"
        r = rows[mid]
        assert r["verdict"] in line, f"{mid}: verdict {r['verdict']} not in {line!r}"
        tok = r["tok_s"]
        tok_s = f"{tok:.3f}" if tok < 0.01 else f"{tok:.2f}"
        assert tok_s in line, f"{mid}: projected {tok_s} tok/s not in {line!r}"
        b = next(m["bytes"] for m in catalog["models"] if m["id"] == mid)
        need = max(1.1 * b, b + 2e9)  # pi/04: MIN_FREE_DISK_FACTOR 1.1, KEEP_FREE_GB 2 on /
        if mid != "31b-q4":
            assert f"{need / 1e9:.2f} GB" in line, f"{mid}: pi/04 need {need / 1e9:.2f} GB not in {line!r}"
            fits = need <= free
            assert ("fits" in line and "refused" not in line) == fits, f"{mid}: disk verdict wrong in {line!r}"
    assert f"{free / 1e9:.2f} GB free" in sec


def test_measured_quotes_exist_in_sources(text: str) -> None:
    for quote, src in MEASURED_QUOTES:
        assert quote in text, f"runbook no longer quotes {quote!r} (update MEASURED_QUOTES)"
        body = (REPO_ROOT / src).read_text(encoding="utf-8")
        assert quote in body, f"{quote!r} is quoted as MEASURED but does not occur in {src}"


def test_labels_and_safety_terms(text: str, catalog: dict) -> None:
    for label in ("MEASURED", "PROJECTED", "ASSUMED"):
        assert text.count(label) >= 5, label
    rules = section(text, r"^2\.2 Refusal rules")
    for host in catalog["protected_hosts"]:
        assert host in rules, host
    for glob in catalog["protected_service_globs"]:
        assert glob in rules, glob
    for term in ("PIGEMMA_I_KNOW=1", "Busy host", "Not enough RAM", "Not enough disk", "AI HAT",
                 "mlock not allowed", "Bench lock held", "exit 4", "exit 3"):
        assert term in rules, term
    prereq = section(text, r"^4\. Prerequisites")
    for term in ("build-essential cmake git fio python3", "Active Cooler", "NVMe HAT", "ulimit -l unlimited",
                 "log out and in again"):
        assert term in prereq, term
    trouble = section(text, r"^12\. Troubleshooting")
    for term in ("OOM on load", "ETIMEDOUT", "EPERM", "read_ahead", "Thermal throttling"):
        assert term in trouble, term


def test_steps_in_order_with_exit_codes(text: str) -> None:
    order = [int(n) for n in re.findall(r"^### Step 0(\d):", text, re.M)]
    assert order == list(range(7)), order
    for n, script in enumerate(USAGE_SCRIPTS[:7]):
        sec = section(text, rf"^Step 0{n}:")
        assert "Exit codes" in sec, f"step 0{n} lists no exit codes"
        assert script in sec, f"step 0{n} never names {script}"
        assert "```bash" in sec, f"step 0{n} shows no command"


def test_tables_are_rectangular(text: str) -> None:
    for start, rows in md_tables(text):
        widths = {len(split_row(r)) for r in rows}
        assert len(widths) == 1, f"table at line {start} has rows of different widths {sorted(widths)}"
        assert re.fullmatch(r"\|?(\s*:?-+:?\s*\|)+\s*:?-*:?\s*\|?", rows[1].strip()), f"line {start + 1}"


def test_relative_links_resolve(text: str, manifest: dict) -> None:
    planned = planned_paths(manifest)
    missing, pending = [], []
    for target in markdown_links(text):
        if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
            continue
        resolved = (RUNBOOK.parent / target.split("#", 1)[0]).resolve()
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
        print("planned, not written yet:", sorted(set(pending)))
    assert not missing, f"links to files that neither exist nor are planned: {missing}"


def test_anchor_links_resolve(text: str) -> None:
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
        doc = (RUNBOOK.parent / path_part).resolve()
        if doc.suffix == ".md" and doc.is_file() and anchor not in heading_slugs(doc.read_text(encoding="utf-8")):
            bad.append(target)
    assert not bad, bad


def test_model_ids_exist_in_catalog(text: str, catalog: dict) -> None:
    known = {m["id"] for m in catalog["models"]} | {"26b", "31b"}
    bad = []
    for cmd in bash_lines(text) + inline_code(text):
        for val in re.findall(r"--model\s+(\S+)", cmd):
            if any(c in val for c in "/\\<.~") or val.startswith("$"):
                continue
            if val not in known:
                bad.append(val)
    for span in inline_code(text):
        if re.fullmatch(r"(?:26b|31b)-[a-z0-9_]+(?:-[a-z0-9_]+)*", span) and span not in known:
            bad.append(span)
    assert not bad, f"unknown model ids: {sorted(set(bad))}"
