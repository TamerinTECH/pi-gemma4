"""Checks for docs/EXPERIMENT_PLAN.md (stdlib + pytest; no models, no network, about 10 s).

Manifest entry docs/EXPERIMENT_PLAN.md, test field: "tests/test_docs.py link check; experiment ids match
the ones used in results files (E0..E8)". tests/test_docs.py is shared by the four documents (README,
HYPOTHESES, EXPERIMENT_PLAN, PI_RUNBOOK) and owned by whoever writes it. This module holds the
EXPERIMENT_PLAN-specific checks so the two files never overwrite each other. The helpers
(unescape_md, norm_ws, markdown_links, github_slug, heading_slugs, planned_paths, md_tables,
fenced_blocks, inline_code) are module-level, so test_docs.py can import and reuse them.

What is asserted:
  * the file is UTF-8 without BOM, LF line endings, newline-terminated;
  * the experiment headings are exactly '## E<n>: <pigemma.results.EXPERIMENT_TITLES[E<n>]>' for
    E0..E8 in order (the same headings `python -m pigemma.results summarize` renders);
  * every manifest experiment id normalises (results.normalize_experiment) to its section, and its
    hypothesis, procedure, runtime_estimate, thresholds and each metric appear verbatim in that section
    (after undoing Markdown backslash escapes and collapsing whitespace); so does run_order_today;
  * every experiment section has the required sub-sections, exactly one results-template header
    'Metric | MEASURED (host, date, result file) | PROJECTED (inputs, assumptions) | ASSUMED (value,
    source)', one 'Verdict vs pre-registered threshold' line and one 'What would change this
    conclusion' line;
  * every GFM table is rectangular (the escaped pipes \\| inside cells are handled);
  * the six controls named by the manifest spec are in the Controls section; the rendered outputs
    (DESKTOP_<date>.md, H1_RESULTS.md, H2_RESULTS.md) are named;
  * relative links resolve to an existing file or to a file planned in manifest.json files[] (planned
    but not yet written files are reported, not failed); '#anchor' links match a heading slug;
  * only experiment ids E0..E8 appear; every --experiment value is valid; every registry id used
    with --model or written alone in a code span exists in pigemma/data/catalog.json;
  * every long flag in the plan's commands exists in the tool it is passed to, for each tool whose
    source exists today:
      - `python -m pigemma.<mod> <sub>`: checked against its --help output;
      - pc/ and pi/ scripts: checked against the script text;
      - pigemma-probe, pinner, membw and the llama.cpp binaries: checked against their C/C++ sources
        (fenced code blocks only).
    Tools that are not written yet are skipped, so this test starts catching drift as soon as they
    land.
"""

from __future__ import annotations

import html
import json
import os
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN = REPO_ROOT / "docs" / "EXPERIMENT_PLAN.md"
MANIFEST = REPO_ROOT / "docs" / "design" / "manifest.json"
CATALOG = REPO_ROOT / "pigemma" / "data" / "catalog.json"

TEMPLATE_HEADER = ("| Metric | MEASURED (host, date, result file) | PROJECTED (inputs, assumptions) "
                   "| ASSUMED (value, source) |")
VERDICT_LINE = "**Verdict vs pre-registered threshold:**"
CHANGE_LINE = "**What would change this conclusion:**"
REQUIRED_CONTROLS = (
    "hybrid_negctrl",
    "KLD of -nr vs default",
    "teacher-forced vs generated routing",
    "stack-distance vs simulator",
    "iso-byte",
    "ffn_out sanity",
)
RENDERED_OUTPUTS = ("docs/results/DESKTOP_<date>.md", "docs/results/H1_RESULTS.md",
                    "docs/results/H2_RESULTS.md")


# ------------------------------------------------------------------------------------------ helpers
_PUNCT_ESC = re.compile(r"\\([!-/:-@\[-`{-~])")
_FENCE = re.compile(r"^(```|~~~)")


def unescape_md(text: str) -> str:
    """Undo CommonMark backslash escapes of ASCII punctuation (\\* \\| \\. ...) and HTML character
    references (&lt; is how the plan writes a '<' that would otherwise start a raw HTML tag)."""
    return html.unescape(_PUNCT_ESC.sub(r"\1", text))


def norm_ws(text: str) -> str:
    return " ".join(text.split())


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


def strip_fenced(text: str) -> str:
    """The text with fenced code blocks replaced by blank lines (line numbers kept)."""
    lines, inside = [], False
    for line in text.splitlines():
        if _FENCE.match(line.strip()):
            inside = not inside
            lines.append("")
            continue
        lines.append("" if inside else line)
    return "\n".join(lines)


def inline_code(text: str) -> list[str]:
    """Contents of inline code spans outside fenced blocks (single-backtick spans)."""
    return re.findall(r"(?<!`)`([^`\n]+)`(?!`)", strip_fenced(text))


def markdown_links(text: str) -> list[str]:
    """Link targets of [text](target) and ![alt](target), outside code blocks and code spans."""
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
    """Repo-relative paths the manifest plans (incl. the '(+ a, b)' suffix files; globs skipped)."""
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
    """[(first line number, table lines)] for GFM tables outside fenced blocks."""
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


def h2_sections(text: str) -> dict[str, str]:
    """'## heading' -> body up to the next '## ' heading (fenced blocks kept)."""
    out: dict[str, str] = {}
    name, buf, inside = None, [], False
    for line in text.splitlines():
        if _FENCE.match(line.strip()):
            inside = not inside
        if not inside and line.startswith("## "):
            if name is not None:
                out[name] = "\n".join(buf)
            name, buf = line[3:].strip(), []
            continue
        if name is not None:
            buf.append(line)
    if name is not None:
        out[name] = "\n".join(buf)
    return out


# ------------------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def text() -> str:
    assert PLAN.is_file(), f"{PLAN} is missing"
    return PLAN.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def results_mod():
    from pigemma import results
    return results


@pytest.fixture(scope="module")
def exp_sections(text: str) -> dict[str, str]:
    out = {}
    for heading, body in h2_sections(text).items():
        m = re.match(r"^(E\d): ", heading)
        if m:
            out[m.group(1)] = body
    return out


# ------------------------------------------------------------------------------------------ tests
def test_plan_is_utf8_lf_no_bom() -> None:
    raw = PLAN.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "BOM at the start"
    assert b"\r" not in raw, "CR line endings"
    raw.decode("utf-8")
    assert raw.endswith(b"\n")


def test_experiment_headings_match_results_module(text: str, results_mod) -> None:
    found = re.findall(r"^## (E\d): (.+)$", strip_fenced(text), re.M)
    expected = [(e, results_mod.EXPERIMENT_TITLES[e]) for e in results_mod.EXPERIMENTS if e != "adhoc"]
    assert found == expected


def test_manifest_experiment_ids_normalise_to_their_section(manifest: dict, results_mod,
                                                            exp_sections: dict[str, str]) -> None:
    ids = [e["id"] for e in manifest["experiments"]]
    assert [results_mod.normalize_experiment(i) for i in ids] == [f"E{n}" for n in range(9)]
    for i in ids:
        short = results_mod.normalize_experiment(i)
        assert f"`{i}`" in exp_sections[short], f"manifest id {i} not named in section {short}"


def test_manifest_text_copied_verbatim(manifest: dict, results_mod, exp_sections: dict[str, str]) -> None:
    problems = []
    for e in manifest["experiments"]:
        short = results_mod.normalize_experiment(e["id"])
        sec = norm_ws(unescape_md(exp_sections[short]))
        fields = [("hypothesis", e["hypothesis"]), ("procedure", e["procedure"]),
                  ("runtime_estimate", e["runtime_estimate"]), ("thresholds", e["thresholds"])]
        fields += [(f"metrics[{k}]", m) for k, m in enumerate(e["metrics"])]
        for name, value in fields:
            if norm_ws(value) not in sec:
                problems.append(f"{short}.{name}")
    assert not problems, f"manifest text not found verbatim: {problems}"


def test_run_order_copied_verbatim(text: str, manifest: dict) -> None:
    body = norm_ws(unescape_md(text))
    missing = [item[:40] for item in manifest["run_order_today"] if norm_ws(item) not in body]
    assert not missing, missing


def test_each_experiment_has_required_parts(exp_sections: dict[str, str]) -> None:
    assert sorted(exp_sections) == [f"E{n}" for n in range(9)]
    for e, sec in exp_sections.items():
        for sub in ("### Hypothesis", "### Procedure", "### Metrics", "### Pre-registered thresholds",
                    f"### Results template ({e})"):
            assert sub in sec, f"{e}: missing '{sub}'"
        assert sec.count(TEMPLATE_HEADER) == 1, f"{e}: results template header missing or repeated"
        assert sec.count(VERDICT_LINE) == 1, f"{e}: verdict line"
        assert sec.count(CHANGE_LINE) == 1, f"{e}: 'what would change' line"
        t = sec.index(f"### Results template ({e})")
        assert t < sec.index(TEMPLATE_HEADER) < sec.index(VERDICT_LINE) < sec.index(CHANGE_LINE)


def test_results_template_rows_have_four_cells(exp_sections: dict[str, str]) -> None:
    for e, sec in exp_sections.items():
        tail = sec[sec.index(TEMPLATE_HEADER):]
        rows = []
        for line in tail.splitlines():
            if not line.startswith("|"):
                break
            rows.append(line)
        assert len(rows) >= 3, f"{e}: template has no metric rows"
        for r in rows:
            assert len(split_row(r)) == 4, f"{e}: row does not have 4 cells: {r[:80]}"


def test_all_tables_are_rectangular(text: str) -> None:
    for start, lines in md_tables(text):
        n = len(split_row(lines[0]))
        assert len(lines) >= 2 and re.fullmatch(r"\|(\s*:?-+:?\s*\|)+", lines[1].strip()), \
            f"table at line {start}: missing separator row"
        assert len(split_row(lines[1])) == n, f"table at line {start}: separator width"
        for k, line in enumerate(lines[2:], start + 2):
            assert len(split_row(line)) == n, f"table row at line {k} has {len(split_row(line))} cells, header {n}"


def test_controls_section_lists_required_controls(text: str) -> None:
    ctrl = h2_sections(text).get("Controls", "")
    missing = [c for c in REQUIRED_CONTROLS if c not in ctrl]
    assert not missing, missing


def test_rendered_outputs_named(text: str) -> None:
    for name in RENDERED_OUTPUTS:
        assert name in text, name


def test_relative_links_resolve(text: str, manifest: dict) -> None:
    planned = planned_paths(manifest)
    missing, pending = [], []
    for target in markdown_links(text):
        if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
            continue
        path_part = target.split("#", 1)[0]
        resolved = (PLAN.parent / path_part).resolve()
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
        doc = (PLAN.parent / path_part).resolve()
        if doc.suffix == ".md" and doc.is_file():
            if anchor not in heading_slugs(doc.read_text(encoding="utf-8")):
                bad.append(target)
    assert not bad, bad


def test_only_known_experiment_ids(text: str, results_mod) -> None:
    ids = {int(n) for n in re.findall(r"\bE(\d+)\b", text)}
    assert ids and max(ids) <= 8, sorted(ids)
    for value in re.findall(r"--experiment\s+([^\s`]+)", text):
        for part in value.split(","):
            assert results_mod.normalize_experiment(part) in results_mod.EXPERIMENTS


def test_model_ids_exist_in_catalog(text: str) -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    known = {m["id"] for m in catalog["models"]} | {"26b", "31b"}  # pi/04 --model, --want take families
    commands = [ln for _, lines in fenced_blocks(text) for ln in lines] + inline_code(text)
    bad = []
    for cmd in commands:
        for val in re.findall(r"--model\s+(\S+)", cmd):
            if any(c in val for c in "/\\<.") or val.startswith("$"):
                continue  # a path or a placeholder
            if val not in known:
                bad.append(val)
    for span in inline_code(text):
        if re.fullmatch(r"(?:26b|31b)-[a-z0-9_]+(?:-[a-z0-9_]+)*", span) and span not in known:
            bad.append(span)
    assert not bad, f"unknown model ids: {sorted(set(bad))}"


# ---------------------------------------------------------------- command flags vs the real tools
_SHELL_STOP = {"|", "||", "&&", ";", ">", ">>", "2>", "&"}
_BINARY_SOURCES = {
    "pigemma-probe": ("tools/probe/probe.cpp", "third_party/llama.cpp/common/arg.cpp"),
    "pinner": ("tools/pinner/pinner.c",),
    "membw": ("tools/membw/membw.c",),
    "llama-bench": ("third_party/llama.cpp/tools/llama-bench/llama-bench.cpp",),
    "llama-completion": ("third_party/llama.cpp/common/arg.cpp",),
    "llama-server": ("third_party/llama.cpp/common/arg.cpp",),
    "llama-perplexity": ("third_party/llama.cpp/common/arg.cpp",),
    "llama-cli": ("third_party/llama.cpp/common/arg.cpp",),
}


def _long_flags(tokens: list[str]) -> list[str]:
    out = []
    for t in tokens:
        if t == "--" or t in _SHELL_STOP:
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


@lru_cache(maxsize=None)
def _source_text(rel_paths: tuple[str, ...]) -> str | None:
    texts = [(REPO_ROOT / p).read_text(encoding="utf-8", errors="replace")
             for p in rel_paths if (REPO_ROOT / p).is_file()]
    return "\n".join(texts) if texts else None


def _check_python_module(cmd: str, problems: list[str], skipped: set[str]) -> None:
    for m in re.finditer(r"python3?\s+-m\s+pigemma\.([\w.]+)((?:\s+\S+)*)", cmd):
        module, rest = m.group(1), m.group(2).split()
        src = REPO_ROOT / "pigemma" / (module.replace(".", "/") + ".py")
        if not src.is_file():
            skipped.add(f"pigemma.{module}")
            continue
        sub = rest[0] if rest and re.fullmatch(r"[a-z][a-z0-9-]*", rest[0]) else None
        code, help_text = _help_text(module, sub)
        if code != 0:
            problems.append(f"`python -m pigemma.{module} {sub or ''} --help` exits {code}")
            continue
        for flag in _long_flags(rest[1:] if sub else rest):
            if flag not in help_text:
                problems.append(f"pigemma.{module} {sub or ''}: unknown flag {flag}")


def _check_scripts(cmd: str, problems: list[str], skipped: set[str]) -> None:
    for m in re.finditer(r"(?:bash\s+)((?:pc|pi)/[\w./-]+\.sh)((?:\s+\S+)*)", cmd):
        script, rest = REPO_ROOT / m.group(1), m.group(2).split()
        if "bash" in rest:  # a second script on the same line is matched separately
            rest = rest[:rest.index("bash")]
        if not script.is_file():
            skipped.add(m.group(1))
            continue
        body = script.read_text(encoding="utf-8", errors="replace")
        for flag in _long_flags(rest):
            if flag not in body:
                problems.append(f"{m.group(1)}: unknown flag {flag}")
    for m in re.finditer(r"\.\\(pc\\[\w-]+\.ps1)((?:\s+\S+)*)", cmd):
        rel = m.group(1).replace("\\", "/")
        script = REPO_ROOT / rel
        if not script.is_file():
            skipped.add(rel)
            continue
        body = script.read_text(encoding="utf-8", errors="replace")
        for tok in m.group(2).split():
            if re.fullmatch(r"-[A-Z][A-Za-z]+", tok) and not re.search(r"\$" + tok[1:] + r"\b", body, re.I):
                problems.append(f"{rel}: unknown parameter {tok}")


def _check_binaries(line: str, problems: list[str], skipped: set[str]) -> None:
    names = "|".join(map(re.escape, _BINARY_SOURCES))
    hits = list(re.finditer(rf"(?:^|[\s/\\])({names})(?:\.exe)?(?=\s)", line))
    for k, m in enumerate(hits):
        end = hits[k + 1].start() if k + 1 < len(hits) else len(line)
        tokens = line[m.end():end].split()
        src = _source_text(_BINARY_SOURCES[m.group(1)])
        if src is None:
            skipped.add(m.group(1))
            continue
        for flag in _long_flags(tokens):
            if f'"{flag}"' not in src and f'"{flag[2:]}"' not in src and flag not in src:
                problems.append(f"{m.group(1)}: unknown flag {flag}")


def test_command_flags_exist_in_written_tools(text: str) -> None:
    problems: list[str] = []
    skipped: set[str] = set()
    code_lines = [ln for _, lines in fenced_blocks(text) for ln in lines]
    for cmd in code_lines + inline_code(text):
        _check_python_module(cmd, problems, skipped)
        _check_scripts(cmd, problems, skipped)
    for line in code_lines:
        _check_binaries(line, problems, skipped)
    if skipped:
        print("not written yet, flags not checked:", sorted(skipped))
    assert not problems, "\n".join(sorted(set(problems)))
