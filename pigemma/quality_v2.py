"""pigemma.quality_v2 - harder held-out Q&A (data/prompts/qa_heldout_v2.json) with objective
graders, asked through llama-server's CHAT endpoint (brief 2026-10-05, section 7).

Scope: MODEL QUALITY only (is the answer right?). Exact-cache correctness - whether a caching
method reproduces the unchanged reference token for token - is a different check and is NOT
decided here; compare the raw ``content`` of the reference run and the method run (same
prompts, greedy) with a separate tool. A method can pass every item here and still be inexact.

Why the chat endpoint: ``POST /v1/chat/completions`` applies the GGUF's own chat template (Gemma
4: ``<|turn>user ... <turn|>``, thinking disabled through ``chat_template_kwargs.enable_thinking
= false``), which removes the hand-written-template markup glitches seen in the 2026-10-05 Pi QA
run. The thinking channel is separated by the server (``reasoning_format: deepseek`` ->
``message.reasoning_content``); anything left in ``content`` is stripped again here
(``<|channel>thought ... <channel|>``, end-of-turn tokens, a fabricated next turn, a stray
leading ``model`` line). Only the visible answer is graded.

Item format (``{"format": "pigemma-qa/2", "items": [...]}``)::

    {"id", "lang": en|he|de, "category": negation|numbers|multilingual|extraction|code,
     "prompt", "n_predict", "grader": {...}, "reference": "<a correct answer>",
     "wrong": ["<answers that must fail>", ...], "notes"}

Graders (``grader.kind``; all are deterministic and objective):

* ``contains`` {any: [..]}            normalised substring (pigemma.quality.normalize_text)
* ``regex`` {pattern}                 re.search on the visible text or either normalised form
* ``not_regex`` {pattern}             must match none of those forms
* ``number`` {value, tolerance, which: first|last}   number in the visible text (quality rules)
* ``script`` {script: hebrew|latin, min_fraction}    share of letters in that script
* ``lines`` / ``words`` {min, max}    non-empty line count / whitespace word count
* ``json_fields`` {fields: {key: spec}}  first JSON object in the answer; every field must match
  (spec = literal | {"any": [..]} | {"number": x, "tol": t} | {"set": [..]}); score = share matched
* ``python_exec`` {entry_point, tests: [{args, expect}], timeout_s, float_tol}  the answer's code
  is screened (AST: stdlib allow-list, no open/exec/eval/getattr/..., no dangerous dunders) and
  run in a separate ``python -I -S -B`` process in a temp dir with a timeout, restricted builtins
  and a guarded ``__import__`` (POSIX: CPU / address-space / file-size rlimits). This is defence
  in depth against accidental damage by model-written code, NOT a hardened security sandbox.
* ``all_of`` {checks: [..]}           every sub-check passes; score = mean of sub-scores

CLI (``python -m pigemma.quality_v2 <cmd>``)::

    selftest [--qa F]                       grade every reference (must pass) and every wrong
                                            answer (must fail); no server, no model; exit 1 on
                                            any mismatch
    run --url http://127.0.0.1:8080 [--qa F] [--ids a,b] [--label L] [--out F.json]
        [--thinking] [--n-predict-scale X] [--timeout S] [--seed 0]
                                            asks an ALREADY RUNNING llama-server (one request per
                                            item, cache_prompt=false, greedy) and writes a JSON
                                            report with raw outputs, timings and grades
    regrade REPORT.json [--qa F] [--out F]  re-grade the saved raw outputs offline

Stdlib only (runs on the Pi's Python 3.13). The ``run`` command never starts or stops a server.
"""

from __future__ import annotations

import argparse
import ast
import collections
import datetime as _dt
import hashlib
import http.client
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
import urllib.parse
from pathlib import Path
from typing import Any, Mapping, Sequence

from pigemma import paths
from pigemma import quality as q1

__all__ = [
    "QA_V2_PATH", "load_qa_v2", "visible_answer", "grade_item", "run_check", "extract_json",
    "extract_code", "screen_code", "run_python_tests", "numbers_in", "script_fraction",
    "ChatClient", "run_items", "summarize", "selftest", "main",
]

QA_V2_PATH = paths.PROMPTS_DIR / "qa_heldout_v2.json"
QA_V2_FORMAT = "pigemma-qa/2"
REPORT_FORMAT = "pigemma-quality-v2/1"
CATEGORIES = ("negation", "numbers", "multilingual", "extraction", "code")
CHECK_KINDS = ("contains", "regex", "not_regex", "number", "script", "lines", "words",
               "json_fields", "python_exec", "all_of")

ALLOWED_IMPORTS = frozenset({
    "math", "cmath", "re", "string", "itertools", "functools", "collections", "heapq", "bisect",
    "statistics", "fractions", "decimal", "datetime", "calendar", "typing", "dataclasses",
    "operator", "copy", "json", "unicodedata", "enum", "abc", "random", "numbers", "textwrap",
    "array", "__future__",
})
BANNED_NAMES = frozenset({
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "memoryview", "exit", "quit", "help", "__builtins__",
    "__loader__", "__spec__",
})
ALLOWED_DUNDERS = frozenset({
    "__init__", "__name__", "__doc__", "__len__", "__iter__", "__next__", "__eq__", "__ne__",
    "__lt__", "__le__", "__gt__", "__ge__", "__repr__", "__str__", "__hash__", "__contains__",
    "__getitem__", "__setitem__", "__post_init__", "__future__", "__main__",
})
RESULT_MARK = "@@PIGEMMA_RESULT@@ "
DEFAULT_TIMEOUT_S = 10.0


class QaV2Error(ValueError):
    """The Q&A file or a grader specification is malformed."""


# --------------------------------------------------------------------------------------------
# Q&A file
# --------------------------------------------------------------------------------------------
def _validate_check(spec: Any, where: str) -> None:
    if not isinstance(spec, Mapping) or spec.get("kind") not in CHECK_KINDS:
        raise QaV2Error(f"{where}: grader kind must be one of {CHECK_KINDS}, got {spec!r:.80}")
    k = spec["kind"]
    need = {"contains": ("any",), "regex": ("pattern",), "not_regex": ("pattern",),
            "number": ("value",), "script": ("script", "min_fraction"), "lines": (),
            "words": (), "json_fields": ("fields",), "python_exec": ("entry_point", "tests"),
            "all_of": ("checks",)}[k]
    miss = [n for n in need if n not in spec]
    if miss:
        raise QaV2Error(f"{where}: {k} needs {miss}")
    if k in ("regex", "not_regex"):
        re.compile(spec["pattern"])
    if k == "all_of":
        if not spec["checks"]:
            raise QaV2Error(f"{where}: all_of needs at least one check")
        for i, c in enumerate(spec["checks"]):
            _validate_check(c, f"{where}.checks[{i}]")
    if k == "python_exec":
        if not spec["tests"]:
            raise QaV2Error(f"{where}: python_exec needs tests")
        for t in spec["tests"]:
            if not isinstance(t, Mapping) or "args" not in t or "expect" not in t:
                raise QaV2Error(f"{where}: every python_exec test needs args and expect")
    if k == "script" and spec["script"] not in ("hebrew", "latin"):
        raise QaV2Error(f"{where}: script must be hebrew or latin")


def load_qa_v2(path: str | os.PathLike[str] = QA_V2_PATH, *,
               ids: Sequence[str] | None = None) -> list[dict[str, Any]]:
    p = Path(os.fspath(path))
    if not p.is_file():
        raise QaV2Error(f"Q&A file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8-sig"))
    if not isinstance(data, Mapping) or data.get("format") != QA_V2_FORMAT:
        raise QaV2Error(f"{p}: expected an object with format {QA_V2_FORMAT!r}")
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise QaV2Error(f"{p}: no items")
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for i, it in enumerate(items):
        if not isinstance(it, Mapping):
            raise QaV2Error(f"{p}: item {i} is not an object")
        for key in ("id", "lang", "category", "prompt", "n_predict", "grader", "reference"):
            if key not in it:
                raise QaV2Error(f"{p}: item {it.get('id', i)} lacks {key!r}")
        if it["id"] in seen:
            raise QaV2Error(f"{p}: duplicate id {it['id']}")
        if it["category"] not in CATEGORIES:
            raise QaV2Error(f"{p}: item {it['id']}: category {it['category']!r} not in {CATEGORIES}")
        if not isinstance(it["n_predict"], int) or it["n_predict"] <= 0:
            raise QaV2Error(f"{p}: item {it['id']}: n_predict must be a positive int")
        seen.add(it["id"])
        _validate_check(it["grader"], f"{p}: item {it['id']}")
        out.append(dict(it))
    if ids:
        unknown = [x for x in ids if x not in seen]
        if unknown:
            raise QaV2Error(f"unknown id(s): {unknown}")
        out = [it for it in out if it["id"] in set(ids)]
    return out


# --------------------------------------------------------------------------------------------
# Visible answer (thinking channel and markup stripped)
# --------------------------------------------------------------------------------------------
_FAKE_TURN_RE = re.compile(r"(?:<\|turn>|<start_of_turn>)\s*(?:user|system|model)\b")
_STRAY_MODEL_RE = re.compile(r"^\s*model[^\S\n]*\n")


def visible_answer(content: str | None, reasoning_content: str | None = None) -> tuple[str, int, list[str]]:
    """(visible text, thinking chars, cleanup notes) of one chat response.

    Uses pigemma.quality.strip_thinking (thinking blocks, orphan closers, end tokens), then cuts
    a fabricated next turn (``<|turn>user`` / ``<start_of_turn>user`` ...) and a stray leading
    ``model`` line. Only markup is removed; the answer text itself is never edited.
    """
    notes: list[str] = []
    raw = str(content or "")
    vis, think = q1.strip_thinking(raw)
    if reasoning_content:
        think += len(str(reasoning_content).strip())
    m = _FAKE_TURN_RE.search(vis)
    if m:
        vis = vis[:m.start()].rstrip()
        notes.append("cut fabricated turn")
    if _STRAY_MODEL_RE.match(vis):
        vis = _STRAY_MODEL_RE.sub("", vis, count=1)
        notes.append("removed stray 'model' line")
    if vis != raw.strip() and not notes:
        notes.append("markup stripped")
    return vis.strip(), think, notes


# --------------------------------------------------------------------------------------------
# Simple checks
# --------------------------------------------------------------------------------------------
def numbers_in(text: str) -> list[float]:
    """Every number of a text, read with pigemma.quality.first_number's rules, in order."""
    s = unicodedata.normalize("NFKC", str(text or ""))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Cf").replace("−", "-")
    s = q1._THOUSANDS_RE.sub(lambda m: m.group(1).replace(",", ""), s)  # noqa: SLF001
    out: list[float] = []
    pos = 0
    while True:
        m = q1._NUM_RE.search(s, pos)  # noqa: SLF001
        if not m:
            return out
        num, pos = m.group("num"), m.end()
        if "." not in num:
            d = q1._DEC_COMMA_RE.match(s, m.end())  # noqa: SLF001
            if d:
                num, pos = f"{num}.{d.group(1)}", d.end()   # '16,8' is one number, not 16 and 8
        try:
            v = float(num)
        except ValueError:
            continue
        out.append(-v if m.group("sign") else v)


def script_fraction(text: str, script: str) -> float | None:
    """Share of letters (Unicode L*) that belong to the script (hebrew: U+05D0-U+05EA and the
    presentation forms; latin: Unicode name starts with LATIN). None when there are no letters."""
    letters = [ch for ch in str(text or "") if unicodedata.category(ch).startswith("L")]
    if not letters:
        return None
    if script == "hebrew":
        hit = sum(1 for ch in letters if "א" <= ch <= "ת" or "יִ" <= ch <= "ﭏ")
    else:
        hit = sum(1 for ch in letters if unicodedata.name(ch, "").startswith("LATIN"))
    return hit / len(letters)


def _forms(vis: str) -> tuple[str, str, str]:
    return vis, q1.normalize_text(vis, "space"), q1.normalize_text(vis, "delete")


def _res(ok: bool, kind: str, reason: str, score: float | None = None, **extra: Any) -> dict[str, Any]:
    d = {"kind": kind, "pass": bool(ok), "score": float(score if score is not None else (1.0 if ok else 0.0)),
         "reason": reason}
    d.update(extra)
    return d


# --------------------------------------------------------------------------------------------
# JSON extraction / field matching
# --------------------------------------------------------------------------------------------
def extract_json(text: str) -> Any | None:
    """The first JSON object (or array) in a text (code fences allowed), or None."""
    s = str(text or "")
    dec = json.JSONDecoder()
    for m in re.finditer(r"[{\[]", s):
        try:
            obj, _ = dec.raw_decode(s, m.start())
        except ValueError:
            continue
        if isinstance(obj, (dict, list)):
            return obj
    return None


def _as_number(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        nums = numbers_in(v)
        return nums[0] if len(nums) == 1 else None
    return None


def _norm_str(v: Any) -> str:
    return q1.normalize_text(str(v), "space")


_TRUE = {"true", "yes", "ja", "כן", "1"}
_FALSE = {"false", "no", "nein", "לא", "0"}


def _field_match(got: Any, spec: Any, tol: float) -> bool:
    if isinstance(spec, Mapping):
        if "any" in spec:
            return any(_field_match(got, alt, tol) for alt in spec["any"])
        if "number" in spec:
            g = _as_number(got)
            return g is not None and abs(g - float(spec["number"])) <= float(spec.get("tol", tol)) + 1e-9
        if "set" in spec:
            if not isinstance(got, list) or len(got) != len(spec["set"]):
                return False
            left = list(spec["set"])
            for g in got:
                j = next((i for i, e in enumerate(left) if _field_match(g, e, tol)), None)
                if j is None:
                    return False
                left.pop(j)
            return True
        raise QaV2Error(f"unknown field spec {spec!r}")
    if isinstance(spec, bool):
        if isinstance(got, bool):
            return got is spec
        return _norm_str(got) in (_TRUE if spec else _FALSE)
    if isinstance(spec, (int, float)):
        g = _as_number(got)
        return g is not None and abs(g - float(spec)) <= tol + 1e-9
    if isinstance(spec, list):
        return (isinstance(got, list) and len(got) == len(spec)
                and all(_field_match(g, e, tol) for g, e in zip(got, spec)))
    if spec is None:
        return got is None
    return _norm_str(got) == _norm_str(spec)


def _check_json_fields(vis: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    obj = extract_json(vis)
    if not isinstance(obj, dict):
        return _res(False, "json_fields", "no JSON object in the answer")
    lower = {str(k).casefold(): v for k, v in obj.items()}
    tol = float(spec.get("tolerance", 0.001))
    per: dict[str, bool] = {}
    for key, fspec in spec["fields"].items():
        if key in obj:
            got = obj[key]
        elif key.casefold() in lower:
            got = lower[key.casefold()]
        else:
            per[key] = False
            continue
        per[key] = _field_match(got, fspec, tol)
    n_ok = sum(per.values())
    bad = [k for k, v in per.items() if not v]
    return _res(n_ok == len(per), "json_fields",
                "all fields match" if not bad else f"mismatched/missing: {', '.join(bad)}",
                score=n_ok / len(per), fields=per)


# --------------------------------------------------------------------------------------------
# Executable code checks
# --------------------------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)(?:```|\Z)", re.DOTALL)


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def extract_code(text: str, entry_point: str) -> str | None:
    """The answer's Python code that defines ``entry_point`` (fenced block preferred; an
    unterminated fence of a truncated answer is accepted; otherwise from the first top-level
    ``def``/``import``/``from``/``@`` line, trimmed at the end until it parses)."""
    s = str(text or "").replace("\r\n", "\n")
    want = f"def {entry_point}"
    blocks = [m.group(2) for m in _FENCE_RE.finditer(s)
              if m.group(1).lower() in ("", "python", "py", "python3")]
    for b in blocks:
        if want in b:
            return b.strip("\n")
    if want not in s:
        return None
    lines = s.split("\n")
    start = next((i for i, ln in enumerate(lines)
                  if re.match(r"(def |import |from |@|class )", ln) and (want in "\n".join(lines[i:]))), None)
    if start is None:
        return None
    chunk = lines[start:]
    while chunk:
        code = "\n".join(chunk)
        if _parses(code) and want in code:
            return code
        chunk = chunk[:-1]
    return None


def screen_code(code: str) -> list[str]:
    """Static safety screen; returns the violations ([] = allowed to run)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error: {e.msg} (line {e.lineno})"]
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    bad.append(f"import {a.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                bad.append(f"from {node.module} import ...")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            bad.append(f"name {node.id}")
        elif isinstance(node, ast.Name) and node.id.startswith("__") and node.id not in ALLOWED_DUNDERS:
            bad.append(f"name {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr not in ALLOWED_DUNDERS:
            bad.append(f"attribute .{node.attr}")
    return sorted(set(bad))


_RUNNER = r'''
import json, sys, copy
MARK = "@@PIGEMMA_RESULT@@ "

def canon(v, depth=0):
    if depth > 50:
        return repr(v)[:200]
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return v if v == v and v not in (float("inf"), float("-inf")) else repr(v)
    if isinstance(v, (list, tuple)):
        return [canon(x, depth + 1) for x in v]
    if isinstance(v, (set, frozenset)):
        items = [canon(x, depth + 1) for x in v]
        try:
            return sorted(items)
        except TypeError:
            return items
    if isinstance(v, dict):
        return {"__dict__": [[canon(k, depth + 1), canon(x, depth + 1)] for k, x in v.items()]}
    return {"__repr__": repr(v)[:200]}

def emit(obj):
    sys.stdout.write("\n" + MARK + json.dumps(obj) + "\n")
    sys.stdout.flush()

def main():
    with open(sys.argv[1], encoding="utf-8") as f:
        payload = json.load(f)
    allowed = set(payload["allowed_imports"])
    import builtins as _b
    real_import = _b.__import__
    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level != 0 or name.split(".")[0] not in allowed:
            raise ImportError("import of %r is not allowed in the grader sandbox" % (name,))
        return real_import(name, globals, locals, fromlist, level)
    safe = {k: v for k, v in vars(_b).items() if k not in set(payload["banned_builtins"])}
    safe["__import__"] = guarded_import
    ns = {"__builtins__": safe, "__name__": "candidate"}
    out = {"loaded": False, "results": []}
    try:
        exec(compile(payload["code"], "<candidate>", "exec"), ns)
    except BaseException as e:
        out["load_error"] = ("%s: %s" % (type(e).__name__, e))[:300]
        emit(out)
        return
    fn = ns.get(payload["entry_point"])
    if not callable(fn):
        out["load_error"] = "entry point %r is not defined" % (payload["entry_point"],)
        emit(out)
        return
    out["loaded"] = True
    for t in payload["tests"]:
        try:
            got = fn(*copy.deepcopy(t["args"]))
            out["results"].append({"ok": True, "got": canon(got)})
        except BaseException as e:
            out["results"].append({"ok": False, "error": ("%s: %s" % (type(e).__name__, e))[:200]})
    emit(out)

main()
'''


def _values_equal(got: Any, exp: Any, tol: float) -> bool:
    if isinstance(exp, bool) or isinstance(got, bool):
        return isinstance(exp, bool) and isinstance(got, bool) and exp is got
    if isinstance(exp, (int, float)) and isinstance(got, (int, float)):
        return math.isclose(float(got), float(exp), rel_tol=1e-9, abs_tol=tol)
    if isinstance(exp, list):
        return (isinstance(got, list) and len(got) == len(exp)
                and all(_values_equal(g, e, tol) for g, e in zip(got, exp)))
    if isinstance(exp, Mapping):
        if isinstance(got, Mapping) and "__dict__" in got:
            pairs = got["__dict__"]
            if len(pairs) != len(exp):
                return False
            return all(str(k) in exp and _values_equal(v, exp[str(k)], tol) for k, v in pairs)
        return False
    return got == exp


def _limits(timeout_s: float) -> Any:
    if os.name != "posix":
        return None

    def _apply() -> None:  # pragma: no cover - runs in the child on POSIX only
        import resource  # noqa: PLC0415
        cpu = int(math.ceil(timeout_s)) + 1
        for lim, val in ((resource.RLIMIT_CPU, cpu), (resource.RLIMIT_AS, 1 << 30),
                         (resource.RLIMIT_FSIZE, 1 << 20)):
            try:
                resource.setrlimit(lim, (val, val))
            except (ValueError, OSError):
                pass
        try:
            os.nice(10)
        except OSError:
            pass
    return _apply


def run_python_tests(code: str, entry_point: str, tests: Sequence[Mapping[str, Any]], *,
                     timeout_s: float = DEFAULT_TIMEOUT_S, float_tol: float = 1e-6) -> dict[str, Any]:
    """Screen ``code``, run it in a sandboxed subprocess and compare ``entry_point(*args)``
    with every test's ``expect``. Returns {ran, passed, total, pass, violations, error,
    timeout, per_test}."""
    total = len(tests)
    out: dict[str, Any] = {"ran": False, "passed": 0, "total": total, "pass": False,
                           "violations": [], "error": None, "timeout": False, "per_test": []}
    viol = screen_code(code)
    if viol:
        out["violations"] = viol
        out["error"] = "sandbox screen refused the code: " + "; ".join(viol[:5])
        return out
    payload = {"code": code, "entry_point": entry_point,
               "tests": [{"args": list(t["args"])} for t in tests],
               "allowed_imports": sorted(ALLOWED_IMPORTS),
               "banned_builtins": sorted(BANNED_NAMES - {"__builtins__", "__loader__", "__spec__"})}
    env = {"PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
    for k in ("SYSTEMROOT", "SystemRoot", "PATH"):
        if k in os.environ:
            env[k] = os.environ[k]
    with tempfile.TemporaryDirectory(prefix="pigemma-qa2-") as td:
        runner = Path(td) / "runner.py"
        runner.write_text(_RUNNER, encoding="utf-8")
        pfile = Path(td) / "payload.json"
        pfile.write_text(json.dumps(payload), encoding="utf-8")
        ofile = Path(td) / "stdout.txt"
        with open(ofile, "wb") as so:
            try:
                cp = subprocess.run([sys.executable, "-I", "-S", "-B", str(runner), str(pfile)],
                                    cwd=td, env=env, stdin=subprocess.DEVNULL, stdout=so,
                                    stderr=subprocess.STDOUT, timeout=timeout_s,
                                    preexec_fn=_limits(timeout_s), check=False)  # noqa: PLW1509
                rc = cp.returncode
            except subprocess.TimeoutExpired:
                out["timeout"] = True
                out["error"] = f"timeout after {timeout_s:g} s"
                return out
        with open(ofile, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 65536))
            tail = f.read().decode("utf-8", errors="replace")
    line = next((ln for ln in reversed(tail.splitlines()) if ln.startswith(RESULT_MARK)), None)
    if line is None:
        out["error"] = f"no result from the sandbox (rc {rc}): {tail[-300:]}"
        return out
    res = json.loads(line[len(RESULT_MARK):])
    out["ran"] = True
    if not res.get("loaded"):
        out["error"] = res.get("load_error") or "code did not load"
        return out
    for t, r in zip(tests, res.get("results", [])):
        ok = bool(r.get("ok")) and _values_equal(r.get("got"), t["expect"], float_tol)
        out["per_test"].append({"args": t["args"], "expect": t["expect"],
                                "got": r.get("got"), "error": r.get("error"), "pass": ok})
    out["passed"] = sum(1 for r in out["per_test"] if r["pass"])
    out["pass"] = out["passed"] == total
    return out


# --------------------------------------------------------------------------------------------
# Grading
# --------------------------------------------------------------------------------------------
def run_check(spec: Mapping[str, Any], vis: str) -> dict[str, Any]:
    """Run one grader specification on a visible answer -> {kind, pass, score, reason, ...}."""
    k = spec["kind"]
    if k == "contains":
        for mode in ("space", "delete"):
            hay = q1.normalize_text(vis, mode)
            for e in spec["any"]:
                needle = q1.normalize_text(str(e), mode)
                if needle and needle in hay:
                    return _res(True, k, f"contains {e!r}")
        return _res(False, k, f"none of {spec['any']!r} found")
    if k in ("regex", "not_regex"):
        rx = re.compile(spec["pattern"])
        hit = any(rx.search(f) for f in _forms(vis))
        ok = hit if k == "regex" else not hit
        return _res(ok, k, f"{'matched' if hit else 'no match'}: {spec['pattern']}")
    if k == "number":
        nums = numbers_in(vis)
        if not nums:
            return _res(False, k, "no number in the answer")
        v = nums[-1] if spec.get("which") == "last" else nums[0]
        tol = float(spec.get("tolerance", 0.0))
        ok = abs(v - float(spec["value"])) <= tol + 1e-9
        return _res(ok, k, f"read {v:g}, expected {spec['value']} +- {tol:g}", value=v)
    if k == "script":
        fr = script_fraction(vis, spec["script"])
        ok = fr is not None and fr >= float(spec["min_fraction"])
        return _res(ok, k, f"{spec['script']} letter share {fr if fr is None else round(fr, 3)}"
                    f" (min {spec['min_fraction']})", fraction=fr)
    if k in ("lines", "words"):
        n = (len([ln for ln in vis.splitlines() if ln.strip()]) if k == "lines" else len(vis.split()))
        lo, hi = spec.get("min", 0), spec.get("max", 10 ** 9)
        return _res(lo <= n <= hi, k, f"{n} {k} (allowed {lo}..{hi})", count=n)
    if k == "json_fields":
        return _check_json_fields(vis, spec)
    if k == "python_exec":
        code = extract_code(vis, spec["entry_point"])
        if code is None:
            return _res(False, k, f"no code defining {spec['entry_point']} in the answer")
        r = run_python_tests(code, spec["entry_point"], spec["tests"],
                             timeout_s=float(spec.get("timeout_s", DEFAULT_TIMEOUT_S)),
                             float_tol=float(spec.get("float_tol", 1e-6)))
        reason = (f"{r['passed']}/{r['total']} tests passed" if r["ran"] and not r["error"]
                  else str(r["error"]))
        return _res(r["pass"], k, reason, score=(r["passed"] / r["total"]) if r["total"] else 0.0,
                    exec=r)
    if k == "all_of":
        subs = [run_check(c, vis) for c in spec["checks"]]
        ok = all(s["pass"] for s in subs)
        failed = [s["reason"] for s in subs if not s["pass"]]
        return _res(ok, k, "all checks passed" if ok else "; ".join(failed),
                    score=sum(s["score"] for s in subs) / len(subs), checks=subs)
    raise QaV2Error(f"unknown grader kind {k!r}")


def grade_item(item: Mapping[str, Any], visible: str) -> dict[str, Any]:
    """Grade an already-visible answer (thinking stripped) against an item's grader."""
    return run_check(item["grader"], str(visible or ""))


def selftest(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Every reference must pass and every 'wrong' answer must fail its own grader."""
    rows: list[dict[str, Any]] = []
    for it in items:
        ref = grade_item(it, it["reference"])
        rows.append({"id": it["id"], "answer": "reference", "expected": True, "pass": ref["pass"],
                     "ok": ref["pass"], "reason": ref["reason"]})
        for j, w in enumerate(it.get("wrong") or []):
            g = grade_item(it, w)
            rows.append({"id": it["id"], "answer": f"wrong[{j}]", "expected": False,
                         "pass": g["pass"], "ok": not g["pass"], "reason": g["reason"]})
    bad = [r for r in rows if not r["ok"]]
    return {"ok": not bad, "n_checked": len(rows), "n_items": len(items), "failures": bad, "rows": rows}


# --------------------------------------------------------------------------------------------
# llama-server chat client
# --------------------------------------------------------------------------------------------
class ChatClient:
    """Minimal stdlib client for llama-server's OpenAI-compatible chat endpoint."""

    def __init__(self, url: str = "http://127.0.0.1:8080", timeout: float = 1800.0) -> None:
        u = urllib.parse.urlparse(url if "://" in url else f"http://{url}")
        self.host = u.hostname or "127.0.0.1"
        self.port = int(u.port or 8080)
        self.timeout = float(timeout)
        self.url = f"http://{self.host}:{self.port}"

    def _request(self, method: str, path: str, body: Mapping[str, Any] | None = None,
                 timeout: float | None = None) -> tuple[int, Any]:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout or self.timeout)
        try:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
            conn.request(method, path, body=data,
                         headers={"Content-Type": "application/json"} if data is not None else {})
            resp = conn.getresponse()
            raw = resp.read()
            try:
                obj = json.loads(raw.decode("utf-8", errors="replace")) if raw else None
            except ValueError:
                obj = {"_raw": raw[:500].decode("utf-8", errors="replace")}
            return resp.status, obj
        finally:
            conn.close()

    def props(self) -> dict[str, Any] | None:
        """GET /props (model path, build info, context size), or None."""
        try:
            st, obj = self._request("GET", "/props", timeout=30)
        except (OSError, http.client.HTTPException):
            return None
        return obj if st == 200 and isinstance(obj, dict) else None

    @staticmethod
    def body(messages: Sequence[Mapping[str, str]], *, n_predict: int, enable_thinking: bool = False,
             cache_prompt: bool = False, seed: int = 0, temperature: float = 0.0) -> dict[str, Any]:
        """The request body: greedy, llama.cpp's own ``n_predict`` bound, thinking per flag."""
        return {"messages": [dict(m) for m in messages], "n_predict": int(n_predict),
                "temperature": float(temperature), "seed": int(seed),
                "cache_prompt": bool(cache_prompt), "stream": False,
                "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
                "reasoning_format": "deepseek"}

    def chat(self, messages: Sequence[Mapping[str, str]], **kw: Any) -> dict[str, Any]:
        """POST /v1/chat/completions -> {content, reasoning_content, finish_reason, timings,
        usage, wall_s, error}."""
        body = self.body(messages, **kw)
        t0 = time.monotonic()
        try:
            st, obj = self._request("POST", "/v1/chat/completions", body)
        except (OSError, http.client.HTTPException) as e:
            return {"error": f"{type(e).__name__}: {e}", "wall_s": time.monotonic() - t0}
        wall = time.monotonic() - t0
        if st != 200 or not isinstance(obj, Mapping) or not obj.get("choices"):
            return {"error": f"HTTP {st}: {str(obj)[:300]}", "wall_s": wall}
        ch = obj["choices"][0]
        msg = ch.get("message") or {}
        return {"content": msg.get("content") or "", "reasoning_content": msg.get("reasoning_content") or "",
                "finish_reason": ch.get("finish_reason"), "timings": obj.get("timings") or {},
                "usage": obj.get("usage") or {}, "wall_s": wall, "error": None}


def run_items(client: ChatClient, items: Sequence[Mapping[str, Any]], *, enable_thinking: bool = False,
              n_predict_scale: float = 1.0, seed: int = 0, log: Any = None) -> list[dict[str, Any]]:
    """One independent greedy chat request per item (cache_prompt=false), graded."""
    rows: list[dict[str, Any]] = []
    for i, it in enumerate(items):
        n_pred = max(1, int(round(int(it["n_predict"]) * n_predict_scale)))
        r = client.chat([{"role": "user", "content": it["prompt"]}], n_predict=n_pred,
                        enable_thinking=enable_thinking, cache_prompt=False, seed=seed)
        row = _grade_row(it, r, n_pred)
        rows.append(row)
        if log:
            log(f"[{i + 1}/{len(items)}] {it['id']}: {'PASS' if row['pass'] else 'FAIL'} "
                f"({row['reason'][:80]})")
    return rows


def _grade_row(it: Mapping[str, Any], r: Mapping[str, Any], n_pred: int) -> dict[str, Any]:
    vis, think, notes = visible_answer(r.get("content"), r.get("reasoning_content"))
    err = r.get("error")
    g = grade_item(it, vis) if not err else _res(False, it["grader"]["kind"], f"request failed: {err}")
    coherent, why = q1.is_coherent(vis) if vis else (False, "empty output")
    t = r.get("timings") or {}
    return {"id": it["id"], "lang": it["lang"], "category": it["category"], "n_predict": n_pred,
            "content": r.get("content"), "reasoning_content": r.get("reasoning_content"),
            "visible": vis, "thinking_chars": think, "cleanup": notes,
            "finish_reason": r.get("finish_reason"), "error": err,
            "pass": g["pass"], "score": g["score"], "reason": g["reason"], "grade": g,
            "coherent": coherent, "coherence_reason": why,
            "prompt_n": t.get("prompt_n"), "predicted_n": t.get("predicted_n"),
            "prompt_per_second": t.get("prompt_per_second"),
            "predicted_per_second": t.get("predicted_per_second"),
            "cache_n": t.get("cache_n"), "wall_s": r.get("wall_s")}


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def agg(key: str) -> dict[str, dict[str, Any]]:
        d: dict[str, dict[str, Any]] = collections.defaultdict(lambda: {"pass": 0, "total": 0, "score": 0.0})
        for r in rows:
            e = d[str(r.get(key))]
            e["total"] += 1
            e["pass"] += int(bool(r.get("pass")))
            e["score"] += float(r.get("score") or 0.0)
        return {k: {"pass": v["pass"], "total": v["total"], "mean_score": round(v["score"] / v["total"], 4)}
                for k, v in sorted(d.items())}
    tps = [float(r["predicted_per_second"]) for r in rows if r.get("predicted_per_second")]
    return {"pass": sum(1 for r in rows if r.get("pass")), "total": len(rows),
            "mean_score": round(sum(float(r.get("score") or 0) for r in rows) / len(rows), 4) if rows else None,
            "by_category": agg("category"), "by_lang": agg("lang"),
            "n_errors": sum(1 for r in rows if r.get("error")),
            "n_truncated": sum(1 for r in rows if r.get("finish_reason") == "length"),
            "n_markup_cleanup": sum(1 for r in rows if r.get("cleanup")),
            "thinking_chars_total": sum(int(r.get("thinking_chars") or 0) for r in rows),
            "median_predicted_per_second": sorted(tps)[len(tps) // 2] if tps else None,
            "note": "tok/s here are short-answer figures from a quality run, not performance results"}


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pigemma.quality_v2", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("selftest", help="grade every reference / wrong answer (no server)")
    st.add_argument("--qa", default=str(QA_V2_PATH))
    st.add_argument("--json", action="store_true")
    rn = sub.add_parser("run", help="ask a running llama-server through /v1/chat/completions")
    rn.add_argument("--url", default="http://127.0.0.1:8080")
    rn.add_argument("--qa", default=str(QA_V2_PATH))
    rn.add_argument("--ids", default=None, help="comma-separated item ids")
    rn.add_argument("--label", required=True, help="what was served, e.g. 26b-qat-q4_0-nr-pi")
    rn.add_argument("--out", default=None, help="report path (default results/quality_v2/<label>.json)")
    rn.add_argument("--thinking", action="store_true", help="enable the thinking channel")
    rn.add_argument("--n-predict-scale", type=float, default=1.0)
    rn.add_argument("--timeout", type=float, default=1800.0, help="per-request HTTP timeout, s")
    rn.add_argument("--seed", type=int, default=0)
    rg = sub.add_parser("regrade", help="re-grade a saved report offline")
    rg.add_argument("report")
    rg.add_argument("--qa", default=str(QA_V2_PATH))
    rg.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

    if a.cmd == "selftest":
        rep = selftest(load_qa_v2(a.qa))
        if a.json:
            print(json.dumps(rep, ensure_ascii=False, indent=1))
        else:
            for r in rep["failures"]:
                print(f"MISMATCH {r['id']} {r['answer']}: graded pass={r['pass']} ({r['reason']})")
            print(f"selftest: {rep['n_items']} items, {rep['n_checked']} answers checked, "
                  f"{len(rep['failures'])} mismatch(es)")
        return 0 if rep["ok"] else 1

    if a.cmd == "run":
        qa_path = Path(a.qa)
        items = load_qa_v2(qa_path, ids=a.ids.split(",") if a.ids else None)
        client = ChatClient(a.url, timeout=a.timeout)
        props = client.props()
        if props is None:
            print(f"pigemma.quality_v2: REFUSED: no llama-server answering GET /props at {client.url}",
                  file=sys.stderr)
            return paths.EXIT_REFUSED
        t0 = time.monotonic()
        rows = run_items(client, items, enable_thinking=a.thinking, n_predict_scale=a.n_predict_scale,
                         seed=a.seed, log=lambda s: print(s, file=sys.stderr, flush=True))
        report = {
            "format": REPORT_FORMAT, "kind": "measured",
            "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "label": a.label, "client_host": paths.host_fingerprint(),
            "qa_set": {"path": paths.repo_relative(qa_path), "sha256": _sha256(qa_path), "n_items": len(items)},
            "server": {"url": client.url,
                       "props": {k: props.get(k) for k in ("model_path", "build_info", "total_slots",
                                                           "model_alias", "chat_template_caps")},
                       "n_ctx": (props.get("default_generation_settings") or {}).get("n_ctx")},
            "request": {"endpoint": "/v1/chat/completions",
                        **{k: v for k, v in ChatClient.body([], n_predict=0, enable_thinking=a.thinking,
                                                            seed=a.seed).items() if k not in ("messages", "n_predict")},
                        "n_predict": "per item x %g" % a.n_predict_scale},
            "wall_s": round(time.monotonic() - t0, 3),
            "summary": summarize(rows), "items": rows,
        }
        out = Path(a.out) if a.out else paths.RESULTS_DIR / "quality_v2" / f"{a.label}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        s = report["summary"]
        print(f"{a.label}: {s['pass']}/{s['total']} pass, mean score {s['mean_score']}, "
              f"errors {s['n_errors']}, truncated {s['n_truncated']} -> {out}")
        return 0

    if a.cmd == "regrade":
        rep = json.loads(Path(a.report).read_text(encoding="utf-8"))
        by_id = {it["id"]: it for it in load_qa_v2(a.qa)}
        rows = []
        for old in rep.get("items", []):
            it = by_id.get(old["id"])
            if it is None:
                continue
            saved = {"content": old.get("content"), "reasoning_content": old.get("reasoning_content"),
                     "finish_reason": old.get("finish_reason"), "error": old.get("error"),
                     "wall_s": old.get("wall_s"),
                     "timings": {k: old.get(k) for k in ("prompt_n", "predicted_n", "prompt_per_second",
                                                         "predicted_per_second", "cache_n")}}
            rows.append(_grade_row(it, saved, int(old.get("n_predict") or it["n_predict"])))
        rep["items"], rep["summary"] = rows, summarize(rows)
        rep["regraded_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        rep["qa_set_regrade"] = {"path": a.qa, "sha256": _sha256(Path(a.qa))}
        out = Path(a.out) if a.out else Path(a.report)
        out.write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"regraded {len(rows)} items: {rep['summary']['pass']}/{rep['summary']['total']} -> {out}")
        return 0
    return paths.EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
