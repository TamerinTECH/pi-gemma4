"""pigemma.quality - cheap quality metrics for the Gemma 4 runs (experiment E2, and E8 on the Pi).

What it measures (all kind='measured' results envelopes, written with pigemma.results):

* KL divergence of a variant (``-nr``, the ``-ot`` negative control, a lower quant) against the
  reference logits of the Q4_0 model, through ``llama-perplexity --kl-divergence``;
* perplexity of a model on the calibration corpus (``data/calibration/all.txt``);
* the fixed Q&A set (``data/prompts/qa_set.json``: 7 English, 6 Hebrew, 3 numeric items) graded
  with Hebrew-aware normalisation on the visible answer (thinking channel stripped), through
  llama-server (the bench.py server-spec machinery) or llama-completion per item;
* the losslessness check of speculative variants (``exact_match_vs_plain``: the fraction of Q&A
  items whose visible greedy answer equals the plain run's);
* a repetition / garbage check (``is_coherent``) that bench.py and trace.py also use.

CLI (``python -m pigemma.quality <cmd> --help``; ``python3`` on the Pi)::

    qa        --model M [--threads 6] [--server-variant plain[,mtp-explicit,...]] [-n 160]
              [--out results/quality/qa_<model>_<variant>.json | DIR] [--no-server]
              [--load-variant V] [--plain-ref QA.json] [--qa-set F] [--ids a,b] [--thinking]
    kld-base  --model M [--text data/calibration/all.txt] [-c 512] [--chunks 6] [--threads 6]
              [--out results/quality/ref_<model>.logits] [--results-dir DIR | --no-envelope]
              [--force] [--n-vocab 262144]
    kld       --model M --base results/quality/ref_<model>.logits --variant V [--out DIR|F.json]
    ppl       --model M [--text ...] [-c 512] [--chunks 6] [--variant V] [--out DIR|F.json]
    corpus    [--out data/calibration/all.txt] [--mode interleave|concat] [--check]
    parse-kld LOG|-        parse-ppl LOG|-        check-logits FILE     check-text FILE|-

Options shared by the model-loading commands: ``--flags "<extra llama flags>"`` (repeatable;
flags after a bare ``--`` also work), ``--bin PATH`` (binary override; a string with spaces that
is not an existing file is shell-split, e.g. "python fake.py"), ``--min-free-gb N`` (bench-lock
RAM gate, default 20, env PIGEMMA_MIN_FREE_GB wins), ``--timeout S``, ``--nice N`` (POSIX),
``--experiment E2|E8|adhoc`` (default E8 on aarch64 hosts, E2 elsewhere), ``--no-sha``,
``--dry-run`` (print the command lines and the disk-gate numbers; runs nothing), ``--log-level``.
Every model-loading path holds ``paths.bench_lock`` for its whole duration.

Exit codes (pigemma.paths): 0 ok; 1 a run failed (its envelope is still written with a
``failed: ...`` note), the logits file failed its size check, or another error; 2 usage;
3 missing model / binary / text / base logits / Q&A set; 4 refused (bench lock busy, not enough
free RAM, not enough free disk for the logits, port in use). Refusals print
``pigemma.quality: REFUSED: ...`` on stderr. A low Q&A score is a measurement, not an error.

Python API::

    strip_thinking(text) -> (visible, thinking_chars); visible_text(text) -> str
    is_coherent(text) -> (bool, reason)
    normalize_text(text, punct='space'|'delete') -> str; first_number(text) -> float | None
    grade(item, visible) -> bool; load_qa_set(path, ids=None, max_items=None) -> [item]
    grade_outputs(items, contents, errors=None) -> report; exact_match_vs_plain(contents, plain)
    chat_wrap(user_text, thinking=False) -> str          (= bench.chat_wrap, Gemma 4 template)
    parse_kld_output(text) -> dict; parse_ppl_output(text) -> dict; perplexity_errors(text)
    kld_block(parsed) -> envelope 'kld' dict; kld_tier(mean); kld_neutral(kld)
    size_estimate(chunks, ctx, n_vocab=262144); expected_logits_bytes(n_ctx, n_vocab, n_chunk)
    disk_gate(free_bytes, chunks, ctx, n_vocab=262144, factor=2.0) -> dict
    read_logits_header(path) / check_logits(path) -> dict
    build_corpus(out=CORPUS_PATH, *, mode='interleave', slice_chars=1200, write=True) -> dict
    perplexity_argv(...); run_perplexity(...) -> PerplexityRun
    load_qa_file(path) -> flat dict of a QA result file
    main(argv) -> int

Decisions (where the manifest left room; also reported as open issues):

* Chat template: Gemma 4's ``<|turn>user\\n...<turn|>\\n<|turn>model\\n`` with the empty thinking
  block (thinking disabled), i.e. ``bench.chat_wrap`` (read from the GGUF's chat template). The
  ``<start_of_turn>`` format in docs/design/FACTS.md is Gemma 2/3's. ``--thinking`` enables the
  thinking channel (give a larger ``-n`` then).
* strip_thinking also drops an echoed chat prompt (everything up to the LAST
  ``<start_of_turn>model`` / ``<|turn>model`` opener when a ``user`` / ``system`` turn opener
  precedes it, as in llama-completion stdout without ``--no-display-prompt``), removes ``<turn|>``, ``<eos>`` and a trailing ``[end of text]`` as well
  as ``<end_of_turn>``, and treats text before an orphan ``<channel|>`` (opener in the prompt) as
  thinking. thinking_chars = characters inside the thinking blocks, whitespace-trimmed.
* Grading: ``contains`` passes when any expectation is a substring of the answer after
  normalisation, with punctuation either turned into spaces or deleted (either reading passes;
  the Hebrew maqaf/paseq/sof-pasuq count as punctuation, not niqqud, so 'כדור־הארץ' matches
  'כדור הארץ'); format characters (bidi marks, zero-width joiners) are dropped; ``regex`` passes
  when a pattern re.search-matches the raw visible text or either normalised form; ``number``
  takes the first number of the NFKC text (thousands separators '1,234' removed, a decimal comma
  '16,8' accepted when it is not part of a list, U+2212 minus accepted, digits glued to letters
  such as 'CO2' or hyphenated to a Latin word such as 'num-01' skipped; a Hebrew
  prefix letter + hyphen such as 'ב-8' or 'ל-1990' is a number) and passes within the item's
  tolerance (+1e-9).
* is_coherent implements the three manifest rules on the text it is given (word n-grams) and one
  addition: a unit of 1-16 characters repeated back-to-back at least 8 times and covering >= 128
  characters (or >= 48 characters and >= 90 % of the text) is a degenerate loop ('!!!!...'); the
  word rules cannot see loops without spaces. Characters counted as 'letters/digits/space/punct'
  are Unicode L*, M*, N*, P*, whitespace and ASCII punctuation/symbols.
* The Q&A result file (``qa`` --out) IS a results envelope (experiment E2, kind measured; tok/s,
  load time, draft statistics from bench.server_fields), with qa_score / qa_total / coherent /
  identical_to_plain at the top level and the Q&A detail under ``extra``: extra.items
  [{id, lang, kind, prompt, expect, visible, thinking_chars, pass, coherent, coherence_reason,
  error, predicted_n, stop_type}], extra.per_language {lang: {score, total}},
  extra.exact_match_vs_plain (None for plain or when no plain reference exists),
  extra.exact_match_by_item, extra.plain_ref, extra.qa_set {path, sha256, n_items},
  extra.thinking_chars_total, extra.thinking_share, extra.threshold_check. ``load_qa_file()``
  flattens it. ``results summarize results/quality`` therefore renders the Q&A scores.
* Plain reference for exact_match_vs_plain: ``--plain-ref FILE``, else the plain run of the
  same invocation (plain always runs first), else ``<out dir>/qa_<model>_plain<+load>.json``.
* Several ``--server-variant`` values run one server each under one bench lock; --out must then
  be a directory (default results/quality/). ``--no-server`` runs llama-completion once per
  item (one model load per item) and records the variant as ``completion[+<load variant>]``.
* KLD variant names: a load variant (bench.VARIANT_FLAGS, '+'-combinable: norepack,
  hybrid_negctrl, ...) adds its flags; a catalog / registry model id (e.g. 26b-ud-q3_k_m) is a
  label for a different GGUF passed with --model; any other label is refused unless
  ``--allow-label`` (a typo must never yield a silent 'default vs default' KLD of zero).
  The KLD context and chunk count come from the base logits header (llama-perplexity reads the
  evaluation tokens from that file; ``--chunks`` is passed but ignored by the tool).
* Logits size: the disk gate uses the manifest's conservative estimate chunks*ctx*n_vocab*2
  bytes (6 x 512 x 262144 x 2 = 1,610,612,736 B) and needs 2x that free. The size check after
  the run uses the exact layout of tools/perplexity/perplexity.cpp at a7b94df:
  20 header bytes + n_chunk*n_ctx*4 token bytes + n_chunk*(n_ctx-1-n_ctx/2)*nv*2 log-prob bytes
  with nv = 2*((n_vocab+1)/2)+4 (802,185,188 B for the Gemma 4 default), read from the file's
  own header. A sidecar ``<logits>.json`` records model, corpus sha256, argv, PPL and the check.
  An existing valid reference with the same model/ctx/chunks/corpus/flags is reused unless
  --force; a valid but different one needs --force; a broken one (size check failed) is replaced.
* llama-perplexity exits 0 even when it fails (e.g. too few tokens, wrong base file), so success
  is decided by the parsed statistics; known error lines are copied into the failure note.
* Corpus: data/calibration/all.txt concatenates every calibration file (sorted by name; README
  excluded; CRLF normalised) as slices of ~1200 characters cut at line boundaries and taken
  round-robin across files ('interleave', default), so the first 6 x 512 tokens that
  llama-perplexity evaluates mix English, Hebrew, German, numbers and code instead of only the
  alphabetically first file ('concat' gives the plain file-after-file concatenation). It is
  rebuilt only when its content would change; sha256 values go into every envelope.
* Raw tool output: kld-base -> ``<logits>.log``; kld / ppl -> ``<out>/raw/<stem>.perplexity.log``;
  qa -> ``<out dir>/raw/<stem>.server.log`` (``results/**/*.log`` is git-ignored).
* Pre-registered E2 thresholds are evaluated into extra.threshold_check (neutral: KLD mean
  < 0.001 and same-top-p > 99.5 %; tiers < 0.05 acceptable, < 0.15 degraded, else not
  recommended; Q&A >= 13/16 and Hebrew >= 4/6). They are labels on the measurement, not new data.
* No idle-machine check here (quality, not speed, is the measurement); tok/s recorded by qa are
  short-answer figures, not E1 numbers.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import dataclasses
import datetime as _dt
import hashlib
import json
import logging
import math
import os
import re
import shlex
import shutil
import statistics
import string
import struct
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

from pigemma import bench, paths, results

log = logging.getLogger("pigemma.quality")

__all__ = [
    "strip_thinking", "visible_text", "is_coherent", "normalize_text", "first_number", "grade",
    "load_qa_set", "grade_outputs", "exact_match_vs_plain", "chat_wrap", "parse_kld_output",
    "parse_ppl_output", "perplexity_errors", "kld_block", "kld_tier", "kld_neutral",
    "size_estimate", "expected_logits_bytes", "disk_gate", "read_logits_header", "check_logits",
    "build_corpus", "corpus_sources", "perplexity_argv", "run_perplexity", "PerplexityRun",
    "load_qa_file", "QualityError", "DiskGateError", "main",
]

# --------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------
QA_SET_PATH = paths.PROMPTS_DIR / "qa_set.json"
CORPUS_PATH = paths.CALIBRATION_DIR / "all.txt"
QUALITY_DIR = paths.RESULTS_DIR / "quality"

GEMMA4_N_VOCAB = 262144
DEFAULT_CTX = 512
DEFAULT_CHUNKS = 6
DEFAULT_THREADS = 6
DEFAULT_QA_N_PREDICT = 160
DEFAULT_QA_N_CTX = 4096
DISK_GATE_FACTOR = 2.0
LOGITS_MAGIC = b"_logits_"
LOGITS_HEADER_BYTES = 20  # magic(8) + n_ctx u32 + n_vocab i32 + n_chunk i32
LOGITS_META_FORMAT = "pigemma-logits-meta/1"

CORPUS_MODES = ("interleave", "concat")
CORPUS_SLICE_CHARS = 1200
CORPUS_SUFFIXES = frozenset({".txt", ".cpp", ".c", ".h", ".hpp", ".py", ".md"})
CORPUS_EXCLUDE = frozenset({"all.txt"})

# pre-registered E2 thresholds (docs/design/manifest.json E2-quality, docs/EXPERIMENT_PLAN.md)
KLD_NEUTRAL_MEAN = 0.001
KLD_NEUTRAL_SAME_TOP_P = 0.995
KLD_TIER_ACCEPTABLE = 0.05
KLD_TIER_DEGRADED = 0.15
QA_PASS_SCORE = 13      # of 16
QA_HE_PASS_SCORE = 4    # of 6

# coherence rules (manifest) + the degenerate-loop extension (module docstring)
COHERENT_MIN_GOOD_FRAC = 0.60
COHERENT_4GRAM_MAX = 6      # a 4-gram repeated >= 6 times -> incoherent
COHERENT_6GRAM_MAX = 3      # a 6-gram repeated > 3 times -> incoherent
LOOP_UNIT_MAX = 16
LOOP_MIN_REPEATS = 8
LOOP_MIN_CHARS = 128
LOOP_MIN_CHARS_SHORT = 48
LOOP_SHORT_FRACTION = 0.9   # ... when the loop is (almost) the whole text

QA_KINDS = ("contains", "regex", "number")

# --------------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------------


class QualityError(RuntimeError):
    """A run could not be completed or its output is unusable (exit 1)."""


class DiskGateError(QualityError):
    """Not enough free disk for the reference logits (exit 4)."""


# --------------------------------------------------------------------------------------------
# Visible text / thinking channel
# --------------------------------------------------------------------------------------------
_MODEL_OPENER_RE = re.compile(r"(?:<start_of_turn>|<\|turn>)model[^\S\n]*\n?")
_PROMPT_OPENER_RE = re.compile(r"(?:<start_of_turn>|<\|turn>)(?:user|system)\b")
_THINK_BLOCK_RE = re.compile(r"<\|channel>[^\S\n]*thought\b(.*?)(?:<channel\|>|\Z)", re.DOTALL)
_ORPHAN_CLOSE = "<channel|>"
_LEADING_CHANNEL_RE = re.compile(r"^\s*(?:<channel\|>|<\|channel>)")
_END_TOKENS_RE = re.compile(r"<end_of_turn>|<turn\|>|<eos>|<bos>")
_END_OF_TEXT_RE = re.compile(r"\s*\[end of text\]\s*$")


def strip_thinking(text: str) -> tuple[str, int]:
    """(visible text, thinking chars) of a model output.

    Removes ``<|channel>thought ... <channel|>`` blocks (DOTALL; an unterminated block runs to
    the end of the text), a leading bare ``<channel|>`` / ``<|channel>`` token, text before an
    orphan ``<channel|>`` (thinking whose opener was in the prompt), an echoed chat prompt (up to
    the last model-turn opener, when a user/system turn opener precedes it), ``<end_of_turn>`` / ``<turn|>`` / ``<eos>`` / ``<bos>`` and a
    trailing ``[end of text]``. thinking_chars counts the whitespace-trimmed characters inside
    the thinking blocks. The visible text is stripped of surrounding whitespace.
    """
    s = str(text or "").replace("\r\n", "\n")
    # an echoed chat prompt: the answer follows the last model-turn opener, provided a user /
    # system turn opener precedes it (a bare model opener inside an answer is left alone)
    last = None
    for m in _MODEL_OPENER_RE.finditer(s):
        last = m
    if last is not None and _PROMPT_OPENER_RE.search(s, 0, last.start()):
        s = s[last.end():]
    thinking = 0
    for m in _THINK_BLOCK_RE.finditer(s):
        thinking += len(m.group(1).strip())
    s = _THINK_BLOCK_RE.sub("", s)
    # an orphan closer: the thinking opener was part of the prompt
    if _ORPHAN_CLOSE in s and "<|channel>" not in s.split(_ORPHAN_CLOSE, 1)[0]:
        head, s = s.split(_ORPHAN_CLOSE, 1)
        thinking += len(head.strip())
    while True:
        s2 = _LEADING_CHANNEL_RE.sub("", s, count=1)
        if s2 == s:
            break
        s = s2
    s = _END_TOKENS_RE.sub("", s)
    s = _END_OF_TEXT_RE.sub("", s)
    return s.strip(), thinking


def visible_text(text: str) -> str:
    """The visible answer (strip_thinking()[0]); idempotent."""
    return strip_thinking(text)[0]


def chat_wrap(user_text: str, thinking: bool = False) -> str:
    """Gemma 4 single-user-turn prompt (bench.chat_wrap): thinking disabled by default."""
    return bench.chat_wrap(user_text, thinking=thinking)


# --------------------------------------------------------------------------------------------
# Coherence
# --------------------------------------------------------------------------------------------
_ASCII_PUNCT = frozenset(string.punctuation)
_LOOP_RE = re.compile(r"(.{1,%d}?)\1{%d,}" % (LOOP_UNIT_MAX, LOOP_MIN_REPEATS - 1), re.DOTALL)


def _good_char(ch: str) -> bool:
    if ch.isspace() or ch in _ASCII_PUNCT:
        return True
    return unicodedata.category(ch)[0] in ("L", "M", "N", "P")


def _degenerate_loop(t: str) -> str | None:
    """Reason string when a short unit repeats back-to-back over a long stretch, else None."""
    n = len(t)
    for m in _LOOP_RE.finditer(t):
        unit = m.group(1)
        if not unit.strip():
            continue
        covered = m.end() - m.start()
        if covered >= LOOP_MIN_CHARS or (covered >= LOOP_MIN_CHARS_SHORT
                                          and covered >= LOOP_SHORT_FRACTION * n):
            return (f"degenerate loop: {unit[:16]!r} repeated {covered // len(unit)} times "
                    f"({covered} chars)")
    return None


def is_coherent(text: str) -> tuple[bool, str]:
    """(coherent, reason) for a generated text (callers pass the visible text).

    Incoherent when: empty; fewer than 60 % of the characters are letters / digits / spaces /
    punctuation; a word 4-gram occurs >= 6 times; a word 6-gram occurs > 3 times; or a short
    unit loops back-to-back (see the module docstring). reason is 'ok' when coherent.
    """
    t = str(text or "").strip()
    if not t:
        return False, "empty output"
    good = sum(1 for ch in t if _good_char(ch))
    frac = good / len(t)
    if frac < COHERENT_MIN_GOOD_FRAC:
        return False, f"only {frac:.0%} of chars are letters/digits/space/punctuation"
    words = t.split()
    for n, limit, strict in ((4, COHERENT_4GRAM_MAX, False), (6, COHERENT_6GRAM_MAX, True)):
        if len(words) < n:
            continue
        cnt = collections.Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
        gram, c = cnt.most_common(1)[0]
        if (c > limit) if strict else (c >= limit):
            return False, f"{n}-gram {' '.join(gram)[:60]!r} repeated {c} times"
    loop = _degenerate_loop(t)
    if loop:
        return False, loop
    return True, "ok"


# --------------------------------------------------------------------------------------------
# Q&A grading
# --------------------------------------------------------------------------------------------
_HEB_FINALS = str.maketrans({"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"})
# Hebrew punctuation inside U+0591-U+05C7: maqaf, paseq, sof pasuq, nun hafukha
_HEB_PUNCT = frozenset("־׀׃׆")
_EXTRA_PUNCT = frozenset("`~^|")
_WS_RE = re.compile(r"\s+")


def _is_niqqud(ch: str) -> bool:
    return "֑" <= ch <= "ׇ" and ch not in _HEB_PUNCT


def normalize_text(text: str, punct: str = "space") -> str:
    """Hebrew-aware normalisation used by the ``contains`` grader.

    NFKC, casefold, drop format characters (bidi marks, ZWJ), strip niqqud/cantillation
    U+0591-U+05C7 (the Hebrew punctuation in that range counts as punctuation), Hebrew final
    forms ךםןףץ -> כמנפצ, punctuation (Unicode P* plus `~^|) -> space (``punct='space'``) or
    removed (``punct='delete'``), whitespace collapsed and trimmed.
    """
    if punct not in ("space", "delete"):
        raise ValueError("punct must be 'space' or 'delete'")
    s = unicodedata.normalize("NFKC", str(text or "")).casefold()
    repl = " " if punct == "space" else ""
    out: list[str] = []
    for ch in s:
        cat = unicodedata.category(ch)
        if cat == "Cf":
            continue
        if ch in _HEB_PUNCT or cat[0] == "P" or ch in _EXTRA_PUNCT:
            out.append(repl)
        elif _is_niqqud(ch):
            continue
        else:
            out.append(ch)
    s = "".join(out).translate(_HEB_FINALS)
    return _WS_RE.sub(" ", s).strip()


_NUM_RE = re.compile(r"(?P<sign>(?<![\w.\-])-)?(?<![\w.])(?<![A-Za-z]-)(?P<num>\d+(?:\.\d+)?)")
_THOUSANDS_RE = re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+)(?!\d|,\d)")
_DEC_COMMA_RE = re.compile(r",(\d{1,2})(?![\d])(?!\s*,\s*\d)")


def first_number(text: str) -> float | None:
    """First number of a text (see the module docstring for the accepted spellings), or None."""
    s = unicodedata.normalize("NFKC", str(text or ""))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Cf")
    s = s.replace("−", "-")
    s = _THOUSANDS_RE.sub(lambda m: m.group(1).replace(",", ""), s)
    m = _NUM_RE.search(s)
    if not m:
        return None
    num = m.group("num")
    if "." not in num:
        d = _DEC_COMMA_RE.match(s, m.end())
        if d:
            num = f"{num}.{d.group(1)}"
    try:
        v = float(num)
    except ValueError:
        return None
    return -v if m.group("sign") else v


_REGEX_CACHE: dict[str, re.Pattern[str]] = {}


def _rx(p: str) -> re.Pattern[str]:
    r = _REGEX_CACHE.get(p)
    if r is None:
        r = _REGEX_CACHE[p] = re.compile(p)
    return r


def grade(item: Mapping[str, Any], visible: str) -> bool:
    """True when ``visible`` (the answer with the thinking channel already stripped) satisfies a
    qa_set.json item: contains / regex / number (see the module docstring)."""
    kind = item.get("kind")
    expect = item.get("expect")
    if not isinstance(expect, (list, tuple)) or not expect:
        raise ValueError(f"item {item.get('id')!r}: 'expect' must be a non-empty list")
    vis = str(visible or "")
    if kind == "contains":
        for mode in ("space", "delete"):
            hay = normalize_text(vis, mode)
            for e in expect:
                needle = normalize_text(str(e), mode)
                if needle and needle in hay:
                    return True
        return False
    if kind == "regex":
        cands = (vis, normalize_text(vis, "space"), normalize_text(vis, "delete"))
        return any(_rx(str(p)).search(c) for p in expect for c in cands)
    if kind == "number":
        v = first_number(vis)
        if v is None:
            return False
        tol = float(item.get("tolerance") or 0.0)
        return any(abs(v - float(e)) <= tol + 1e-9 for e in expect)
    raise ValueError(f"item {item.get('id')!r}: unknown kind {kind!r} (known: {', '.join(QA_KINDS)})")


def load_qa_set(path: str | os.PathLike[str] = QA_SET_PATH, *, ids: Sequence[str] | None = None,
                max_items: int | None = None) -> list[dict[str, Any]]:
    """Validated Q&A items from qa_set.json (optionally filtered by id / truncated)."""
    p = Path(os.fspath(path))
    if not p.is_file():
        raise bench.MissingInputError(f"Q&A set not found: {p}")
    with open(p, encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, Mapping):
        data = data.get("items") or data.get("prompts")
    if not isinstance(data, list) or not data:
        raise QualityError(f"{p}: expected a non-empty JSON list of Q&A items")
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, it in enumerate(data):
        if not isinstance(it, Mapping):
            raise QualityError(f"{p}: item {i} is not an object")
        d = dict(it)
        missing = [k for k in ("id", "prompt", "kind", "expect") if k not in d]
        if missing:
            raise QualityError(f"{p}: item {i} lacks {', '.join(missing)}")
        if d["kind"] not in QA_KINDS:
            raise QualityError(f"{p}: item {d['id']} has unknown kind {d['kind']!r}")
        if not isinstance(d["expect"], list) or not d["expect"]:
            raise QualityError(f"{p}: item {d['id']} needs a non-empty 'expect' list")
        d["id"] = str(d["id"])
        d.setdefault("lang", "unknown")
        if d["id"] in seen:
            raise QualityError(f"{p}: duplicate id {d['id']}")
        seen.add(d["id"])
        items.append(d)
    if ids:
        want = [str(x) for x in ids]
        unknown = [x for x in want if x not in seen]
        if unknown:
            raise ValueError(f"unknown Q&A id(s): {', '.join(unknown)}")
        items = [it for it in items if it["id"] in want]
    if max_items:
        items = items[: int(max_items)]
    return items


def grade_outputs(items: Sequence[Mapping[str, Any]], contents: Mapping[str, str | None],
                  errors: Mapping[str, str | None] | None = None,
                  extra_by_id: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Grade raw model outputs ({id: content}) against Q&A items.

    Returns {items: [{id, lang, kind, prompt, expect, visible, thinking_chars, pass, coherent,
    coherence_reason, error, ...extra_by_id[id]}], qa_score, qa_total, per_language
    {lang: {score, total}}, coherent (all answered items coherent; None if none answered),
    visible_chars_total, thinking_chars_total, thinking_share}. A missing or failed output
    counts as a fail.
    """
    errors = errors or {}
    out: list[dict[str, Any]] = []
    per: dict[str, dict[str, int]] = {}
    vis_total = think_total = 0
    coh_all: list[bool] = []
    for it in items:
        iid = str(it["id"])
        raw = contents.get(iid)
        err = errors.get(iid)
        rec: dict[str, Any] = {"id": iid, "lang": it.get("lang"), "kind": it.get("kind"),
                               "prompt": it.get("prompt"), "expect": list(it.get("expect") or [])}
        if raw is None:
            rec.update(visible=None, thinking_chars=None, coherent=None,
                       coherence_reason="no output", error=err or "no output")
            rec["pass"] = False
        else:
            vis, th = strip_thinking(raw)
            ok, why = is_coherent(vis)
            rec.update(visible=vis, thinking_chars=th, coherent=ok, coherence_reason=why,
                       error=err)
            rec["pass"] = bool(grade(it, vis))
            vis_total += len(vis)
            think_total += th
            coh_all.append(ok)
        if extra_by_id and iid in extra_by_id:
            for k, v in extra_by_id[iid].items():
                rec.setdefault(k, v)
        lang = str(it.get("lang") or "unknown")
        b = per.setdefault(lang, {"score": 0, "total": 0})
        b["total"] += 1
        b["score"] += 1 if rec["pass"] else 0
        out.append(rec)
    denom = vis_total + think_total
    return {
        "items": out,
        "qa_score": sum(1 for r in out if r["pass"]),
        "qa_total": len(out),
        "per_language": per,
        "coherent": all(coh_all) if coh_all else None,
        "visible_chars_total": vis_total,
        "thinking_chars_total": think_total,
        "thinking_share": (think_total / denom) if denom else None,
    }


def exact_match_vs_plain(contents: Mapping[str, str | None], plain: Mapping[str, str | None] | None
                         ) -> tuple[float | None, dict[str, bool] | None]:
    """(fraction of items whose visible text equals the plain run's, {id: equal}).

    Both sides go through visible_text (idempotent, so already-visible strings work too).
    (None, None) when there is no plain reference or no item in common.
    """
    if not plain:
        return None, None
    by: dict[str, bool] = {}
    for iid, c in contents.items():
        p = plain.get(iid)
        if c is None or p is None:
            continue
        by[str(iid)] = visible_text(c) == visible_text(p)
    if not by:
        return None, None
    return sum(by.values()) / len(by), by


def qa_threshold_check(report: Mapping[str, Any]) -> dict[str, Any]:
    """Pre-registered E2 Q&A thresholds applied to a graded report (labels, not new data)."""
    total = report.get("qa_total") or 0
    he = (report.get("per_language") or {}).get("he") or {}
    full = total == 16
    return {
        "rule": "E2 (pre-registered): qa_score >= 13/16 and Hebrew subset >= 4/6",
        "qa_pass": (report.get("qa_score", 0) >= QA_PASS_SCORE) if full else None,
        "hebrew_pass": (he.get("score", 0) >= QA_HE_PASS_SCORE) if he.get("total") == 6 else None,
        "applies": full,
    }


def load_qa_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    """A Q&A result file (envelope written by ``qa``) flattened to {model_id, variant, qa_score,
    qa_total, per_language, items, exact_match_vs_plain, coherent, visible_by_id, path}."""
    p = Path(os.fspath(path))
    with open(p, encoding="utf-8-sig") as f:
        d = json.load(f)
    if not isinstance(d, Mapping):
        raise QualityError(f"{p}: not a JSON object")
    ex = d.get("extra") if isinstance(d.get("extra"), Mapping) else {}
    items = ex.get("items") if isinstance(ex.get("items"), list) else d.get("items")
    if not isinstance(items, list):
        raise QualityError(f"{p}: no Q&A items (extra.items) - not a quality qa result")
    model = d.get("model") if isinstance(d.get("model"), Mapping) else {}
    return {
        "path": str(p), "model_id": model.get("id"), "variant": d.get("variant"),
        "qa_score": d.get("qa_score"), "qa_total": d.get("qa_total"),
        "per_language": ex.get("per_language"), "items": items,
        "exact_match_vs_plain": ex.get("exact_match_vs_plain"),
        "coherent": d.get("coherent"),
        "visible_by_id": {str(r.get("id")): r.get("visible") for r in items
                          if isinstance(r, Mapping) and r.get("visible") is not None},
    }


# --------------------------------------------------------------------------------------------
# llama-perplexity output parsers
# --------------------------------------------------------------------------------------------
_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?nan(?:\([a-z]+\))?|[-+]?inf(?:inity)?"
_PM = r"\s*(?:±|\+/-|\+-|Â±|�|\?)\s*"
_PFX_RE = re.compile(r"^\s*\d+\.\d{2}\.\d{3}\.\d{3}\s+(?:[IWEDT]\s+)?")
_DP = r"\S{0,3}p"  # 'Δp', mangled spellings ('?p', 'Î”p') included


def _val(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return (v + 0.0) if math.isfinite(v) else None  # + 0.0: '-0.000000' -> 0.0


def _lines(text: str) -> list[str]:
    return [_PFX_RE.sub("", ln.rstrip("\r")) for ln in str(text or "").replace("\r\n", "\n").split("\n")]


def _line_re(label: str, *, pm: bool = True, pct: bool = False) -> re.Pattern[str]:
    tail = rf"(?:{_PM}({_NUM}))?" if pm else ""
    return re.compile(rf"^\s*{label}\s*:\s*({_NUM})\s*%?{tail}\s*{'%' if pct else ''}",
                      re.IGNORECASE)


_KLD_LINES: dict[str, re.Pattern[str]] = {
    "mean": _line_re(r"Mean\s+KLD"),
    "max": _line_re(r"Maximum\s+KLD", pm=False),
    "median": _line_re(r"Median\s+KLD", pm=False),
    "min": _line_re(r"Minimum\s+KLD", pm=False),
    "ppl_q": _line_re(r"Mean\s+PPL\s*\(\s*Q\s*\)"),
    "ppl_base": _line_re(r"Mean\s+PPL\s*\(\s*base\s*\)"),
    "cor_ln_ppl_pct": _line_re(r"Cor\s*\(\s*ln\s*\(\s*PPL\s*\(\s*Q\s*\)\s*\)\s*,\s*ln\s*\(\s*PPL\s*\(\s*base\s*\)\s*\)\s*\)", pm=False),
    "mean_ln_ppl_ratio": _line_re(r"Mean\s+ln\s*\(\s*PPL\s*\(\s*Q\s*\)\s*/\s*PPL\s*\(\s*base\s*\)\s*\)"),
    "mean_ppl_ratio": _line_re(r"Mean\s+PPL\s*\(\s*Q\s*\)\s*/\s*PPL\s*\(\s*base\s*\)"),
    "mean_ppl_diff": _line_re(r"Mean\s+PPL\s*\(\s*Q\s*\)\s*-\s*PPL\s*\(\s*base\s*\)"),
    "mean_dp_pct": _line_re(rf"Mean\s+{_DP}", pct=True),
    "max_dp_pct": _line_re(rf"Maximum\s+{_DP}", pm=False),
    "median_dp_pct": _line_re(rf"Median\s+{_DP}", pm=False),
    "min_dp_pct": _line_re(rf"Minimum\s+{_DP}", pm=False),
    "rms_dp_pct": _line_re(rf"RMS\s+{_DP}", pct=True),
    "same_top_p_pct": _line_re(r"Same\s+top\s+p", pct=True),
}
_KLD_PCTL_RE = re.compile(rf"^\s*(\d+(?:\.\d+)?)\s*%\s*KLD\s*:\s*({_NUM})", re.IGNORECASE)
_DP_PCTL_RE = re.compile(rf"^\s*(\d+(?:\.\d+)?)\s*%\s*{_DP}\s*:\s*({_NUM})\s*%", re.IGNORECASE)
_CHUNK_ROW_RE = re.compile(
    rf"^\s*(\d+)\s+({_NUM}){_PM}({_NUM})\s+({_NUM}){_PM}({_NUM})\s+({_NUM}){_PM}({_NUM})"
    rf"\s+({_NUM}){_PM}({_NUM})\s*%\s+({_NUM}){_PM}({_NUM})\s*%", re.IGNORECASE)
_FINAL_RE = re.compile(rf"Final\s+estimate\s*:\s*PPL\s*=\s*({_NUM})(?:{_PM}({_NUM}))?", re.IGNORECASE)
_PPL_PROGRESS_RE = re.compile(rf"\[(\d+)\]\s*({_NUM})\s*,")
_COMPUTING_RE = re.compile(
    r"(?:kl_divergence|perplexity): (?:computing|calculating perplexity) over (\d+) chunks, "
    r"n_ctx=(\d+), batch_size=(\d+), n_seq=(\d+)")
_ERROR_PATTERNS = (
    "you need at least", "does not look like a file containing log-probabilities",
    "inconsistent vocabulary", "has been computed with", "failed to open", "unable to load model",
    "failed to create context", "failed reading", "failed to load model", "error:",
    "Unexpected negative standard deviation",
)


def parse_kld_output(text: str) -> dict[str, Any]:
    """Parse ``llama-perplexity --kl-divergence`` output (stdout + stderr, any spacing).

    Keys (None when a line is missing or the value is not finite): mean, mean_sd, max, p999,
    p99, p95, p90, median, p10, p05, p01, p001, min, ppl_q, ppl_q_sd, ppl_base, ppl_base_sd,
    cor_ln_ppl_pct, mean_ln_ppl_ratio(_sd), mean_ppl_ratio(_sd), mean_ppl_diff(_sd),
    mean_dp_pct(_sd), max_dp_pct, median_dp_pct, min_dp_pct, rms_dp_pct(_sd),
    same_top_p_pct(_sd), same_top_p (fraction = pct/100), kld_percentiles {'99.9': ..},
    dp_percentiles_pct, chunks [{chunk, ppl, ppl_sd, ln_ratio, kld, kld_sd, dp_rms_pct,
    same_top_p_pct}], n_chunks, computing {n_chunk, n_ctx, n_batch, n_seq}, ok (mean found).
    """
    out: dict[str, Any] = {}
    for k in _KLD_LINES:
        out[k] = None
        if k not in ("max", "median", "min", "cor_ln_ppl_pct", "max_dp_pct", "median_dp_pct",
                     "min_dp_pct"):
            out[f"{k}_sd"] = None
    pct: dict[str, float | None] = {}
    dp_pct: dict[str, float | None] = {}
    chunks: list[dict[str, Any]] = []
    computing = None
    for ln in _lines(text):
        if not ln.strip():
            continue
        m = _COMPUTING_RE.search(ln)
        if m and computing is None:
            computing = {"n_chunk": int(m.group(1)), "n_ctx": int(m.group(2)),
                         "n_batch": int(m.group(3)), "n_seq": int(m.group(4))}
        m = _CHUNK_ROW_RE.match(ln)
        if m:
            g = m.groups()
            chunks.append({"chunk": int(g[0]), "ppl": _val(g[1]), "ppl_sd": _val(g[2]),
                           "ln_ratio": _val(g[3]), "kld": _val(g[5]), "kld_sd": _val(g[6]),
                           "dp_rms_pct": _val(g[7]), "same_top_p_pct": _val(g[9])})
            continue
        m = _KLD_PCTL_RE.match(ln)
        if m:
            pct[f"{float(m.group(1)):.1f}"] = _val(m.group(2))
            continue
        m = _DP_PCTL_RE.match(ln)
        if m:
            dp_pct[f"{float(m.group(1)):.1f}"] = _val(m.group(2))
            continue
        for k, rx in _KLD_LINES.items():
            if out[k] is not None:
                continue
            m = rx.match(ln)
            if m:
                out[k] = _val(m.group(1))
                if f"{k}_sd" in out and m.lastindex and m.lastindex >= 2:
                    out[f"{k}_sd"] = _val(m.group(2))
                break
    out["kld_percentiles"] = pct or None
    out["dp_percentiles_pct"] = dp_pct or None
    for key, p in (("p999", "99.9"), ("p99", "99.0"), ("p95", "95.0"), ("p90", "90.0"),
                   ("p10", "10.0"), ("p05", "5.0"), ("p01", "1.0"), ("p001", "0.1")):
        out[key] = pct.get(p)
    sp = out.get("same_top_p_pct")
    out["same_top_p"] = round(sp / 100.0, 10) if sp is not None else None
    out["chunks"] = chunks or None
    out["n_chunks"] = chunks[-1]["chunk"] if chunks else (computing or {}).get("n_chunk")
    out["computing"] = computing
    out["ok"] = out["mean"] is not None
    return out


def parse_ppl_output(text: str) -> dict[str, Any]:
    """Parse ``Final estimate: PPL = x +/- y`` (+ the '[i]ppl,' progress) -> {ppl, ppl_sd,
    chunk_ppl [..], n_chunks, computing, ok}."""
    t = str(text or "")
    ppl = sd = None
    for ln in _lines(t):
        m = _FINAL_RE.search(ln)
        if m:
            ppl, sd = _val(m.group(1)), _val(m.group(2))
    prog = [(int(a), _val(b)) for a, b in _PPL_PROGRESS_RE.findall(t)]
    computing = None
    m = _COMPUTING_RE.search(t)
    if m:
        computing = {"n_chunk": int(m.group(1)), "n_ctx": int(m.group(2)),
                     "n_batch": int(m.group(3)), "n_seq": int(m.group(4))}
    return {"ppl": ppl, "ppl_sd": sd, "chunk_ppl": [v for _, v in prog] or None,
            "n_chunks": prog[-1][0] if prog else (computing or {}).get("n_chunk"),
            "computing": computing, "ok": ppl is not None}


def perplexity_errors(text: str) -> list[str]:
    """Error-looking lines of llama-perplexity output (it exits 0 on most of them)."""
    out: list[str] = []
    for ln in _lines(text):
        s = ln.strip()
        low = s.lower()
        if s and any(p.lower() in low for p in _ERROR_PATTERNS):
            out.append(s[:300])
    return out[:8]


def kld_block(parsed: Mapping[str, Any]) -> dict[str, Any]:
    """The envelope ``kld`` object {mean, max, p999, same_top_p (fraction), ppl_q, ppl_base}."""
    return {k: parsed.get(k) for k in ("mean", "max", "p999", "same_top_p", "ppl_q", "ppl_base")}


def kld_tier(mean: float | None) -> str | None:
    """Pre-registered quant tier: < 0.05 'acceptable', < 0.15 'degraded', else 'not recommended'."""
    if mean is None:
        return None
    if mean < KLD_TIER_ACCEPTABLE:
        return "acceptable"
    if mean < KLD_TIER_DEGRADED:
        return "degraded"
    return "not recommended"


def kld_neutral(kld: Mapping[str, Any]) -> bool | None:
    """Pre-registered neutrality check: KLD mean < 0.001 and same-top-p > 99.5 %."""
    mean, stp = kld.get("mean"), kld.get("same_top_p")
    if mean is None or stp is None:
        return None
    return bool(mean < KLD_NEUTRAL_MEAN and stp > KLD_NEUTRAL_SAME_TOP_P)


# --------------------------------------------------------------------------------------------
# Logits file: size estimate, disk gate, header / size check
# --------------------------------------------------------------------------------------------
def size_estimate(chunks: int, ctx: int, n_vocab: int = GEMMA4_N_VOCAB) -> int:
    """The manifest's conservative logits size: chunks * ctx * n_vocab * 2 bytes."""
    return int(chunks) * int(ctx) * int(n_vocab) * 2


def expected_logits_bytes(n_ctx: int, n_vocab: int, n_chunk: int) -> int:
    """Exact size of a ``--save-all-logits`` file at a7b94df (tools/perplexity/perplexity.cpp)."""
    nv = 2 * ((int(n_vocab) + 1) // 2) + 4
    per_chunk_tokens = int(n_ctx) - 1 - int(n_ctx) // 2
    return (LOGITS_HEADER_BYTES + int(n_chunk) * int(n_ctx) * 4
            + int(n_chunk) * per_chunk_tokens * nv * 2)


def disk_gate(free_bytes: int, chunks: int, ctx: int, n_vocab: int = GEMMA4_N_VOCAB,
              factor: float = DISK_GATE_FACTOR) -> dict[str, Any]:
    """{estimate_bytes, required_bytes (= factor x estimate), free_bytes, ok, exact_bytes}."""
    est = size_estimate(chunks, ctx, n_vocab)
    req = int(math.ceil(est * float(factor)))
    return {"estimate_bytes": est, "required_bytes": req, "free_bytes": int(free_bytes),
            "factor": float(factor), "ok": int(free_bytes) >= req,
            "exact_bytes": expected_logits_bytes(ctx, n_vocab, chunks)}


def read_logits_header(path: str | os.PathLike[str]) -> dict[str, Any]:
    """{magic_ok, n_ctx, n_vocab, n_chunk} of a logits file (ValueError when too short)."""
    with open(os.fspath(path), "rb") as f:
        head = f.read(LOGITS_HEADER_BYTES)
    if len(head) < LOGITS_HEADER_BYTES:
        raise ValueError(f"{path}: only {len(head)} bytes, not a logits file")
    n_ctx, n_vocab, n_chunk = struct.unpack("<Iii", head[8:20])
    return {"magic_ok": head[:8] == LOGITS_MAGIC, "n_ctx": n_ctx, "n_vocab": n_vocab,
            "n_chunk": n_chunk}


def check_logits(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Size check of a logits file against its own header: {path, exists, bytes, magic_ok,
    n_ctx, n_vocab, n_chunk, expected_bytes, size_ok, reason}."""
    p = Path(os.fspath(path))
    info: dict[str, Any] = {"path": paths.repo_relative(p), "exists": p.is_file(), "bytes": None,
                            "magic_ok": None, "n_ctx": None, "n_vocab": None, "n_chunk": None,
                            "expected_bytes": None, "size_ok": False, "reason": None}
    if not p.is_file():
        info["reason"] = "missing"
        return info
    info["bytes"] = p.stat().st_size
    try:
        h = read_logits_header(p)
    except (OSError, ValueError) as e:
        info["reason"] = str(e)
        return info
    info.update(h)
    if not h["magic_ok"]:
        info["reason"] = "bad magic (not a llama-perplexity logits file)"
        return info
    if h["n_ctx"] <= 1 or h["n_vocab"] <= 0 or h["n_chunk"] <= 0:
        info["reason"] = "implausible header values"
        return info
    info["expected_bytes"] = expected_logits_bytes(h["n_ctx"], h["n_vocab"], h["n_chunk"])
    info["size_ok"] = info["bytes"] == info["expected_bytes"]
    if not info["size_ok"]:
        info["reason"] = (f"size {info['bytes']} B != expected {info['expected_bytes']} B "
                          "(truncated or interrupted run)")
    return info


def _meta_path(logits: Path) -> Path:
    return logits.with_name(logits.name + ".json")


def read_logits_meta(logits: str | os.PathLike[str]) -> dict[str, Any] | None:
    """The ``<logits>.json`` sidecar written by kld-base, or None."""
    p = _meta_path(Path(os.fspath(logits)))
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, p)


# --------------------------------------------------------------------------------------------
# Calibration corpus
# --------------------------------------------------------------------------------------------
def corpus_sources(src_dir: str | os.PathLike[str] = paths.CALIBRATION_DIR) -> list[Path]:
    """Calibration files that go into all.txt (sorted by name; README*, all.txt, dotfiles and
    unknown suffixes excluded)."""
    d = Path(os.fspath(src_dir))
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.iterdir(), key=lambda x: x.name):
        if not p.is_file() or p.name.startswith(".") or p.name.upper().startswith("README"):
            continue
        if p.name in CORPUS_EXCLUDE or p.suffix.lower() not in CORPUS_SUFFIXES:
            continue
        out.append(p)
    return out


def _read_source(p: Path) -> str:
    s = p.read_bytes().decode("utf-8")
    if s.startswith("﻿"):
        s = s[1:]
    return s.replace("\r\n", "\n")


def _slices(text: str, slice_chars: int) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    n = 0
    for ln in text.splitlines(keepends=True):
        cur.append(ln)
        n += len(ln)
        if n >= slice_chars:
            out.append("".join(cur))
            cur, n = [], 0
    if cur and "".join(cur).strip():
        out.append("".join(cur))
    return out


def corpus_text(sources: Sequence[Path], mode: str = "interleave",
                slice_chars: int = CORPUS_SLICE_CHARS) -> tuple[str, list[dict[str, Any]]]:
    """(text, layout [{file, slice, start, chars}]) of the corpus built from ``sources``."""
    if mode not in CORPUS_MODES:
        raise ValueError(f"corpus mode must be one of {', '.join(CORPUS_MODES)}")
    texts = {p.name: _read_source(p) for p in sources}
    if mode == "concat":
        pieces = [(name, 0, t) for name, t in texts.items()]
    else:
        sl = {name: _slices(t, max(1, int(slice_chars))) for name, t in texts.items()}
        rounds = max((len(v) for v in sl.values()), default=0)
        pieces = [(name, i, sl[name][i]) for i in range(rounds) for name in sl if i < len(sl[name])]
    parts: list[str] = []
    layout: list[dict[str, Any]] = []
    pos = 0
    for name, i, t in pieces:
        if not t.endswith("\n"):
            t += "\n"
        if parts:
            parts.append("\n")
            pos += 1
        layout.append({"file": name, "slice": i, "start": pos, "chars": len(t)})
        parts.append(t)
        pos += len(t)
    return "".join(parts), layout


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_corpus(out: str | os.PathLike[str] | None = CORPUS_PATH, *,
                 src_dir: str | os.PathLike[str] = paths.CALIBRATION_DIR,
                 mode: str = "interleave", slice_chars: int = CORPUS_SLICE_CHARS,
                 write: bool = True) -> dict[str, Any]:
    """Build data/calibration/all.txt from the calibration files (see the module docstring).

    The file is (re)written only when ``write`` and its content would change. Returns {path,
    sha256, bytes, chars, mode, slice_chars, n_slices, written, up_to_date, files [{name,
    sha256, bytes}]}. Raises MissingInputError when there are no source files.
    """
    srcs = corpus_sources(src_dir)
    if not srcs:
        raise bench.MissingInputError(f"no calibration files in {src_dir}")
    text, layout = corpus_text(srcs, mode, slice_chars)
    data = text.encode("utf-8")
    sha = _sha256_bytes(data)
    info: dict[str, Any] = {
        "path": paths.repo_relative(out) if out is not None else None, "sha256": sha,
        "bytes": len(data), "chars": len(text), "mode": mode, "slice_chars": int(slice_chars),
        "n_slices": len(layout), "written": False, "up_to_date": None,
        "files": [{"name": p.name, "sha256": _sha256_file(p), "bytes": p.stat().st_size}
                  for p in srcs],
    }
    if out is not None:
        op = Path(os.fspath(out))
        same = op.is_file() and op.read_bytes() == data
        info["up_to_date"] = same
        if write and not same:
            op.parent.mkdir(parents=True, exist_ok=True)
            tmp = op.with_name(f".{op.name}.tmp{os.getpid()}")
            tmp.write_bytes(data)
            os.replace(tmp, op)
            info["written"] = True
            log.info("corpus: wrote %s (%d bytes, %d slices, sha256 %s)", op, len(data),
                     len(layout), sha[:16])
    return info


def text_info(p: Path) -> dict[str, Any]:
    """{path (repo-relative), bytes, sha256} of an input text."""
    return {"path": paths.repo_relative(p), "bytes": p.stat().st_size, "sha256": _sha256_file(p)}


# --------------------------------------------------------------------------------------------
# llama-perplexity runner
# --------------------------------------------------------------------------------------------
@dataclasses.dataclass
class PerplexityRun:
    argv: list[str]
    returncode: int | None
    output: str
    wall_s: float
    mem: dict[str, Any]
    log_path: str | None = None
    timed_out: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and self.error is None


def perplexity_argv(cmd: Sequence[str], model: str | os.PathLike[str],
                    text: str | os.PathLike[str], *, ctx: int, chunks: int, threads: int,
                    mode: str = "ppl", logits: str | os.PathLike[str] | None = None,
                    flags: Sequence[str] = ()) -> list[str]:
    """``llama-perplexity -m F -f TEXT -c C --chunks N -t T`` + mode flags + flags.

    mode 'ppl': nothing; 'base': ``--save-all-logits OUT``; 'kld': ``--kl-divergence
    --kl-divergence-base BASE``.
    """
    a = list(cmd) + ["-m", os.fspath(model), "-f", os.fspath(text), "-c", str(int(ctx)),
                     "--chunks", str(int(chunks)), "-t", str(int(threads))]
    if mode == "base":
        if logits is None:
            raise ValueError("mode 'base' needs the logits output path")
        a += ["--save-all-logits", os.fspath(logits)]
    elif mode == "kld":
        if logits is None:
            raise ValueError("mode 'kld' needs the base logits path")
        a += ["--kl-divergence", "--kl-divergence-base", os.fspath(logits)]
    elif mode != "ppl":
        raise ValueError(f"unknown perplexity mode {mode!r}")
    return a + [str(f) for f in flags]


def _preexec(nice: int | None) -> Any:
    if not nice or paths.is_windows():
        return None
    n = int(nice)

    def _pre() -> None:
        with contextlib.suppress(OSError):
            os.nice(n)
    return _pre


def _stop_child(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=bench.STOP_GRACE_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=bench.STOP_GRACE_S)


def run_perplexity(argv: Sequence[str], *, log_path: Path | None = None,
                   timeout: float | None = None, nice: int | None = None,
                   sample_interval: float = bench.DEFAULT_SAMPLE_INTERVAL_S) -> PerplexityRun:
    """Run one llama-perplexity command; stdout + stderr go to ``log_path`` (or a temp file)
    and are returned as text. Only ever stops the child it started."""
    tmp_log = None
    if log_path is None:
        fd, name = tempfile.mkstemp(prefix="pigemma-ppl-", suffix=".log")
        os.close(fd)
        tmp_log = Path(name)
    lp = log_path or tmp_log
    assert lp is not None
    lp.parent.mkdir(parents=True, exist_ok=True)
    log.info("run: %s", _cmd_str(argv))
    log.info("output -> %s", lp)
    t0 = time.monotonic()
    timed_out = False
    err: str | None = None
    mem: dict[str, Any] = {}
    rc: int | None = None
    try:
        with open(lp, "wb") as f:
            kw: dict[str, Any] = {}
            pre = _preexec(nice)
            if pre is not None:
                kw["preexec_fn"] = pre
            try:
                proc = subprocess.Popen(list(argv), stdin=subprocess.DEVNULL, stdout=f,
                                        stderr=subprocess.STDOUT, **kw)
            except OSError as e:
                return PerplexityRun(list(argv), None, "", 0.0, {}, paths.repo_relative(lp),
                                     error=f"cannot start: {type(e).__name__}: {e}")
            mon = bench.ProcessMonitor(proc.pid, interval=sample_interval, t0=t0).start()
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                log.warning("llama-perplexity exceeded %.0f s; stopping it", timeout or 0)
                _stop_child(proc)
                rc = proc.returncode
            except BaseException:
                _stop_child(proc)
                mon.stop()
                raise
            mon.stop()
            mem = mon.summary()
        output = lp.read_bytes().decode("utf-8", errors="replace").replace("\r\n", "\n")
    finally:
        if tmp_log is not None:
            with contextlib.suppress(OSError):
                tmp_log.unlink()
    if rc not in (0, None) and not timed_out:
        err = f"exit code {rc}"
    return PerplexityRun(list(argv), rc, output, round(time.monotonic() - t0, 3), mem,
                         None if tmp_log is not None else paths.repo_relative(lp),
                         timed_out=timed_out, error=err)


# --------------------------------------------------------------------------------------------
# CLI helpers
# --------------------------------------------------------------------------------------------
def _cmd_str(argv: Sequence[str]) -> str:
    if paths.is_windows():
        return subprocess.list2cmdline([str(a) for a in argv])
    return shlex.join([str(a) for a in argv])


def _split(s: str) -> list[str]:
    if not paths.is_windows():
        return shlex.split(s)
    out = []
    for x in shlex.split(s, posix=False):
        out.append(x[1:-1] if len(x) >= 2 and x[0] == x[-1] and x[0] in "\"'" else x)
    return out


def _tool_cmd(tool: str, override: str | Sequence[str] | None) -> list[str]:
    """Binary command prefix: paths.bin(tool), an existing file, or a shell-split string."""
    if override is None:
        return [str(paths.bin(tool))]
    if isinstance(override, (list, tuple)):
        return [str(x) for x in override]
    s = os.fspath(override)
    if Path(s).is_file() or " " not in s.strip():
        return [s]
    return _split(s)


def _check_cmd(cmd: Sequence[str]) -> None:
    exe = cmd[0]
    if os.path.sep in exe or "/" in exe or (paths.is_windows() and ":" in exe):
        if not Path(exe).is_file():
            raise bench.MissingInputError(f"binary not found: {exe}")
    elif shutil.which(exe) is None:
        raise bench.MissingInputError(f"binary not found on PATH: {exe}")


def _extra_flags(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    for s in getattr(args, "flags", None) or []:
        out.extend(_split(s))
    out.extend(getattr(args, "extra", None) or [])
    return out


def default_experiment() -> str:
    """E8 on aarch64 hosts (the Pi, the emulated Pi), E2 elsewhere."""
    return "E8" if paths.machine() == "aarch64" else "E2"


def _resolve_model(spec: str, *, require: bool) -> tuple[Path, str | None]:
    p, mid = bench.resolve_model(spec)
    if require and not p.is_file():
        hint = ("; " + paths.download_hint(mid)) if mid in paths.MODELS else ""
        raise bench.MissingInputError(f"model {spec!r} not found at {p}{hint}")
    return p, mid


def _model_id(model: Mapping[str, Any] | None, path: Path) -> str:
    mid = (model or {}).get("id") or path.stem
    return re.sub(r"[^A-Za-z0-9.+\-_]+", "-", str(mid)).strip("-") or "model"


def _resolve_text(arg: str | None, *, write: bool = True) -> tuple[Path, dict[str, Any] | None]:
    """(text path, corpus build info or None). The default corpus is (re)built when needed
    (``write=False``: only checked, e.g. for --dry-run)."""
    if arg is None or Path(arg).resolve() == CORPUS_PATH.resolve():
        info = build_corpus(CORPUS_PATH, write=write)
        return CORPUS_PATH, info
    p = Path(arg).expanduser()
    if not p.is_file():
        raise bench.MissingInputError(f"text file not found: {p}")
    return p, None


def _out_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg).expanduser()
    return paths.RESULTS_DIR / _dt.date.today().isoformat()


def _is_dir_arg(s: str | None) -> bool:
    if not s:
        return False
    p = Path(s)
    return p.is_dir() or s.endswith(("/", "\\")) or (not p.exists() and p.suffix == "")


def _lock(args: argparse.Namespace, cmd: str) -> Any:
    mf = args.min_free_gb if getattr(args, "min_free_gb", None) is not None else paths.DEFAULT_MIN_FREE_GB
    return paths.bench_lock(mf, cmd=cmd)


def _build(cmd: Sequence[str], output: str | None = None) -> tuple[Any, Any]:
    bn, commit = bench.parse_build_info(output or "")
    if bn is None:
        info = bench.binary_version(cmd)
        bn, commit = info.get("build_number"), info.get("commit")
    return bn, commit


def _mb(b: Any) -> float | None:
    if b is None:
        return None
    try:
        return round(float(b) / bench.MB, 1)
    except (TypeError, ValueError):
        return None


def _stem(experiment: str, mid: str, variant: str, threads: Any) -> str:
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return "_".join(re.sub(r"[^A-Za-z0-9.+\-]+", "-", str(x)).strip("-.") or "na"
                    for x in (experiment, mid, variant, threads, ts))


def _write_envelope(out: Path | str, fields: Mapping[str, Any]) -> Path:
    p = results.write_result(out, **fields)
    print(paths.repo_relative(p), flush=True)
    return p


def _common_notes(build: tuple[Any, Any]) -> list[str]:
    notes: list[str] = []
    bn, commit = build
    if bn is not None and commit is not None:
        if str(commit).lower().startswith(paths.LLAMA_COMMIT) and bn != paths.LLAMA_BUILD:
            notes.append(f"llama.cpp reported build_number {bn} for the pinned commit "
                         f"{paths.LLAMA_COMMIT} (shallow-clone build count; the pinned build is "
                         f"b{paths.LLAMA_BUILD})")
        elif not str(commit).lower().startswith(paths.LLAMA_COMMIT):
            notes.append(f"llama.cpp commit {commit} differs from the pinned {paths.LLAMA_COMMIT}")
    if os.environ.get("PIGEMMA_EMULATED", "").strip() == "1":
        notes.append("emulated host (qemu): a functional proof, not a Pi measurement")
    return notes


def _perplexity_failure(run: PerplexityRun, what: str) -> str:
    errs = perplexity_errors(run.output)
    why = ("timed out" if run.timed_out else run.error or
           ("; ".join(errs) if errs else f"no {what} in the output"))
    note = bench.failure_note(run.output, extra=f"llama-perplexity: {why}")
    return note


# --------------------------------------------------------------------------------------------
# Subcommands: corpus / parse / check
# --------------------------------------------------------------------------------------------
def cmd_corpus(args: argparse.Namespace) -> int:
    out = Path(args.out).expanduser() if args.out else CORPUS_PATH
    info = build_corpus(out, src_dir=args.src_dir or paths.CALIBRATION_DIR, mode=args.mode,
                        slice_chars=args.slice_chars, write=not args.check)
    print(json.dumps(info, indent=1, ensure_ascii=False))
    if args.check and not info["up_to_date"]:
        print(f"pigemma.quality: {out} is missing or out of date", file=sys.stderr)
        return paths.EXIT_ERROR
    return paths.EXIT_OK


def _read_input(arg: str) -> str:
    if arg == "-":
        return sys.stdin.read()
    p = Path(arg).expanduser()
    if not p.is_file():
        raise bench.MissingInputError(f"input file not found: {p}")
    return p.read_bytes().decode("utf-8", errors="replace")


def cmd_parse_kld(args: argparse.Namespace) -> int:
    d = parse_kld_output(_read_input(args.input))
    d["kld"] = kld_block(d)
    d["tier"] = kld_tier(d.get("mean"))
    d["neutral"] = kld_neutral(d)
    print(json.dumps(d, indent=1, ensure_ascii=False))
    return paths.EXIT_OK if d["ok"] else paths.EXIT_ERROR


def cmd_parse_ppl(args: argparse.Namespace) -> int:
    d = parse_ppl_output(_read_input(args.input))
    print(json.dumps(d, indent=1, ensure_ascii=False))
    return paths.EXIT_OK if d["ok"] else paths.EXIT_ERROR


def cmd_check_logits(args: argparse.Namespace) -> int:
    p = Path(args.file).expanduser()
    if not p.is_file():
        raise bench.MissingInputError(f"logits file not found: {p}")
    info = check_logits(p)
    info["meta"] = read_logits_meta(p)
    print(json.dumps(info, indent=1, ensure_ascii=False))
    return paths.EXIT_OK if info["size_ok"] else paths.EXIT_ERROR


def cmd_check_text(args: argparse.Namespace) -> int:
    raw = _read_input(args.input)
    vis, th = strip_thinking(raw)
    ok, why = is_coherent(vis)
    print(json.dumps({"visible": vis, "thinking_chars": th, "coherent": ok, "reason": why},
                     indent=1, ensure_ascii=False))
    return paths.EXIT_OK


# --------------------------------------------------------------------------------------------
# Subcommands: kld-base / kld / ppl
# --------------------------------------------------------------------------------------------
def _ppl_common_fields(*, experiment: str, model: Mapping[str, Any], variant: str,
                       flags: Sequence[str], threads: int, ctx: int, run: PerplexityRun,
                       build: tuple[Any, Any], inputs: Sequence[str]) -> dict[str, Any]:
    return {
        "experiment": experiment, "kind": "measured", "llama_build": build[0],
        "llama_commit": build[1], "model": dict(model), "variant": variant,
        "flags": [str(x) for x in flags], "threads": int(threads), "n_ctx": int(ctx),
        "rss_peak_mb": _mb(run.mem.get("rss_peak_bytes")),
        "private_peak_mb": _mb(run.mem.get("private_peak_bytes")),
        "mapped_mb": _mb(run.mem.get("mapped_peak_bytes")),
        "inputs": [str(x) for x in inputs],
    }


def _variant_flags(variant: str, *, allow_label: bool, model_id: str | None,
                   base_model_id: str | None) -> tuple[list[str], str]:
    """(flags, kind 'load'|'model'|'label') of a KLD / PPL variant name (see the docstring)."""
    try:
        return bench.variant_flags(variant, "run"), "load"
    except ValueError:
        pass
    if variant in paths.MODELS or bench.catalog_model(variant) is not None:
        if model_id and variant != model_id:
            log.warning("variant %s names a model but --model resolved to %s", variant, model_id)
        return [], "model"
    if allow_label:
        return [], "label"
    if model_id and base_model_id and model_id != base_model_id:
        return [], "label"
    raise ValueError(
        f"variant {variant!r} is neither a load variant ({', '.join(bench.VARIANT_FLAGS)}; "
        "'+' combines) nor a catalog model id; pass --allow-label to use it as a free label")


def cmd_kld_base(args: argparse.Namespace) -> int:
    extra = _extra_flags(args)
    vflags, vkind = _variant_flags(args.variant, allow_label=True, model_id=None, base_model_id=None)
    flags = vflags + extra
    experiment = args.experiment or default_experiment()
    cmd = _tool_cmd("llama-perplexity", args.bin)
    model_path, mid = _resolve_model(args.model, require=not args.dry_run)
    model = bench.model_block(model_path, mid, with_sha=not args.no_sha and model_path.is_file())
    model_id = _model_id(model, model_path)
    text, corpus = _resolve_text(args.text, write=not args.dry_run)
    if args.out and _is_dir_arg(args.out):
        out = Path(args.out).expanduser() / f"ref_{model_id}.logits"
    else:
        out = Path(args.out).expanduser() if args.out else QUALITY_DIR / f"ref_{model_id}.logits"
    existing = out.stat().st_size if out.is_file() else 0
    free = paths.free_disk_bytes(out.parent) + existing
    gate = disk_gate(free, args.chunks, args.ctx, args.n_vocab, DISK_GATE_FACTOR)
    print(f"pigemma.quality: logits size estimate {gate['estimate_bytes']:,} B "
          f"({gate['estimate_bytes'] / 1e9:.2f} GB; exact layout {gate['exact_bytes']:,} B); "
          f"free disk at {paths.repo_relative(out.parent)}: {gate['free_bytes']:,} B "
          f"({gate['free_bytes'] / 1e9:.2f} GB); need >= {DISK_GATE_FACTOR:g} x estimate = "
          f"{gate['required_bytes']:,} B", file=sys.stderr)
    argv = perplexity_argv(cmd, model_path, text, ctx=args.ctx, chunks=args.chunks,
                           threads=args.threads, mode="base", logits=out, flags=flags)
    if args.dry_run:
        print(_cmd_str(argv))
        print(f"disk gate: {'ok' if gate['ok'] else 'REFUSED'} (estimate "
              f"{gate['estimate_bytes']} B, free {gate['free_bytes']} B, required "
              f"{gate['required_bytes']} B)")
        return paths.EXIT_OK
    tinfo = text_info(text)
    # reuse a valid, matching reference unless --force
    if out.exists() and not args.force:
        chk = check_logits(out)
        meta = read_logits_meta(out) or {}
        same = (chk["size_ok"] and chk["n_ctx"] == args.ctx and meta.get("chunks") == args.chunks
                and (meta.get("model") or {}).get("sha_prefix") == model.get("sha_prefix")
                and (meta.get("model") or {}).get("bytes") == model.get("bytes")
                and (meta.get("text") or {}).get("sha256") == tinfo["sha256"]
                and meta.get("flags") == flags)
        if same:
            print(f"pigemma.quality: {paths.repo_relative(out)} is a valid reference for "
                  f"{model_id} (ctx {args.ctx}, {args.chunks} chunks, corpus "
                  f"{tinfo['sha256'][:12]}); reusing it (--force recomputes)", file=sys.stderr)
            print(paths.repo_relative(out), flush=True)
            return paths.EXIT_OK
        if chk["size_ok"]:
            raise QualityError(
                f"{out} is a valid logits file but does not match this run (different model / "
                "ctx / chunks / corpus / flags); pass --force to overwrite it")
        # a broken file (interrupted / failed run) is never worth keeping: replace it
        log.warning("existing %s is not a usable logits file (%s); replacing it", out,
                    chk.get("reason"))
    if not gate["ok"]:
        raise DiskGateError(
            f"not enough free disk for the reference logits: {gate['free_bytes']:,} B free, "
            f"need >= {gate['required_bytes']:,} B (2 x estimate {gate['estimate_bytes']:,} B)")
    _check_cmd(cmd)
    log_path = out.with_name(out.name + ".log")
    with _lock(args, f"pigemma.quality kld-base {model_id}"):
        run = run_perplexity(argv, log_path=log_path, timeout=args.timeout, nice=args.nice)
    build = _build(cmd, run.output)
    parsed = parse_ppl_output(run.output)
    chk = check_logits(out)
    ok = run.ok and parsed["ok"] and chk["size_ok"]
    notes = _common_notes(build)
    if not (run.ok and parsed["ok"]):
        notes.insert(0, _perplexity_failure(run, "'Final estimate' line"))
    elif not chk["size_ok"]:
        notes.insert(0, f"failed: logits file check: {chk['reason']}")
    if chk.get("n_chunk") is not None and chk["n_chunk"] < args.chunks:
        notes.append(f"the text gave only {chk['n_chunk']} chunks of {args.ctx} tokens "
                     f"({args.chunks} requested)")
    meta = {
        "format": LOGITS_META_FORMAT, "created_at": results.utc_now_iso(),
        "logits": paths.repo_relative(out), "check": chk, "model": model, "variant": args.variant,
        "flags": flags, "ctx": args.ctx, "chunks": args.chunks, "threads": args.threads,
        "n_vocab_assumed": args.n_vocab, "text": tinfo, "corpus": corpus, "argv": run.argv,
        "llama_build": build[0], "llama_commit": build[1], "ppl": parsed["ppl"],
        "ppl_sd": parsed["ppl_sd"], "estimate_bytes": gate["estimate_bytes"],
        "log": run.log_path, "ok": ok, "wall_s": run.wall_s,
    }
    if out.exists():
        _write_json(_meta_path(out), meta)
    print(f"pigemma.quality: kld-base {model_id}: PPL {parsed['ppl']} +/- {parsed['ppl_sd']}; "
          f"logits {chk.get('bytes')} B (expected {chk.get('expected_bytes')}) -> "
          f"{'ok' if chk['size_ok'] else 'SIZE CHECK FAILED'}", file=sys.stderr)
    if not args.no_envelope:
        f = _ppl_common_fields(experiment=experiment, model=model, variant=args.variant,
                               flags=flags, threads=args.threads, ctx=args.ctx, run=run,
                               build=build, inputs=[tinfo["path"]])
        f["ppl"] = parsed["ppl"]
        f["extra"] = {
            "source": "llama-perplexity", "role": "kld-base", "variant_kind": vkind,
            "argv": run.argv, "raw_log": run.log_path, "wall_s": run.wall_s,
            "returncode": run.returncode, "ppl_sd": parsed["ppl_sd"],
            "chunk_ppl": parsed["chunk_ppl"], "n_chunks": parsed["n_chunks"],
            "computing": parsed["computing"], "chunks_requested": args.chunks,
            "logits": chk, "logits_meta": paths.repo_relative(_meta_path(out)),
            "estimate_bytes": gate["estimate_bytes"], "text": tinfo, "corpus": corpus,
            "mem": run.mem,
        }
        if notes:
            f["notes"] = notes
        _write_envelope(_out_dir(args.results_dir), f)
    elif notes:
        for n in notes:
            print(f"pigemma.quality: {n}", file=sys.stderr)
    print(paths.repo_relative(out), flush=True)
    return paths.EXIT_OK if ok else paths.EXIT_ERROR


def cmd_kld(args: argparse.Namespace) -> int:
    base = Path(args.base).expanduser()
    base_present = base.is_file()
    if base_present:
        chk = check_logits(base)
        if not chk["magic_ok"]:
            raise QualityError(f"{base}: {chk['reason']}")
        if not chk["size_ok"]:
            raise QualityError(f"{base}: {chk['reason']}; recompute it with kld-base --force")
        meta = read_logits_meta(base) or {}
    elif args.dry_run:
        # a dry run may precede kld-base: show the command with the requested -c / --chunks
        chk = {"n_ctx": args.ctx or DEFAULT_CTX, "n_chunk": args.chunks}
        meta = {}
        log.warning("base logits %s not present yet (dry run): showing -c %d --chunks %d",
                    base, chk["n_ctx"], chk["n_chunk"])
    else:
        raise bench.MissingInputError(f"base logits not found: {base} (run kld-base first)")
    base_model = meta.get("model") or {}
    extra = _extra_flags(args)
    experiment = args.experiment or default_experiment()
    cmd = _tool_cmd("llama-perplexity", args.bin)
    model_path, mid = _resolve_model(args.model, require=not args.dry_run)
    model = bench.model_block(model_path, mid, with_sha=not args.no_sha and model_path.is_file())
    model_id = _model_id(model, model_path)
    vflags, vkind = _variant_flags(args.variant, allow_label=args.allow_label, model_id=model_id,
                                   base_model_id=base_model.get("id"))
    flags = vflags + extra
    ctx = int(chk["n_ctx"])
    chunks = int(chk["n_chunk"])
    if base_present and args.ctx is not None and args.ctx != ctx:
        log.warning("-c %d ignored: the base logits were computed with n_ctx %d", args.ctx, ctx)
    text_arg = args.text
    if text_arg is None:
        mt = (meta.get("text") or {}).get("path")
        if mt and (paths.REPO_ROOT / mt).is_file():
            text_arg = str(paths.REPO_ROOT / mt)
    text, corpus = _resolve_text(text_arg, write=not args.dry_run)
    tinfo = text_info(text) if text.is_file() else {"path": paths.repo_relative(text)}
    if (meta.get("text") or {}).get("sha256") not in (None, tinfo.get("sha256")):
        log.warning("text %s differs from the base run's (KLD uses the base file's tokens anyway)",
                    tinfo["path"])
    argv = perplexity_argv(cmd, model_path, text, ctx=ctx, chunks=chunks, threads=args.threads,
                           mode="kld", logits=base, flags=flags)
    if args.dry_run:
        print(_cmd_str(argv))
        return paths.EXIT_OK
    _check_cmd(cmd)
    out = _out_dir(args.out) if (args.out is None or _is_dir_arg(args.out)) else Path(args.out)
    raw_dir = (out if _is_dir_arg(str(out)) else out.parent) / "raw"
    log_path = raw_dir / f"{_stem(experiment, model_id, args.variant, args.threads)}.perplexity.log"
    with _lock(args, f"pigemma.quality kld {model_id} {args.variant}"):
        run = run_perplexity(argv, log_path=log_path, timeout=args.timeout, nice=args.nice)
    build = _build(cmd, run.output)
    parsed = parse_kld_output(run.output)
    kld = kld_block(parsed)
    ok = run.ok and parsed["ok"]
    notes = []
    if not ok:
        notes.append(_perplexity_failure(run, "'Mean KLD' line"))
    notes += _common_notes(build)
    if base_model.get("id") and base_model.get("id") != model.get("id"):
        notes.append(f"KLD of {model.get('id')} against the reference logits of {base_model.get('id')}")
    f = _ppl_common_fields(experiment=experiment, model=model, variant=args.variant, flags=flags,
                           threads=args.threads, ctx=ctx, run=run, build=build,
                           inputs=[tinfo["path"], paths.repo_relative(base)])
    if ok:
        f["kld"] = kld
        f["ppl"] = parsed["ppl_q"]
    f["extra"] = {
        "source": "llama-perplexity", "role": "kld", "variant_kind": vkind,
        "argv": run.argv, "raw_log": run.log_path, "wall_s": run.wall_s,
        "returncode": run.returncode,
        "base": {"logits": paths.repo_relative(base), "bytes": chk["bytes"],
                 "n_ctx": chk["n_ctx"], "n_vocab": chk["n_vocab"], "n_chunk": chk["n_chunk"],
                 "model": base_model or None, "variant": meta.get("variant"),
                 "flags": meta.get("flags"), "text_sha256": (meta.get("text") or {}).get("sha256")},
        "kld_stats": {k: v for k, v in parsed.items() if k not in ("chunks", "ok")},
        "chunks": parsed["chunks"], "text": tinfo, "corpus": corpus, "mem": run.mem,
        "threshold_check": {
            "rule": ("E2 (pre-registered): neutral = KLD mean < 0.001 and same-top-p > 99.5 %; "
                     "quant tier: mean < 0.05 acceptable, < 0.15 degraded, else not recommended"),
            "neutral": kld_neutral(kld), "tier": kld_tier(kld.get("mean")),
        },
    }
    if notes:
        f["notes"] = notes
    _write_envelope(out, f)
    print(f"pigemma.quality: kld {model_id} {args.variant}: mean {kld['mean']} max {kld['max']} "
          f"p99.9 {kld['p999']} same_top_p {kld['same_top_p']} PPL(Q) {kld['ppl_q']} "
          f"PPL(base) {kld['ppl_base']} -> neutral={kld_neutral(kld)} tier={kld_tier(kld['mean'])}",
          file=sys.stderr)
    return paths.EXIT_OK if ok else paths.EXIT_ERROR


def cmd_ppl(args: argparse.Namespace) -> int:
    extra = _extra_flags(args)
    vflags, vkind = _variant_flags(args.variant, allow_label=True, model_id=None, base_model_id=None)
    flags = vflags + extra
    experiment = args.experiment or default_experiment()
    cmd = _tool_cmd("llama-perplexity", args.bin)
    model_path, mid = _resolve_model(args.model, require=not args.dry_run)
    model = bench.model_block(model_path, mid, with_sha=not args.no_sha and model_path.is_file())
    model_id = _model_id(model, model_path)
    text, corpus = _resolve_text(args.text, write=not args.dry_run)
    argv = perplexity_argv(cmd, model_path, text, ctx=args.ctx, chunks=args.chunks,
                           threads=args.threads, mode="ppl", flags=flags)
    if args.dry_run:
        print(_cmd_str(argv))
        return paths.EXIT_OK
    _check_cmd(cmd)
    tinfo = text_info(text)
    out = _out_dir(args.out) if (args.out is None or _is_dir_arg(args.out)) else Path(args.out)
    raw_dir = (out if _is_dir_arg(str(out)) else out.parent) / "raw"
    log_path = raw_dir / f"{_stem(experiment, model_id, args.variant, args.threads)}.perplexity.log"
    with _lock(args, f"pigemma.quality ppl {model_id} {args.variant}"):
        run = run_perplexity(argv, log_path=log_path, timeout=args.timeout, nice=args.nice)
    build = _build(cmd, run.output)
    parsed = parse_ppl_output(run.output)
    ok = run.ok and parsed["ok"]
    notes = [] if ok else [_perplexity_failure(run, "'Final estimate' line")]
    notes += _common_notes(build)
    f = _ppl_common_fields(experiment=experiment, model=model, variant=args.variant, flags=flags,
                           threads=args.threads, ctx=args.ctx, run=run, build=build,
                           inputs=[tinfo["path"]])
    f["ppl"] = parsed["ppl"]
    f["extra"] = {"source": "llama-perplexity", "role": "ppl", "variant_kind": vkind,
                  "argv": run.argv, "raw_log": run.log_path, "wall_s": run.wall_s,
                  "returncode": run.returncode, "ppl_sd": parsed["ppl_sd"],
                  "chunk_ppl": parsed["chunk_ppl"], "n_chunks": parsed["n_chunks"],
                  "chunks_requested": args.chunks, "computing": parsed["computing"],
                  "text": tinfo, "corpus": corpus, "mem": run.mem}
    if notes:
        f["notes"] = notes
    _write_envelope(out, f)
    print(f"pigemma.quality: ppl {model_id} {args.variant}: {parsed['ppl']} +/- {parsed['ppl_sd']}",
          file=sys.stderr)
    return paths.EXIT_OK if ok else paths.EXIT_ERROR


# --------------------------------------------------------------------------------------------
# Subcommand: qa
# --------------------------------------------------------------------------------------------
def _qa_set_info(p: Path, items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"path": paths.repo_relative(p), "sha256": _sha256_file(p), "n_items": len(items),
            "ids": [str(it["id"]) for it in items]}


def _plain_from_file(p: Path) -> dict[str, str]:
    """{id: visible} from a QA result file, or (fallback) from a server-spec envelope."""
    try:
        return {k: v for k, v in load_qa_file(p)["visible_by_id"].items() if v is not None}
    except QualityError:
        return {k: v for k, v in bench.load_plain_ref_file(p).items() if v is not None}


def _qa_out_path(out_arg: str | None, model_id: str, variant: str, multi: bool) -> Path:
    name = f"qa_{model_id}_{re.sub(r'[^A-Za-z0-9.+-]+', '-', variant)}.json"
    if out_arg is None:
        return QUALITY_DIR / name
    if _is_dir_arg(out_arg):
        return Path(out_arg).expanduser() / name
    if multi:
        raise ValueError("--out must be a directory when several --server-variant values are given")
    return Path(out_arg).expanduser()


def _qa_finish(fields: dict[str, Any], report: Mapping[str, Any], *, contents: Mapping[str, Any],
               plain: Mapping[str, str] | None, plain_source: str | None, is_reference: bool,
               qa_info: Mapping[str, Any], thinking: bool, n_predict: int) -> dict[str, Any]:
    """Add the Q&A fields to bench-style envelope kwargs."""
    if is_reference:
        frac, by = None, None
    else:
        frac, by = exact_match_vs_plain(contents, plain)
    fields["qa_score"] = report["qa_score"]
    fields["qa_total"] = report["qa_total"]
    fields["coherent"] = report["coherent"]
    if not is_reference:
        fields["identical_to_plain"] = frac
    ex = dict(fields.get("extra") or {})
    ex.update({
        "items": report["items"], "per_language": report["per_language"],
        "exact_match_vs_plain": frac, "exact_match_by_item": by,
        "plain_ref": None if is_reference else plain_source,
        "qa_set": dict(qa_info), "thinking_enabled": thinking, "n_predict": n_predict,
        "visible_chars_total": report["visible_chars_total"],
        "thinking_chars_total": report["thinking_chars_total"],
        "thinking_share": report["thinking_share"],
        "threshold_check": qa_threshold_check(report),
        "grader": "pigemma.quality.grade (contains: NFKC+casefold+niqqud/finals+punct; "
                  "regex: re.search raw or normalised; number: first number within tolerance)",
    })
    fields["extra"] = ex
    notes = list(fields.get("notes") or [])
    if not is_reference and plain is None:
        notes.append("no plain Q&A reference available: exact_match_vs_plain n/a")
    notes = [n for n in notes if not n.startswith("no plain reference available: identical_to_plain")]
    if notes:
        fields["notes"] = notes
    return fields


def _qa_server_variant(args: argparse.Namespace, *, v: str, items: Sequence[Mapping[str, Any]],
                       model_path: Path, mid: str | None, model: Mapping[str, Any],
                       model_id: str, experiment: str, base_flags: Sequence[str],
                       load_suffix: str, out_path: Path, build: tuple[Any, Any],
                       plain: Mapping[str, str] | None, plain_source: str | None,
                       qa_info: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str | None]]:
    """One llama-server run over the Q&A items -> (envelope kwargs, {id: content})."""
    env_variant = v + load_suffix
    dtype, st = bench.SERVER_VARIANT_SPEC[v]
    dp: Path | None = None
    did: str | None = None
    if dtype is not None:
        if args.draft:
            dp, did = Path(args.draft).expanduser(), paths.model_id_for(args.draft)
        else:
            dp, did = bench.resolve_draft(mid, model_path, dtype)
    flags = list(base_flags)
    draft_info = None
    if dp is not None:
        flags = ["-md", paths.repo_relative(dp)] + flags
        draft_info = {"id": did, "path": paths.repo_relative(dp),
                      "bytes": dp.stat().st_size if dp.is_file() else None}
    if st:
        flags = flags + ["--spec-type", st]
    if args.draft_n_max is not None and (dp is not None or st):
        flags = flags + ["--spec-draft-n-max", str(args.draft_n_max)]
    stem = _stem(experiment, model_id, env_variant, args.threads)
    log_path = out_path.parent / "raw" / f"{stem}.server.log"
    ex_common: dict[str, Any] = {"source": "llama-server", "engine": "llama-server",
                                 "server_log": paths.repo_relative(log_path),
                                 "load_variant": args.load_variant, "port": args.port,
                                 "warmup_tokens": args.warmup_tokens}
    kw = dict(experiment=experiment, model=model, variant=env_variant, flags=flags,
              threads=args.threads, n_ctx=args.ctx, server_variant=v, spec_type=st,
              draft=draft_info, plain=None if v == "plain" else plain,
              plain_source=None if v == "plain" else plain_source, build=build)
    if dtype is not None and (dp is None or not dp.is_file()):
        note = (f"failed: no '{dtype}' draft found for model {model.get('id')!r} "
                f"({dp if dp else 'not in the catalog'})")
        f = bench.server_fields([], load_time_s=None, mem=None, failure=note, extra=ex_common, **kw)
        return f, {}
    srv = bench.LlamaServer(model_path, threads=args.threads, n_ctx=args.ctx, port=args.port,
                            draft=dp, spec_type=st,
                            draft_n_max=args.draft_n_max if (dp is not None or st) else None,
                            flags=base_flags, binary=_tool_cmd("llama-server", args.bin),
                            log_path=log_path, health_timeout=args.health_timeout,
                            nice=args.nice)
    ex_common["argv"] = [str(a) for a in srv.argv]
    wrapped = [{**it, "prompt": chat_wrap(str(it["prompt"]), thinking=args.thinking)} for it in items]
    records: list[dict[str, Any]] = []
    failure: str | None = None
    try:
        srv.start()
        records = bench.run_prompt_set(srv, wrapped, n_predict=args.n, chat=False,
                                       warmup_tokens=args.warmup_tokens,
                                       request_timeout=args.request_timeout)
    except bench.ServerStartError as e:
        failure = str(e)
    finally:
        srv.stop()
    mem = srv.monitor.summary() if srv.monitor else None
    ex = dict(ex_common)
    ex.update(server_returncode=srv.returncode, health_wait_s=srv.health_wait_s,
              load_time_source=srv.load_time_source)
    f = bench.server_fields(records, load_time_s=srv.load_time_s, mem=mem, failure=failure,
                            extra=ex, **kw)
    contents = {str(r["id"]): r.get("content") for r in records}
    errs = {str(r["id"]): r.get("error") for r in records if r.get("error")}
    f["_errors"] = errs
    f["_per_item"] = {str(r["id"]): {"predicted_n": r.get("predicted_n"),
                                     "stop_type": r.get("stop_type"),
                                     "tg_tok_s": r.get("predicted_per_second")} for r in records}
    return f, contents


def _qa_completion_run(args: argparse.Namespace, *, items: Sequence[Mapping[str, Any]],
                       model_path: Path, model: Mapping[str, Any], model_id: str,
                       experiment: str, base_flags: Sequence[str], load_suffix: str,
                       out_path: Path, build: tuple[Any, Any]
                       ) -> tuple[dict[str, Any], dict[str, str | None]]:
    """llama-completion once per item (one model load each) -> (envelope kwargs, contents)."""
    variant = "completion" + load_suffix
    binary = _tool_cmd("llama-completion", args.bin)
    stem = _stem(experiment, model_id, variant, args.threads)
    contents: dict[str, str | None] = {}
    errs: dict[str, str | None] = {}
    runs: list[dict[str, Any]] = []
    per: dict[str, dict[str, Any]] = {}
    rss = priv = mapped = None
    eval_ms = n_eval = prompt_ms = n_prompt = 0.0
    tg_rates: list[float] = []
    pp_rates: list[float] = []
    loads: list[float] = []
    n_ok = 0
    for it in items:
        iid = str(it["id"])
        prompt = chat_wrap(str(it["prompt"]), thinking=args.thinking)
        err_path = out_path.parent / "raw" / f"{stem}.{iid}.stderr.log"
        run = bench.run_completion(model_path, prompt, n_predict=args.n, threads=args.threads,
                                   n_ctx=args.ctx, flags=base_flags, chat=False, binary=binary,
                                   timeout=args.timeout, nice=args.nice, stderr_path=err_path)
        p = run.parsed
        if run.returncode == 0 and not run.timed_out and run.error is None:
            contents[iid] = run.stdout_text
            n_ok += 1
        else:
            contents[iid] = None
            errs[iid] = bench.failure_note(run.stderr_text, extra=(
                "timed out" if run.timed_out else run.error or f"exit code {run.returncode}"))
        if p.get("tg_tok_s") is not None and p.get("eval_ms") and p.get("n_eval"):
            tg_rates.append(float(p["tg_tok_s"]))
            eval_ms += float(p["eval_ms"])
            n_eval += float(p["n_eval"])
        if p.get("pp_tok_s") is not None and p.get("prompt_ms") and p.get("n_prompt"):
            pp_rates.append(float(p["pp_tok_s"]))
            prompt_ms += float(p["prompt_ms"])
            n_prompt += float(p["n_prompt"])
        if p.get("load_s") is not None:
            loads.append(float(p["load_s"]))
        for key, cur in (("rss_peak_bytes", "rss"), ("private_peak_bytes", "priv"),
                         ("mapped_peak_bytes", "mapped")):
            v = run.mem.get(key)
            if v is not None:
                if cur == "rss":
                    rss = max(rss or 0, int(v))
                elif cur == "priv":
                    priv = max(priv or 0, int(v))
                else:
                    mapped = max(mapped or 0, int(v))
        per[iid] = {"predicted_n": p.get("n_eval"), "stop_type": None, "tg_tok_s": p.get("tg_tok_s")}
        runs.append({"id": iid, "returncode": run.returncode, "wall_s": run.wall_s,
                     "load_s": p.get("load_s"), "tg_tok_s": p.get("tg_tok_s"),
                     "n_eval": p.get("n_eval"), "pp_tok_s": p.get("pp_tok_s"),
                     "n_prompt": p.get("n_prompt"), "stderr_log": paths.repo_relative(err_path),
                     "timed_out": run.timed_out})
    f: dict[str, Any] = {
        "experiment": experiment, "kind": "measured", "llama_build": build[0],
        "llama_commit": build[1], "model": dict(model), "variant": variant,
        "flags": [str(x) for x in base_flags], "threads": args.threads, "n_ctx": args.ctx,
        "load_s": round(statistics.median(loads), 3) if loads else None,
        "rss_peak_mb": _mb(rss), "private_peak_mb": _mb(priv), "mapped_mb": _mb(mapped),
        "output_sample": next((c for c in contents.values() if c), None),
        "extra": {"source": "llama-completion", "engine": "llama-completion",
                  "load_variant": args.load_variant, "runs": runs,
                  "argv_example": [str(a) for a in bench.completion_argv(
                      binary, model_path, "<temp prompt file>", n_predict=args.n,
                      threads=args.threads, n_ctx=args.ctx, flags=base_flags)]},
    }
    if n_ok and eval_ms > 0:
        f["tg"] = {"tok_s": n_eval / (eval_ms / 1000.0),
                   "sd": statistics.stdev(tg_rates) if len(tg_rates) >= 2 else None,
                   "n": len(tg_rates)}
    if n_ok and prompt_ms > 0:
        f["pp"] = {"tok_s": n_prompt / (prompt_ms / 1000.0),
                   "sd": statistics.stdev(pp_rates) if len(pp_rates) >= 2 else None,
                   "n": len(pp_rates)}
    if n_ok < len(items):
        f["notes"] = [f"failed: {len(items) - n_ok} of {len(items)} llama-completion runs failed"
                      if n_ok == 0 else
                      f"{len(items) - n_ok} of {len(items)} llama-completion runs failed"]
    f["_errors"] = errs
    f["_per_item"] = per
    return f, contents


def cmd_qa(args: argparse.Namespace) -> int:
    variants: list[str] = []
    for v in str(args.server_variant).split(","):
        v = v.strip()
        if not v:
            continue
        if v not in bench.SERVER_VARIANT_SPEC:
            raise ValueError(f"unknown server variant {v!r} (known: {', '.join(bench.SERVER_VARIANTS)})")
        if v not in variants:
            variants.append(v)
    if not variants:
        raise ValueError("no --server-variant given")
    if "plain" in variants:
        variants.remove("plain")
        variants.insert(0, "plain")
    if args.no_server:
        if variants != ["plain"]:
            raise ValueError("--no-server runs plain greedy llama-completion only "
                             "(speculative variants need llama-server)")
    if args.n is None or args.n <= 0:
        raise ValueError("-n must be > 0")
    bench.split_variant(args.load_variant)
    load_suffix = "" if args.load_variant == "default" else f"+{args.load_variant}"
    base_flags = bench.variant_flags(args.load_variant, "run") + _extra_flags(args)
    experiment = args.experiment or default_experiment()
    qa_path = Path(args.qa_set).expanduser()
    items = load_qa_set(qa_path, ids=[x for x in (args.ids or "").split(",") if x.strip()] or None,
                        max_items=args.max_items)
    qa_info = _qa_set_info(qa_path, items)
    model_path, mid = _resolve_model(args.model, require=not args.dry_run)
    model = bench.model_block(model_path, mid, with_sha=not args.no_sha and model_path.is_file())
    model_id = _model_id(model, model_path)
    run_variants = ["completion"] if args.no_server else variants
    outs = {v: _qa_out_path(args.out, model_id, v + load_suffix, len(run_variants) > 1)
            for v in run_variants}
    tool = "llama-completion" if args.no_server else "llama-server"
    cmd = _tool_cmd(tool, args.bin)
    if args.dry_run:
        for v in run_variants:
            if args.no_server:
                print(_cmd_str(bench.completion_argv(cmd, model_path, "<temp prompt file>",
                                                     n_predict=args.n, threads=args.threads,
                                                     n_ctx=args.ctx, flags=base_flags)))
                print(f"  x {len(items)} items (one load each) -> {paths.repo_relative(outs[v])}")
                continue
            dtype, st = bench.SERVER_VARIANT_SPEC[v]
            dp = None
            if dtype is not None:
                dp = Path(args.draft) if args.draft else bench.resolve_draft(mid, model_path, dtype)[0]
            srv = bench.LlamaServer(model_path, threads=args.threads, n_ctx=args.ctx,
                                    port=args.port, draft=dp, spec_type=st,
                                    draft_n_max=args.draft_n_max if (dp is not None or st) else None,
                                    flags=base_flags, binary=cmd)
            print(_cmd_str(srv.argv))
            print(f"  then: POST /completion x {len(items)} (n_predict {args.n}, temperature 0) "
                  f"-> {paths.repo_relative(outs[v])}")
        return paths.EXIT_OK
    _check_cmd(cmd)
    build = bench.binary_version(cmd)
    build_t = (build.get("build_number"), build.get("commit"))
    plain: dict[str, str] | None = None
    plain_source: str | None = None
    if args.plain_ref:
        pr = Path(args.plain_ref).expanduser()
        if not pr.is_file():
            raise bench.MissingInputError(f"plain reference not found: {pr}")
        plain, plain_source = _plain_from_file(pr), paths.repo_relative(pr)
    rc_all = paths.EXIT_OK
    with _lock(args, f"pigemma.quality qa {model_id} {','.join(run_variants)}"):
        for v in run_variants:
            out_path = outs[v]
            is_ref = v == "plain"
            if plain is None and not is_ref:
                cand = out_path.parent / f"qa_{model_id}_plain{load_suffix}.json"
                if cand.is_file():
                    try:
                        plain, plain_source = _plain_from_file(cand), paths.repo_relative(cand)
                    except (QualityError, bench.BenchError, OSError, ValueError) as e:
                        log.warning("cannot use %s as the plain reference: %s", cand, e)
            if args.no_server:
                f, contents = _qa_completion_run(
                    args, items=items, model_path=model_path, model=model, model_id=model_id,
                    experiment=experiment, base_flags=base_flags, load_suffix=load_suffix,
                    out_path=out_path, build=build_t)
                is_ref = False
            else:
                f, contents = _qa_server_variant(
                    args, v=v, items=items, model_path=model_path, mid=mid, model=model,
                    model_id=model_id, experiment=experiment, base_flags=base_flags,
                    load_suffix=load_suffix, out_path=out_path, build=build_t, plain=plain,
                    plain_source=plain_source, qa_info=qa_info)
            errs = f.pop("_errors", {})
            per_item = f.pop("_per_item", {})
            report = grade_outputs(items, contents, errs, per_item)
            f = _qa_finish(f, report, contents=contents, plain=plain, plain_source=plain_source,
                           is_reference=is_ref, qa_info=qa_info, thinking=args.thinking,
                           n_predict=args.n)
            for n in _common_notes(build_t):
                have = f.get("notes") or []
                # bench.server_fields words its build / emulated notes slightly differently
                if not any(h == n or h.split(" (", 1)[0] == n.split(" (", 1)[0] for h in have):
                    f.setdefault("notes", []).append(n)
            _write_envelope(out_path, f)
            per = report["per_language"]
            langs = ", ".join(f"{k} {b['score']}/{b['total']}" for k, b in per.items())
            em = f["extra"].get("exact_match_vs_plain")
            print(f"pigemma.quality: qa {model_id} {f['variant']}: {report['qa_score']}/"
                  f"{report['qa_total']} ({langs}); exact_match_vs_plain="
                  f"{'n/a' if em is None else f'{em:.3f}'}; coherent={report['coherent']}",
                  file=sys.stderr)
            if v == "plain" and any(c is not None for c in contents.values()):
                plain = {k: visible_text(c) for k, c in contents.items() if c is not None}
                plain_source = "this invocation"
            failed = any(str(n).startswith("failed") for n in f.get("notes") or [])
            if failed or errs:
                rc_all = paths.EXIT_ERROR
    return rc_all


# --------------------------------------------------------------------------------------------
# Argument parser / main
# --------------------------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pigemma.quality",
        description="Quality metrics: KL divergence / perplexity through llama-perplexity, the "
                    "fixed Q&A set with Hebrew-aware grading, thinking stripping, coherence and "
                    "the speculative-decoding losslessness check (results envelopes, E2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exit codes: 0 ok, 1 run failed / size check failed / error, 2 usage, 3 missing "
               "input, 4 refused (bench lock busy, low RAM, low disk, port in use).")
    p.add_argument("--log-level", default="INFO", help="logging level (default INFO)")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<cmd>")

    run = argparse.ArgumentParser(add_help=False)
    run.add_argument("--model", "-m", required=True, help="registry/catalog id or GGUF path")
    run.add_argument("--threads", "-t", type=int, default=DEFAULT_THREADS,
                     help=f"threads (default {DEFAULT_THREADS})")
    run.add_argument("--experiment", help="E2|E8|adhoc (default E8 on aarch64, else E2)")
    run.add_argument("--flags", action="append", default=[],
                     help="extra llama.cpp flags (one string, shell-split; repeatable)")
    run.add_argument("--bin", help="binary override (path, or a shell-split command string)")
    run.add_argument("--min-free-gb", type=float, default=None,
                     help="bench-lock free-RAM gate in GB (default 20; env PIGEMMA_MIN_FREE_GB wins)")
    run.add_argument("--timeout", type=float, default=None, help="seconds before a child is stopped")
    run.add_argument("--nice", type=int, default=None, help="POSIX niceness for the child")
    run.add_argument("--no-sha", action="store_true", help="skip the 16 MiB sha256 prefix of the model")
    run.add_argument("--dry-run", action="store_true", help="print the command lines; run nothing")

    ppl_common = argparse.ArgumentParser(add_help=False)
    ppl_common.add_argument("--text", default=None,
                            help="evaluation text (default data/calibration/all.txt, built on demand)")
    ppl_common.add_argument("--chunks", type=int, default=DEFAULT_CHUNKS,
                            help=f"chunks (default {DEFAULT_CHUNKS})")

    q = sub.add_parser("qa", parents=[run], help="fixed Q&A set via llama-server (or llama-completion)")
    q.add_argument("--server-variant", default="plain",
                   help=f"comma list of {', '.join(bench.SERVER_VARIANTS)} (plain runs first)")
    q.add_argument("--load-variant", default="default", help="load variant flags (e.g. norepack)")
    q.add_argument("-n", "--n-predict", dest="n", type=int, default=DEFAULT_QA_N_PREDICT,
                   help=f"tokens per answer (default {DEFAULT_QA_N_PREDICT})")
    q.add_argument("-c", "--ctx", type=int, default=DEFAULT_QA_N_CTX, help="context (default 4096)")
    q.add_argument("--out", help="QA result file, or a directory (default results/quality/"
                                 "qa_<model>_<variant>.json)")
    q.add_argument("--qa-set", default=str(QA_SET_PATH), help="Q&A set JSON")
    q.add_argument("--ids", help="only these item ids (comma list)")
    q.add_argument("--max-items", type=int, default=None, help="only the first N items")
    q.add_argument("--no-server", action="store_true",
                   help="llama-completion per item instead of one llama-server")
    q.add_argument("--thinking", action="store_true", help="enable the thinking channel")
    q.add_argument("--plain-ref", help="plain QA result file for exact_match_vs_plain")
    q.add_argument("--port", type=int, default=bench.DEFAULT_PORT)
    q.add_argument("--draft", help="draft GGUF override (default: catalog draft)")
    q.add_argument("--draft-n-max", type=int, default=None, help="--spec-draft-n-max N")
    q.add_argument("--warmup-tokens", type=int, default=bench.WARMUP_TOKENS)
    q.add_argument("--health-timeout", type=float, default=bench.HEALTH_TIMEOUT_S)
    q.add_argument("--request-timeout", type=float, default=None)
    q.set_defaults(func=cmd_qa)

    kb = sub.add_parser("kld-base", parents=[run, ppl_common],
                        help="reference logits (--save-all-logits) + PPL of the base model")
    kb.add_argument("-c", "--ctx", type=int, default=DEFAULT_CTX, help=f"context (default {DEFAULT_CTX})")
    kb.add_argument("--out", help="logits file (default results/quality/ref_<model>.logits) or a dir")
    kb.add_argument("--variant", default="default", help="load variant of the base run (default 'default')")
    kb.add_argument("--results-dir", help="envelope directory (default results/<date>/)")
    kb.add_argument("--no-envelope", action="store_true", help="write no results envelope")
    kb.add_argument("--n-vocab", type=int, default=GEMMA4_N_VOCAB,
                    help=f"vocabulary size for the disk estimate (default {GEMMA4_N_VOCAB})")
    kb.add_argument("--force", action="store_true", help="recompute an existing logits file")
    kb.set_defaults(func=cmd_kld_base)

    k = sub.add_parser("kld", parents=[run, ppl_common],
                       help="KL divergence of a variant against reference logits")
    k.add_argument("--base", required=True, help="reference logits from kld-base")
    k.add_argument("--variant", required=True,
                   help="load variant (norepack, hybrid_negctrl, ...) or a model id label")
    k.add_argument("--allow-label", action="store_true",
                   help="accept a variant name that is neither a load variant nor a model id")
    k.add_argument("-c", "--ctx", type=int, default=None, help="ignored: taken from the base file")
    k.add_argument("--out", help="envelope file or directory (default results/<date>/)")
    k.set_defaults(func=cmd_kld)

    pp = sub.add_parser("ppl", parents=[run, ppl_common], help="perplexity of a model on the corpus")
    pp.add_argument("-c", "--ctx", type=int, default=DEFAULT_CTX, help=f"context (default {DEFAULT_CTX})")
    pp.add_argument("--variant", default="default", help="load variant (default 'default')")
    pp.add_argument("--out", help="envelope file or directory (default results/<date>/)")
    pp.set_defaults(func=cmd_ppl)

    c = sub.add_parser("corpus", help="build / check data/calibration/all.txt")
    c.add_argument("--out", help="output path (default data/calibration/all.txt)")
    c.add_argument("--src-dir", help="calibration directory (default data/calibration)")
    c.add_argument("--mode", choices=CORPUS_MODES, default="interleave")
    c.add_argument("--slice-chars", type=int, default=CORPUS_SLICE_CHARS)
    c.add_argument("--check", action="store_true", help="exit 1 if missing or out of date; write nothing")
    c.set_defaults(func=cmd_corpus)

    pk = sub.add_parser("parse-kld", help="parse a llama-perplexity --kl-divergence log -> JSON")
    pk.add_argument("input", help="log file or -")
    pk.set_defaults(func=cmd_parse_kld)
    pq = sub.add_parser("parse-ppl", help="parse a llama-perplexity log (Final estimate) -> JSON")
    pq.add_argument("input", help="log file or -")
    pq.set_defaults(func=cmd_parse_ppl)
    cl = sub.add_parser("check-logits", help="header + exact size check of a logits file")
    cl.add_argument("file")
    cl.set_defaults(func=cmd_check_logits)
    ct = sub.add_parser("check-text", help="strip thinking + coherence of a model output -> JSON")
    ct.add_argument("input", help="text file or -")
    ct.set_defaults(func=cmd_check_text)
    return p


def _utf8_stdio() -> None:
    for s in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    _utf8_stdio()
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else paths.EXIT_USAGE
    args.extra = extra
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO),
                        format="%(name)s: %(levelname)s: %(message)s", stream=sys.stderr)
    try:
        return int(args.func(args))
    except (paths.LockBusyError, paths.InsufficientMemoryError, bench.RefusedError,
            DiskGateError) as e:
        print(f"pigemma.quality: REFUSED: {e}", file=sys.stderr)
        return paths.EXIT_REFUSED
    except (bench.MissingInputError, FileNotFoundError) as e:
        print(f"pigemma.quality: MISSING: {e}", file=sys.stderr)
        return paths.EXIT_MISSING
    except results.ResultSchemaError as e:
        print(f"pigemma.quality: result schema error: {e}", file=sys.stderr)
        return paths.EXIT_ERROR
    except ValueError as e:
        print(f"pigemma.quality: usage: {e}", file=sys.stderr)
        return paths.EXIT_USAGE
    except (QualityError, bench.BenchError) as e:
        print(f"pigemma.quality: ERROR: {e}", file=sys.stderr)
        return paths.EXIT_ERROR
    except KeyboardInterrupt:
        print("pigemma.quality: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
