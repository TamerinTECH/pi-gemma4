"""Unit tests for the fixed evaluation prompts in data/prompts/.

Files under test (manifest entry "data/prompts/qa_set.json (+ data/prompts/gen_prompts.json)"):

* ``data/prompts/qa_set.json`` -- a JSON array of 16 scored Q&A items
  ``{id, lang: 'en'|'he'|'num', prompt, kind: 'contains'|'regex'|'number', expect: [...],
  tolerance (number kind only)}``: 7 English factual/one-word, 6 Hebrew (answers expected in
  Hebrew), 3 arithmetic/list.
* ``data/prompts/gen_prompts.json`` -- a JSON array of 8 generation prompts
  ``{id, lang, prompt, max_tokens: 256}``: 3 'en' (explain/story), 2 'he', 1 'code',
  2 'mixed' (English question that asks for a Hebrew answer).

Prompts are plain user text; ``pigemma.quality.chat_wrap`` applies the Gemma chat template.

Besides the shape checks required by the manifest (valid UTF-8 JSON, 16 items with >= 1
expectation each, both languages present, Hebrew items contain RTL characters, unique ids), the
test runs every Q&A item through a small *reference grader* that mirrors the grading rules of the
quality.py spec (casefold, NFKC, Hebrew final forms -> regular forms, niqqud U+0591-U+05C7
stripped, punctuation stripped, whitespace collapsed; regex = re.search; number = first number in
the text within tolerance) against plausible good and bad model answers. Where the spec leaves a
detail open (punctuation deleted vs replaced by a space; regex applied to the raw or the normalised
text) the reference grader checks BOTH readings, so the data stays valid whichever one quality.py
picks. When ``pigemma.quality.grade`` exists it is cross-checked on the same samples.

Stdlib + pytest only; runs on Windows (Python 3.14) and the Pi (Python 3.13) without models.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = REPO_ROOT / "data" / "prompts"
QA_PATH = PROMPTS_DIR / "qa_set.json"
GEN_PATH = PROMPTS_DIR / "gen_prompts.json"

QA_KEYS_REQUIRED = {"id", "lang", "prompt", "kind", "expect"}
QA_KEYS_ALLOWED = QA_KEYS_REQUIRED | {"tolerance"}
QA_LANG_COUNTS = {"en": 7, "he": 6, "num": 3}
QA_KINDS = {"contains", "regex", "number"}

GEN_KEYS = {"id", "lang", "prompt", "max_tokens"}
GEN_LANG_COUNTS = {"en": 3, "he": 2, "code": 1, "mixed": 2}

ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
RTL_RE = re.compile(r"[֐-׿יִ-ﭏ]")  # Hebrew block + presentation forms
TEMPLATE_MARKERS = ("<start_of_turn>", "<end_of_turn>", "<|channel>", "<channel|>", "<bos>", "<eos>")

# ---------------------------------------------------------------------------------------------
# Reference grader (mirrors the quality.py spec; kept deliberately small and independent of it)
# ---------------------------------------------------------------------------------------------
_FINALS = str.maketrans({"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"})
_NIQQUD_RE = re.compile(r"[֑-ׇ]")
_WS_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def normalise(text: str, punct_to_space: bool) -> str:
    """Casefold + NFKC + Hebrew finals + strip niqqud/punctuation + collapse whitespace."""
    s = unicodedata.normalize("NFKC", text).casefold()
    s = _NIQQUD_RE.sub("", s)
    s = s.translate(_FINALS)
    repl = " " if punct_to_space else ""
    s = "".join(repl if unicodedata.category(ch).startswith("P") else ch for ch in s)
    return _WS_RE.sub(" ", s).strip()


def first_number(text: str) -> float | None:
    """First decimal number in the (raw) text, or None."""
    m = _NUMBER_RE.search(text)
    return float(m.group(0)) if m else None


def reference_grade(item: dict[str, Any], visible: str) -> bool:
    """Return True only if the answer passes under EVERY reading of the spec checked here."""
    kind = item["kind"]
    if kind == "contains":
        verdicts = []
        for p2s in (False, True):
            text_n = normalise(visible, p2s)
            verdicts.append(any(normalise(e, p2s) in text_n for e in item["expect"]))
        return all(verdicts)
    if kind == "regex":
        verdicts = []
        for candidate in (visible, normalise(visible, False), normalise(visible, True)):
            verdicts.append(any(re.search(p, candidate) for p in item["expect"]))
        return all(verdicts)
    if kind == "number":
        value = first_number(visible)
        if value is None:
            return False
        tol = float(item.get("tolerance", 0) or 0)
        return any(abs(value - float(e)) <= tol + 1e-12 for e in item["expect"])
    raise AssertionError(f"unknown kind {kind!r}")


def reference_fails(item: dict[str, Any], visible: str) -> bool:
    """Return True only if the answer FAILS under every reading of the spec checked here."""
    kind = item["kind"]
    if kind == "contains":
        return not any(
            any(normalise(e, p2s) in normalise(visible, p2s) for e in item["expect"])
            for p2s in (False, True)
        )
    if kind == "regex":
        return not any(
            any(re.search(p, candidate) for p in item["expect"])
            for candidate in (visible, normalise(visible, False), normalise(visible, True))
        )
    return not reference_grade(item, visible)


# Plausible greedy answers per item (incl. markdown bold, trailing punctuation, niqqud, maqaf,
# final-form variants) and clearly wrong ones. Every qa_set id MUST have an entry here.
SAMPLES: dict[str, tuple[list[str], list[str]]] = {
    "en-01": (["Paris", "The capital of France is **Paris**."], ["Lyon", "Marseille."]),
    "en-02": (["Jupiter.", "**Jupiter**"], ["Saturn", "Earth"]),
    "en-03": (["Au", "The chemical symbol for gold is **Au**."], ["Ag", "Because gold is a metal."]),
    "en-04": (["Shakespeare", "William Shakespeare."], ["Marlowe", "Dickens"]),
    "en-05": (["Eight.", "8", "A spider has **eight** legs."], ["Six", "18"]),
    "en-06": (["Thursday", "**Thursday**."], ["Wednesday", "Friday"]),
    "en-07": (["Carbon dioxide", "CO₂", "CO2."], ["Oxygen", "Nitrogen."]),
    "he-01": (["פריז", "פָּרִיז.", "**פאריס**"], ["לונדון", "ליון"]),
    "he-02": (["חם", "חַם.", "חמה"], ["קר", "פושר"]),
    "he-03": (["כדור הארץ", "כדור־הארץ.", "**כדור הארץ**"], ["מאדים", "נוגה"]),
    "he-04": (["ירוק", "ירוק."], ["סגול", "כתום"]),
    "he-05": (["אפריקה", "ביבשת אפריקה."], ["אסיה", "אירופה"]),
    "he-06": (["ירושלים", "בירושלים.", "ירושלם"], ["תל אביב", "חיפה"]),
    "num-01": (["391", "**391**", "391."], ["381", "401"]),
    "num-02": (["16.8", "16.80", "16.80 shekels", "₪16.80"], ["16.4", "14"]),
    "num-03": (["2, 3, 5, 7, 11", "**2, 3, 5, 7, 11**", "2,3,5,7,11"],
               ["1, 2, 3, 5, 7", "2, 3, 5, 7, 9"]),
}


# ---------------------------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------------------------
def _load(path: Path) -> Any:
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), f"{path.name} must not start with a UTF-8 BOM"
    text = raw.decode("utf-8")  # strict: raises on invalid UTF-8
    return json.loads(text)


@pytest.fixture(scope="module")
def qa() -> list[dict[str, Any]]:
    return _load(QA_PATH)


@pytest.fixture(scope="module")
def gen() -> list[dict[str, Any]]:
    return _load(GEN_PATH)


def _qa_items() -> list[dict[str, Any]]:
    return _load(QA_PATH)


# ---------------------------------------------------------------------------------------------
# Shape tests (manifest test field)
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("path", [QA_PATH, GEN_PATH], ids=lambda p: p.name)
def test_files_are_valid_utf8_json_arrays(path: Path) -> None:
    data = _load(path)
    assert isinstance(data, list) and data, f"{path.name} must be a non-empty JSON array"
    assert all(isinstance(it, dict) for it in data)


def test_qa_has_16_items_with_expectations(qa: list[dict[str, Any]]) -> None:
    assert len(qa) == 16
    for it in qa:
        keys = set(it)
        assert QA_KEYS_REQUIRED <= keys, f"{it.get('id')}: missing {QA_KEYS_REQUIRED - keys}"
        assert keys <= QA_KEYS_ALLOWED, f"{it['id']}: unexpected keys {keys - QA_KEYS_ALLOWED}"
        assert it["kind"] in QA_KINDS, it["id"]
        assert isinstance(it["prompt"], str) and it["prompt"].strip(), it["id"]
        assert isinstance(it["expect"], list) and len(it["expect"]) >= 1, it["id"]


def test_qa_ids_unique_and_safe(qa: list[dict[str, Any]]) -> None:
    ids = [it["id"] for it in qa]
    assert len(ids) == len(set(ids)), [i for i, n in Counter(ids).items() if n > 1]
    for i in ids:
        assert ID_RE.match(i), f"id {i!r} must be lowercase ASCII words joined by '-'"


def test_qa_language_mix(qa: list[dict[str, Any]]) -> None:
    counts = Counter(it["lang"] for it in qa)
    assert dict(counts) == QA_LANG_COUNTS
    assert counts["en"] > 0 and counts["he"] > 0  # both languages present


def test_hebrew_items_contain_rtl(qa: list[dict[str, Any]]) -> None:
    for it in qa:
        if it["lang"] != "he":
            continue
        assert RTL_RE.search(it["prompt"]), f"{it['id']}: prompt has no Hebrew characters"
        for e in it["expect"]:
            assert isinstance(e, str) and RTL_RE.search(e), f"{it['id']}: expectation {e!r} not Hebrew"


def test_english_and_numeric_prompts_have_no_rtl(qa: list[dict[str, Any]]) -> None:
    for it in qa:
        if it["lang"] in ("en", "num"):
            assert not RTL_RE.search(it["prompt"]), it["id"]


def test_number_items_and_tolerance(qa: list[dict[str, Any]]) -> None:
    for it in qa:
        if it["kind"] == "number":
            assert "tolerance" in it, f"{it['id']}: number items need a tolerance"
            tol = it["tolerance"]
            assert isinstance(tol, (int, float)) and not isinstance(tol, bool) and tol >= 0, it["id"]
            for e in it["expect"]:
                assert isinstance(e, (int, float)) and not isinstance(e, bool), it["id"]
        else:
            assert "tolerance" not in it, f"{it['id']}: tolerance is for number items only"
            for e in it["expect"]:
                assert isinstance(e, str) and e.strip(), it["id"]


def test_regex_items_compile_and_have_no_control_chars(qa: list[dict[str, Any]]) -> None:
    # A JSON "\b" (backspace) instead of "\\b" would silently break word boundaries.
    for it in qa:
        if it["kind"] != "regex":
            continue
        for pattern in it["expect"]:
            assert not any(unicodedata.category(ch) == "Cc" for ch in pattern), it["id"]
            re.compile(pattern)


@pytest.mark.parametrize("path", [QA_PATH, GEN_PATH], ids=lambda p: p.name)
def test_prompts_are_plain_user_text(path: Path) -> None:
    for it in _load(path):
        p = it["prompt"]
        assert p == p.strip(), f"{it['id']}: leading/trailing whitespace"
        for marker in TEMPLATE_MARKERS:
            assert marker not in p, f"{it['id']}: prompt must not contain chat template token {marker}"


# ---------------------------------------------------------------------------------------------
# gen_prompts.json
# ---------------------------------------------------------------------------------------------
def test_gen_prompts_shape(gen: list[dict[str, Any]]) -> None:
    assert len(gen) == 8
    for it in gen:
        assert set(it) == GEN_KEYS, f"{it.get('id')}: keys must be exactly {sorted(GEN_KEYS)}"
        assert it["max_tokens"] == 256 and isinstance(it["max_tokens"], int), it["id"]
        assert isinstance(it["prompt"], str) and len(it["prompt"]) >= 40, it["id"]
    assert dict(Counter(it["lang"] for it in gen)) == GEN_LANG_COUNTS


def test_gen_prompt_languages(gen: list[dict[str, Any]]) -> None:
    for it in gen:
        has_rtl = bool(RTL_RE.search(it["prompt"]))
        if it["lang"] == "he":
            assert has_rtl, f"{it['id']}: Hebrew prompt without Hebrew characters"
        else:
            # en / code / mixed prompts are written in English
            assert not has_rtl, f"{it['id']}: unexpected Hebrew characters"
        if it["lang"] == "mixed":
            assert "hebrew" in it["prompt"].lower(), f"{it['id']}: must ask for a Hebrew answer"
        if it["lang"] == "code":
            assert "python" in it["prompt"].lower(), it["id"]


def test_ids_unique_across_files(qa: list[dict[str, Any]], gen: list[dict[str, Any]]) -> None:
    ids = [it["id"] for it in qa] + [it["id"] for it in gen]
    assert len(ids) == len(set(ids))
    for i in ids:
        assert ID_RE.match(i), i


# ---------------------------------------------------------------------------------------------
# Grading self-consistency
# ---------------------------------------------------------------------------------------------
def test_every_item_has_grading_samples(qa: list[dict[str, Any]]) -> None:
    assert set(SAMPLES) == {it["id"] for it in qa}


def test_reference_normalisation_hebrew() -> None:
    # final forms, niqqud, maqaf and punctuation are neutralised
    assert normalise("חַם!", False) == normalise("חמ", False)
    assert normalise("כדור־הארץ.", False) == "כדורהארצ"
    assert normalise("**Paris**.", True) == "paris"
    assert normalise("CO₂", False) == "co2"


@pytest.mark.parametrize("item", _qa_items(), ids=lambda it: it["id"])
def test_reference_grader_accepts_good_and_rejects_bad(item: dict[str, Any]) -> None:
    good, bad = SAMPLES[item["id"]]
    for answer in good:
        assert reference_grade(item, answer), f"{item['id']}: good answer {answer!r} rejected"
    for answer in bad:
        assert reference_fails(item, answer), f"{item['id']}: wrong answer {answer!r} accepted"


def test_real_grader_agrees_when_available(qa: list[dict[str, Any]]) -> None:
    """Cross-check against pigemma.quality.grade once quality.py exists (skipped before)."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        quality = importlib.import_module("pigemma.quality")
    except Exception as exc:  # module not written yet, or its optional deps unavailable
        pytest.skip(f"pigemma.quality not importable: {exc!r}")
    grade = getattr(quality, "grade", None)
    if not callable(grade):
        pytest.skip("pigemma.quality.grade not defined")
    for item in qa:
        good, bad = SAMPLES[item["id"]]
        for answer in good:
            assert grade(item, answer), f"quality.grade rejected good answer {answer!r} for {item['id']}"
        for answer in bad:
            assert not grade(item, answer), f"quality.grade accepted wrong answer {answer!r} for {item['id']}"
