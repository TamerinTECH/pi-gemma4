"""Unit tests for pigemma.quality (stdlib + pytest; no model and no llama.cpp binary needed).

Manifest test field: strip_thinking on the checked-in fixtures (incl. an unterminated block and a
leading ``<channel|>``); the grader on Hebrew (final forms, niqqud, maqaf) and on numbers with a
tolerance; is_coherent flags repetition loops; the KLD / PPL line parsers on the captured
llama-perplexity output (tests/fixtures/perplexity_*_sample.txt); the disk-gate arithmetic;
exact_match_vs_plain on the captured llama-server outputs (tests/fixtures/bench/server_moe_*.json).

Also covered: the exact ``--save-all-logits`` size check, the corpus builder, grade_outputs and
the pre-registered E2 thresholds, and the CLI end to end (kld-base -> kld, the disk-gate refusal,
a truncated logits file, qa through llama-server and through llama-completion, the bench lock)
with stdlib test doubles that are written to tmp_path at test time.
"""

from __future__ import annotations

import json
import socket
import struct
import sys
from pathlib import Path
from typing import Any

import pytest

from pigemma import bench, paths, quality, results

FIX = Path(__file__).resolve().parent / "fixtures"
BENCH_FIX = FIX / "bench"
QA_SET = paths.PROMPTS_DIR / "qa_set.json"
QA_HARD = paths.PROMPTS_DIR / "qa_hard.json"


def fx(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


def server_content(name: str) -> str:
    return json.loads((BENCH_FIX / name).read_text(encoding="utf-8"))["content"]


def qa_item(item_id: str, path: Path = QA_SET) -> dict[str, Any]:
    return next(it for it in quality.load_qa_set(path) if it["id"] == item_id)


# =============================================================================================
# strip_thinking / visible_text
# =============================================================================================
def test_strip_thinking_completion_stdout_fixture() -> None:
    """llama-completion stdout with the echoed <start_of_turn> prompt and ' [end of text]'."""
    vis, th = quality.strip_thinking(fx("thinking_sample.txt"))
    assert vis.startswith("A 26B parameter model requires massive amounts of VRAM")
    assert vis.endswith("שלום")
    assert th == 0  # the captured thinking block is empty
    for marker in ("<start_of_turn>", "<end_of_turn>", "<|channel>", "<channel|>", "[end of text]",
                   "Explain in 3 short sentences"):
        assert marker not in vis


def test_strip_thinking_pi_hebrew_fixture_cut_at_token_cap() -> None:
    vis, th = quality.strip_thinking(fx("thinking_pi31b_he_sample.txt"))
    assert vis.startswith("הוא קטן וצורך מעט מאוד חשמל.")
    assert vis.endswith("הוא מאפשר לבנות פרויקטים")
    assert th == 0 and "ענה בעברית" not in vis


@pytest.mark.parametrize("name", ["server_moe_plain.json", "server_moe_mtp.json",
                                  "server_moe_dflash.json"])
def test_strip_thinking_server_fixtures(name: str) -> None:
    raw = server_content(name)
    assert raw.startswith("<|channel>thought\n<channel|>")
    vis, th = quality.strip_thinking(raw)
    assert vis.startswith("In a standard Transformer") and th == 0
    assert "<|channel>" not in vis and "<channel|>" not in vis


def test_strip_thinking_nonempty_and_unterminated_blocks() -> None:
    # no captured sample has a non-empty block (tests/fixtures/README.md): build them inline
    assert quality.strip_thinking("<|channel>thought\nlet me think about it<channel|>Paris") == (
        "Paris", len("let me think about it"))
    assert quality.strip_thinking("<|channel>thought\nI am still thinking about") == (
        "", len("I am still thinking about"))  # unterminated: runs to the end
    vis, th = quality.strip_thinking("Answer first.\n<|channel>thought\nnever closed")
    assert (vis, th) == ("Answer first.", len("never closed"))
    vis, th = quality.strip_thinking("<|channel>thought\na<channel|>B<|channel>thought\ncc<channel|>D")
    assert (vis, th) == ("BD", 3)


def test_strip_thinking_leading_channel_tokens_and_end_markers() -> None:
    assert quality.strip_thinking("<channel|>Shalom <end_of_turn>") == ("Shalom", 0)
    assert quality.strip_thinking("  <|channel>Shalom")[0] == "Shalom"
    assert quality.strip_thinking("Paris<turn|>")[0] == "Paris"
    assert quality.strip_thinking("Paris<eos>")[0] == "Paris"
    assert quality.strip_thinking("Paris [end of text]\n\n")[0] == "Paris"
    assert quality.strip_thinking("Paris\r\nLyon\r\n")[0] == "Paris\nLyon"
    assert quality.strip_thinking("") == ("", 0)
    assert quality.strip_thinking(None) == ("", 0)  # type: ignore[arg-type]


def test_strip_thinking_orphan_closer_counts_as_thinking() -> None:
    # the thinking opener was part of the prompt (chat_wrap(thinking=True) variants)
    assert quality.strip_thinking("reasoning here<channel|>The answer") == (
        "The answer", len("reasoning here"))


def test_strip_thinking_echo_cut_needs_a_user_turn() -> None:
    echoed = "<|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>4"
    assert quality.strip_thinking(echoed) == ("4", 0)
    # a model opener inside an answer (no echoed user turn before it) is not a prompt echo
    vis = quality.visible_text("Gemma 4 replies after <|turn>model\nlike this.")
    assert vis.startswith("Gemma 4 replies after")


def test_visible_text_is_idempotent_and_bench_delegates() -> None:
    for raw in (fx("thinking_sample.txt"), fx("thinking_pi31b_he_sample.txt"),
                server_content("server_moe_dflash.json"), "<channel|>Shalom <end_of_turn>"):
        v = quality.visible_text(raw)
        assert quality.visible_text(v) == v
        assert bench.strip_thinking(raw) == quality.strip_thinking(raw)
        assert bench.visible_text(raw) == v
    ok, _reason, impl = bench.coherence(server_content("server_moe_plain.json"))
    assert ok is True and impl == "quality"


def test_chat_wrap_is_bench_gemma4_template() -> None:
    w = quality.chat_wrap("  Hi  ")
    assert w == bench.chat_wrap("Hi") == (
        "<|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
    assert quality.chat_wrap("Hi", thinking=True) == "<|turn>user\nHi<turn|>\n<|turn>model\n"
    assert "<start_of_turn>" not in w


# =============================================================================================
# is_coherent
# =============================================================================================
def _sep_words(gram: str, times: int) -> str:
    return " ".join(f"{gram} unique{i}" for i in range(times))


def test_is_coherent_repetition_rules_and_boundaries() -> None:
    # 4-gram repeated >= 6 times -> incoherent; 5 times is still fine
    assert quality.is_coherent(_sep_words("alpha beta gamma delta", 5))[0] is True
    ok, why = quality.is_coherent(_sep_words("alpha beta gamma delta", 6))
    assert ok is False and "4-gram" in why and "6 times" in why
    # 6-gram repeated > 3 times -> incoherent; exactly 3 times is fine
    assert quality.is_coherent(_sep_words("a1 b2 c3 d4 e5 f6", 3))[0] is True
    ok, why = quality.is_coherent(_sep_words("a1 b2 c3 d4 e5 f6", 4))
    assert ok is False and "6-gram" in why
    # a classic greedy loop
    ok, why = quality.is_coherent("The model is fast. " + "the cat sat on the mat " * 12)
    assert ok is False and "repeated" in why


def test_is_coherent_character_rules() -> None:
    assert quality.is_coherent("") == (False, "empty output")
    assert quality.is_coherent("   \n ")[0] is False
    ok, why = quality.is_coherent("\ufffd\ufffd\u05f5\u05f5}}}}")
    assert ok is False and "letters/digits/space/punctuation" in why
    assert quality.is_coherent("\u2588" * 30 + " hi")[0] is False
    # spaceless loops are caught by the degenerate-loop extension
    assert quality.is_coherent("Answer: " + "!" * 200)[0] is False
    assert quality.is_coherent("ok " + "ab" * 80)[0] is False


@pytest.mark.parametrize("text", [
    "Experts are routed per token. שלום לכולם.",
    "הוא קטן וצורך מעט מאוד חשמל. הוא זול ונגיש למטרות למידה וניסויים.",
    "def f(x: int) -> int:\n    return x * 2\n\nprint(f(3))  # 6\n" + "=" * 60,
    "| a | b | c |\n|---|---|---|\n| 1 | 2 | 3 |\n",
    "391",
    "2, 3, 5, 7, 11",
])
def test_is_coherent_accepts_normal_answers(text: str) -> None:
    assert quality.is_coherent(text) == (True, "ok")


def test_is_coherent_on_fixtures() -> None:
    for raw in (fx("thinking_sample.txt"), fx("thinking_pi31b_he_sample.txt"),
                server_content("server_moe_plain.json"), server_content("server_moe_dflash.json")):
        assert quality.is_coherent(quality.visible_text(raw)) == (True, "ok")


# =============================================================================================
# Grading
# =============================================================================================
def test_normalize_text_hebrew_and_unicode() -> None:
    n = quality.normalize_text
    assert n("כָּדוּר־הָאָרֶץ.", "space") == "כדור הארצ"      # niqqud gone, maqaf = punctuation
    assert n("כָּדוּר־הָאָרֶץ.", "delete") == "כדורהארצ"
    assert n("ירושלים") == "ירושלימ" and n("ךםןףץ") == "כמנפצ"   # final forms
    assert n("\u200fפריז\u200f!") == "פריז"                     # bidi marks dropped
    assert n("ＰＡＲＩＳ") == "paris" and n("**Paris**.") == "paris"  # NFKC + casefold
    assert n("CO₂") == "co2"
    assert n("  a \t\n b  ") == "a b"
    with pytest.raises(ValueError):
        n("x", "keep")


def test_grade_hebrew_contains_with_finals_niqqud_and_maqaf() -> None:
    he06 = qa_item("he-06")
    for good in ("ירושלים", "בירושלים.", "יְרוּשָׁלַיִם", "ירושלם", "**ירושלים**"):
        assert quality.grade(he06, good), good
    for bad in ("תל אביב", "חיפה", ""):
        assert not quality.grade(he06, bad), bad
    he03 = qa_item("he-03")
    for good in ("כדור הארץ", "כדור־הארץ.", "כַּדּוּר הָאָרֶץ", "על כדור הארץ"):
        assert quality.grade(he03, good), good
    assert not quality.grade(he03, "מאדים")
    he02 = qa_item("he-02")
    assert quality.grade(he02, "חַם.") and quality.grade(he02, "חמ")
    assert not quality.grade(he02, "קר")


def test_grade_number_with_tolerance() -> None:
    num02 = qa_item("num-02")  # 16.8, tolerance 0.001
    for good in ("16.8", "16.80", "16.80 shekels", "₪16.80", "**16.8**", "16,8", "16.801"):
        assert quality.grade(num02, good), good
    for bad in ("16.81", "16.4", "14", "about sixteen", ""):
        assert not quality.grade(num02, bad), bad
    num01 = qa_item("num-01")  # 391, tolerance 0
    assert quality.grade(num01, "391") and quality.grade(num01, "The answer is **391**.")
    assert not quality.grade(num01, "390")
    # the rule is the FIRST number of the visible text
    assert not quality.grade(num01, "17*23 = 391")


@pytest.mark.parametrize("text,value", [
    ("391", 391.0), ("16.80 shekels", 16.8), ("₪16.80", 16.8), ("16,8", 16.8),
    ("2, 3, 5", 2.0), ("1,234 items", 1234.0), ("-5 degrees", -5.0), ("Answer: \u22123", -3.0),
    ("ב-8 שקלים, נשארו 21", 8.0), ("ל-1990 שנה", 1990.0), ("num-01: 391", 391.0),
    ("COVID-19 is 5", 5.0), ("CO2 then 7", 7.0), ("１２", 12.0), ("no digits", None), ("", None),
])
def test_first_number_spellings(text: str, value: float | None) -> None:
    assert quality.first_number(text) == value


def test_grade_regex_uses_raw_and_normalised_text() -> None:
    au = qa_item("en-03")
    assert quality.grade(au, "**Au**") and quality.grade(au, "The symbol is Au.")
    assert not quality.grade(au, "Because gold is a metal.") and not quality.grade(au, "Ag")
    primes = qa_item("num-03")
    assert quality.grade(primes, "2, 3, 5, 7, 11") and quality.grade(primes, "2,3,5,7,11")
    assert not quality.grade(primes, "2, 3, 5, 7, 9")
    # qa_hard: an anchored multi-line regex only matches the raw text (newlines kept)
    lst = qa_item("ins-he-01", QA_HARD)
    assert quality.grade(lst, "1. תפוח\n2. בננה\n3. אגס")
    assert not quality.grade(lst, "1. תפוח 2. בננה 3. אגס")
    no = qa_item("ins-en-02", QA_HARD)
    assert quality.grade(no, "NO") and quality.grade(no, " NO.") and not quality.grade(no, "YES")
    hhmm = qa_item("ext-he-01", QA_HARD)
    assert quality.grade(hhmm, "07:30") and not quality.grade(hhmm, "08:30")


def test_grade_errors() -> None:
    with pytest.raises(ValueError):
        quality.grade({"id": "x", "kind": "fuzzy", "expect": ["a"]}, "a")
    with pytest.raises(ValueError):
        quality.grade({"id": "x", "kind": "contains", "expect": []}, "a")


def test_load_qa_sets() -> None:
    items = quality.load_qa_set()
    assert len(items) == 16 and {it["lang"] for it in items} == {"en", "he", "num"}
    hard = quality.load_qa_set(QA_HARD)
    assert len(hard) == 20 and {"de", "code"} <= {it["lang"] for it in hard}
    sel = quality.load_qa_set(QA_SET, ids=["he-01", "num-02"])
    assert [it["id"] for it in sel] == ["he-01", "num-02"]
    assert len(quality.load_qa_set(QA_SET, max_items=3)) == 3
    with pytest.raises(ValueError):
        quality.load_qa_set(QA_SET, ids=["nope-01"])
    with pytest.raises(bench.MissingInputError):
        quality.load_qa_set(QA_SET.with_name("missing.json"))


# =============================================================================================
# grade_outputs / exact_match_vs_plain / thresholds
# =============================================================================================
GOOD = {"en-01": "Paris", "en-02": "Jupiter", "en-03": "Au", "en-04": "Shakespeare",
        "en-05": "Eight.", "en-06": "Thursday", "en-07": "Carbon dioxide", "he-01": "פריז",
        "he-02": "חם", "he-03": "כדור הארץ", "he-04": "ירוק", "he-05": "אפריקה",
        "he-06": "ירושלים", "num-01": "391", "num-02": "16.80", "num-03": "2, 3, 5, 7, 11"}


def test_grade_outputs_scores_per_language_and_thinking() -> None:
    items = quality.load_qa_set()
    contents: dict[str, str | None] = {
        k: "<|channel>thought\n" + ("x" * 10) + "<channel|>" + v for k, v in GOOD.items()}
    contents["he-02"] = "<|channel>thought\n<channel|>קר"   # wrong
    contents["num-03"] = None                               # request failed
    rep = quality.grade_outputs(items, contents, errors={"num-03": "HTTP 500"})
    assert (rep["qa_score"], rep["qa_total"]) == (14, 16)
    assert rep["per_language"] == {"en": {"score": 7, "total": 7}, "he": {"score": 5, "total": 6},
                                   "num": {"score": 2, "total": 3}}
    by = {r["id"]: r for r in rep["items"]}
    assert by["en-01"]["visible"] == "Paris" and by["en-01"]["thinking_chars"] == 10
    assert by["en-01"]["pass"] is True and by["he-02"]["pass"] is False
    assert by["num-03"]["pass"] is False and by["num-03"]["error"] == "HTTP 500"
    assert by["num-03"]["visible"] is None
    for r in rep["items"]:
        assert {"id", "lang", "visible", "thinking_chars", "pass"} <= set(r)
    assert rep["thinking_chars_total"] == 140 and rep["coherent"] is True
    assert rep["thinking_share"] == pytest.approx(140 / (140 + rep["visible_chars_total"]))
    chk = quality.qa_threshold_check(rep)
    assert chk["qa_pass"] is True and chk["hebrew_pass"] is True and chk["applies"] is True


def test_qa_threshold_boundaries() -> None:
    def rep(score: int, he: int, total: int = 16) -> dict[str, Any]:
        return {"qa_score": score, "qa_total": total,
                "per_language": {"he": {"score": he, "total": 6}}}
    assert quality.qa_threshold_check(rep(13, 4))["qa_pass"] is True
    assert quality.qa_threshold_check(rep(12, 4))["qa_pass"] is False
    assert quality.qa_threshold_check(rep(16, 3))["hebrew_pass"] is False
    assert quality.qa_threshold_check(rep(18, 4, total=20))["qa_pass"] is None  # not the 16 set


def test_exact_match_vs_plain_on_server_fixtures() -> None:
    plain = server_content("server_moe_plain.json")
    mtp = server_content("server_moe_mtp.json")        # never drafted: identical to plain
    dflash = server_content("server_moe_dflash.json")  # drafted: diverges after ~10 tokens
    assert quality.exact_match_vs_plain({"p": mtp}, {"p": plain}) == (1.0, {"p": True})
    assert quality.exact_match_vs_plain({"p": dflash}, {"p": plain}) == (0.0, {"p": False})
    frac, by = quality.exact_match_vs_plain({"a": mtp, "b": dflash, "c": None, "d": mtp},
                                            {"a": plain, "b": plain, "c": plain})
    assert frac == 0.5 and by == {"a": True, "b": False}   # c failed, d has no reference
    # the plain side may already be visible text (as stored in a QA result file)
    assert quality.exact_match_vs_plain({"p": mtp}, {"p": quality.visible_text(plain)})[0] == 1.0
    assert quality.exact_match_vs_plain({"p": mtp}, None) == (None, None)
    assert quality.exact_match_vs_plain({"p": mtp}, {"q": plain}) == (None, None)


# =============================================================================================
# llama-perplexity parsers
# =============================================================================================
def test_parse_kld_fixture() -> None:
    d = quality.parse_kld_output(fx("perplexity_kld_sample.txt"))
    assert d["ok"] is True
    assert d["mean"] == pytest.approx(0.060170) and d["mean_sd"] == pytest.approx(0.001545)
    assert d["max"] == pytest.approx(0.116063) and d["p999"] == pytest.approx(0.115574)
    assert d["p99"] == pytest.approx(0.110928) and d["median"] == pytest.approx(0.057377)
    assert d["min"] == pytest.approx(0.035381)
    assert d["ppl_q"] == pytest.approx(1100.227588) and d["ppl_q_sd"] == pytest.approx(179.440353)
    assert d["ppl_base"] == pytest.approx(1165.064883)
    assert d["same_top_p_pct"] == pytest.approx(77.419)
    assert d["same_top_p_pct_sd"] == pytest.approx(3.770)
    assert d["same_top_p"] == pytest.approx(0.77419)          # a fraction for the envelope
    assert d["mean_dp_pct"] == pytest.approx(0.039) and d["mean_dp_pct_sd"] == pytest.approx(0.033)
    assert d["rms_dp_pct"] == pytest.approx(0.363) and d["max_dp_pct"] == pytest.approx(3.315)
    assert d["cor_ln_ppl_pct"] == pytest.approx(98.18)
    assert d["mean_ln_ppl_ratio"] == pytest.approx(-0.057260)
    assert d["n_chunks"] == 4 and len(d["chunks"]) == 4
    assert d["chunks"][3] == {
        "chunk": 4, "ppl": pytest.approx(1100.2276), "ppl_sd": pytest.approx(179.4404),
        "ln_ratio": pytest.approx(-0.05726), "kld": pytest.approx(0.06017),
        "kld_sd": pytest.approx(0.00154), "dp_rms_pct": pytest.approx(0.363),
        "same_top_p_pct": pytest.approx(77.419)}
    assert d["computing"] == {"n_chunk": 4, "n_ctx": 64, "n_batch": 2048, "n_seq": 32}
    blk = quality.kld_block(d)
    assert set(blk) == {"mean", "max", "p999", "same_top_p", "ppl_q", "ppl_base"}
    assert blk["same_top_p"] == pytest.approx(0.77419)


def test_parse_kld_is_tolerant_to_spacing_crlf_and_mangled_plusminus() -> None:
    text = ("Mean PPL(Q)   :  12.5 +/- 0.25\r\n"
            "Mean PPL(base):12.4 \u00c2\u00b1 0.2\r\n"
            "  Mean KLD:0.000123 ± 0.000010\r\n"
            "Maximum   KLD:   0.0042\r\n"
            "99.9 %  KLD : 0.0031\r\n"
            "Same top p:99.871 ± 0.050 %\r\n")
    d = quality.parse_kld_output(text)
    assert d["ppl_q"] == 12.5 and d["ppl_q_sd"] == 0.25 and d["ppl_base"] == 12.4
    assert d["mean"] == pytest.approx(0.000123) and d["mean_sd"] == pytest.approx(0.00001)
    assert d["max"] == pytest.approx(0.0042) and d["p999"] == pytest.approx(0.0031)
    assert d["same_top_p"] == pytest.approx(0.99871)
    assert d["median"] is None and d["chunks"] is None   # missing lines -> None
    empty = quality.parse_kld_output("")
    assert empty["ok"] is False and empty["mean"] is None and empty["same_top_p"] is None
    nan = quality.parse_kld_output("Mean    KLD:        nan ±        nan\n")
    assert nan["mean"] is None and nan["ok"] is False
    # a variant identical to the base prints '-0.000000' (measured with -nr on a tiny model)
    zero = quality.parse_kld_output("Mean    KLD:  -0.000000 +/-   0.000000\n")["mean"]
    assert zero == 0.0 and str(zero) == "0.0"


def test_parse_ppl_fixture_and_errors() -> None:
    d = quality.parse_ppl_output(fx("perplexity_base_sample.txt"))
    assert d["ok"] is True and d["ppl"] == pytest.approx(1165.0598)
    assert d["ppl_sd"] == pytest.approx(189.16265)
    assert d["chunk_ppl"] == pytest.approx([505.3665, 802.5845, 958.9822, 1165.0598])
    assert d["n_chunks"] == 4
    assert quality.parse_ppl_output("Final estimate:PPL=7.25 +/- 0.1")["ppl"] == 7.25
    no = quality.parse_ppl_output(fx("perplexity_kld_sample.txt"))
    assert no["ok"] is False and no["ppl"] is None
    errs = quality.perplexity_errors(
        "0.00.020.000 E perplexity: you need at least 1024 tokens to evaluate perplexity with a "
        "context of 512\nnormal line\n")
    assert len(errs) == 1 and errs[0].startswith("perplexity: you need at least 1024 tokens")
    assert quality.perplexity_errors(fx("perplexity_kld_sample.txt")) == []


def test_kld_tier_and_neutrality_thresholds() -> None:
    assert quality.kld_tier(None) is None
    assert quality.kld_tier(0.0499) == "acceptable"
    assert quality.kld_tier(0.05) == "degraded" and quality.kld_tier(0.1499) == "degraded"
    assert quality.kld_tier(0.15) == "not recommended"
    assert quality.kld_neutral({"mean": 0.0009, "same_top_p": 0.996}) is True
    assert quality.kld_neutral({"mean": 0.0009, "same_top_p": 0.995}) is False  # must be > 99.5 %
    assert quality.kld_neutral({"mean": 0.001, "same_top_p": 1.0}) is False     # must be < 0.001
    assert quality.kld_neutral({"mean": None, "same_top_p": 1.0}) is None


# =============================================================================================
# Disk gate and logits file
# =============================================================================================
def test_disk_gate_arithmetic() -> None:
    est = quality.size_estimate(6, 512)
    assert est == 6 * 512 * 262144 * 2 == 1_610_612_736          # "6 chunks = 1.61 GB"
    g = quality.disk_gate(2 * est, 6, 512)
    assert g["ok"] is True and g["required_bytes"] == 2 * est == 3_221_225_472
    assert g["estimate_bytes"] == est and g["free_bytes"] == 2 * est
    assert quality.disk_gate(2 * est - 1, 6, 512)["ok"] is False
    # the file llama-perplexity really writes is smaller than the conservative estimate
    assert g["exact_bytes"] == quality.expected_logits_bytes(512, 262144, 6) == 802_185_188
    assert g["exact_bytes"] < est
    g3 = quality.disk_gate(10**12, 1, 512, n_vocab=1000, factor=3.0)
    assert g3["required_bytes"] == 3 * 512 * 1000 * 2


def _write_logits(p: Path, n_ctx: int, n_vocab: int, n_chunk: int, *, delta: int = 0,
                  magic: bytes = b"_logits_") -> int:
    size = quality.expected_logits_bytes(n_ctx, n_vocab, n_chunk) + delta
    with open(p, "wb") as f:
        f.write(magic + struct.pack("<Iii", n_ctx, n_vocab, n_chunk))
        f.write(b"\0" * (size - 20))
    return size


def test_expected_logits_bytes_matches_perplexity_cpp_layout() -> None:
    # header 20 B + n_chunk*n_ctx int32 tokens + per chunk (n_ctx-1-n_ctx/2) rows of nv uint16,
    # nv = 2*((n_vocab+1)/2)+4 (tools/perplexity/perplexity.cpp at a7b94df)
    assert quality.expected_logits_bytes(64, 354, 4) == 20 + 4 * 64 * 4 + 4 * 31 * 358 * 2 == 89_828
    assert quality.expected_logits_bytes(64, 355, 1) == 20 + 64 * 4 + 31 * 360 * 2


def test_check_logits(tmp_path: Path) -> None:
    good = tmp_path / "ref.logits"
    size = _write_logits(good, 64, 354, 4)
    c = quality.check_logits(good)
    assert c["size_ok"] is True and c["bytes"] == size == c["expected_bytes"]
    assert (c["n_ctx"], c["n_vocab"], c["n_chunk"], c["magic_ok"]) == (64, 354, 4, True)
    assert quality.read_logits_header(good) == {"magic_ok": True, "n_ctx": 64, "n_vocab": 354,
                                                "n_chunk": 4}
    short = tmp_path / "short.logits"
    _write_logits(short, 64, 354, 4, delta=-100)
    c = quality.check_logits(short)
    assert c["size_ok"] is False and "truncated" in c["reason"]
    bad = tmp_path / "bad.logits"
    _write_logits(bad, 64, 354, 4, magic=b"NOTLOGIT")
    assert quality.check_logits(bad)["reason"].startswith("bad magic")
    tiny = tmp_path / "tiny.logits"
    tiny.write_bytes(b"_logits_")
    assert quality.check_logits(tiny)["size_ok"] is False
    assert quality.check_logits(tmp_path / "missing.logits")["reason"] == "missing"


# =============================================================================================
# Corpus
# =============================================================================================
def test_build_corpus_interleave_concat_and_idempotence(tmp_path: Path) -> None:
    src = tmp_path / "cal"
    src.mkdir()
    (src / "a_en.txt").write_bytes(("\ufeff" + "A line one\r\n" * 30).encode("utf-8"))
    (src / "b_he.txt").write_text("שורה בעברית\n" * 30, encoding="utf-8")
    (src / "c_code.cpp").write_text("int main() { return 0; }\n" * 30, encoding="utf-8")
    (src / "README.md").write_text("not corpus", encoding="utf-8")
    (src / "all.txt").write_text("old output", encoding="utf-8")
    (src / "notes.bin").write_bytes(b"\0\1")
    assert [p.name for p in quality.corpus_sources(src)] == ["a_en.txt", "b_he.txt", "c_code.cpp"]
    out = tmp_path / "all.txt"
    info = quality.build_corpus(out, src_dir=src, slice_chars=200)
    text = out.read_text(encoding="utf-8")
    assert info["written"] is True and info["up_to_date"] is False
    assert info["sha256"] == quality._sha256_file(out) and info["bytes"] == out.stat().st_size
    assert [f["name"] for f in info["files"]] == ["a_en.txt", "b_he.txt", "c_code.cpp"]
    assert b"\r" not in out.read_bytes()
    assert "\ufeff" not in text and "not corpus" not in text
    assert "old output" not in text
    # interleaved: the first slices come from different files
    _, layout = quality.corpus_text(quality.corpus_sources(src), "interleave", 200)
    assert [s["file"] for s in layout[:3]] == ["a_en.txt", "b_he.txt", "c_code.cpp"]
    assert text.count("A line one") == 30 and text.count("שורה בעברית") == 30
    again = quality.build_corpus(out, src_dir=src, slice_chars=200)
    assert again["written"] is False and again["up_to_date"] is True
    assert again["sha256"] == info["sha256"]
    cat = quality.build_corpus(None, src_dir=src, mode="concat")
    assert cat["path"] is None and cat["written"] is False and cat["sha256"] != info["sha256"]
    with pytest.raises(ValueError):
        quality.corpus_text(quality.corpus_sources(src), "shuffle")
    with pytest.raises(bench.MissingInputError):
        quality.build_corpus(None, src_dir=tmp_path / "empty")


def test_real_calibration_corpus_mixes_languages_early() -> None:
    srcs = quality.corpus_sources()
    names = {p.name for p in srcs}
    assert "he_discussion_llm_efficiency.txt" in names and "en_tech_moe_streaming.txt" in names
    assert not any(n.upper().startswith("README") or n == "all.txt" for n in names)
    info = quality.build_corpus(None)     # never writes into the repo here
    assert info["written"] is False and len(info["sha256"]) == 64
    _, layout = quality.corpus_text(srcs, "interleave")
    first = {s["file"] for s in layout if s["start"] < 6000}
    assert len(first) >= 5   # the first ~2k tokens already mix code, English, Hebrew, German


# =============================================================================================
# CLI end to end with test doubles
# =============================================================================================
FAKE_PERPLEXITY = r'''
import os, struct, sys
from pathlib import Path
a = sys.argv
if "--version" in a:
    print("version: 11398 (a7b94df2c)")
    sys.exit(0)
def arg(n, d=None):
    return a[a.index(n) + 1] if n in a else d
log = os.environ.get("FAKE_PPL_LOG")
if log:
    with open(log, "a", encoding="utf-8") as f:
        f.write(" ".join(a[1:]) + "\n")
fix = Path(os.environ["FAKE_PPL_FIXDIR"])
if "--kl-divergence" in a:
    sys.stdout.buffer.write((fix / "perplexity_kld_sample.txt").read_bytes())
    sys.exit(0)
out = arg("--save-all-logits")
n_ctx, n_chunk = int(arg("-c", "512")), int(arg("--chunks", "4"))
n_vocab = int(os.environ.get("FAKE_NVOCAB", "354"))
nv = 2 * ((n_vocab + 1) // 2) + 4
size = 20 + n_chunk * n_ctx * 4 + n_chunk * (n_ctx - 1 - n_ctx // 2) * nv * 2
if os.environ.get("FAKE_TRUNCATE") == "1":
    size -= 1000
if out:
    with open(out, "wb") as f:
        f.write(b"_logits_" + struct.pack("<Iii", n_ctx, n_vocab, n_chunk))
        f.write(b"\0" * (size - 20))
sys.stdout.buffer.write((fix / "perplexity_base_sample.txt").read_bytes())
'''

FAKE_QA_SERVER = r'''
import json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
a = sys.argv
if "--version" in a:
    print("version: 11398 (a7b94df2c)")
    sys.exit(0)
def arg(n, d=None):
    return a[a.index(n) + 1] if n in a else d
HOST, PORT = arg("--host", "127.0.0.1"), int(arg("--port", "8080"))
DRAFT, SPEC = arg("-md"), arg("--spec-type")
with open(os.environ["FAKE_QA_ANSWERS"], encoding="utf-8") as f:
    ANS = json.load(f)
DIV = os.environ.get("FAKE_QA_DIVERGE", "")
print("0.00.100.000 I srv  llama_server: model loaded", flush=True)
class H(BaseHTTPRequestHandler):
    def log_message(self, *x):
        pass
    def _send(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {})
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n).decode("utf-8"))
        prompt = str(body.get("prompt", ""))
        answer = next((v for k, v in ANS.items() if k in prompt), "I do not know.")
        drafting = bool(DRAFT or SPEC)
        if drafting and DIV and DIV in prompt:
            answer += " (diverged)"
        t = {"prompt_n": 20, "prompt_ms": 10.0, "prompt_per_second": 2000.0, "predicted_n": 5,
             "predicted_ms": 50.0, "predicted_per_second": 80.0, "cache_n": 0}
        if drafting:
            t.update(draft_n=4, draft_n_accepted=3)
        types = ("none," + SPEC) if SPEC else "none"
        self._send(200, {"content": "<|channel>thought\n<channel|>" + answer, "stop_type": "eos",
                         "timings": t, "tokens_predicted": 5, "tokens_evaluated": 20,
                         "generation_settings": {"speculative.types": types},
                         "truncated": False})
srv = ThreadingHTTPServer((HOST, PORT), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
while True:
    time.sleep(0.2)
'''


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture()
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_lock: Path) -> dict[str, Any]:
    for k in ("PIGEMMA_EMULATED", "FAKE_TRUNCATE", "FAKE_QA_DIVERGE", "FAKE_STDOUT", "FAKE_RC",
              "FAKE_STDERR", "FAKE_HANG"):
        monkeypatch.delenv(k, raising=False)
    ppl = tmp_path / "fake_perplexity.py"
    ppl.write_text(FAKE_PERPLEXITY, encoding="utf-8")
    srv = tmp_path / "fake_qa_server.py"
    srv.write_text(FAKE_QA_SERVER, encoding="utf-8")
    fakes = {"llama-perplexity": ppl, "llama-server": srv,
             "llama-completion": BENCH_FIX / "fake_llama_completion.py"}
    real = quality._tool_cmd

    def tool_cmd(tool: str, override: Any = None) -> list[str]:
        if tool in fakes and override is None:
            return [sys.executable, str(fakes[tool])]
        return real(tool, override)
    monkeypatch.setattr(quality, "_tool_cmd", tool_cmd)
    monkeypatch.setattr(bench, "HEALTH_POLL_S", 0.05)
    monkeypatch.setattr(bench, "PORT_PROBE_TIMEOUT_S", 0.05)
    monkeypatch.setenv("FAKE_PPL_FIXDIR", str(FIX))
    monkeypatch.setenv("FAKE_PPL_LOG", str(tmp_path / "ppl_calls.txt"))
    monkeypatch.setenv("FAKE_NVOCAB", "354")
    model = tmp_path / "fake-26B-q4_0.gguf"
    model.write_bytes(b"GGUF" + b"\0" * 1000)
    text = tmp_path / "corpus.txt"
    text.write_text("Experts are routed per token.\nשלום לכולם.\n" * 50, encoding="utf-8")
    return {"tmp": tmp_path, "model": model, "text": text, "lock": isolated_lock,
            "common": ["--min-free-gb", "0", "--no-sha", "-t", "2"],
            "calls": tmp_path / "ppl_calls.txt"}


def _n_calls(p: Path) -> int:
    return len(p.read_text(encoding="utf-8").splitlines()) if p.is_file() else 0


def _envelopes(d: Path) -> list[dict[str, Any]]:
    return results.load_results(d, recursive=False)


def test_cli_kld_base_then_kld(cli_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
                               capsys: pytest.CaptureFixture[str]) -> None:
    e = cli_env
    monkeypatch.setattr(paths, "free_disk_bytes", lambda *_a, **_k: 10**12)
    ref = e["tmp"] / "quality" / "ref_fake.logits"
    base_args = ["kld-base", "--model", str(e["model"]), "--text", str(e["text"]), "-c", "64",
                 "--chunks", "4", "--out", str(ref), "--results-dir", str(e["tmp"] / "res")]
    rc = quality.main(base_args + e["common"])
    cap = capsys.readouterr()
    assert rc == 0, cap.err
    assert "logits size estimate 134,217,728 B" in cap.err          # 4 x 64 x 262144 x 2
    assert ref.stat().st_size == quality.expected_logits_bytes(64, 354, 4)
    meta = quality.read_logits_meta(ref)
    assert meta and meta["ok"] is True and meta["chunks"] == 4 and meta["ctx"] == 64
    assert meta["ppl"] == pytest.approx(1165.0598) and meta["text"]["sha256"]
    [base_doc] = _envelopes(e["tmp"] / "res")
    assert base_doc["experiment"] == quality.default_experiment()
    assert base_doc["kind"] == "measured" and base_doc["ppl"] == pytest.approx(1165.0598)
    assert base_doc["extra"]["logits"]["size_ok"] is True and base_doc["n_ctx"] == 64
    assert base_doc["llama_build"] == 11398 and base_doc["llama_commit"].startswith("a7b94df")
    assert "--save-all-logits" in base_doc["extra"]["argv"]
    assert results.validate_result(base_doc) == []
    assert _n_calls(e["calls"]) == 1
    # a valid matching reference is reused, not recomputed
    assert quality.main(base_args + e["common"]) == 0
    assert _n_calls(e["calls"]) == 1
    capsys.readouterr()
    # KLD of the -nr variant against it
    rc = quality.main(["kld", "--model", str(e["model"]), "--base", str(ref), "--text",
                       str(e["text"]), "--variant", "norepack", "--out",
                       str(e["tmp"] / "kld") + "/"] + e["common"])
    cap = capsys.readouterr()
    assert rc == 0, cap.err
    [doc] = _envelopes(e["tmp"] / "kld")
    assert doc["variant"] == "norepack" and doc["flags"] == ["-nr"]
    assert doc["kld"] == {"mean": pytest.approx(0.06017), "max": pytest.approx(0.116063),
                          "p999": pytest.approx(0.115574), "same_top_p": pytest.approx(0.77419),
                          "ppl_q": pytest.approx(1100.227588),
                          "ppl_base": pytest.approx(1165.064883)}
    argv = doc["extra"]["argv"]
    assert argv[argv.index("-c") + 1] == "64" and argv[argv.index("--chunks") + 1] == "4"
    assert "--kl-divergence" in argv and argv[argv.index("--kl-divergence-base") + 1] == str(ref)
    assert argv[-1] == "-nr"
    assert doc["extra"]["threshold_check"]["neutral"] is False
    assert doc["extra"]["threshold_check"]["tier"] == "degraded"
    assert results.validate_result(doc) == []
    # a variant name that is neither a load variant nor a model id is a usage error
    assert quality.main(["kld", "--model", str(e["model"]), "--base", str(ref), "--variant",
                         "norepak"] + e["common"]) == 2


def test_cli_kld_base_disk_gate_refuses(cli_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
                                        capsys: pytest.CaptureFixture[str]) -> None:
    e = cli_env
    monkeypatch.setattr(paths, "free_disk_bytes", lambda *_a, **_k: 3_000_000_000)
    ref = e["tmp"] / "q" / "ref.logits"
    rc = quality.main(["kld-base", "--model", str(e["model"]), "--text", str(e["text"]),
                       "--out", str(ref), "--no-envelope"] + e["common"])
    err = capsys.readouterr().err
    assert rc == paths.EXIT_REFUSED == 4
    assert "REFUSED" in err and "1,610,612,736" in err and "3,000,000,000" in err
    assert "3,221,225,472" in err
    assert not ref.exists() and _n_calls(e["calls"]) == 0


def test_cli_kld_base_truncated_file_fails_then_is_replaced(cli_env: dict[str, Any],
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    e = cli_env
    monkeypatch.setattr(paths, "free_disk_bytes", lambda *_a, **_k: 10**12)
    monkeypatch.setenv("FAKE_TRUNCATE", "1")
    ref = e["tmp"] / "ref.logits"
    args = ["kld-base", "--model", str(e["model"]), "--text", str(e["text"]), "-c", "64",
            "--chunks", "4", "--out", str(ref), "--results-dir", str(e["tmp"] / "res")]
    assert quality.main(args + e["common"]) == 1
    [doc] = _envelopes(e["tmp"] / "res")
    assert doc["notes"][0].startswith("failed: logits file check")
    assert quality.check_logits(ref)["size_ok"] is False
    # kld refuses a broken base
    assert quality.main(["kld", "--model", str(e["model"]), "--base", str(ref), "--variant",
                         "norepack"] + e["common"]) == 1
    # a broken reference is replaced without --force
    monkeypatch.delenv("FAKE_TRUNCATE")
    assert quality.main(args + ["--no-envelope"] + e["common"]) == 0
    assert quality.check_logits(ref)["size_ok"] is True


def test_cli_qa_server_plain_and_mtp_explicit(cli_env: dict[str, Any],
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    e = cli_env
    items = quality.load_qa_set()
    answers = {it["prompt"]: GOOD[it["id"]] for it in items}
    answers[qa_item("he-02")["prompt"]] = "קר"          # one wrong Hebrew answer
    ans_file = e["tmp"] / "answers.json"
    ans_file.write_text(json.dumps(answers, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("FAKE_QA_ANSWERS", str(ans_file))
    monkeypatch.setenv("FAKE_QA_DIVERGE", "בירת צרפת")   # he-01 changes when drafting
    draft = e["tmp"] / "mtp-fake-Q4_0.gguf"
    draft.write_bytes(b"GGUF" + b"\0" * 100)
    out = e["tmp"] / "qa"
    rc = quality.main(["qa", "--model", str(e["model"]), "--server-variant", "mtp-explicit,plain",
                       "--draft", str(draft), "--port", str(free_port()), "-n", "32",
                       "--warmup-tokens", "4", "--out", str(out) + "/"] + e["common"])
    assert rc == 0
    plain = json.loads((out / "qa_fake-26B-q4_0_plain.json").read_text(encoding="utf-8"))
    mtp = json.loads((out / "qa_fake-26B-q4_0_mtp-explicit.json").read_text(encoding="utf-8"))
    for doc in (plain, mtp):
        assert results.validate_result(doc) == []
        assert doc["experiment"] == quality.default_experiment() and doc["kind"] == "measured"
        assert (doc["qa_score"], doc["qa_total"]) == (15, 16)
        ex = doc["extra"]
        assert ex["per_language"] == {"en": {"score": 7, "total": 7},
                                      "he": {"score": 5, "total": 6},
                                      "num": {"score": 3, "total": 3}}
        assert ex["threshold_check"]["qa_pass"] is True
        assert ex["threshold_check"]["hebrew_pass"] is True
        by = {r["id"]: r for r in ex["items"]}
        assert by["en-01"]["visible"] == "Paris" and by["en-01"]["thinking_chars"] == 0
        assert by["he-02"]["pass"] is False and by["he-06"]["pass"] is True
        assert all("<|channel>" not in (r["visible"] or "") for r in ex["items"])
        assert ex["qa_set"]["n_items"] == 16 and len(ex["qa_set"]["sha256"]) == 64
        assert doc["coherent"] is True and doc["tg"]["tok_s"] > 0
    assert plain["variant"] == "plain" and plain["extra"]["exact_match_vs_plain"] is None
    assert plain["identical_to_plain"] is None
    assert mtp["variant"] == "mtp-explicit"
    assert mtp["extra"]["exact_match_vs_plain"] == pytest.approx(15 / 16)
    assert mtp["identical_to_plain"] == pytest.approx(15 / 16)
    assert mtp["extra"]["exact_match_by_item"]["he-01"] is False
    assert mtp["extra"]["plain_ref"] == "this invocation"
    assert mtp["spec_active"] is True and mtp["spec_type_effective"] == "explicit:draft-mtp"
    assert mtp["accept_rate"] == pytest.approx(0.75)
    assert {r["id"]: r["visible"] for r in mtp["extra"]["items"]}["he-01"] == "פריז (diverged)"
    # a later speculative-only run finds the plain file in the same directory
    rc = quality.main(["qa", "--model", str(e["model"]), "--server-variant", "mtp-explicit",
                       "--draft", str(draft), "--port", str(free_port()), "-n", "32",
                       "--ids", "en-01,he-01", "--out", str(e["tmp"] / "qa" / "again.json")]
                      + e["common"])
    assert rc == 0
    again = quality.load_qa_file(e["tmp"] / "qa" / "again.json")
    assert again["qa_score"] == 2 and again["exact_match_vs_plain"] == 0.5
    assert again["visible_by_id"] == {"en-01": "Paris", "he-01": "פריז (diverged)"}


def test_cli_qa_no_server_uses_llama_completion(cli_env: dict[str, Any],
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    e = cli_env
    monkeypatch.setenv("FAKE_STDOUT", "Paris")
    out = e["tmp"] / "qa.json"
    rc = quality.main(["qa", "--model", str(e["model"]), "--no-server", "--ids", "en-01,he-01",
                       "-n", "16", "--out", str(out)] + e["common"])
    assert rc == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert results.validate_result(doc) == []
    assert doc["variant"] == "completion" and (doc["qa_score"], doc["qa_total"]) == (1, 2)
    assert {r["id"]: r["pass"] for r in doc["extra"]["items"]} == {"en-01": True, "he-01": False}
    assert len(doc["extra"]["runs"]) == 2 and doc["tg"]["tok_s"] > 0
    assert doc["extra"]["exact_match_vs_plain"] is None


def test_cli_refuses_when_bench_lock_is_held(cli_env: dict[str, Any],
                                             capsys: pytest.CaptureFixture[str]) -> None:
    e = cli_env
    args = ["ppl", "--model", str(e["model"]), "--text", str(e["text"]), "-c", "64",
            "--chunks", "4", "--out", str(e["tmp"] / "ppl") + "/"] + e["common"]
    with paths.bench_lock(0.0, cmd="another model-loading job"):
        rc = quality.main(args)
    assert rc == paths.EXIT_REFUSED
    assert "REFUSED" in capsys.readouterr().err and _n_calls(e["calls"]) == 0
    assert quality.main(args) == 0
    [doc] = _envelopes(e["tmp"] / "ppl")
    assert doc["ppl"] == pytest.approx(1165.0598)
    assert doc["extra"]["chunk_ppl"][0] == pytest.approx(505.3665)


def test_cli_usage_and_missing_inputs(cli_env: dict[str, Any]) -> None:
    e = cli_env
    m = ["--model", str(e["model"])] + e["common"]
    assert quality.main(["qa", "--server-variant", "turbo"] + m) == 2
    assert quality.main(["qa", "--no-server", "--server-variant", "mtp"] + m) == 2
    assert quality.main(["qa", "--ids", "zz-99"] + m) == 2
    assert quality.main(["qa", "--qa-set", str(e["tmp"] / "none.json")] + m) == 3
    assert quality.main(["ppl", "--model", str(e["tmp"] / "missing.gguf"), "--text",
                         str(e["text"])] + e["common"]) == 3
    assert quality.main(["kld", "--base", str(e["tmp"] / "none.logits"), "--variant",
                         "norepack"] + m) == 3
    assert quality.main(["nonsense"]) == 2


def test_cli_parse_and_check_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert quality.main(["parse-kld", str(FIX / "perplexity_kld_sample.txt")]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["kld"]["same_top_p"] == pytest.approx(0.77419) and d["tier"] == "degraded"
    assert quality.main(["parse-ppl", str(FIX / "perplexity_base_sample.txt")]) == 0
    assert json.loads(capsys.readouterr().out)["ppl"] == pytest.approx(1165.0598)
    assert quality.main(["parse-ppl", str(FIX / "perplexity_kld_sample.txt")]) == 1
    capsys.readouterr()
    assert quality.main(["check-text", str(FIX / "thinking_sample.txt")]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["visible"].endswith("שלום") and d["coherent"] is True and d["thinking_chars"] == 0
    good = tmp_path / "ok.logits"
    _write_logits(good, 64, 354, 2)
    assert quality.main(["check-logits", str(good)]) == 0
    capsys.readouterr()
    bad = tmp_path / "bad.logits"
    _write_logits(bad, 64, 354, 2, delta=-1)
    assert quality.main(["check-logits", str(bad)]) == 1
    assert quality.main(["check-logits", str(tmp_path / "none.logits")]) == 3
