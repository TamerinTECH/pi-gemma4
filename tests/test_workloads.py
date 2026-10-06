"""Unit tests for pigemma.workloads and data/workloads/*.jsonl (P3, 2026-10-05). No model."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from pigemma import workloads as wl

REPO = Path(__file__).resolve().parents[1]
WDIR = REPO / "data" / "workloads"
TEMPLATE = REPO / "tests" / "fixtures" / "gemma4_chat_template.jinja"


@pytest.fixture(scope="module")
def sets():
    return wl.load_all()


def test_repo_workloads_validate():
    rep = wl.validate_all()
    assert rep["ok"], rep["errors"]
    s = rep["summary"]
    assert 20 <= s["calibration"]["n_conversations"] <= 28
    assert 6 <= s["tuning"]["n_conversations"] <= 10
    assert 16 <= s["heldout"]["n_conversations"] <= 20
    assert s["heldout"]["generation_budget_tokens"] <= wl.HELDOUT_GEN_BUDGET


@pytest.mark.parametrize("name", ["calibration", "heldout"])
def test_coverage(sets, name):
    convs = sets[name]
    langs_in_turns = set()
    for c in convs:
        langs_in_turns.add(c["lang"])
    assert {"en", "he", "de"} <= langs_in_turns
    cats = {c["category"] for c in convs}
    assert {"chat", "code", "instruction_following", "extraction", "topic_switch", "follow_up"} <= cats
    # abrupt topic changes: >= 3 topic_switch conversations with >= 2 user turns
    assert sum(1 for c in convs if c["category"] == "topic_switch") >= 3
    # follow-ups with explicit reuse
    fu = [c for c in convs if c["context_reuse"] == "follow_up_same_conversation"]
    assert len(fu) >= 4 and all(len(wl.user_turns(c)) >= 2 for c in fu)
    assert all(wl.MAX_TOKENS_MIN <= c["max_tokens"] <= wl.MAX_TOKENS_MAX for c in convs)


def test_tuning_mirrors_heldout_mix(sets):
    cats = {c["category"] for c in sets["tuning"]}
    assert {"chat", "code", "instruction_following", "extraction", "topic_switch", "follow_up"} <= cats
    assert {c["lang"] for c in sets["tuning"]} >= {"en", "he", "de"}


def test_token_counts_file_matches_and_has_long_and_short(sets):
    p = WDIR / "token_counts.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    # the counts were measured on the current files
    import hashlib
    for s in wl.SETS:
        assert d["files_sha256"][s] == hashlib.sha256((WDIR / f"{s}.jsonl").read_bytes()).hexdigest(), s
    for s in ("calibration", "heldout"):
        lc = d["by_set"][s]["length_classes"]
        assert lc.get("long", 0) >= 3 and lc.get("short", 0) >= 8, (s, lc)
    for cid, e in d["conversations"].items():
        if e["length_class"] == "long":
            assert 500 <= e["first_prompt_tokens"] <= 1000, cid
    assert d["thought_block_tokens"] == 4


def test_disjointness_detects_shared_text(tmp_path):
    for f in WDIR.glob("*.jsonl"):
        shutil.copy(f, tmp_path / f.name)
    assert wl.validate_all(tmp_path, check_external=False)["ok"]
    # copy a held-out document into a calibration request -> must fail
    ho = wl.load_set(tmp_path / "heldout.jsonl")
    doc = next(c for c in ho if c["id"] == "ho-12")["turns"][0]["content"]
    cal = wl.load_set(tmp_path / "calibration.jsonl")
    cal[0]["turns"][0]["content"] = "Summarise this. " + doc[200:900]
    (tmp_path / "calibration.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cal), encoding="utf-8")
    rep = wl.validate_all(tmp_path, check_external=False)
    assert not rep["ok"] and any("shares text with ho-12" in e for e in rep["errors"])


def test_external_corpus_is_not_in_heldout_or_tuning(tmp_path):
    for f in WDIR.glob("*.jsonl"):
        shutil.copy(f, tmp_path / f.name)
    ho = wl.load_set(tmp_path / "heldout.jsonl")
    corpus = (REPO / "data" / "calibration" / "en_narrative_bakery.txt").read_text(encoding="utf-8")
    ho[0]["turns"][0]["content"] = "Continue: " + corpus[:400].strip()
    (tmp_path / "heldout.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in ho), encoding="utf-8")
    rep = wl.validate_all(tmp_path)
    assert any("en_narrative_bakery" in e for e in rep["errors"])


def test_schema_errors():
    base = {"id": "ho-99", "set": "heldout", "lang": "en", "category": "chat",
            "turns": [{"role": "user", "content": "hi"}], "max_tokens": 64,
            "context_reuse": "none", "notes": ""}
    assert wl.validate_conversation(base, "heldout") == []
    assert wl.validate_conversation(dict(base, max_tokens=200), "heldout")
    assert wl.validate_conversation(dict(base, context_reuse="follow_up_same_conversation"), "heldout")
    assert wl.validate_conversation(dict(base, turns=[{"role": "assistant", "content": "x"}]), "heldout")
    assert wl.validate_conversation(dict(base, turns=[{"role": "user", "content": " x "}]), "heldout")
    assert wl.validate_conversation(dict(base, extra=1), "heldout")
    assert wl.validate_conversation(base, "tuning")


def test_request_messages_and_render(sets):
    conv = next(c for c in sets["heldout"] if c["id"] == "ho-15")
    m0 = wl.request_messages(conv, 0, [])
    assert m0 == [{"role": "user", "content": conv["turns"][0]["content"]}]
    m2 = wl.request_messages(conv, 2, ["r0", "r1"])
    assert [m["role"] for m in m2] == ["user", "assistant", "user", "assistant", "user"]
    with pytest.raises(ValueError):
        wl.request_messages(conv, 2, ["r0"])
    p0 = wl.render_gemma4(m0)
    assert p0.startswith("<|turn>user\n") and p0.endswith("<|turn>model\n<|channel>thought\n<channel|>")
    # reusable prefix: request 1 starts with request 0 minus the empty thought block
    p1 = wl.render_gemma4(wl.request_messages(conv, 1, ["  reply  "]))
    assert p1.startswith(p0[: -len(wl.EMPTY_THOUGHT)])
    assert not p1.startswith(p0)
    assert "<|turn>model\nreply<turn|>\n" in p1


def test_render_matches_the_gguf_template_via_jinja2(sets):
    jinja2 = pytest.importorskip("jinja2")
    env = jinja2.Environment()

    def _raise(msg):
        raise RuntimeError(msg)
    env.globals["raise_exception"] = _raise
    tmpl = env.from_string(TEMPLATE.read_text(encoding="utf-8"))
    cases = []
    for s in wl.SETS:
        for c in sets[s]:
            n = len(wl.user_turns(c))
            cases.append(wl.request_messages(c, n - 1, [f"reply {j} <|channel>thought\nx<channel|>ok "
                                                        for j in range(n - 1)]))
    cases.append([{"role": "system", "content": " Be brief. "}, {"role": "user", "content": "hi"}])
    for msgs in cases:
        for think in (False, True):
            ref = tmpl.render(messages=msgs, add_generation_prompt=True, bos_token="<bos>",
                              enable_thinking=think)
            assert wl.render_gemma4(msgs, enable_thinking=think, bos=True) == ref


def test_cli_validate_and_render(capsys):
    assert wl.main(["validate"]) == 0
    assert wl.main(["render", "ho-17", "--turn", "1"]) == 0
    out = capsys.readouterr().out
    assert "<REPLY 0>" in out and "Grace Mensah" in out
