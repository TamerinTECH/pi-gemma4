"""Unit tests for pigemma.quality_v2 and data/prompts/qa_heldout_v2.json (P3, 2026-10-05).

No model, no llama.cpp binary: the chat endpoint is a tiny in-process fake HTTP server. The
python_exec graders really run code in a sandboxed subprocess (a few seconds in total).
"""

from __future__ import annotations

import http.server
import json
import socket
import threading
from pathlib import Path

import pytest

from pigemma import quality_v2 as q2
from pigemma import workloads as wl

REPO = Path(__file__).resolve().parents[1]
QA_PATH = REPO / "data" / "prompts" / "qa_heldout_v2.json"


@pytest.fixture(scope="module")
def items():
    return q2.load_qa_v2(QA_PATH)


# --------------------------------------------------------------------------------------------
# The Q&A file
# --------------------------------------------------------------------------------------------
def test_qa_file_size_and_coverage(items):
    assert len(items) >= 40
    cats = {}
    for it in items:
        cats.setdefault(it["category"], []).append(it)
    for c in ("negation", "numbers", "multilingual", "extraction", "code"):
        assert len(cats.get(c, [])) >= 6, c
    for c in ("negation", "numbers", "multilingual", "extraction"):
        assert {it["lang"] for it in cats[c]} >= {"en", "he", "de"}, c
    # structured extraction is graded by JSON field match, code by execution
    assert all(it["grader"]["kind"] == "json_fields" for it in cats["extraction"])
    assert all(it["grader"]["kind"] == "python_exec" for it in cats["code"])
    assert {it["lang"] for it in cats["code"]} >= {"en", "he", "de"}
    # every item has at least one answer that must fail
    assert all(it.get("wrong") for it in items)


def test_selftest_references_pass_and_wrong_answers_fail(items):
    rep = q2.selftest(items)
    assert rep["ok"], rep["failures"]
    assert rep["n_checked"] == len(items) + sum(len(it["wrong"]) for it in items)


def test_qa_items_are_new_and_disjoint_from_workloads():
    """No shared 8-word run with any workload set (calibration/tuning/held-out) or with the
    older smoke-test prompt sets."""
    qa = {it["id"]: wl.shingles(it["prompt"]) for it in q2.load_qa_v2(QA_PATH)}
    other = {}
    for s, convs in wl.load_all().items():
        for c in convs:
            sh = set()
            for t in c["turns"]:
                sh |= wl.shingles(t["content"])
            other[f"{s}:{c['id']}"] = sh
    for k, v in wl.external_corpus_texts().items():
        if k.startswith("data/prompts/"):
            other[k] = wl.shingles(v)
    hits = wl._overlaps(qa, other)  # noqa: SLF001
    assert not hits, hits


def test_loader_rejects_bad_files(tmp_path):
    p = tmp_path / "qa.json"
    p.write_text(json.dumps([{"id": "a"}]), encoding="utf-8")
    with pytest.raises(q2.QaV2Error):
        q2.load_qa_v2(p)
    good = {"id": "x", "lang": "en", "category": "numbers", "prompt": "p", "n_predict": 8,
            "grader": {"kind": "number", "value": 1}, "reference": "1"}
    p.write_text(json.dumps({"format": "pigemma-qa/2", "items": [good, good]}), encoding="utf-8")
    with pytest.raises(q2.QaV2Error, match="duplicate"):
        q2.load_qa_v2(p)
    bad = dict(good, id="y", grader={"kind": "nope"})
    p.write_text(json.dumps({"format": "pigemma-qa/2", "items": [good, bad]}), encoding="utf-8")
    with pytest.raises(q2.QaV2Error):
        q2.load_qa_v2(p)
    assert [it["id"] for it in q2.load_qa_v2(QA_PATH, ids=["code-01", "neg-en-01"])] == ["neg-en-01", "code-01"]


# --------------------------------------------------------------------------------------------
# Visible answer
# --------------------------------------------------------------------------------------------
def test_visible_answer_strips_thinking_and_markup():
    vis, think, notes = q2.visible_answer("<|channel>thought\nlet me think<channel|>Paris<turn|>")
    assert vis == "Paris" and think == len("let me think")
    vis, think, _ = q2.visible_answer("42", reasoning_content="  some reasoning  ")
    assert vis == "42" and think == len("some reasoning")
    vis, _, notes = q2.visible_answer("NO\n<|turn>user\nnext question?")
    assert vis == "NO" and "cut fabricated turn" in notes
    vis, _, notes = q2.visible_answer("model\nNO")
    assert vis == "NO" and any("model" in n for n in notes)
    # orphan closer (thinking opener was in the prompt), unterminated block
    assert q2.visible_answer("thinking...<channel|>16,00")[0] == "16,00"
    assert q2.visible_answer("<|channel>thought\nnever closed")[0] == ""


def test_numbers_and_script():
    assert q2.numbers_in("Total: 1,234.5 and -7, then ב-8 שקלים") == [1234.5, -7.0, 8.0]
    assert q2.numbers_in("16,8 Euro") == [16.8]
    assert q2.numbers_in("no digits") == []
    assert q2.script_fraction("שלום world", "hebrew") == pytest.approx(4 / 9)
    assert q2.script_fraction("Größe", "latin") == 1.0
    assert q2.script_fraction("123 !", "latin") is None


# --------------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------------
def test_simple_checks():
    assert q2.run_check({"kind": "contains", "any": ["כדור הארץ"]}, "כדור־הארץ.")["pass"]
    assert not q2.run_check({"kind": "not_regex", "pattern": r"(?i)\bzinc\b"}, "Zinc")["pass"]
    r = q2.run_check({"kind": "number", "value": 27, "which": "last"}, "40 -> 30 -> 27")
    assert r["pass"] and r["value"] == 27
    assert not q2.run_check({"kind": "number", "value": 27}, "40 -> 30 -> 27")["pass"]
    assert q2.run_check({"kind": "lines", "min": 3, "max": 3}, "a\n\nb\nc\n")["pass"]
    assert not q2.run_check({"kind": "words", "max": 2}, "one two three")["pass"]
    r = q2.run_check({"kind": "all_of", "checks": [{"kind": "regex", "pattern": "a"},
                                                   {"kind": "regex", "pattern": "z"}]}, "abc")
    assert not r["pass"] and r["score"] == 0.5


def test_extract_json_and_field_match():
    assert q2.extract_json('Sure! ```json\n{"a": 1, "b": [1, 2]}\n``` done') == {"a": 1, "b": [1, 2]}
    assert q2.extract_json("x = {not json} then {\"k\": \"v\"}") == {"k": "v"}
    assert q2.extract_json("nothing here") is None
    spec = {"kind": "json_fields", "fields": {"n": 47, "d": "2025-03-03", "ok": True, "who": {"any": ["Jana Weiß"]},
                                              "price": {"number": 120, "tol": 0.5}, "tags": {"set": ["a", "b"]}}}
    ans = '{"N": "47", "d": "2025-03-03", "ok": "yes", "who": "jana weiss", "price": "120,00", "tags": ["b", "a"]}'
    r = q2.run_check(spec, ans)
    assert r["pass"], r
    r = q2.run_check(spec, '{"n": 46, "d": "2025-3-3", "ok": false, "who": "Jana", "price": 121, "tags": ["a"]}')
    assert not r["pass"] and r["score"] == 0.0
    r = q2.run_check(spec, '{"n": 47, "d": "2025-03-03"}')
    assert not r["pass"] and r["score"] == pytest.approx(2 / 6)


# --------------------------------------------------------------------------------------------
# Code extraction, screening and the sandbox
# --------------------------------------------------------------------------------------------
def test_extract_code_variants():
    t = "Here:\n```python\ndef f(x):\n    return x + 1\n```\nExample: f(1) == 2"
    assert q2.extract_code(t, "f") == "def f(x):\n    return x + 1"
    t = "```js\nfunction f(){}\n```\n```\nimport math\ndef f(x):\n    return math.sqrt(x)\n```"
    assert q2.extract_code(t, "f").startswith("import math")
    truncated = "```python\ndef f(x):\n    return 2 * x\n"
    assert "return 2 * x" in q2.extract_code(truncated, "f")
    plain = "def f(x):\n    return x\n\nThis function returns its input."
    assert q2.extract_code(plain, "f") == "def f(x):\n    return x\n"
    assert q2.extract_code("no code at all", "f") is None


def test_screen_code():
    assert q2.screen_code("import math\nfrom collections import Counter\nif __name__ == '__main__':\n    pass") == []
    assert q2.screen_code("import os") == ["import os"]
    assert "name open" in q2.screen_code("def f():\n    return open('x').read()")
    assert "attribute .__class__" in q2.screen_code("x = ().__class__.__bases__")
    assert "name getattr" in q2.screen_code("getattr(1, 'real')")
    assert "name eval" in q2.screen_code("eval('1')")
    assert q2.screen_code("from . import x")
    assert q2.screen_code("def f(:")[0].startswith("syntax error")


def test_run_python_tests_pass_fail_and_errors():
    code = "def add(a, b):\n    print('noise')\n    return a + b\nprint('module-level noise')"
    r = q2.run_python_tests(code, "add", [{"args": [1, 2], "expect": 3}, {"args": [0.1, 0.2], "expect": 0.3},
                                          {"args": [1, 1], "expect": 3}])
    assert r["ran"] and r["passed"] == 2 and not r["pass"]
    r = q2.run_python_tests("def g(x):\n    raise ValueError('boom')", "g", [{"args": [1], "expect": 1}])
    assert r["passed"] == 0 and "ValueError" in r["per_test"][0]["error"]
    r = q2.run_python_tests("def g(x):\n    return x", "h", [{"args": [1], "expect": 1}])
    assert not r["pass"] and "not defined" in r["error"]
    r = q2.run_python_tests("def g(x):\n    return (x, {x})", "g", [{"args": [2], "expect": [2, [2]]}])
    assert r["pass"], r   # tuples and sets are canonicalised to lists
    r = q2.run_python_tests("def g():\n    return True", "g", [{"args": [], "expect": 1}])
    assert not r["pass"]  # bool is not accepted for an int and vice versa


def test_run_python_tests_timeout():
    r = q2.run_python_tests("def spin():\n    while True:\n        pass", "spin",
                            [{"args": [], "expect": None}], timeout_s=2.0)
    assert r["timeout"] and not r["pass"]


def test_runtime_import_guard(monkeypatch):
    """Even when the static screen is bypassed, the runtime __import__ guard refuses modules
    outside the allow-list, and the banned builtins are absent."""
    monkeypatch.setattr(q2, "screen_code", lambda code: [])
    r = q2.run_python_tests("import os\ndef f():\n    return 1", "f", [{"args": [], "expect": 1}])
    assert not r["pass"] and "ImportError" in (r["error"] or "")
    r = q2.run_python_tests("def f():\n    return open", "f", [{"args": [], "expect": None}])
    assert not r["pass"] and "NameError" in (r["per_test"][0]["error"] or "")


# --------------------------------------------------------------------------------------------
# Chat endpoint client against a fake llama-server
# --------------------------------------------------------------------------------------------
class _FakeServer:
    def __init__(self, answers):
        self.answers = answers
        self.bodies = []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def _send(self, obj, status=200):
                data = json.dumps(obj).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/props":
                    self._send({"model_path": "/m/fake.gguf", "build_info": "b11398-a7b94df",
                                "total_slots": 1, "default_generation_settings": {"n_ctx": 4096}})
                else:
                    self._send({"error": "nope"}, 404)

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n).decode("utf-8"))
                outer.bodies.append((self.path, body))
                prompt = body["messages"][-1]["content"]
                content, reasoning = outer.answers.get(prompt, ("I do not know", ""))
                self._send({"choices": [{"index": 0, "finish_reason": "stop",
                                         "message": {"role": "assistant", "content": content,
                                                     "reasoning_content": reasoning}}],
                            "usage": {"prompt_tokens": 20, "completion_tokens": 3},
                            "timings": {"prompt_n": 20, "prompt_ms": 100.0, "prompt_per_second": 200.0,
                                        "predicted_n": 3, "predicted_ms": 300.0,
                                        "predicted_per_second": 10.0, "cache_n": 0}})

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_chat_client_request_body_and_grading(items, tmp_path):
    by_id = {it["id"]: it for it in items}
    sel = [by_id["neg-en-01"], by_id["ext-en-03"], by_id["code-07"]]
    answers = {
        by_id["neg-en-01"]["prompt"]: ("<|channel>thought\nhmm<channel|>91", ""),
        by_id["ext-en-03"]["prompt"]: ('{"winner": "Maya", "winner_points": 5, "second_place": "Lior", '
                                       '"second_place_points": 3}', "reasoning text"),
        by_id["code-07"]["prompt"]: (by_id["code-07"]["reference"], ""),
    }
    with _FakeServer(answers) as srv:
        client = q2.ChatClient(f"http://127.0.0.1:{srv.port}")
        assert client.props()["build_info"] == "b11398-a7b94df"
        rows = q2.run_items(client, sel)
        assert [r["pass"] for r in rows] == [True, False, True]
        assert rows[0]["visible"] == "91" and rows[0]["thinking_chars"] == 3
        assert rows[1]["score"] == pytest.approx(0.5) and rows[1]["thinking_chars"] == len("reasoning text")
        assert rows[2]["predicted_per_second"] == 10.0
        path, body = srv.bodies[0]
        assert path == "/v1/chat/completions"
        assert body["messages"] == [{"role": "user", "content": by_id["neg-en-01"]["prompt"]}]
        assert body["n_predict"] == by_id["neg-en-01"]["n_predict"]
        assert body["temperature"] == 0.0 and body["cache_prompt"] is False and body["stream"] is False
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert "max_tokens" not in body
        # CLI run + regrade write a full report
        out = tmp_path / "rep.json"
        rc = q2.main(["run", "--url", f"http://127.0.0.1:{srv.port}", "--qa", str(QA_PATH),
                      "--ids", "neg-en-01,ext-en-03,code-07", "--label", "fake", "--out", str(out)])
        assert rc == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["format"] == "pigemma-quality-v2/1" and rep["kind"] == "measured"
    assert rep["summary"]["pass"] == 2 and rep["summary"]["total"] == 3
    assert rep["server"]["props"]["model_path"] == "/m/fake.gguf"
    assert rep["request"]["endpoint"] == "/v1/chat/completions"
    assert rep["qa_set"]["n_items"] == 3 and len(rep["qa_set"]["sha256"]) == 64
    assert all(r["content"] is not None for r in rep["items"])
    out2 = tmp_path / "rep2.json"
    assert q2.main(["regrade", str(out), "--qa", str(QA_PATH), "--out", str(out2)]) == 0
    rep2 = json.loads(out2.read_text(encoding="utf-8"))
    assert [r["pass"] for r in rep2["items"]] == [r["pass"] for r in rep["items"]]
    assert rep2["items"][0]["predicted_per_second"] == 10.0


def test_run_refuses_without_server(tmp_path):
    rc = q2.main(["run", "--url", f"http://127.0.0.1:{_free_port()}", "--label", "none",
                  "--out", str(tmp_path / "x.json"), "--ids", "neg-en-01"])
    assert rc == 4 and not (tmp_path / "x.json").exists()


def test_quality_module_does_not_decide_exactness():
    """Model-quality grading is kept separate from exact-cache correctness."""
    assert not hasattr(q2, "exact_match_vs_plain")
    assert "Exact-cache correctness" in q2.__doc__
