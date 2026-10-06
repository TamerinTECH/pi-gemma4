"""Unit tests for pigemma.bench (no model files; Windows and Linux).

Manifest test field: llama-bench JSON rows parsed (avg_ts/stddev_ts, n_threads); completion
stderr parsed to load_s/pp/tg; buffer_split picks CPU_REPACK and CPU_Mapped MiB; server timings
-> accept_rate and spec_active; a fake local HTTP server (http.server.ThreadingHTTPServer)
exercises server-spec request/response shaping and identical_to_plain without a model; the
variant -> flags mapping as exact strings; membw_sample.json parses.

Fixtures (tests/fixtures/bench/, provenance in its README.md) are real captures from the
llama.cpp a7b94df binaries (desktop, Docker, the real Pi) plus three stdlib test doubles that
stand in for llama-bench / llama-completion / llama-server in the subprocess + CLI tests. The
test marked ``integration`` runs the REAL binaries on a 76 KB random-weight GGUF.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from pigemma import bench, paths, results

FIX = Path(__file__).resolve().parent / "fixtures" / "bench"
SHARED = Path(__file__).resolve().parent / "fixtures"


def fx(name: str) -> str:
    return (FIX / name).read_bytes().decode("utf-8", errors="replace")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# --------------------------------------------------------------------------------------------
# variants
# --------------------------------------------------------------------------------------------
def test_variant_flags_exact_strings() -> None:
    assert bench.variant_flags("default", "bench") == []
    assert bench.variant_flags("default", "run") == []
    assert bench.variant_flags("norepack", "bench") == ["--repack", "0"]
    assert bench.variant_flags("norepack", "run") == ["-nr"]
    assert bench.variant_flags("hybrid_negctrl", "bench") == ["-ot", "ffn_(gate_up|down)_exps=CPU"]
    assert bench.variant_flags("hybrid_negctrl", "run") == ["-ot", "ffn_(gate_up|down)_exps=CPU"]
    assert bench.variant_flags("nommap", "bench") == ["-lm", "none"]
    assert bench.variant_flags("nommap", "run") == ["-lm", "none"]
    assert bench.variant_flags("mlock", "run") == ["-lm", "mmap+mlock"]
    assert bench.variant_flags("kvq8", "run") == ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"]
    assert bench.variant_flags("kvq8", "bench") == ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"]
    assert bench.variant_flags("norepack+kvq8", "bench") == [
        "--repack", "0", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"]
    with pytest.raises(ValueError):
        bench.variant_flags("hybrid")
    with pytest.raises(ValueError):
        bench.variant_flags("norepack", "cli")
    assert bench.SERVER_VARIANT_SPEC == {
        "plain": (None, None), "mtp": ("mtp", None), "mtp-explicit": ("mtp", "draft-mtp"),
        "dflash": ("dflash", None), "dflash-explicit": ("dflash", "draft-dflash"),
        "ngram": (None, "ngram-simple"),
    }


def test_infer_variant_from_rows() -> None:
    assert bench.infer_variant({"repack": True, "load_mode": "auto",
                                "tensor_buft_overrides": "none", "type_k": "f16"}) == "default"
    assert bench.infer_variant({"repack": False}) == "norepack"
    assert bench.infer_variant({"tensor_buft_overrides": "ffn_(gate_up|down)_exps=CPU"}) == "hybrid_negctrl"
    assert bench.infer_variant({"load_mode": "none"}) == "nommap"
    assert bench.infer_variant({"repack": False, "type_k": "q8_0", "type_v": "q8_0"}) == "norepack+kvq8"


# --------------------------------------------------------------------------------------------
# llama-bench JSON
# --------------------------------------------------------------------------------------------
def test_parse_bench_json_real_rows() -> None:
    rows = bench.parse_bench_json(fx("llama_bench_tiny.json"))
    assert [(r["kind"], r["n_threads"]) for r in rows] == [("pp", 2), ("tg", 2), ("pp", 4), ("tg", 4)]
    assert rows[0]["avg_ts"] == pytest.approx(116091.932755)
    assert rows[0]["stddev_ts"] == pytest.approx(13094.433152)
    assert rows[3]["avg_ts"] == pytest.approx(22637.074029)
    assert rows[1]["test"] == "tg16" and rows[0]["test"] == "pp32"
    assert all(r["n_samples"] == 3 for r in rows)
    assert rows[0]["build_number"] == 11398 and rows[0]["build_commit"] == "a7b94df2c"
    assert rows[0]["cpu_info"] == "AMD Ryzen 5 8400F 6-Core Processor"  # trailing spaces stripped


def test_parse_bench_json_truncated_nan_jsonl_and_noise() -> None:
    # real failing run: stdout was just "[" (context creation failed)
    assert bench.parse_bench_json(fx("llama_bench_fail.stdout.txt")) == []
    full = fx("llama_bench_tiny.json")
    # killed mid-way: only complete rows survive
    cut = full[: full.index('"n_threads": 4') + 5]
    rows = bench.parse_bench_json(cut)
    assert [r["n_threads"] for r in rows] == [2, 2]
    # std::to_string(NaN) -> bare -nan / nan
    bad = full.replace('"stddev_ts": 13094.433152', '"stddev_ts": -nan').replace(
        '"avg_ts": 18353.348019', '"avg_ts": nan')
    rows = bench.parse_bench_json(bad)
    assert rows[0]["stddev_ts"] is None and rows[1]["avg_ts"] is None and len(rows) == 4
    # JSONL (-o jsonl) and log noise before the array
    objs = json.loads(full)
    jsonl = "load_backend: loaded CPU backend\n" + "\n".join(json.dumps(o) for o in objs)
    assert len(bench.parse_bench_json(jsonl)) == 4
    assert len(bench.parse_bench_json("load_backend: x\n[ Prompt: 1 ]\n" + full)) == 4
    assert bench.parse_bench_json("") == []


def test_bench_rows_to_fields_build_valid_envelopes() -> None:
    rows = bench.parse_bench_json(fx("llama_bench_tiny.json"))
    model = {"id": "26b-q4", "file": "models/x.gguf", "quant": "Q4_0", "bytes": 1, "sha_prefix": None}
    fields = bench.bench_rows_to_fields(rows, experiment="E0", model=model, variant="norepack",
                                        flags=["--repack", "0"], extra={"argv": ["llama-bench"]})
    assert len(fields) == 4
    docs = [results.build_result(**f) for f in fields]
    pp, tg = docs[0], docs[1]
    assert pp["pp"] == {"tok_s": pytest.approx(116091.932755), "sd": pytest.approx(13094.433152), "n": 3}
    assert "tg" not in pp and "pp" not in tg
    assert tg["tg"]["tok_s"] == pytest.approx(18353.348019)
    assert pp["threads"] == 2 and docs[2]["threads"] == 4
    assert pp["n_ctx"] == 32 and tg["n_ctx"] == 16
    assert pp["llama_build"] == 11398 and pp["llama_commit"] == "a7b94df2c"
    assert pp["kind"] == "measured" and pp["flags"] == ["--repack", "0"]
    assert pp["extra"]["test"] == "pp32" and pp["extra"]["samples_ts"] == [100978, 123267, 124031]
    for d in docs:
        assert results.check_result(d)[0] == []
    # the inferred variant when none is given
    f2 = bench.bench_rows_to_fields(rows[:1], experiment="E8", model=None, variant=None, flags=[])
    assert f2[0]["variant"] == "default"
    # a shallow-clone build count is noted, not hidden
    r = dict(rows[0], build_number=1, build_commit="a7b94df")
    f3 = bench.bench_rows_to_fields([r], experiment="E8", model=None, variant=None, flags=[])
    assert any("shallow-clone" in n for n in f3[0]["notes"])


# --------------------------------------------------------------------------------------------
# llama-completion stderr
# --------------------------------------------------------------------------------------------
def test_parse_completion_stderr_real_logs() -> None:
    p = bench.parse_completion_stderr(fx("completion_26b_smoke.stderr.txt"))  # real 26B run
    assert p["load_s"] == pytest.approx(12.748)
    assert p["load_ms"] == pytest.approx(12747.86)
    assert p["pp_tok_s"] == pytest.approx(50.92) and p["n_prompt"] == 61
    assert p["tg_tok_s"] == pytest.approx(8.71) and p["n_eval"] == 74
    assert p["total_ms"] == pytest.approx(9736.62) and p["n_total"] == 135
    assert (p["n_ctx"], p["n_batch"], p["n_predict"], p["n_threads"]) == (2048, 2048, 160, 6)
    assert p["buffer_split"] == {}  # default verbosity hides the buffer lines
    pi = bench.parse_completion_stderr(fx("completion_pi_31b.stderr.txt"))  # real Pi 5 run
    assert pi["tg_tok_s"] == pytest.approx(0.53) and pi["pp_tok_s"] == pytest.approx(0.63)
    assert pi["load_s"] == pytest.approx(4.861) and pi["n_threads"] == 4 and pi["n_eval"] == 47
    t = bench.parse_completion_stderr(fx("completion_tiny_lv4.stderr.txt"))
    assert t["tg_tok_s"] == pytest.approx(16358.46) and t["n_eval"] == 23
    assert t["pp_tok_s"] == pytest.approx(115131.58) and t["n_prompt"] == 70
    assert t["buffers_other"]["KV_CPU_mib"] == pytest.approx(0.25)
    assert t["has_perf"] is True
    assert bench.parse_completion_stderr("garbage")["has_perf"] is False


def test_parse_buffer_split_repack_vs_mapped() -> None:
    assert bench.parse_buffer_split(fx("completion_tiny_lv4.stderr.txt")) == {
        "CPU_Mapped_mib": 0.05, "CPU_REPACK_mib": 0.05}
    nr = bench.parse_buffer_split(fx("completion_tiny_norepack.stderr.txt"))
    assert nr == {"CPU_Mapped_mib": 0.06} and "CPU_REPACK_mib" not in nr
    text = ("0.00.100.000 I load_tensors:   CPU_REPACK model buffer size = 12251.53 MiB\n"
            "0.00.100.001 I load_tensors:   CPU_Mapped model buffer size =  1534.50 MiB\n"
            "0.00.100.002 I load_tensors:   CPU_Mapped model buffer size =    10.00 MiB\n"
            "0.00.100.003 I load_tensors:          CPU model buffer size =     1.25 MiB\n"
            "0.00.100.004 I common_memory_breakdown_print: |   - CPU_REPACK  |  0 = 0 + 0 |\n")
    assert bench.parse_buffer_split(text) == {
        "CPU_REPACK_mib": 12251.53, "CPU_Mapped_mib": 1544.5, "CPU_mib": 1.25}


def test_failure_note_names_the_cause() -> None:
    n = bench.failure_note(fx("llama_bench_fail.stderr.txt"), extra="llama-bench exit code 1")
    assert n.startswith("failed: llama-bench exit code 1;")
    assert "does not divide n_embd_head_k=16" in n and "failed to create context" in n
    assert "load_backend" not in n
    assert bench.failure_note("") == "failed: (no stderr output)"


# --------------------------------------------------------------------------------------------
# llama-server timings
# --------------------------------------------------------------------------------------------
def _server_fixture(name: str) -> dict[str, Any]:
    return json.loads(fx(name))


def test_parse_timings_and_aggregate_real_responses() -> None:
    d = bench.parse_timings(_server_fixture("server_moe_dflash.json"))
    assert (d["draft_n"], d["draft_n_accepted"]) == (321, 91)
    assert d["accept_rate"] == pytest.approx(91 / 321)
    assert d["speculative_types"] == "draft-dflash" and d["draft_reported"] is True
    assert d["predicted_per_second"] == pytest.approx(11.834139363106837)
    m = bench.parse_timings(_server_fixture("server_moe_mtp.json"))
    assert m["draft_n"] is None and m["draft_reported"] is False and m["accept_rate"] is None
    assert m["speculative_types"] == "none"
    # MTP configured (-md) but never drafted: measured 0, inactive
    a = bench.aggregate_timings([m], spec_configured=True)
    assert (a["draft_n"], a["draft_n_accepted"], a["accept_rate"], a["spec_active"]) == (0, 0, None, False)
    # DFlash: active, sum-based acceptance
    a = bench.aggregate_timings([d, d], spec_configured=True)
    assert a["spec_active"] is True and a["draft_n"] == 642
    assert a["accept_rate"] == pytest.approx(182 / 642)
    # server convention: n - 1 decode steps (first token comes from the prompt logits)
    assert a["tg"]["tok_s"] == pytest.approx(11.834139363106837) and a["tg"]["n"] == 2
    assert a["tg"]["tokens"] == 400 and a["tg"]["steps"] == 398
    assert a["pp"]["tok_s"] == pytest.approx(61.713025917342854)
    # one drafting + one non-drafting request -> not active on every request
    a = bench.aggregate_timings([d, m], spec_configured=True)
    assert a["spec_active"] is False and a["n_drafting"] == 1
    # plain: no speculative fields at all; inference from the records
    p = bench.parse_timings(_server_fixture("server_moe_plain.json"))
    a = bench.aggregate_timings([p], spec_configured=False)
    assert a["draft_n"] is None and a["spec_active"] is None and a["accept_rate"] is None
    assert bench.aggregate_timings([d])["spec_configured"] is True
    assert bench.aggregate_timings([p])["spec_configured"] is False
    # nothing succeeded -> nothing measured
    a = bench.aggregate_timings([{"error": "boom"}], spec_configured=True)
    assert a["draft_n"] is None and a["spec_active"] is None and a["tg"] is None


def test_identical_to_plain_on_real_outputs() -> None:
    plain = {"q": _server_fixture("server_moe_plain.json")["content"]}
    mtp = [{"id": "q", "content": _server_fixture("server_moe_mtp.json")["content"]}]
    dfl = [{"id": "q", "content": _server_fixture("server_moe_dflash.json")["content"]}]
    c = bench.compare_to_plain(mtp, plain)
    assert c["identical_to_plain"] == 1.0 and c["similarity_by_prompt"] == {"q": 1.0}
    c = bench.compare_to_plain(dfl, plain)  # the recorded DFlash run diverged after ~10 tokens
    assert c["identical_to_plain"] == 0.0 and 0.3 < c["similarity_by_prompt"]["q"] < 0.95
    assert c["n_similar_ge_095"] == 0
    assert bench.compare_to_plain(dfl, None)["identical_to_plain"] is None
    # thinking channel is ignored for the comparison
    c = bench.compare_to_plain([{"id": "q", "content": "<|channel>thought\nabc<channel|>Same."}],
                               {"q": "Same."})
    assert c["identical_to_plain"] == 1.0


def test_server_fields_spec_type_effective_and_validity() -> None:
    model = {"id": "26b-q4", "file": "m.gguf", "quant": "Q4_0", "bytes": 1, "sha_prefix": None}
    d = bench.parse_timings(_server_fixture("server_moe_dflash.json"))
    rec = dict(d, id="q", content=_server_fixture("server_moe_dflash.json")["content"], error=None)
    plain = {"q": _server_fixture("server_moe_plain.json")["content"]}
    kw = dict(experiment="E1", model=model, flags=["-md", "d.gguf"], threads=6, n_ctx=4096,
              draft={"id": "26b-dflash", "path": "d.gguf", "bytes": 1}, load_time_s=12.5,
              mem={"rss_peak_bytes": 2_000_000_000, "private_peak_bytes": 900_000_000,
                   "mapped_peak_bytes": 1_500_000_000}, plain=plain, plain_source="x.json")
    f = bench.server_fields([rec], variant="dflash", server_variant="dflash", spec_type=None, **kw)
    doc = results.build_result(**f)
    assert doc["spec_type_effective"] == "inferred" and doc["spec_active"] is True
    assert doc["accept_rate"] == pytest.approx(91 / 321) and doc["draft_n"] == 321
    assert doc["identical_to_plain"] == 0.0 and doc["load_s"] == 12.5
    assert doc["rss_peak_mb"] == 2000.0 and doc["private_peak_mb"] == 900.0
    assert doc["tg"]["tok_s"] == pytest.approx(11.834139363106837) and doc["tg"]["n"] == 1
    assert any("lossless" in n for n in doc["notes"])
    assert doc["extra"]["requests"][0]["thinking_chars"] == 0
    f = bench.server_fields([rec], variant="dflash-explicit", server_variant="dflash-explicit",
                            spec_type="draft-dflash", **kw)
    assert f["spec_type_effective"] == "explicit:draft-dflash"
    m = bench.parse_timings(_server_fixture("server_moe_mtp.json"))
    rec_m = dict(m, id="q", content=_server_fixture("server_moe_mtp.json")["content"], error=None)
    f = bench.server_fields([rec_m], variant="mtp", server_variant="mtp", spec_type=None, **kw)
    assert f["spec_type_effective"] == "none" and f["spec_active"] is False and f["draft_n"] == 0
    assert f["identical_to_plain"] == 1.0
    assert any("no token was drafted" in n for n in f["notes"])
    p = bench.parse_timings(_server_fixture("server_moe_plain.json"))
    rec_p = dict(p, id="q", content=plain["q"], error=None)
    kw_plain = dict(kw, plain=None, plain_source=None, draft=None, flags=[])
    f = bench.server_fields([rec_p], variant="plain", server_variant="plain", spec_type=None, **kw_plain)
    doc = results.build_result(**f)
    assert doc["spec_active"] is None and doc["identical_to_plain"] is None
    assert doc["draft_n"] is None and doc["spec_type_effective"] == "none"
    assert doc["coherent"] is True and doc["tg"]["tok_s"] == pytest.approx(16.030119224615884)
    # a variant that never ran records failure and nothing measured
    f = bench.server_fields([], variant="mtp", server_variant="mtp", spec_type=None,
                            **dict(kw, load_time_s=None, mem=None), failure="failed: no draft")
    doc = results.build_result(**f)
    assert "tg" not in doc and doc["draft_n"] is None and doc["spec_active"] is None
    assert doc["spec_type_effective"] is None and doc["notes"][0] == "failed: no draft"


# --------------------------------------------------------------------------------------------
# text helpers, membw
# --------------------------------------------------------------------------------------------
def test_chat_wrap_is_the_gemma4_template() -> None:
    assert bench.chat_wrap("  Hi there \n") == (
        "<|turn>user\nHi there<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
    assert bench.chat_wrap("x", thinking=True) == "<|turn>user\nx<turn|>\n<|turn>model\n"


def test_strip_thinking_and_coherence_fallbacks() -> None:
    vis, th = bench._strip_thinking_fallback(_server_fixture("server_moe_plain.json")["content"])
    assert vis.startswith("In a standard Transformer model") and th == 0
    vis, th = bench._strip_thinking_fallback("<|channel>thought\nlet me think<channel|>Answer. [end of text]")
    assert vis == "Answer." and th == len("let me think")
    assert bench._strip_thinking_fallback("<|channel>thought\nnever closed")[0] == ""
    assert bench._strip_thinking_fallback("<channel|>Shalom <end_of_turn>")[0] == "Shalom"
    assert bench._coherent_fallback("Experts are routed per token. שלום לכולם.")[0] is True
    ok, why = bench._coherent_fallback("the cat sat on " * 10)
    assert ok is False and "repeated" in why
    assert bench._coherent_fallback("��׵׵}}}}")[0] is False
    assert bench._coherent_fallback("")[0] is False
    ok, reason, impl = bench.coherence(_server_fixture("server_moe_plain.json")["content"])
    assert ok is True and impl in ("quality", "bench-fallback")


def test_parse_membw_json() -> None:
    m = bench.parse_membw_json(fx("membw_sample.json"))
    assert m["read_gbps_by_threads"] == {"1": 30.608, "2": 35.703}
    assert m["copy_gbps"] == pytest.approx(33.207) and m["triad_gbps"] == pytest.approx(28.38)
    assert m["source"] == "measured" and m["size_mb"] == 64
    shared = SHARED / "membw_sample.json"
    if shared.is_file():  # the conftest agent's copy, when it exists
        assert bench.parse_membw_json(shared.read_text(encoding="utf-8"))["read_gbps_by_threads"]


@pytest.mark.parametrize("name", ["llama_bench_sample.json", "completion_stderr_sample.txt",
                                  "server_timings_sample.json"])
def test_shared_fixtures_parse_when_present(name: str) -> None:
    p = SHARED / name
    if not p.is_file():
        pytest.skip(f"shared fixture {name} not checked in (tests/conftest.py agent)")
    text = p.read_text(encoding="utf-8", errors="replace")
    if name.startswith("llama_bench"):
        rows = bench.parse_bench_json(text)
        assert rows and all(r["n_threads"] for r in rows)
    elif name.startswith("completion"):
        assert bench.parse_completion_stderr(text)["has_perf"]
    else:
        obj = json.loads(text)
        obj = obj[0] if isinstance(obj, list) else obj
        assert bench.parse_timings(obj)["predicted_per_second"] is not None


def test_load_prompts_formats(tmp_path: Path) -> None:
    items = bench.load_prompts(paths.PROMPTS_DIR / "gen_prompts.json")
    assert len(items) == 8 and items[0]["id"] == "gen-en-explain-01" and items[0]["max_tokens"] == 256
    p = tmp_path / "p.json"
    p.write_text(json.dumps(["a", "b"]), encoding="utf-8")
    assert [i["id"] for i in bench.load_prompts(p)] == ["p00", "p01"]
    p.write_text(json.dumps({"prompts": [{"prompt": "x", "id": "k"}]}), encoding="utf-8")
    assert bench.load_prompts(p)[0]["id"] == "k"
    p.write_text("[]", encoding="utf-8")
    with pytest.raises(bench.BenchError):
        bench.load_prompts(p)
    with pytest.raises(bench.MissingInputError):
        bench.load_prompts(tmp_path / "missing.json")


def test_process_monitor_reads_this_process() -> None:
    mon = bench.ProcessMonitor(os.getpid(), interval=0.05)
    mon.start()
    mon.snapshot("x")
    time.sleep(0.15)
    mon.stop()
    s = mon.summary()
    assert s["source"] in ("procfs", "win32", "psutil")
    assert s["rss_peak_bytes"] and s["rss_peak_bytes"] > 1_000_000
    assert "x" in mon.marks and s["n_samples"] >= 2


# --------------------------------------------------------------------------------------------
# fake HTTP server (in-process): request/response shaping + identical_to_plain
# --------------------------------------------------------------------------------------------
class _FakeHandler(BaseHTTPRequestHandler):
    bodies: list[dict[str, Any]] = []
    drafting = False
    diverge = "ZZZ"

    def log_message(self, *a: object) -> None:
        pass

    def _send(self, code: int, obj: object) -> None:
        b = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self) -> None:  # noqa: N802
        self._send(200, {"status": "ok"})

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).bodies.append(body)
        prompt = body["prompt"]
        text = "<|channel>thought\n<channel|>Answer about " + prompt[12:30]
        n = body["n_predict"]
        timings = {"prompt_n": 10, "prompt_ms": 100.0, "prompt_per_second": 100.0,
                   "predicted_n": n, "predicted_ms": 50.0 * n,  # server rate uses n - 1 steps
                   "predicted_per_second": (n - 1) / (0.05 * n)}
        if type(self).drafting:
            timings.update(draft_n=8, draft_n_accepted=6)
            if type(self).diverge in prompt:
                text += " but different"
        self._send(200, {"content": text, "timings": timings, "stop_type": "limit",
                         "generation_settings": {"speculative.types":
                                                 "draft-mtp" if type(self).drafting else "none"}})


def test_fake_http_server_request_shaping_and_identity() -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        client = bench.ServerClient("127.0.0.1", srv.server_address[1])
        assert client.health() == (True, "ok")
        items = [{"id": "a", "prompt": "Explain MoE."}, {"id": "b", "prompt": "Story ZZZ.", "max_tokens": 7}]
        _FakeHandler.bodies = []
        _FakeHandler.drafting = False
        plain = bench.run_prompt_set(client, items, n_predict=None, warmup_tokens=16,
                                     default_n_predict=12)
        b = _FakeHandler.bodies
        assert len(b) == 3  # warm-up + 2
        assert b[0]["n_predict"] == 16 and b[0]["prompt"] == bench.chat_wrap("Explain MoE.")
        assert b[1] == {"prompt": bench.chat_wrap("Explain MoE."), "n_predict": 12,
                        "temperature": 0.0, "cache_prompt": False}
        assert b[2]["n_predict"] == 7
        assert [r["id"] for r in plain] == ["a", "b"] and all(r["error"] is None for r in plain)
        assert plain[0]["predicted_per_second"] == pytest.approx(11 / 0.6)
        assert plain[0]["draft_n"] is None
        _FakeHandler.drafting = True
        spec = bench.run_prompt_set(client, items, n_predict=5, warmup_tokens=0)
        assert len(_FakeHandler.bodies) == 5 and _FakeHandler.bodies[-1]["n_predict"] == 5
        ref = {r["id"]: r["content"] for r in plain}
        # n_predict differs between the two runs here, which only shortens the fake's text
        f = bench.server_fields(spec, experiment="E1", model=None, variant="mtp", flags=[],
                                threads=6, n_ctx=4096, server_variant="mtp", spec_type=None,
                                draft={"id": "26b-mtp", "path": "x", "bytes": 1},
                                load_time_s=1.0, mem=None, plain=ref, plain_source="test")
        doc = results.build_result(**f)
        assert doc["identical_to_plain"] == 0.5
        assert doc["extra"]["identical_by_prompt"] == {"a": True, "b": False}
        assert doc["spec_active"] is True and doc["accept_rate"] == pytest.approx(0.75)
        assert doc["spec_type_effective"] == "inferred"
        assert doc["tg"] == {"tok_s": pytest.approx(16.0), "sd": 0.0, "n": 2}
    finally:
        srv.shutdown()
        srv.server_close()
    # a dead server -> recorded request errors, not an exception
    recs = bench.run_prompt_set(bench.ServerClient("127.0.0.1", free_port()),
                                [{"id": "a", "prompt": "x"}], n_predict=4, warmup_tokens=0,
                                request_timeout=0.3)
    assert recs[0]["error"] and recs[0]["content"] is None


# --------------------------------------------------------------------------------------------
# CLI end to end with the test doubles (subprocess, lock, monitor, result files)
# --------------------------------------------------------------------------------------------
@pytest.fixture()
def fake_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("PIGEMMA_LOCK_FILE", str(tmp_path / "bench.lock"))
    for k in ("PIGEMMA_LOCK_HELD", "PIGEMMA_MIN_FREE_GB", "PIGEMMA_EMULATED", "FAKE_RC",
              "FAKE_HANG", "FAKE_SERVER_MODE", "FAKE_DIVERGE", "FAKE_DRAFT_N", "FAKE_INFERRED"):
        monkeypatch.delenv(k, raising=False)
    fakes = {"llama-bench": "fake_llama_bench.py", "llama-completion": "fake_llama_completion.py",
             "llama-server": "fake_llama_server.py"}
    real = bench._tool_cmd

    def tool_cmd(tool: str, override: Any = None) -> list[str]:
        if tool in fakes and override is None:
            return [sys.executable, str(FIX / fakes[tool])]
        return real(tool, override)
    monkeypatch.setattr(bench, "_tool_cmd", tool_cmd)
    monkeypatch.setattr(bench, "HEALTH_POLL_S", 0.05)
    monkeypatch.setattr(bench, "PORT_PROBE_TIMEOUT_S", 0.05)
    model = tmp_path / "fake-26B-q4_0.gguf"
    model.write_bytes(b"GGUF" + b"\0" * 1000)
    draft = tmp_path / "fake-dflash-Q8_0.gguf"
    draft.write_bytes(b"GGUF" + b"\0" * 100)
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps([
        {"id": "a", "prompt": "Explain experts."}, {"id": "b", "prompt": "Tell a ZZZ story."},
        {"id": "c", "prompt": "Count to three."}]), encoding="utf-8")
    common = ["--min-free-gb", "0", "--no-sha", "--idle-check-s", "0"]
    return {"tmp": tmp_path, "model": model, "draft": draft, "prompts": prompts,
            "out": tmp_path / "out", "common": common}


def _docs(out: Path) -> list[dict[str, Any]]:
    return results.load_results(out, recursive=False)


def test_cli_matrix_with_fake_llama_bench(fake_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
                                          capsys: pytest.CaptureFixture[str]) -> None:
    e = fake_env
    rc = bench.main(["matrix", "--model", str(e["model"]), "--threads", "2,4", "--pp", "32",
                     "--tg", "16", "-r", "3", "--variants", "default,norepack",
                     "--out", str(e["out"])] + e["common"])
    assert rc == 0
    docs = _docs(e["out"])
    assert len(docs) == 8 and capsys.readouterr().out.count(".json") == 8
    nr = [d for d in docs if d["variant"] == "norepack"]
    assert all(d["flags"] == ["--repack", "0"] for d in nr)
    assert sorted((d["threads"], "pp" if "pp" in d else "tg") for d in nr) == [
        (2, "pp"), (2, "tg"), (4, "pp"), (4, "tg")]
    assert all(d["model"]["id"] == "fake-26B-q4_0" and d["model"]["quant"] == "Q4_0" for d in docs)
    err_log = (paths.REPO_ROOT / nr[0]["extra"]["raw_stderr"]) if not Path(
        nr[0]["extra"]["raw_stderr"]).is_absolute() else Path(nr[0]["extra"]["raw_stderr"])
    assert "-t 2,4 -p 32 -n 16 -r 3 -o json --repack 0" in err_log.read_text(encoding="utf-8")
    # failing llama-bench: one failure result per expected (threads, test), exit 1
    monkeypatch.setenv("FAKE_BENCH_JSON", str(FIX / "llama_bench_fail.stdout.txt"))
    monkeypatch.setenv("FAKE_BENCH_STDERR", str(FIX / "llama_bench_fail.stderr.txt"))
    monkeypatch.setenv("FAKE_RC", "1")
    out2 = e["tmp"] / "out2"
    rc = bench.main(["matrix", "--model", str(e["model"]), "--threads", "2,4", "--pp", "32",
                     "--tg", "16", "--variants", "kvq8", "--out", str(out2)] + e["common"])
    assert rc == 1
    docs = _docs(out2)
    assert len(docs) == 4 and all("pp" not in d and "tg" not in d for d in docs)
    assert all("failed to create context" in d["notes"][0] for d in docs)
    assert {d["variant"] for d in docs} == {"kvq8"}


def test_cli_completion_with_fake_llama_completion(fake_env: dict[str, Any],
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    e = fake_env
    monkeypatch.setenv("FAKE_STDERR", str(FIX / "completion_tiny_norepack.stderr.txt"))
    rc = bench.main(["completion", "--model", str(e["model"]), "--prompt", "Why are MoEs fast?",
                     "-n", "64", "-t", "2", "--variant", "norepack", "--chat", "-c", "2048",
                     "--out", str(e["out"])] + e["common"])
    assert rc == 0
    (d,) = _docs(e["out"])
    assert d["experiment"] in ("E0", "E8") and d["variant"] == "norepack" and d["flags"] == ["-nr"]
    assert d["tg"] == {"tok_s": pytest.approx(16548.46), "sd": None, "n": 1}
    assert d["pp"]["tok_s"] == pytest.approx(165945.17) and d["load_s"] == pytest.approx(0.004)
    assert d["buffer_split"] == {"CPU_Mapped_mib": 0.06}
    assert d["n_ctx"] == 2048 and d["threads"] == 2
    assert d["coherent"] is True and d["output_sample"].startswith("Experts are chosen")
    ex = d["extra"]
    assert ex["n_eval_runs"] == 63 and ex["n_prompt_tokens"] == 115
    mk = ex["marks_s"]
    assert mk["prompt_start"] <= mk["gen_start"] <= mk["gen_end"]
    assert d["rss_peak_mb"] and d["rss_peak_mb"] >= 30  # the fake touches 40 MB
    raw_err = Path(ex["raw_stderr"])
    raw_err = raw_err if raw_err.is_absolute() else paths.REPO_ROOT / raw_err
    log_text = raw_err.read_text(encoding="utf-8")
    assert "-n 64 --temp 0 -t 2 -c 2048 -no-cnv --no-display-prompt -lv 4 -nr" in log_text
    assert 'fake-prompt: "<|turn>user\\nWhy are MoEs fast?<turn|>\\n<|turn>model\\n' in log_text
    if sys.platform.startswith("linux"):
        assert d["read_bytes_per_token"] is not None and d["majflt_per_token"] is not None
        assert ex["io"]["gen"]["wall_s"] is not None
    else:
        assert d["read_bytes_per_token"] is None
    # non-zero exit -> failed result, no tok/s, exit 1
    monkeypatch.setenv("FAKE_RC", "3")
    out2 = e["tmp"] / "o2"
    rc = bench.main(["completion", "--model", str(e["model"]), "--prompt", "x", "-n", "8",
                     "--out", str(out2)] + e["common"])
    assert rc == 1
    (d,) = _docs(out2)
    assert "tg" not in d and d["notes"][0].startswith("failed: llama-completion exit code 3")


def test_cli_completion_timeout_kills_only_its_child(fake_env: dict[str, Any],
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    e = fake_env
    monkeypatch.setenv("FAKE_HANG", "1")
    t0 = time.monotonic()
    rc = bench.main(["completion", "--model", str(e["model"]), "--prompt", "x", "-n", "8",
                     "--timeout", "0.8", "--out", str(e["out"])] + e["common"])
    assert rc == 1 and time.monotonic() - t0 < 25
    (d,) = _docs(e["out"])
    assert "timed out" in d["notes"][0] and d["extra"]["timed_out"] is True


def test_cli_server_spec_with_fake_server(fake_env: dict[str, Any],
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    e = fake_env
    port = free_port()
    monkeypatch.setenv("FAKE_DIVERGE", "ZZZ")
    monkeypatch.setenv("FAKE_LOG", str(e["tmp"] / "requests.jsonl"))
    rc = bench.main(["server-spec", "--model", str(e["model"]), "--variant", "dflash,plain",
                     "--draft", str(e["draft"]), "--prompts", str(e["prompts"]), "-n", "20",
                     "-t", "2", "--port", str(port), "--out", str(e["out"])] + e["common"])
    assert rc == 0
    docs = {d["variant"]: d for d in _docs(e["out"])}
    assert set(docs) == {"plain", "dflash"}
    p, s = docs["plain"], docs["dflash"]
    assert p["experiment"] == "E1" and p["spec_active"] is None and p["identical_to_plain"] is None
    # the fake server answers /health 503 for FAKE_LOAD_S (0.1 s) after its own start, so the
    # measured health wait can never be shorter than that (0.2 was a timing guess that flaked
    # at 0.191 s on a loaded machine)
    assert p["load_s"] and p["load_s"] >= 0.1 and p["tg"]["n"] == 3
    assert s["flags"][0] == "-md" and s["spec_type_effective"] == "inferred"
    assert s["spec_active"] is True and s["draft_n"] == 36 and s["draft_n_accepted"] == 27
    assert s["accept_rate"] == pytest.approx(0.75)
    assert s["identical_to_plain"] == pytest.approx(2 / 3)
    assert s["extra"]["identical_by_prompt"] == {"a": True, "b": False, "c": True}
    assert s["extra"]["plain_ref"] == "this invocation"
    assert any("lossless" in n for n in s["notes"])
    assert s["extra"]["server_returncode"] is not None  # our child was stopped
    assert p["extra"]["server_returncode"] is not None
    sent = [json.loads(x) for x in (e["tmp"] / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(sent) == 8  # (1 warm-up + 3) per variant
    assert sent[0]["n_predict"] == 16 and sent[1]["n_predict"] == 20
    assert sent[1]["prompt"] == bench.chat_wrap("Explain experts.")
    assert all(b["temperature"] == 0.0 and b["cache_prompt"] is False for b in sent)
    # second invocation: mtp never drafts -> plain found in --out, --auto-explicit adds mtp-explicit
    monkeypatch.delenv("FAKE_DIVERGE")
    monkeypatch.setenv("FAKE_DRAFT_N", "0")
    monkeypatch.setenv("FAKE_INFERRED", "none")
    rc = bench.main(["server-spec", "--model", str(e["model"]), "--variant", "mtp",
                     "--auto-explicit", "--draft", str(e["draft"]), "--prompts", str(e["prompts"]),
                     "-n", "20", "-t", "2", "--port", str(port), "--warmup-tokens", "0",
                     "--out", str(e["out"])] + e["common"])
    assert rc == 0
    docs = {d["variant"]: d for d in _docs(e["out"])}
    m, mx = docs["mtp"], docs["mtp-explicit"]
    assert m["spec_active"] is False and m["draft_n"] == 0 and m["spec_type_effective"] == "none"
    assert m["identical_to_plain"] == 1.0 and m["extra"]["plain_ref"].endswith(".json")
    assert any("no token was drafted" in n for n in m["notes"])
    assert mx["spec_type_effective"] == "explicit:draft-mtp"
    assert mx["flags"][-2:] == ["--spec-type", "draft-mtp"]


def test_cli_server_spec_refuses_busy_port_and_reports_crash(
        fake_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    e = fake_env
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        rc = bench.main(["server-spec", "--model", str(e["model"]), "--prompts", str(e["prompts"]),
                         "--port", str(port), "--out", str(e["out"])] + e["common"])
    assert rc == 4
    assert not e["out"].exists() or not list(e["out"].glob("*.json"))
    monkeypatch.setenv("FAKE_SERVER_MODE", "crash")
    rc = bench.main(["server-spec", "--model", str(e["model"]), "--prompts", str(e["prompts"]),
                     "--port", str(free_port()), "--out", str(e["out"])] + e["common"])
    assert rc == 1
    (d,) = _docs(e["out"])
    assert "tg" not in d and "exited with code 1" in d["notes"][0]
    assert "fake crash" in d["notes"][0]
    monkeypatch.setenv("FAKE_SERVER_MODE", "never-healthy")
    out2 = e["tmp"] / "o2"
    rc = bench.main(["server-spec", "--model", str(e["model"]), "--prompts", str(e["prompts"]),
                     "--port", str(free_port()), "--health-timeout", "0.6", "--out", str(out2)]
                    + e["common"])
    assert rc == 1
    (d,) = _docs(out2)
    assert "not healthy after 1 s" in d["notes"][0] or "not healthy after 0 s" in d["notes"][0]


def test_cli_server_spec_rounds_interleave(fake_env: dict[str, Any],
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    e = fake_env
    rc = bench.main(["server-spec", "--model", str(e["model"]), "--variant", "mtp-explicit,plain",
                     "--rounds", "2", "--draft", str(e["draft"]), "--prompts", str(e["prompts"]),
                     "--max-prompts", "1", "-n", "8", "--warmup-tokens", "0",
                     "--port", str(free_port()), "--out", str(e["out"])] + e["common"])
    assert rc == 0
    docs = sorted(_docs(e["out"]), key=lambda d: d["created_at"] + d["run_id"])
    order = sorted(((d["extra"]["round"], d["variant"]) for d in docs))
    assert order == [(1, "mtp-explicit"), (1, "plain"), (2, "mtp-explicit"), (2, "plain")]
    assert all(d["extra"]["rounds"] == 2 for d in docs)
    assert all(d["spec_type_effective"] == "explicit:draft-mtp"
               for d in docs if d["variant"] == "mtp-explicit")
    assert all(d["identical_to_plain"] == 1.0 for d in docs if d["variant"] == "mtp-explicit")


def test_idle_check_note_and_refusal(fake_env: dict[str, Any]) -> None:
    info = bench.measure_background_cpu(0.4, step=0.1)
    assert info["source"] in ("procfs", "win32", "psutil")
    assert info["mean_pct"] is not None and 0.0 <= info["mean_pct"] <= 100.0
    assert info["n_windows"] >= 2
    e = fake_env
    busy = ["--min-free-gb", "0", "--no-sha", "--idle-check-s", "0.3", "--idle-max-pct", "-1"]
    rc = bench.main(["completion", "--model", str(e["model"]), "--prompt", "x", "-n", "8",
                     "--out", str(e["out"])] + busy)
    assert rc == 0
    (d,) = _docs(e["out"])
    bg = d["extra"]["background_cpu"]
    assert bg["idle"] is False and bg["max_allowed_pct"] == -1.0 and bg["seconds"] >= 0.25
    assert any("not an idle-machine timing" in n for n in d["notes"])
    out2 = e["tmp"] / "o2"
    rc = bench.main(["completion", "--model", str(e["model"]), "--prompt", "x", "-n", "8",
                     "--require-idle", "--out", str(out2)] + busy)
    assert rc == 4
    assert not out2.exists() or not list(out2.glob("*.json"))
    ns = bench._build_parser().parse_args(["completion", "-m", "x", "-p", "y"])
    assert ns.idle_check_s == 10.0 and ns.idle_max_pct == 5.0  # FACTS: < 5 % for 10 s
    assert bench.idle_check(argparse_ns(idle_check_s=0), "x") is None


def argparse_ns(**kw: Any) -> Any:
    import argparse
    return argparse.Namespace(**kw)


def test_cli_lock_busy_refuses(fake_env: dict[str, Any]) -> None:
    e = fake_env
    with paths.bench_lock(0, cmd="other job"):
        rc = bench.main(["completion", "--model", str(e["model"]), "--prompt", "x",
                         "--out", str(e["out"])] + e["common"])
    assert rc == 4 and not e["out"].exists()


def test_cli_parse_subcommands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = bench.main(["parse-bench", str(FIX / "llama_bench_tiny.json"), "--no-sha",
                     "--experiment", "E8", "--out", str(tmp_path)])
    assert rc == 0
    docs = results.load_results(tmp_path)
    assert len(docs) == 4 and {d["variant"] for d in docs} == {"default"}
    assert {d["experiment"] for d in docs} == {"E8"} and docs[0]["model"]["id"] == "tiny-q4_0"
    capsys.readouterr()
    assert bench.main(["parse-bench", str(FIX / "llama_bench_tiny.json"), "--no-sha",
                       "--variant", "norepack"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert len(printed) == 4 and printed[0]["flags"] == ["--repack", "0"]
    assert bench.main(["parse-bench", str(FIX / "llama_bench_fail.stdout.txt")]) == 1
    assert bench.main(["parse-bench", str(tmp_path / "nope.json")]) == 3
    capsys.readouterr()
    assert bench.main(["parse-completion", str(FIX / "completion_pi_31b.stderr.txt")]) == 0
    assert json.loads(capsys.readouterr().out)["tg_tok_s"] == pytest.approx(0.53)
    out = tmp_path / "pc"
    assert bench.main(["parse-completion", str(FIX / "completion_26b_smoke.stderr.txt"),
                       "--model", "26b-q4", "--no-sha", "--threads", "6", "--out", str(out)]) == 0
    (d,) = results.load_results(out)
    assert d["model"]["id"] == "26b-q4" and d["tg"]["tok_s"] == pytest.approx(8.71)
    assert d["load_s"] == pytest.approx(12.748) and "rss_peak_mb" not in d
    capsys.readouterr()
    assert bench.main(["parse-server", str(FIX / "server_moe_dflash.json"),
                       str(FIX / "server_moe_mtp.json")]) == 0
    agg = json.loads(capsys.readouterr().out)["aggregate"]
    assert agg["draft_n"] == 321 and agg["spec_active"] is False and agg["n_drafting"] == 1
    assert bench.main(["variants", "--json"]) == 0
    v = json.loads(capsys.readouterr().out)
    assert v["load_variants"]["norepack"]["bench"] == ["--repack", "0"]
    assert bench.main(["matrix", "--model", "26b-q4", "--variants", "bogus", "--dry-run"]) == 2
    assert bench.main(["nonsense"]) == 2


def test_cli_dry_run_prints_exact_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert bench.main(["--log-level", "WARNING", "matrix", "--model", "26b-q4", "--threads", "4,6",
                       "--variants", "default,hybrid_negctrl", "--dry-run"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2 and lines[0].endswith("-t 4,6 -p 128 -n 32 -r 3 -o json")
    assert "ffn_(gate_up|down)_exps=CPU" in lines[1]
    assert bench.main(["--log-level", "WARNING", "server-spec", "--model", "26b-q4", "--variant",
                       "ngram,plain,dflash-explicit", "--dry-run", "--max-prompts", "1"]) == 0
    out = capsys.readouterr().out
    cmds = [ln for ln in out.splitlines() if not ln.startswith("  then:")]
    assert "--spec-type" not in cmds[0] and "-md" not in cmds[0]  # plain first
    assert cmds[1].endswith("--spec-type ngram-simple")
    assert "dflash-gemma-4-26B-A4B-it-Q8_0.gguf --spec-type draft-dflash" in cmds[2]
    assert "--host 127.0.0.1 --port 18080" in cmds[0] and "-np 1" in cmds[0]


# --------------------------------------------------------------------------------------------
# integration: the REAL llama.cpp binaries on a tiny random-weight GGUF (seconds, no big model)
# --------------------------------------------------------------------------------------------
def _write_tiny_gguf(path: Path) -> None:
    np = pytest.importorskip("numpy")
    gguf = pytest.importorskip("gguf")
    q = gguf.GGMLQuantizationType
    rng = np.random.default_rng(0)
    n_embd, n_ff, n_layer, n_head = 64, 128, 2, 2
    tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)]
    tokens += ["▁", "▁the", "e", "t", "a"]
    while len(tokens) % 32:
        tokens.append(f"<extra_{len(tokens)}>")
    w = gguf.GGUFWriter(str(path), "llama")
    w.add_context_length(512)
    w.add_embedding_length(n_embd)
    w.add_block_count(n_layer)
    w.add_feed_forward_length(n_ff)
    w.add_head_count(n_head)
    w.add_head_count_kv(n_head)
    w.add_rope_dimension_count(n_embd // n_head)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_file_type(2)
    w.add_tokenizer_model("llama")
    w.add_token_list(tokens)
    w.add_token_scores([0.0] * 259 + [-float(i) for i in range(len(tokens) - 259)])
    w.add_token_types([2, 3, 3] + [6] * 256 + [1] * (len(tokens) - 259))
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_unk_token_id(0)

    def q4(name: str, ne0: int, ne1: int) -> None:
        data = (rng.standard_normal((ne1, ne0)) * 0.05).astype(np.float32)
        w.add_tensor(name, gguf.quants.quantize(data, q.Q4_0), raw_dtype=q.Q4_0)

    q4("token_embd.weight", n_embd, len(tokens))
    w.add_tensor("output_norm.weight", np.ones((n_embd,), dtype=np.float32))
    q4("output.weight", n_embd, len(tokens))
    for il in range(n_layer):
        w.add_tensor(f"blk.{il}.attn_norm.weight", np.ones((n_embd,), dtype=np.float32))
        for t in ("attn_q", "attn_k", "attn_v", "attn_output"):
            q4(f"blk.{il}.{t}.weight", n_embd, n_embd)
        w.add_tensor(f"blk.{il}.ffn_norm.weight", np.ones((n_embd,), dtype=np.float32))
        q4(f"blk.{il}.ffn_gate.weight", n_embd, n_ff)
        q4(f"blk.{il}.ffn_up.weight", n_embd, n_ff)
        q4(f"blk.{il}.ffn_down.weight", n_ff, n_embd)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


@pytest.mark.integration
def test_real_binaries_on_tiny_gguf(tmp_path: Path) -> None:
    for tool in ("llama-bench", "llama-completion", "llama-server"):
        if not paths.bin(tool).is_file():
            pytest.skip(f"{tool} not built for this platform")
    model = tmp_path / "tiny-q4_0.gguf"
    _write_tiny_gguf(model)
    run = bench.run_llama_bench(model, threads=[2], pp=16, tg=8, reps=2,
                                flags=bench.variant_flags("norepack+kvq8", "bench"), timeout=120)
    assert run.ok, run.stderr[-500:]
    assert [(r["kind"], r["repack"], r["type_k"]) for r in run.rows] == [
        ("pp", False, "q8_0"), ("tg", False, "q8_0")]
    c = bench.run_completion(model, "Hello there", n_predict=16, threads=2, n_ctx=512,
                             flags=bench.variant_flags("default", "run"), chat=True, timeout=120)
    assert c.ok, c.stderr_text[-500:]
    assert "CPU_Mapped_mib" in c.parsed["buffer_split"] and "CPU_REPACK_mib" in c.parsed["buffer_split"]
    assert set(c.marks) >= {"prompt_start", "gen_start", "gen_end"}
    c2 = bench.run_completion(model, "Hello there", n_predict=16, threads=2, n_ctx=512,
                              flags=["-nr"], timeout=120)
    assert c2.ok and "CPU_REPACK_mib" not in c2.parsed["buffer_split"]
    srv = bench.LlamaServer(model, threads=2, n_ctx=512, port=free_port(), health_timeout=60)
    try:
        srv.start()
        recs = bench.run_prompt_set(srv, [{"id": "a", "prompt": "hi"}], n_predict=8)
    finally:
        srv.stop()
        srv.cleanup()
    assert recs[0]["error"] is None and recs[0]["predicted_n"] == 8
    assert recs[0]["speculative_types"] == "none" and srv.returncode is not None
