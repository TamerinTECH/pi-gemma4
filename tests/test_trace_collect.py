"""Tests for pc/trace_collect.sh (E3 driver).

Without Docker or models: `bash -n`, --help/--version, the --dry-run plans of the three tiers,
usage errors, the missing-input gates, and the whole host pipeline (validate, meta, merge, stats,
acceptance checks, E3 envelope, --resume, --keep-going) driven by a FAKE pc/docker_run.sh
(test hook PIGEMMA_TRACE_DOCKER_RUN) that writes small synthetic .moetrace files the way the
real probe does (same sidecars, same PIGEMMA_SUMMARY line).

The pipeline tests write into a throw-away directory INSIDE the repository
(traces/_pytest_tc_<uuid>/): the driver only accepts repo paths because the container sees the
repository as /work. The directory is removed afterwards.

The real --smoke run (72 records, all 30 layers) needs Docker + the 26B GGUF and is not run here.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = "pc/trace_collect.sh"
CORPUS_DIR = REPO / "data" / "calibration"


def _corpus_files() -> list[str]:
    exts = {"txt", "c", "cc", "cpp", "cxx", "h", "hh", "hpp", "py", "js", "ts", "rs", "go", "java", "sh"}
    out = []
    for p in sorted(CORPUS_DIR.iterdir(), key=lambda q: q.name.encode()):
        n = p.name
        if not p.is_file() or n.startswith(".") or n.lower().startswith("readme") or n.startswith("all."):
            continue
        if "." not in n or n.rsplit(".", 1)[1].lower() not in exts or p.stat().st_size == 0:
            continue
        out.append(n)
    return out


def _env(**extra: str) -> dict[str, str]:
    env = dict(os.environ)
    for k in list(env):
        if k.startswith("PIGEMMA_TRACE_") or k in ("PIGEMMA_LOCK_HELD",):
            env.pop(k)
    env["PIGEMMA_PYTHON"] = Path(sys.executable).as_posix()
    env["PYTHONUTF8"] = "1"
    env["PIGEMMA_TRACE_BGCPU_S"] = "0"
    env.update(extra)
    return env


def run(bash: str, *args: str, env: dict[str, str] | None = None, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run([bash, SCRIPT, *args], cwd=REPO, env=env or _env(), capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=timeout, check=False)


def _docker_lines(stdout: str) -> list[str]:
    return [ln for ln in stdout.splitlines() if ln.startswith("bash pc/docker_run.sh ")]


# --------------------------------------------------------------------------------------------
# static checks
# --------------------------------------------------------------------------------------------
def test_bash_syntax(bash: str) -> None:
    r = subprocess.run([bash, "-n", SCRIPT], cwd=REPO, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr


def test_help_and_version(bash: str) -> None:
    r = run(bash, "--help")
    assert r.returncode == 0
    for word in ("--smoke", "--quick", "--full", "--gen", "--threads", "--probs-top", "--out", "--dry-run"):
        assert word in r.stdout
    r = run(bash, "--version")
    assert r.returncode == 0 and r.stdout.startswith("pc/trace_collect.sh ")


# --------------------------------------------------------------------------------------------
# dry-run plans (no docker, nothing written)
# --------------------------------------------------------------------------------------------
def test_dry_run_quick_plan(bash: str) -> None:
    r = run(bash, "--quick", "--dry-run", "--no-lock", "--ignore-busy")
    assert r.returncode == 0, r.stderr
    lines = _docker_lines(r.stdout)
    files = _corpus_files()
    assert len(lines) == len(files) + 1
    for ln, name in zip(lines, files):
        assert f"-f /work/data/calibration/{name} " in ln
        assert " -- python -m pigemma.paths lock --hold -- tools/build-linux-x86/bin/pigemma-probe --mode moe-trace " in ln
        assert " -nr -t 12 -c 4096 -b 2048 -ub 256 --chunk-tokens 1024 --gen 128 " in ln
        stem = name.rsplit(".", 1)[0]
        assert ln.endswith(f"--corpus-id {stem} --trace-out /work/traces/26b/{stem}.moetrace")
    gp = lines[-1]
    assert "--gen-prompts /work/data/prompts/gen_prompts.json --chat " in gp and " --gen 256 " in gp
    assert "--chunk-tokens" not in gp
    assert "all.txt" not in r.stdout and "README" not in r.stdout
    merge = [ln for ln in r.stdout.splitlines() if ln.startswith("python -m pigemma.trace merge ")]
    assert len(merge) == 1
    ins = merge[0].split()[5:]
    assert merge[0].split()[4] == "traces/26b/all.moetrace" and "traces/26b/all.moetrace" not in ins
    assert len(ins) == len(files) + 1
    assert "python -m pigemma.trace stats traces/26b/all.moetrace --out traces/26b/all.stats.json --csv-dir results/h1" \
        in r.stdout
    assert not (REPO / "traces" / "26b" / "all.moetrace.tmp").exists()


def test_dry_run_smoke_and_full(bash: str) -> None:
    r = run(bash, "--smoke", "--dry-run", "--no-lock", "--ignore-busy")
    assert r.returncode == 0, r.stderr
    (ln,) = _docker_lines(r.stdout)
    assert "-f /work/data/calibration/en_tech_moe_streaming.txt " in ln
    assert "--chunk-tokens 64 --gen 8 --tokens 72 --max-chunks 1 " in ln
    assert ln.endswith("--trace-out /work/traces/26b/smoke/en_tech_moe_streaming.moetrace")
    assert "pigemma.trace merge" not in r.stdout
    r = run(bash, "--full", "--dry-run", "--no-lock", "--ignore-busy")
    assert r.returncode == 0, r.stderr
    lines = _docker_lines(r.stdout)
    assert all("--chunk-tokens 2048 --gen 256 " in ln for ln in lines[:-1])
    assert "--gen-prompts" in lines[-1] and " --gen 256 " in lines[-1]


def test_dry_run_overrides(bash: str) -> None:
    r = run(bash, "--quick", "--dry-run", "--no-lock", "--ignore-busy", "--gen=5", "--threads", "6",
            "--probs-top", "4", "--out", "traces/x26", "--files", "mixed_qa_numbers,en_tech_moe_streaming.txt",
            "--no-prompts", "--chunk-tokens", "4000")
    assert r.returncode == 0, r.stderr
    lines = _docker_lines(r.stdout)
    assert len(lines) == 2
    for ln in lines:
        # -c is raised to hold chunk 4000 + gen 5 + 8 -> 4013 -> 4096 (already enough): unchanged
        assert " -nr -t 6 -c 4096 " in ln and "--chunk-tokens 4000 --gen 5 --probs-top 4 " in ln
        assert "--trace-out /work/traces/x26/" in ln
    r = run(bash, "--quick", "--dry-run", "--no-lock", "--ignore-busy", "--chunk-tokens", "4100", "--no-prompts",
            "--files", "mixed_qa_numbers")
    assert r.returncode == 0
    assert " -c 4352 " in _docker_lines(r.stdout)[0]  # 4100 + 128 + 8 = 4236 -> rounded up to 4352


def test_dry_run_model_volume(bash: str) -> None:
    # D14: --model-volume mounts the volume read-only and points -m at /models/<basename>
    r = run(bash, "--quick", "--dry-run", "--no-lock", "--ignore-busy", "--no-prompts",
            "--files", "mixed_qa_numbers", "--model-volume", "pigemma-models")
    assert r.returncode == 0, r.stderr
    ln = _docker_lines(r.stdout)[0]
    assert "--volume pigemma-models:/models:ro" in ln
    assert " -m /models/gemma-4-26B_q4_0-it.gguf " in ln and "-m /work/models/" not in ln
    r = run(bash, "--quick", "--dry-run", "--no-lock", "--ignore-busy", "--model-volume", "bad/name")
    assert r.returncode == 2


@pytest.mark.parametrize("args", [
    ("--bogus",),
    ("--smoke", "--full"),
    ("--threads", "0"),
    ("--probs-top", "300"),
    ("--gen", "x"),
    ("--files", "no_such_file", "--dry-run"),
    ("--model", "31b-q4", "--dry-run"),
])
def test_usage_errors(bash: str, args: tuple[str, ...]) -> None:
    r = run(bash, *args, "--no-lock", "--ignore-busy")
    assert r.returncode == 2, (r.stdout, r.stderr)


def test_out_outside_repo_is_usage_error(bash: str, tmp_path: Path) -> None:
    r = run(bash, "--dry-run", "--out", str(tmp_path / "t"), "--no-lock", "--ignore-busy")
    assert r.returncode == 2
    assert "outside the repository" in r.stderr


# --------------------------------------------------------------------------------------------
# fake docker runner + pipeline tests
# --------------------------------------------------------------------------------------------
FAKE_RUNNER = r'''
"""Fake pc/docker_run.sh for tests/test_trace_collect.py: preflight + a synthetic moe-trace probe."""
import json, os, sys, zlib
from pathlib import Path
sys.path.insert(0, os.getcwd())
import numpy as np
from pigemma import paths
from pigemma import trace as tr

L, K, E = 4, 4, 16
WORDS = ("river stone garden window lantern harbor meadow pencil violin orchard cloud bridge "
         "candle forest mirror ladder basket comet desert engine feather glacier hammer island "
         "jacket kettle lemon marble needle ocean pillow quartz ribbon saddle tunnel umbrella "
         "valley wagon yarn zebra anchor blossom cabin dolphin ember falcon ginger hollow").split()


def repo(p):
    return Path(p[len("/work/"):]) if p.startswith("/work/") else Path(p)


def calls(entry):
    f = os.environ.get("FAKE_CALLS")
    if f:
        with open(f, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")


def flag(cmd, name, default=None):
    for i, a in enumerate(cmd):
        if a == name and i + 1 < len(cmd):
            return cmd[i + 1]
    return default


def main(argv):
    if "--" not in argv:
        print("fake docker_run: no '--'", file=sys.stderr)
        return 2
    cmd = argv[argv.index("--") + 1:]
    if cmd[0] == "sh":
        calls({"op": "preflight"})
        model = repo(cmd[5])
        host = dict(paths.host_fingerprint())
        host["hostname"] = "fake-container"
        host["in_container"] = True
        print("@@HOST")
        print(json.dumps(host))
        print("@@KV")
        print("probe_exec=1")
        print("probe_version=fake-probe 0.0")
        print(f"model_size={model.stat().st_size if model.is_file() else ''}")
        print("mem_total_kb=32000000")
        print("mem_avail_kb=30000000")
        print("nproc=12")
        return 0
    if "--" in cmd:
        cmd = cmd[cmd.index("--") + 1:]
    cid = flag(cmd, "--corpus-id")
    calls({"op": "probe", "cid": cid, "argv": cmd})
    if os.environ.get("FAKE_FAIL_ITEM") == cid:
        print("pigemma-probe: ERROR: fake failure", file=sys.stderr)
        print("PIGEMMA_SUMMARY " + json.dumps({"mode": "moe-trace", "exit_code": 3, "error": "fake failure",
                                               "n_records": 0}), file=sys.stderr)
        return 3
    out = repo(flag(cmd, "--trace-out"))
    model = repo(flag(cmd, "-m"))
    gen = int(flag(cmd, "--gen", "0"))
    chunk = int(flag(cmd, "--chunk-tokens", "1024"))
    max_chunks = int(flag(cmd, "--max-chunks", "0"))
    limit = int(flag(cmd, "--tokens", "-1"))
    bpt = int(os.environ.get("FAKE_BYTES_PER_TOKEN", "3"))
    plan = []  # (n_prompt incl. BOS, n_gen)
    if flag(cmd, "-f"):
        n_text = max(1, repo(flag(cmd, "-f")).stat().st_size // bpt)
        while n_text > 0:
            take = min(n_text, chunk - 1)
            plan.append((take + 1, gen))
            n_text -= take
    else:
        items = json.loads(repo(flag(cmd, "--gen-prompts")).read_text(encoding="utf-8"))
        for it in items:
            plan.append((16 + len(it["prompt"].encode("utf-8")) // 4, gen or int(it.get("max_tokens", 0))))
    if max_chunks:
        plan = plan[:max_chunks]
    hdr = tr.make_header(n_expert=E, n_expert_used=K, moe_layers=list(range(L)), model_name="fake",
                         gguf_path=flag(cmd, "-m"), gguf_size_bytes=model.stat().st_size, corpus_id=cid,
                         n_ctx=int(flag(cmd, "-c", "4096")), n_batch=int(flag(cmd, "-b", "2048")),
                         n_ubatch=int(flag(cmd, "-ub", "512")), n_threads=int(flag(cmd, "-t", "4")),
                         flags=list(cmd), chat="--chat" in cmd)
    rng = np.random.default_rng(zlib.crc32(cid.encode()))
    n_p = n_g = 0
    gen_text = []
    bad = os.environ.get("FAKE_INCOHERENT_ITEM") == cid
    out.parent.mkdir(parents=True, exist_ok=True)
    with tr.TraceWriter(out, hdr) as w:
        for ci, (np_, ng) in enumerate(plan):
            if 0 <= limit <= n_p + n_g:
                break
            if limit >= 0:
                np_ = min(np_, limit - n_p - n_g)
                ng = min(ng, limit - n_p - n_g - np_)
            n = np_ + ng
            w.begin_chunk(source=cid)
            ids = rng.random((n, L, E)).argsort(axis=2)[:, :, :K].astype(np.uint8)
            weights = np.full((n, L, K), 1.0 / K, dtype=np.float32)
            phase = np.array([0] * np_ + [1] * ng, dtype=np.uint8)
            toks = rng.integers(3, 50000, size=n).astype(np.int32)
            toks[0] = 2
            w.write_many(ids, weights, phase=phase, pos=np.arange(n), token_ids=toks, chunk_ids=np.full(n, ci))
            n_p += np_
            n_g += ng
            words = ["and then again"] * 12 if bad else [WORDS[int(x)] for x in rng.integers(0, len(WORDS), max(1, ng))]
            gen_text.append(tr.GEN_CHUNK_SEPARATOR.format(i=ci) + " ".join(words) + ".")
    Path(str(out) + ".gen.txt").write_bytes("".join(gen_text).encode("utf-8"))  # LF, as the probe on Linux
    t_p, t_e = n_p / 50.0 * 1000.0, max(n_g, 1) / 12.0 * 1000.0
    print(f"pigemma-probe: loading {flag(cmd, '-m')} (mode moe-trace; warm-up disabled)", file=sys.stderr)
    print("llama_model_loader: lots of noise that the console filter hides", file=sys.stderr)
    s = {"mode": "moe-trace", "exit_code": 0, "n_prompt": n_p, "n_gen": n_g, "n_records": n_p + n_g,
         "n_chunks": len(gen_text), "t_load_ms": 2000.0, "t_p_eval_ms": t_p, "t_eval_ms": t_e,
         "tok_s_prompt": 50.0, "tok_s_gen": 12.0, "callback_ms": 42.0, "t_prefill_wall_ms": t_p,
         "t_gen_wall_ms": t_e, "wall_tok_s_prompt": 49.0, "wall_tok_s_gen": 11.5, "trace_out": flag(cmd, "--trace-out"),
         "act_out": None, "io_csv": None, "n_p_eval": n_p, "n_eval": n_g, "wall_ms": 2000.0 + t_p + t_e,
         "n_ctx": 4096, "n_threads": 12, "interrupted": False, "error": None, "probe_version": "fake",
         "llama_build": 11398, "llama_commit": "a7b94df"}
    print("PIGEMMA_SUMMARY " + json.dumps(s, separators=(",", ":")), file=sys.stderr)
    return 0


sys.exit(main(sys.argv[1:]))
'''


@pytest.fixture()
def fake(tmp_path: Path, isolated_lock: Path) -> dict:
    pytest.importorskip("numpy")
    ws_rel = f"traces/_pytest_tc_{uuid.uuid4().hex[:10]}"
    ws = REPO / ws_rel
    ws.mkdir(parents=True)
    runner = tmp_path / "fake_runner.py"
    runner.write_text(FAKE_RUNNER, encoding="utf-8", newline="\n")
    sh = tmp_path / "fake_docker_run.sh"
    sh.write_text(f'#!/usr/bin/env bash\nexec "{Path(sys.executable).as_posix()}" "{runner.as_posix()}" "$@"\n',
                  encoding="utf-8", newline="\n")
    model = ws / "fake-model.gguf"
    model.write_bytes(b"GGUF" + bytes(range(256)) * 512)
    calls = tmp_path / "calls.jsonl"
    env = _env(PIGEMMA_TRACE_DOCKER_RUN=str(sh), FAKE_CALLS=str(calls))
    common = ["--model", f"{ws_rel}/fake-model.gguf", "--out", f"{ws_rel}/traces", "--csv-dir", f"{ws_rel}/csv",
              "--results-dir", f"{ws_rel}/results", "--log-dir", f"{ws_rel}/logs", "--no-lock", "--ignore-busy"]
    try:
        yield {"ws": ws, "rel": ws_rel, "env": env, "common": common, "calls": calls}
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def _calls(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _envelopes(d: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("E3_*.json"))]


def test_pipeline_quick_pass(bash: str, fake: dict) -> None:
    env = dict(fake["env"], FAKE_BYTES_PER_TOKEN="2")
    r = run(bash, "--quick", *fake["common"], "--files", "he_discussion_llm_efficiency,code_cpp_llamacpp_debug", env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    t = fake["ws"] / "traces"
    for stem in ("he_discussion_llm_efficiency", "code_cpp_llamacpp_debug", "gen_prompts"):
        assert (t / f"{stem}.moetrace").is_file()
        meta = json.loads((t / f"{stem}.moetrace.meta.json").read_text(encoding="utf-8"))
        assert meta["host"]["hostname"] == "fake-container"
        assert meta["coherent"] is True and meta["tok_s"]["prompt"] == 50.0
        assert meta["gguf_sha256_prefix"] and meta["corpus"][0]["sha256"]
    from pigemma import results
    from pigemma import trace as tr
    assert tr.validate(t / "all.moetrace") == []
    st = json.loads((t / "all.stats.json").read_text(encoding="utf-8"))
    assert st["n_prefill"] >= 10000 and st["n_gen"] >= 2000
    assert list((fake["ws"] / "csv").glob("all.*.csv"))
    assert "TRACE VALID" in r.stdout and "CHECK prefill tokens: PASS" in r.stdout
    assert "E3 preview" in r.stdout and "PROJECTED --full" in r.stdout
    assert "lots of noise" not in r.stderr  # console filter
    (env_doc,) = _envelopes(fake["ws"] / "results")
    assert results.validate_result(env_doc) == []
    assert env_doc["experiment"] == "E3" and env_doc["variant"] == "moe-trace-quick"
    assert env_doc["coherent"] is True and env_doc["pp"]["tok_s"] == pytest.approx(50.0, rel=1e-3)
    assert env_doc["host"]["hostname"] == "fake-container"
    assert env_doc["extra"]["trace_valid"] is True and len(env_doc["extra"]["runs"]) == 3
    probes = [c for c in _calls(fake["calls"]) if c["op"] == "probe"]
    # --files keeps the user's order; the prompts run is always last
    assert [c["cid"] for c in probes] == ["he_discussion_llm_efficiency", "code_cpp_llamacpp_debug", "gen_prompts"]
    logs = list((fake["ws"] / "logs").glob("trace_collect_*"))
    assert any(p.name.endswith(".probe.log") for p in logs) and any(p.name.endswith(".runs.log") for p in logs)


def test_pipeline_checks_fail_exit_5(bash: str, fake: dict) -> None:
    r = run(bash, "--quick", *fake["common"], "--files", "mixed_qa_numbers", "--no-prompts", env=fake["env"])
    assert r.returncode == 5, r.stdout + r.stderr
    assert "CHECK prefill tokens: FAIL" in r.stdout and "TRACE NOT VALID" in r.stdout
    (env_doc,) = _envelopes(fake["ws"] / "results")
    assert env_doc["extra"]["trace_valid"] is False


def test_pipeline_incoherent_chunk(bash: str, fake: dict) -> None:
    env = dict(fake["env"], FAKE_BYTES_PER_TOKEN="2", FAKE_INCOHERENT_ITEM="gen_prompts")
    r = run(bash, "--quick", *fake["common"], "--files", "he_discussion_llm_efficiency,code_cpp_llamacpp_debug",
            "--no-envelope", env=env)
    assert r.returncode == 5, r.stdout + r.stderr
    assert "CHECK coherent: FAIL" in r.stdout and "gen_prompts#0" in r.stdout
    assert not (fake["ws"] / "results").exists()


def test_pipeline_failure_stops_or_keeps_going(bash: str, fake: dict) -> None:
    env = dict(fake["env"], FAKE_FAIL_ITEM="code_cpp_llamacpp_eval_callback")
    sel = ["--files", "code_cpp_llamacpp_eval_callback,mixed_qa_numbers", "--no-prompts"]
    r = run(bash, "--quick", *fake["common"], *sel, env=env)
    assert r.returncode == 1
    assert [c["cid"] for c in _calls(fake["calls"]) if c["op"] == "probe"] == ["code_cpp_llamacpp_eval_callback"]
    assert not (fake["ws"] / "traces" / "all.moetrace").exists()
    fake["calls"].unlink()
    r = run(bash, "--quick", *fake["common"], *sel, "--keep-going", env=env)
    assert r.returncode == 1, r.stdout + r.stderr
    assert [c["cid"] for c in _calls(fake["calls"]) if c["op"] == "probe"] == \
        ["code_cpp_llamacpp_eval_callback", "mixed_qa_numbers"]
    from pigemma import trace as tr
    h = tr.read_header(fake["ws"] / "traces" / "all.moetrace")
    assert {c["source"] for c in h["chunks"]} == {"mixed_qa_numbers"}
    assert "CHECK runs: FAIL" in r.stdout


def test_smoke_and_resume(bash: str, fake: dict) -> None:
    r = run(bash, "--smoke", *fake["common"], env=fake["env"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SMOKE PASS" in r.stdout and "CHECK smoke records: PASS (72 records, expected 72)" in r.stdout
    trace = fake["ws"] / "traces" / "smoke" / "en_tech_moe_streaming.moetrace"
    assert trace.is_file() and (fake["ws"] / "traces" / "smoke" / "en_tech_moe_streaming.moetrace.meta.json").is_file()
    assert not (fake["ws"] / "traces" / "all.moetrace").exists()
    assert "cp " in r.stdout and "golden_26b_72tok" in r.stdout  # hint only
    fake["calls"].unlink()
    r = run(bash, "--smoke", "--resume", *fake["common"], env=fake["env"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert [c["op"] for c in _calls(fake["calls"])] == ["preflight"]
    assert "resume: reusing" in r.stderr
    # a different --gen is not reusable
    fake["calls"].unlink()
    r = run(bash, "--smoke", "--resume", "--gen", "4", *fake["common"], env=fake["env"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert [c["op"] for c in _calls(fake["calls"])] == ["preflight", "probe"]
    assert "(68 records, expected 68)" in r.stdout


def test_missing_inputs_exit_3(bash: str, fake: dict) -> None:
    args = [a for a in fake["common"]]
    i = args.index("--model")
    args[i + 1] = f"{fake['rel']}/no-such-model.gguf"
    r = run(bash, "--smoke", *args, env=fake["env"])
    assert r.returncode == 3 and "missing" in r.stderr
    env = dict(fake["env"], PIGEMMA_TRACE_PROBE="tools/no-such-dir/pigemma-probe")
    r = run(bash, "--smoke", *fake["common"], env=env)
    assert r.returncode == 3 and "build-tools" in r.stderr
    assert _calls(fake["calls"]) == []


def test_no_merge_skips_merged_check(bash: str, fake: dict) -> None:
    env = dict(fake["env"], FAKE_BYTES_PER_TOKEN="2")
    r = run(bash, "--quick", *fake["common"], "--files", "he_discussion_llm_efficiency,code_cpp_llamacpp_debug",
            "--no-merge", env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "CHECK merged trace valid: SKIP" in r.stdout and "CHECK prefill tokens: PASS" in r.stdout
    assert not (fake["ws"] / "traces" / "all.moetrace").exists()
    assert not (fake["ws"] / "results").exists()
