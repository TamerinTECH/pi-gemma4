"""Tests for pi/06_pin_hotset.sh (manifest files[] 'pi/06_pin_hotset.sh').

Manifest test field: `bash -n`; hotset unit tests (tests/test_gguf_map.py covers `gguf_map hotset`); the
same code path exercised in Docker by pc/pagecache_limit.sh --pin.

What runs where:
- everywhere (Windows Git Bash, Linux, the Pi): syntax, usage line == manifest, option parsing, the embedded
  stdlib helper as a unit (byte parsing, budget, map / stats checks, model resolution, pinner summaries,
  the state file and the A/B report incl. the E6 threshold logic), `--dry-run` plans with the
  PIGEMMA_FAKE_PROC fixtures, every refusal path (protected host, override without mlock permission, the
  pinner's own protected-host guard, memlock limit, RAM, budget below the non-expert set, a pinner already
  running, real runs under a fake root), status / stop without a pinner and against a fake pinner pid.
- Linux only (`linux` marker; the Pi or a container with pytest): start / status / stop with the real
  tools/pinner on a synthetic MoE GGUF, and `ab` end to end through pi/05_run.sh with the stdlib
  llama-completion double of tests/fixtures/bench/. The same sequence was run in the pi-gemma4-build
  Docker image (no pytest there) when the script was written.

Nothing here locks memory of the test machine outside the `linux` tests, loads a model or touches models/,
results/ or pi/out/: every run gets its own PIGEMMA_OUT, MODEL_DIR, PIGEMMA_TOOLS_BIN and PIGEMMA_LOCK_FILE.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures"
SCRIPT = "pi/06_pin_hotset.sh"
MAP26 = REPO / "results" / "map26.json"
Q4_FILE = "gemma-4-26B_q4_0-it.gguf"
Q4_BYTES = 14439363584
Q4_NONEXPERT = 1577172088
CONFIG_VARS = (
    "PIGEMMA_HOME", "MODEL_DIR", "LLAMA_DIR", "LLAMA_COMMIT", "PIGEMMA_OUT", "THREADS", "CTX", "QUANT_OVERRIDE",
    "OS_RESERVE_GIB", "MIN_FREE_DISK_FACTOR", "KEEP_FREE_GB", "NICE_LEVEL", "DENY_HOSTS",
    "PROTECTED_SERVICE_GLOBS", "ALLOW_MLOCK", "TUNE_APPLY", "TUNE_NVME_GEN3", "TUNE_GOVERNOR", "TUNE_SWAPPINESS",
    "TUNE_READAHEAD_KB", "TUNE_MEMLOCK", "BUILD_JOBS", "HF_LIMIT_RATE", "PIN_BUDGET_BYTES",
)
FAKE_PID = 999_983  # a pid that names a fake pinner inside a fake system root (never signalled)

needs_map26 = pytest.mark.skipif(not MAP26.is_file(), reason="results/map26.json (tracked artifact) is missing")


# ------------------------------------------------------------------------------------------- helpers
def _p(path: Path | str) -> str:
    return Path(path).as_posix()


def _env(tmp_path: Path, fake: str | Path | None = "pi_proc_16gb", **extra: str) -> dict[str, str]:
    """Hermetic environment: config vars and PIGEMMA_* of the caller removed, every output under tmp."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("PIGEMMA_", "PI05_", "FAKE_")) and k not in CONFIG_VARS and k != "PY"}
    for d in ("out", "models", "bin", "results"):
        (tmp_path / d).mkdir(exist_ok=True)
    env["PIGEMMA_PYTHON"] = _p(sys.executable)
    env["PIGEMMA_OUT"] = _p(tmp_path / "out")
    env["MODEL_DIR"] = _p(tmp_path / "models")
    env["PIGEMMA_TOOLS_BIN"] = _p(tmp_path / "bin")
    env["PIGEMMA_LOCK_FILE"] = _p(tmp_path / "bench.lock")
    env["LC_ALL"] = env.get("LC_ALL") or "C.UTF-8"
    if fake:
        env["PIGEMMA_FAKE_PROC"] = _p(fake if isinstance(fake, Path) else FIX / fake)
    env.update(extra)
    return env


def run(bash: str, args: list[str], env: dict[str, str], *, timeout: float = 240) -> subprocess.CompletedProcess:
    return subprocess.run([bash, SCRIPT, *args], cwd=REPO, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def fake_root(tmp_path: Path, base: str = "pi_proc_16gb", *, mem_available_kb: int | None = None,
              ulimit_l: str | None = None, pinner_pid: int | None = None, vmlck_kb: int = 1024) -> Path:
    """Copy of a fake system root; optionally MemAvailable / ulimit -l rewritten and a fake pinner process."""
    dst = tmp_path / f"proc_{base}"
    shutil.copytree(FIX / base, dst)
    if mem_available_kb is not None:
        mi = dst / "meminfo"
        text = re.sub(r"(?m)^MemAvailable:\s+\d+ kB$", f"MemAvailable:   {mem_available_kb} kB",
                      mi.read_text(encoding="utf-8"))
        mi.write_bytes(text.encode("utf-8"))
    if ulimit_l is not None:
        (dst / "ulimit_l").write_bytes(f"{ulimit_l}\n".encode())
    if pinner_pid is not None:
        d = dst / str(pinner_pid)
        d.mkdir()
        (d / "comm").write_bytes(b"pinner\n")
        (d / "status").write_bytes(f"Name:\tpinner\nState:\tS (sleeping)\nVmLck:\t{vmlck_kb} kB\n".encode())
    return dst


def kv(stdout: str) -> dict[str, str]:
    """NAME='value' lines (shlex-quoted) -> dict."""
    out: dict[str, str] = {}
    for line in stdout.splitlines():
        m = re.match(r"^([A-Z][A-Z0-9_]*)=(.*)$", line)
        if m:
            words = shlex.split(m.group(2)) if m.group(2) else [""]
            out[m.group(1)] = words[0] if words else ""
    return out


def helper_source() -> str:
    text = (REPO / SCRIPT).read_text(encoding="utf-8")
    marker = "<<'PIGEMMA_PY06_EOF'\n"
    start = text.index(marker) + len(marker)
    end = text.index("\nPIGEMMA_PY06_EOF\n", start)
    return text[start:end + 1]


@pytest.fixture(scope="module")
def helper(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("pi06_helper") / "pi06_helper.py"
    p.write_text(helper_source(), encoding="utf-8")
    return p


def run_helper(helper: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, str(helper), *args], cwd=REPO, env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=120)


def stats_for_map26(path: Path, *, n_expert: int = 128, layer_shift: int = 0) -> Path:
    m = json.loads(MAP26.read_text(encoding="utf-8"))
    pl = [{"layer": il + layer_shift, "freq": [(il * 7 + e * 13) % 101 for e in range(n_expert)]}
          for il in m["moe_layers"]]
    path.write_text(json.dumps({"n_tokens": 1234, "per_layer": pl}), encoding="utf-8")
    return path


def _lines(stdout: str) -> list[str]:
    return [ln for ln in stdout.splitlines() if ln.strip()]


# ------------------------------------------------------------------------------------- static checks
def test_bash_syntax(bash: str) -> None:
    r = subprocess.run([bash, "-n", SCRIPT], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_lf_line_endings_and_shebang() -> None:
    raw = (REPO / SCRIPT).read_bytes()
    assert b"\r\n" not in raw
    assert raw.startswith(b"#!/usr/bin/env bash\n")
    assert b"set -euo pipefail" in raw


def test_usage_line_matches_manifest(bash: str, tmp_path: Path) -> None:
    manifest = json.loads((REPO / "docs/design/manifest.json").read_text(encoding="utf-8"))
    entry = next(f for f in manifest["files"] if f["path"] == SCRIPT)
    m = re.match(r"`([^`]+)`", entry["spec"])
    assert m is not None
    usage = m.group(1)
    assert f'PI06_USAGE_LINE="{usage}"' in (REPO / SCRIPT).read_text(encoding="utf-8")
    r = run(bash, ["--help"], _env(tmp_path, fake=None))
    assert r.returncode == 0
    assert usage in r.stdout and "Exit codes" in r.stdout and "--max-hold" in r.stdout


def test_version(bash: str, tmp_path: Path) -> None:
    r = run(bash, ["--version"], _env(tmp_path, fake=None))
    assert r.returncode == 0 and r.stdout.strip() == "pi/06_pin_hotset.sh 1.0.0"


@pytest.mark.parametrize("args,needle", [
    ([], "missing subcommand"),
    (["start", "--bogus"], "unknown option"),
    (["start", "stop"], "more than one subcommand"),
    (["start", "--model"], "needs a value"),
    (["frobnicate"], "unexpected argument"),
    (["start", "--kv-type", "q9"], "--kv-type must be one of"),
    (["start", "--rank", "magic"], "--rank must be count or weight"),
    (["start", "--budget-bytes", "lots"], "not a byte count"),
    (["start", "--max-hold", "x"], "--max-hold must be an integer"),
    (["start", "--run-arg", "x"], "belong to 'ab' only"),
    (["status", "--out", "x.json"], "belong to 'ab' only"),
    (["ab", "--rounds", "0"], "--rounds must be an integer >= 1"),
])
def test_usage_errors_exit_1(bash: str, tmp_path: Path, args: list[str], needle: str) -> None:
    r = run(bash, args, _env(tmp_path))
    assert r.returncode == 1, (r.returncode, r.stderr)
    assert needle in r.stderr and r.stdout == ""


# ------------------------------------------------------------------------------------ helper (unit)
@pytest.mark.parametrize("spec,expected", [
    ("2GiB", 2 * 2**30), ("5.2GB", 5_200_000_000), ("5.2e9", 5_200_000_000), ("5_200_000_000", 5_200_000_000),
    ("512MiB", 512 * 2**20), ("1000", 1000),
])
def test_helper_explicit_budget(helper: Path, spec: str, expected: int) -> None:
    r = run_helper(helper, "budget", f"--spec={spec}", "--spec-source=--budget-bytes", f"--file-size={10**12}")
    assert r.returncode == 0, r.stderr
    d = kv(r.stdout)
    assert int(d["BUDGET_BYTES"]) == expected and d["BUDGET_SOURCE"] == "--budget-bytes"


def test_helper_budget_errors(helper: Path) -> None:
    r = run_helper(helper, "budget", "--spec=lots")
    assert r.returncode == 1 and "not a byte count" in r.stderr
    r = run_helper(helper, "budget", "--spec=auto", "--avail=900000000", "--kv-fallback=160000000")
    assert r.returncode == 3 and "no RAM left for a hot set" in r.stderr
    r = run_helper(helper, "budget", "--spec=auto", "--avail=9000000000", "--kv-type=q9")
    assert r.returncode == 1 and "unknown --kv-type" in r.stderr


@needs_map26
def test_helper_auto_budget_uses_the_map_kv(helper: Path) -> None:
    from pigemma import gguf_map as gm
    kv_q8 = gm.GgufMap.load(MAP26).kv_bytes(4096, "q8_0")
    r = run_helper(helper, "budget", "--spec=auto", "--avail=8000000000", "--ctx=4096", "--kv-type=q8_0",
                   f"--map={MAP26}", f"--file-size={Q4_BYTES}")
    assert r.returncode == 0, r.stderr
    d = kv(r.stdout)
    assert int(d["BUDGET_BYTES"]) == 8_000_000_000 - kv_q8 - 300_000_000 - 500_000_000
    assert int(d["KV_BYTES"]) == kv_q8 and d["KV_SOURCE"].startswith("measured") and d["BUDGET_CAPPED"] == "0"
    # capped at the file size on a big host
    r = run_helper(helper, "budget", "--spec=auto", "--avail=64000000000", f"--map={MAP26}", f"--file-size={Q4_BYTES}")
    d = kv(r.stdout)
    assert int(d["BUDGET_BYTES"]) == Q4_BYTES and d["BUDGET_CAPPED"] == "1"
    # no map yet (dry run): the catalog KV estimate, labelled assumed; f16 costs 64/34 of q8_0
    r = run_helper(helper, "budget", "--spec=auto", "--avail=8000000000", "--ctx=8192", "--kv-type=f16",
                   "--kv-fallback=160000000")
    d = kv(r.stdout)
    kv_est = round(160_000_000 * 2 * 64 / 34)
    assert int(d["KV_BYTES"]) == kv_est and d["KV_SOURCE"].startswith("assumed")
    assert int(d["BUDGET_BYTES"]) == 8_000_000_000 - kv_est - 800_000_000


@needs_map26
def test_helper_check_map(helper: Path, tmp_path: Path) -> None:
    r = run_helper(helper, "check-map", f"--map={MAP26}", f"--gguf={tmp_path / Q4_FILE}", f"--expect-size={Q4_BYTES}")
    d = kv(r.stdout)
    assert d["MAP_MATCH"] == "1" and d["MAP_IS_MOE"] == "1" and int(d["MAP_NONEXPERT"]) == Q4_NONEXPERT
    assert int(d["MAP_BPT_WORST"]) > Q4_NONEXPERT
    r = run_helper(helper, "check-map", f"--map={MAP26}", f"--gguf={tmp_path / 'other.gguf'}")
    d = kv(r.stdout)
    assert d["MAP_MATCH"] == "0" and "built for gemma-4-26B_q4_0-it.gguf" in d["MAP_REASON"]
    r = run_helper(helper, "check-map", f"--map={MAP26}", f"--gguf={tmp_path / Q4_FILE}", "--expect-size=123")
    assert kv(r.stdout)["MAP_MATCH"] == "0"
    r = run_helper(helper, "check-map", f"--map={tmp_path / 'nope.json'}", f"--gguf={tmp_path / Q4_FILE}")
    d = kv(r.stdout)
    assert d["MAP_MATCH"] == "0" and d["MAP_REASON"] == "does not exist"


@needs_map26
def test_helper_check_stats(helper: Path, tmp_path: Path) -> None:
    good = stats_for_map26(tmp_path / "good.json")
    d = kv(run_helper(helper, "check-stats", f"--stats={good}", f"--map={MAP26}").stdout)
    assert d["STATS_OK"] == "1" and d["STATS_N_TOKENS"] == "1234"
    bad = stats_for_map26(tmp_path / "bad.json", n_expert=64)
    d = kv(run_helper(helper, "check-stats", f"--stats={bad}", f"--map={MAP26}").stdout)
    assert d["STATS_OK"] == "0" and "64 experts in the stats, 128 in the map" in d["STATS_REASON"]
    shifted = stats_for_map26(tmp_path / "shift.json", layer_shift=100)
    d = kv(run_helper(helper, "check-stats", f"--stats={shifted}", f"--map={MAP26}").stdout)
    assert d["STATS_OK"] == "0" and "not a MoE layer" in d["STATS_REASON"]
    (tmp_path / "junk.json").write_text("{}", encoding="utf-8")
    d = kv(run_helper(helper, "check-stats", f"--stats={tmp_path / 'junk.json'}", f"--map={MAP26}").stdout)
    assert d["STATS_OK"] == "0" and "per_layer" in d["STATS_REASON"]


def test_helper_resolve_model(helper: Path, tmp_path: Path) -> None:
    cat = REPO / "pigemma/data/catalog.json"
    (tmp_path / "models").mkdir()
    r = run_helper(helper, "resolve-model", "--model=26b-ud-q3_k_m", f"--model-dir={tmp_path / 'models'}",
                   f"--catalog={cat}")
    d = kv(r.stdout)
    assert r.returncode == 0 and d["MODEL_ID"] == "26b-ud-q3_k_m" and d["MODEL_KIND"] == "moe"
    assert d["MODEL_PATH"].endswith("gemma-4-26B-A4B-it-UD-Q3_K_M.gguf") and d["MODEL_EXISTS"] == "0"
    assert int(d["MODEL_FILE_BYTES"]) == int(d["MODEL_CATALOG_BYTES"]) > 0
    # an existing file under MODEL_DIR is found; a wrong size is flagged (incomplete download)
    f = tmp_path / "models" / "gemma-4-26B-A4B-it-UD-Q3_K_M.gguf"
    f.write_bytes(b"GGUF" + b"\0" * 100)
    d = kv(run_helper(helper, "resolve-model", "--model=26b-ud-q3_k_m", f"--model-dir={tmp_path / 'models'}",
                      f"--catalog={cat}").stdout)
    assert d["MODEL_EXISTS"] == "1" and d["MODEL_SIZE"] == "104" and d["MODEL_SIZE_MISMATCH"] == "1"
    # a path is matched to its catalog entry by basename; an unknown file is uncatalogued
    d = kv(run_helper(helper, "resolve-model", f"--model={f}", f"--catalog={cat}").stdout)
    assert d["MODEL_ID"] == "26b-ud-q3_k_m" and d["MODEL_STEM"] == "gemma-4-26B-A4B-it-UD-Q3_K_M"
    other = tmp_path / "models" / "mine.gguf"
    other.write_bytes(b"GGUF")
    d = kv(run_helper(helper, "resolve-model", f"--model={other}", f"--catalog={cat}").stdout)
    assert d["MODEL_ID"] == "" and "uncatalogued" in d["MODEL_NOTES"]
    r = run_helper(helper, "resolve-model", "--model=26b-mtp", f"--catalog={cat}")
    assert r.returncode == 1 and "draft sidecar" in r.stderr
    r = run_helper(helper, "resolve-model", "--model=27b-q4", f"--catalog={cat}")
    assert r.returncode == 1 and "unknown model id" in r.stderr
    r = run_helper(helper, "resolve-model", f"--catalog={cat}", f"--feasibility={tmp_path / 'none.json'}")
    assert r.returncode == 1 and "no --model given" in r.stderr
    feas = tmp_path / "feasibility.json"
    feas.write_text(json.dumps({"recommended": {"model_id": "26b-q4"}}), encoding="utf-8")
    d = kv(run_helper(helper, "resolve-model", f"--catalog={cat}", f"--feasibility={feas}",
                      f"--model-dir={tmp_path / 'models'}").stdout)
    assert d["MODEL_ID"] == "26b-q4" and "feasibility recommendation" in d["MODEL_NOTES"]


PINNER_JSON = {
    "tool": "pinner", "version": "1.0.0", "status": "partial", "exit_code": 2, "mode": "daemon", "pid": 4242,
    "requested_bytes": 5_000_000_000, "locked_bytes": 4_000_000_000, "resident_bytes": 4_100_000_000,
    "resident_before_bytes": 1_000_000_000, "ranges": 812, "failed": 1, "not_attempted": 20, "page_size": 16384,
    "memlock_limit": None, "vm_lck_kb": 3_906_250, "lock_s": 12.5, "mincore_reliable": True, "verify": "ok",
    "reason": "RLIMIT_MEMLOCK reached", "kinds": {"expert": {"ranges": 800, "requested_bytes": 3_400_000_000,
                                                            "locked_bytes": 2_400_000_000,
                                                            "resident_bytes": 2_500_000_000}},
}


def test_helper_pinner_summary(helper: Path, tmp_path: Path) -> None:
    log = tmp_path / "pinner.log"
    log.write_text("not json\n" + json.dumps(PINNER_JSON) + "\n" + '{"tool":"pinner","status":"ok"', encoding="utf-8")
    d = kv(run_helper(helper, "pinner-summary", f"--file={log}").stdout)
    assert d["PS_FOUND"] == "1" and d["PS_STATUS"] == "partial" and d["PS_EXIT"] == "2" and d["PS_MODE"] == "daemon"
    assert d["PS_LOCKED"] == "4000000000" and d["PS_MEMLOCK"] == "unlimited" and d["PS_REASON"] == "RLIMIT_MEMLOCK reached"
    log.write_text('{"tool":"pinner"', encoding="utf-8")  # partial line only: no summary yet
    assert kv(run_helper(helper, "pinner-summary", f"--file={log}").stdout)["PS_FOUND"] == "0"
    assert kv(run_helper(helper, "pinner-summary", f"--file={tmp_path / 'none.log'}").stdout)["PS_FOUND"] == "0"


def test_helper_state_and_reports(helper: Path, tmp_path: Path) -> None:
    log = tmp_path / "pinner.log"
    log.write_text(json.dumps({**PINNER_JSON, "status": "ok", "exit_code": 0}) + "\n", encoding="utf-8")
    hs = tmp_path / "hotset.json"
    hs.write_text(json.dumps({"total_bytes": 5_000_000_000, "budget_bytes": 5_200_000_000, "n_ranges": 812,
                              "n_experts": 1053, "n_expert_items_total": 3840, "expert_coverage_fraction": 0.61,
                              "nonexpert_included": True, "nonexpert_bytes_aligned": 1_580_000_000,
                              "page_size": 16384, "stats": "traces/26b/all.stats.json", "ranges": []}),
                  encoding="utf-8")
    state = tmp_path / "pin_state.json"
    r = run_helper(helper, "state-write", f"--out={state}", "--state=running", "--version=1.0.0", "--pid=4242",
                   f"--summary-file={log}", f"--hotset={hs}", "--budget-bytes=5200000000", "--budget-source=auto",
                   "--max-hold=600", "--note=hello", "--model=/m/x.gguf", "--model-bytes=14439363584")
    assert r.returncode == 0, r.stderr
    st = json.loads(state.read_text(encoding="utf-8"))
    assert st["state"] == "running" and st["pid"] == 4242 and st["summary"]["locked_bytes"] == 4_000_000_000
    assert st["hotset_info"]["n_experts"] == 1053 and st["budget"]["bytes"] == 5_200_000_000
    assert st["max_hold_s"] == 600 and st["notes"] == ["hello"]
    text = run_helper(helper, "report-start", f"--state={state}").stdout
    assert "pinner: running, pid 4242" in text and "1053 of 3840 expert-layer items (27.4%" in text
    assert "locked     4.00 GB of 5.00 GB requested" in text and "expert_coverage_fraction 0.6100" in text
    assert json.loads(run_helper(helper, "report-start", f"--state={state}", "--json=1").stdout)["pid"] == 4242
    run_helper(helper, "state-update", f"--file={state}", "--set=state=stopped", '--set=stopped_at="x"',
               "--set=vm_lck_bytes_at_stop=7")
    st = json.loads(state.read_text(encoding="utf-8"))
    assert st["state"] == "stopped" and st["stopped_at"] == "x" and st["vm_lck_bytes_at_stop"] == 7
    ver = tmp_path / "verify.json"
    ver.write_text(json.dumps({**PINNER_JSON, "mode": "verify-only"}) + "\n", encoding="utf-8")
    text = run_helper(helper, "status-report", f"--state={state}", "--running=1", "--pid=4242",
                      "--vmlck=4000000000", f"--verify={ver}").stdout
    assert "pinner: running, pid 4242, VmLck 4.00 GB" in text and "4.10 GB of 5.00 GB hot-set bytes now" in text
    assert "expert" in text and "2.50 GB of 3.40 GB" in text
    doc = json.loads(run_helper(helper, "status-report", "--running=0", "--why=no pidfile", "--json=1").stdout)
    assert doc["running"] is False and doc["vm_lck_bytes"] is None


def _ab_fixture(tmp_path: Path, label: str, rnd: int, *, tg: float, rb: float, rc: int = 0,
                mode: str = "stream_moe", pinned: bool | None = None) -> dict[str, str]:
    """Synthetic outputs of one pi/05 stream-bench run, as ab-report reads them."""
    env_file = tmp_path / f"E8_{label}_{rnd}.json"
    env_file.write_text(json.dumps({"experiment": "E8", "kind": "measured", "extra": {"pi05": {
        "mode": mode, "effective_flags": "-nr -lm mmap -t 4", "pin": {"pid": 1, "vm_lck_bytes": 9} if pinned else None}}}),
        encoding="utf-8")
    last = tmp_path / f"r{rnd}_{label}.last.json"
    last.write_text(json.dumps({"metrics": [{"file": _p(env_file), "label": "stream-bench", "tg_tok_s": tg,
                                             "pp_tok_s": 9.0, "read_bytes_per_token": rb, "majflt_per_token": rb / 16384,
                                             "load_s": 3.0, "coherent": True, "failed": rc != 0}]}), encoding="utf-8")
    vb = tmp_path / f"r{rnd}_{label}.vb.json"
    vb.write_text(json.dumps({"tool": "pinner", "requested_bytes": 5e9, "resident_bytes": 2e9,
                              "mincore_reliable": True}) + "\n", encoding="utf-8")
    ps = ""
    if label == "pinned":
        p = tmp_path / f"r{rnd}_{label}.pin_state.json"
        p.write_text(json.dumps({"pid": 7, "state": "stopped", "summary": PINNER_JSON}), encoding="utf-8")
        ps = _p(p)
    return {"label": label, "round": str(rnd), "rc": str(rc), "last": _p(last), "files": _p(env_file),
            "vb": _p(vb), "va": _p(vb), "ps": ps}


def _ab_report(helper: Path, tmp_path: Path, runs: list[dict[str, str]], bpt: int = 3_000_000_000
               ) -> tuple[subprocess.CompletedProcess, dict[str, Any]]:
    hs = tmp_path / "hotset.json"
    hs.write_text(json.dumps({"total_bytes": 5e9, "n_experts": 10, "ranges": []}), encoding="utf-8")
    out = tmp_path / "pin_ab.json"
    args = ["ab-report", f"--out={out}", "--version=1.0.0", f"--hotset={hs}", f"--bpt-worst={bpt}", "--rounds=1",
            "--budget-bytes=5200000000", "--budget-source=auto", "--model=/m/x.gguf", "--model-id=26b-q4"]
    for r in runs:
        args += [f"--run-label={r['label']}", f"--run-round={r['round']}", f"--run-rc={r['rc']}",
                 f"--run-last={r['last']}", f"--run-files={r['files']}", f"--run-res-before={r['vb']}",
                 f"--run-res-after={r['va']}", f"--run-pin-state={r['ps']}", "--run-started=t0", "--run-ended=t1"]
    res = run_helper(helper, *args)
    doc = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
    return res, doc


def test_helper_ab_report_pass_fail_inconclusive(helper: Path, tmp_path: Path) -> None:
    from pigemma import results as rs
    # PASS: 1.3x tok/s
    res, doc = _ab_report(helper, tmp_path, [_ab_fixture(tmp_path, "unpinned", 1, tg=1.0, rb=300e6),
                                             _ab_fixture(tmp_path, "pinned", 1, tg=1.3, rb=280e6, pinned=True)])
    assert res.returncode == 0, res.stderr
    assert doc["threshold_check"]["result"] == "PASS" and doc["ratio"]["tok_s_pinned_over_unpinned"] == 1.3
    assert doc["ratio"]["read_bytes_per_token_pinned_over_unpinned"] == round(280 / 300, 4)
    assert doc["runs"][1]["pinner"]["locked_bytes"] == 4_000_000_000 and doc["runs"][0]["pinner"] is None
    assert doc["runs"][0]["hot_set_resident_before_bytes"] == 2e9 and doc["kind"] == "measured"
    assert not rs.is_envelope(doc) and "E6 check  PASS" in res.stderr
    # PASS on bytes alone: 25 % fewer bytes/token at equal speed
    _, doc = _ab_report(helper, tmp_path, [_ab_fixture(tmp_path, "unpinned", 1, tg=1.0, rb=400e6),
                                           _ab_fixture(tmp_path, "pinned", 1, tg=1.0, rb=300e6, pinned=True)])
    assert doc["threshold_check"]["result"] == "PASS" and doc["ratio"]["bytes_per_token_reduction"] == 0.25
    # FAIL: +5 % tok/s, -5 % bytes
    _, doc = _ab_report(helper, tmp_path, [_ab_fixture(tmp_path, "unpinned", 1, tg=1.0, rb=400e6),
                                           _ab_fixture(tmp_path, "pinned", 1, tg=1.05, rb=380e6, pinned=True)])
    assert doc["threshold_check"]["result"] == "FAIL"
    # INCONCLUSIVE: served from RAM (no storage reads unpinned)
    _, doc = _ab_report(helper, tmp_path, [_ab_fixture(tmp_path, "unpinned", 1, tg=3.0, rb=0.0, mode="in_ram"),
                                           _ab_fixture(tmp_path, "pinned", 1, tg=3.0, rb=0.0, mode="in_ram",
                                                       pinned=True)])
    assert doc["threshold_check"]["result"] == "INCONCLUSIVE" and doc["in_ram"] is True
    assert any("mode in_ram" in n for n in doc["notes"])
    # INCONCLUSIVE: the 'pinned' run had no live pinner
    _, doc = _ab_report(helper, tmp_path, [_ab_fixture(tmp_path, "unpinned", 1, tg=1.0, rb=400e6),
                                           _ab_fixture(tmp_path, "pinned", 1, tg=1.5, rb=100e6, pinned=False)])
    assert doc["threshold_check"]["result"] == "INCONCLUSIVE" and "no live pinner" in doc["threshold_check"]["reason"]
    # a failed run -> exit 1, INCONCLUSIVE
    res, doc = _ab_report(helper, tmp_path, [_ab_fixture(tmp_path, "unpinned", 1, tg=1.0, rb=400e6),
                                             _ab_fixture(tmp_path, "pinned", 1, tg=1.5, rb=100e6, rc=1, pinned=True)])
    assert res.returncode == 1 and doc["threshold_check"]["result"] == "INCONCLUSIVE"
    assert doc["pinned"]["n"] == 0 and doc["pinned"]["n_failed"] == 1


# ----------------------------------------------------------------------------- dry runs (fake roots)
@needs_map26
def test_dry_run_start_16gb_auto_budget(bash: str, tmp_path: Path) -> None:
    stats = stats_for_map26(tmp_path / "stats.json")
    r = run(bash, ["start", "--dry-run", "--model", "26b-q4", "--stats", _p(stats)], _env(tmp_path))
    assert r.returncode == 0, r.stderr
    lines = _lines(r.stdout)
    assert len(lines) == 2
    plan, pin = shlex.split(lines[0]), lines[1]
    assert plan[1:4] == ["-m", "pigemma.gguf_map", "hotset"] and plan[plan.index("--map") + 1].endswith("results/map26.json")
    assert plan[plan.index("--stats") + 1].endswith("stats.json") and plan[plan.index("--page-size") + 1] == "16384"
    # fake MemAvailable 16,000,000 kB = 16.38 GB > the file: the auto budget is capped at the file size
    assert plan[plan.index("--budget-bytes") + 1] == str(Q4_BYTES)
    assert plan[plan.index("--out") + 1] == _p(tmp_path / "out" / "hotset.json")
    assert pin.startswith("nohup nice -n 10 ") and "/pinner --gguf " in pin and pin.rstrip().endswith("&")
    assert "--verify --json --pidfile" in pin and _p(tmp_path / "out" / "hotset.ranges.txt") in pin
    assert "capped at the file size" in r.stderr and "pinner not found" in r.stderr
    assert not (tmp_path / "out" / "hotset.json").exists() and not (tmp_path / "out" / "pin_state.json").exists()


@needs_map26
def test_dry_run_explicit_budget_no_stats_and_config_budget(bash: str, tmp_path: Path) -> None:
    r = run(bash, ["start", "--dry-run", "--model", "26b-q4", "--budget-bytes", "5.2GB", "--no-stats",
                   "--rank", "weight", "--per-layer-min", "2"], _env(tmp_path))
    assert r.returncode == 0, r.stderr
    plan = shlex.split(_lines(r.stdout)[0])
    assert plan[plan.index("--budget-bytes") + 1] == "5200000000" and "--stats" not in plan
    assert plan[plan.index("--rank") + 1] == "weight" and plan[plan.index("--per-layer-min") + 1] == "2"
    r = run(bash, ["start", "--dry-run", "--model", "26b-q4", "--no-stats"], _env(tmp_path, PIN_BUDGET_BYTES="3GiB"))
    assert r.returncode == 0, r.stderr
    plan = shlex.split(_lines(r.stdout)[0])
    assert plan[plan.index("--budget-bytes") + 1] == str(3 * 2**30) and "PIN_BUDGET_BYTES" in r.stderr


@needs_map26
def test_dry_run_map_of_another_gguf_is_rebuilt_locally(bash: str, tmp_path: Path) -> None:
    m = json.loads(MAP26.read_text(encoding="utf-8"))
    m["gguf_basename"] = "something-else.gguf"
    other = tmp_path / "other_map.json"
    other.write_text(json.dumps(m), encoding="utf-8")
    r = run(bash, ["start", "--dry-run", "--model", "26b-q4", "--map", _p(other), "--no-stats"], _env(tmp_path))
    assert r.returncode == 0, r.stderr
    lines = _lines(r.stdout)
    build, plan = shlex.split(lines[0]), shlex.split(lines[1])
    target = _p(tmp_path / "out" / "map_gemma-4-26B_q4_0-it.json")
    assert build[1:4] == ["-m", "pigemma.gguf_map", "build"] and build[4].endswith(Q4_FILE)
    assert build[build.index("--out") + 1] == target and build[build.index("--page-size") + 1] == "16384"
    assert plan[plan.index("--map") + 1] == target
    assert "does not describe gemma-4-26B_q4_0-it.gguf" in r.stderr
    # a --map path that does not exist yet is built there
    want = tmp_path / "out" / "my_map.json"
    r = run(bash, ["start", "--dry-run", "--model", "26b-q4", "--map", _p(want), "--no-stats"], _env(tmp_path))
    build = shlex.split(_lines(r.stdout)[0])
    assert r.returncode == 0 and build[build.index("--out") + 1] == _p(want)


@needs_map26
def test_dry_run_stats_problems(bash: str, tmp_path: Path) -> None:
    r = run(bash, ["start", "--dry-run", "--model", "26b-q4", "--stats", _p(tmp_path / "nope.json")], _env(tmp_path))
    assert r.returncode == 3 and "stats file not found" in r.stderr
    bad = stats_for_map26(tmp_path / "bad.json", n_expert=64)
    r = run(bash, ["start", "--dry-run", "--model", "26b-q4", "--stats", _p(bad)], _env(tmp_path))
    assert r.returncode == 1 and "does not fit the map" in r.stderr


@needs_map26
def test_dry_run_low_ram_refuses_budget_below_nonexpert(bash: str, tmp_path: Path) -> None:
    fake = fake_root(tmp_path, mem_available_kb=2_000_000)
    r = run(bash, ["start", "--dry-run", "--model", "26b-q4", "--no-stats"], _env(tmp_path, fake))
    assert r.returncode == 3, r.stderr
    assert "smaller than the mandatory non-expert set" in r.stderr and r.stdout == ""
    # an explicit budget larger than MemAvailable - 512 MiB trips the pinner's RAM guard (checked up front)
    r = run(bash, ["start", "--dry-run", "--model", "26b-q4", "--no-stats", "--budget-bytes", "1.9e9"],
            _env(tmp_path, fake))
    assert r.returncode == 3 and "not enough available RAM to pin" in r.stderr


@needs_map26
def test_memlock_limit(bash: str, tmp_path: Path) -> None:
    fake = fake_root(tmp_path, ulimit_l="8192")
    r = run(bash, ["start", "--dry-run", "--model", "26b-q4", "--no-stats", "--budget-bytes", "4e9"],
            _env(tmp_path, fake))
    assert r.returncode == 3 and "ulimit -l is 8192 kB" in r.stderr and "pi/01_tune_os.sh --apply" in r.stderr
    r = run(bash, ["start", "--dry-run", "--model", "26b-q4", "--no-stats", "--budget-bytes", "4e9", "--best-effort"],
            _env(tmp_path, fake))
    assert r.returncode == 0, r.stderr
    assert "--best-effort" in _lines(r.stdout)[-1] and "partial pin" in r.stderr
    # a finite limit that covers the hot set is accepted with a note
    fake2 = fake_root(tmp_path / "f2", ulimit_l="8000000")
    r = run(bash, ["start", "--dry-run", "--model", "26b-q4", "--no-stats", "--budget-bytes", "4e9"],
            _env(tmp_path, fake2))
    assert r.returncode == 0 and "enough for this hot set" in r.stderr


def test_protected_host_paths(bash: str, tmp_path: Path) -> None:
    r = run(bash, ["start", "--dry-run", "--model", "26b-q4"], _env(tmp_path, "pi_proc_cf1_16gb"))
    assert r.returncode == 4 and "protected host 'context-friend-1'" in r.stderr and r.stdout == ""
    r = run(bash, ["start", "--dry-run", "--model", "26b-q4"], _env(tmp_path, "pi_proc_cf1_16gb", PIGEMMA_I_KNOW="1"))
    assert r.returncode == 4 and "not allowed on this protected host" in r.stderr
    env = _env(tmp_path, "pi_proc_cf1_16gb", PIGEMMA_I_KNOW="1", ALLOW_MLOCK="1")
    r = run(bash, ["start", "--dry-run", "--model", "26b-q4"], env)
    assert r.returncode == 4 and "unless --allow-protected-host is also given" in r.stderr
    r = run(bash, ["ab", "--dry-run", "--model", "26b-q4"], env)
    assert r.returncode == 4 and "--allow-protected-host" in r.stderr
    # with the explicit flag the bounded-experiment default (--max-hold 7200) applies; this fixture's
    # ulimit -l of 8192 kB then stops it (exit 3), never silently
    r = run(bash, ["start", "--dry-run", "--model", "26b-q4", "--allow-protected-host", "--no-stats",
                   "--budget-bytes", "4e9"], env)
    assert r.returncode == 3 and "after 7200s" in r.stderr and "ulimit -l is 8192 kB" in r.stderr
    r = run(bash, ["start", "--dry-run", "--model", "26b-q4"], _env(tmp_path, ALLOW_MLOCK="0"))
    assert r.returncode == 4 and "ALLOW_MLOCK=0 forbids it" in r.stderr


@needs_map26
def test_max_hold_and_protected_flag_reach_the_pinner(bash: str, tmp_path: Path) -> None:
    r = run(bash, ["start", "--dry-run", "--model", "26b-q4", "--no-stats", "--budget-bytes", "4e9", "--max-hold",
                   "600", "--allow-protected-host"], _env(tmp_path))
    assert r.returncode == 0, r.stderr
    pin = shlex.split(_lines(r.stdout)[-1].split(" < /dev/null")[0])
    assert pin[pin.index("--hold-seconds") + 1] == "600" and "--allow-protected-host" in pin


def test_split_gguf_is_refused(bash: str, tmp_path: Path) -> None:
    env = _env(tmp_path)
    part = tmp_path / "models" / "gemma-4-26B_q4_0-it-00001-of-00005.gguf"
    part.write_bytes(b"GGUF" + b"\0" * 60)
    r = run(bash, ["start", "--dry-run", "--model", _p(part)], env)
    assert r.returncode == 3 and "one part of a split GGUF" in r.stderr and r.stdout == ""


def test_real_runs_refused_under_a_fake_root(bash: str, tmp_path: Path) -> None:
    for sub in ("start", "ab"):
        r = run(bash, [sub, "--model", "26b-q4"], _env(tmp_path))
        assert r.returncode == 1 and "only runs with --dry-run here" in r.stderr


def test_status_and_stop_without_a_pinner(bash: str, tmp_path: Path) -> None:
    r = run(bash, ["status"], _env(tmp_path))
    assert r.returncode == 3 and r.stdout.startswith("pinner: not running")
    r = run(bash, ["status", "--json"], _env(tmp_path))
    assert r.returncode == 3 and json.loads(r.stdout)["running"] is False
    r = run(bash, ["stop"], _env(tmp_path))
    assert r.returncode == 0 and r.stdout.strip() == "pinner: not running"
    (tmp_path / "out" / "pinner.pid").write_text(f"{FAKE_PID}\n", encoding="utf-8")  # stale: no such process
    r = run(bash, ["stop"], _env(tmp_path))
    assert r.returncode == 0 and "stale" in r.stderr
    r = run(bash, ["stop"], _env(tmp_path, fake=None))  # the real system, no pidfile process: nothing signalled
    assert r.returncode == 0 and r.stdout.strip() == "pinner: not running"


@needs_map26
def test_fake_running_pinner(bash: str, tmp_path: Path) -> None:
    fake = fake_root(tmp_path, pinner_pid=FAKE_PID, vmlck_kb=4096)
    env = _env(tmp_path, fake)
    (tmp_path / "out" / "pinner.pid").write_text(f"{FAKE_PID}\n", encoding="utf-8")
    r = run(bash, ["status"], env)
    assert r.returncode == 0 and f"pinner: running, pid {FAKE_PID}, VmLck 4.2 MB locked" in r.stdout
    r = run(bash, ["status", "--json"], env)
    doc = json.loads(r.stdout)
    assert r.returncode == 0 and doc["running"] is True and doc["vm_lck_bytes"] == 4096 * 1024
    r = run(bash, ["stop"], env)
    assert r.returncode == 0 and "would send SIGTERM to pinner pid" in r.stdout and "no signal sent" in r.stdout
    r = run(bash, ["start", "--dry-run", "--model", "26b-q4"], env)
    assert r.returncode == 4 and "a pinner is already running" in r.stderr


@needs_map26
def test_dry_run_ab_plan(bash: str, tmp_path: Path) -> None:
    stats = stats_for_map26(tmp_path / "stats.json")
    r = run(bash, ["ab", "--dry-run", "--model", "26b-q4", "--stats", _p(stats), "--budget-bytes", "5.2e9",
                   "--rounds", "2", "-n", "32", "--max-hold", "900", "--run-arg=--want", "--run-arg=26b",
                   "--results-dir", _p(tmp_path / "results"), "--", "--idle-check-s", "0"], _env(tmp_path))
    assert r.returncode == 0, r.stderr
    lines = _lines(r.stdout)
    plan = shlex.split(lines[0])
    assert plan[plan.index("--budget-bytes") + 1] == "5200000000"
    runs = [shlex.split(ln) for ln in lines if "05_run.sh" in ln]
    verifies = [ln for ln in lines if "--verify-only --json" in ln]
    assert len(runs) == 4 and len(verifies) == 4
    labels = ["pinned" if "--pin" in w else "unpinned" for w in runs]
    assert labels == ["unpinned", "pinned", "pinned", "unpinned"]  # ABBA over two rounds
    for w in runs:
        assert w[0] == "bash" and w[1].endswith("pi/05_run.sh") and w[2] == "stream-bench"
        assert w[w.index("--model") + 1].endswith(Q4_FILE) and "--no-calibrate" in w and "--no-he" in w
        assert w[w.index("-n") + 1] == "32" and w[w.index("--out") + 1] == _p(tmp_path / "results")
        assert w[w.index("--want") + 1] == "26b" and w[-2:] == ["--idle-check-s", "0"]
    pinned = runs[1]
    assert pinned[pinned.index("--pin-arg=--hotset") + 1] == "--pin-arg=" + _p(tmp_path / "out" / "hotset.json")
    assert "--pin-arg=--max-hold" in pinned and "--pin-arg=900" in pinned and "--pin-arg=--pinner" in pinned
    assert "pin_ab.json" in r.stderr and not (tmp_path / "results" / "pin_ab.json").exists()


# ------------------------------------------------------------------------- Linux: the real pinner
def _real_pinner() -> Path | None:
    for p in (REPO / "tools/build/bin/pinner",
              REPO / "tools/build-linux-x86/bin/pinner" if platform.machine() == "x86_64" else None):
        if p is not None and p.is_file() and os.access(p, os.X_OK):
            return p
    return None


def _tiny_catalog(tmp_path: Path, gguf: Path, summary: dict[str, Any]) -> Path:
    cat = json.loads((REPO / "pigemma/data/catalog.json").read_text(encoding="utf-8"))
    size = gguf.stat().st_size
    cat["models"].append({
        "id": "tiny-moe-test", "family": "26b", "model": "tiny-test", "kind": "moe", "quant": "Q4_0", "qat": False,
        "repo": "test/tiny", "revision": None, "file": gguf.name, "local_path": None, "bytes": size,
        "bytes_exact": True, "estimated": False, "nonexpert_bytes": summary["nonexpert_bytes"],
        "expert_bytes": summary["expert_bytes"], "expert_unit_bytes": summary["expert_unit_bytes"],
        "expert_bytes_per_token_worst": summary["expert_bytes_per_token_worst"],
        "nonexpert_bytes_per_token": summary["nonexpert_bytes"], "n_layer": 4, "n_moe_layers": 4, "n_expert": 16,
        "n_expert_used": 4, "drafts": [], "sources": {"bytes": "assumed: test fixture"}})
    p = tmp_path / "catalog.json"
    p.write_text(json.dumps(cat, indent=1), encoding="utf-8")
    return p


@pytest.mark.linux
def test_start_status_stop_and_ab_with_the_real_pinner(bash: str, tmp_path: Path) -> None:
    pinner = _real_pinner()
    if pinner is None:
        pytest.skip("no pinner binary (pi/02_build_llama.sh --with-tools or pc/docker_run.sh build-tools)")
    pytest.importorskip("numpy")
    pytest.importorskip("gguf")
    import synth
    from pigemma import gguf_map as gm
    gguf = tmp_path / "models" / "tiny-moe-q4_0.gguf"
    synth.write_tiny_gguf(gguf, n_layer=4, n_embd=256, n_ff_exp=512, n_expert=16, n_expert_used=4)
    summary = gm.build_map(gguf, page_size=4096, with_sha=False).to_dict()["summary"]
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"n_tokens": 100, "per_layer": [
        {"layer": il, "freq": [(il + 3 * e) % 17 for e in range(16)]} for il in range(4)]}), encoding="utf-8")
    cat = _tiny_catalog(tmp_path, gguf, summary)
    env = _env(tmp_path, fake=None, PIGEMMA_TOOLS_BIN=_p(pinner.parent), PIGEMMA_CATALOG=_p(cat),
               PIGEMMA_QUIET="1")
    out = tmp_path / "out"
    r = run(bash, ["start", "--model", _p(gguf), "--stats", _p(stats), "--budget-bytes", "4MiB"], env)
    assert r.returncode == 0, r.stderr
    pid = int((out / "pinner.pid").read_text().strip())
    assert Path(f"/proc/{pid}/comm").read_text().strip().startswith("pinner")
    vmlck = next(int(ln.split()[1]) for ln in Path(f"/proc/{pid}/status").read_text().splitlines()
                 if ln.startswith("VmLck:"))
    st = json.loads((out / "pin_state.json").read_text(encoding="utf-8"))
    assert st["state"] == "running" and st["summary"]["status"] == "ok" and vmlck * 1024 == st["summary"]["locked_bytes"]
    assert st["hotset_info"]["total_bytes"] <= 4 * 2**20 and (out / "map_tiny-moe-q4_0.json").is_file()
    try:
        r = run(bash, ["status", "--json"], env)
        assert r.returncode == 0 and json.loads(r.stdout)["running"] is True
        r = run(bash, ["start", "--model", _p(gguf), "--stats", _p(stats), "--budget-bytes", "4MiB"], env)
        assert r.returncode == 4 and "already running" in r.stderr
    finally:
        r = run(bash, ["stop"], env)
    assert r.returncode == 0 and "pinner: stopped" in r.stdout and not Path(f"/proc/{pid}").exists()
    assert json.loads((out / "pin_state.json").read_text(encoding="utf-8"))["state"] == "stopped"
    assert run(bash, ["status"], env).returncode == 3
    # ab end to end through pi/05 with the llama-completion double
    fake = FIX / "bench" / "fake_llama_completion.py"
    comp = tmp_path / "bin" / "llama-completion"
    comp.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake}" "$@"\n', encoding="utf-8")
    comp.chmod(0o755)
    env["PIGEMMA_LLAMA_BIN"] = _p(tmp_path / "bin")
    r = run(bash, ["ab", "--model", _p(gguf), "--stats", _p(stats), "--budget-bytes", "4MiB", "--results-dir",
                   _p(tmp_path / "results"), "-n", "16", "--", "--want", "tiny-moe-test", "--idle-check-s", "0",
                   "--feasibility-arg=--io-mem", "--feasibility-arg=none", "--feasibility-arg=--calibration",
                   "--feasibility-arg=none", f"--feasibility-arg=--catalog", f"--feasibility-arg={_p(cat)}"], env,
            timeout=600)
    assert r.returncode == 0, r.stderr[-4000:]
    doc = json.loads((tmp_path / "results" / "pin_ab.json").read_text(encoding="utf-8"))
    assert [x["label"] for x in doc["runs"]] == ["unpinned", "pinned"]
    assert doc["runs"][1]["pinned_by_pi05"] is True and doc["runs"][0]["pinned_by_pi05"] is False
    assert doc["runs"][1]["pinner"]["locked_bytes"] > 0 and doc["ratio"]["tok_s_pinned_over_unpinned"] is not None
    assert not (out / "pinner.pid").exists()
    assert str(tmp_path / "results" / "pin_ab.json") in r.stdout.replace("\\", "/") or "pin_ab.json" in r.stdout
