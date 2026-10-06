"""Tests for pi/05_run.sh (manifest files[] 'pi/05_run.sh').

Manifest test field: `bash -n`; argument parsing and refusal paths with the fake-proc fixtures; bench
parse covered by tests/test_bench_parse.py (pi/05 delegates the llama-bench / llama-completion runs and
their parsing to `python3 -m pigemma.bench`, so that module's tests cover the parsers).

What runs where:
- everywhere (Windows Git Bash, Linux, the Pi): syntax, usage line == manifest, option parsing and usage
  errors, the embedded stdlib helper (model selection, variant planning, calibration, envelope
  annotation, last_run.json) as a unit, `--dry-run` command lines with the PIGEMMA_FAKE_PROC fixtures,
  every refusal path (protected host, busy host, not viable, RAM gate in fits and streaming mode, model or
  binary missing, port in use, held bench lock, --pin without mlock permission, every bench variant
  refused), and chat / server end to end against bash test doubles of llama-cli / llama-server.
- Linux only (`linux` marker; the Pi, a container): bench and stream-bench end to end through
  `pigemma.bench` with the stdlib llama-bench / llama-completion doubles of tests/fixtures/bench/
  (Windows Python cannot execute an extensionless script as a binary), including the calibration file.

Nothing here loads a model or touches models/, results/ or pi/out/: every run gets its own PIGEMMA_OUT,
MODEL_DIR, PIGEMMA_LLAMA_BIN, PIGEMMA_LOCK_FILE and --out / --calibration-out under tmp_path, and the
feasibility verdict is hermetic (--io-mem none --calibration none).
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures"
SCRIPT = "pi/05_run.sh"
HERMETIC = ["--feasibility-arg=--io-mem", "--feasibility-arg=none",
            "--feasibility-arg=--calibration", "--feasibility-arg=none"]
CONFIG_VARS = (
    "PIGEMMA_HOME", "MODEL_DIR", "LLAMA_DIR", "LLAMA_COMMIT", "PIGEMMA_OUT", "THREADS", "CTX", "QUANT_OVERRIDE",
    "OS_RESERVE_GIB", "MIN_FREE_DISK_FACTOR", "KEEP_FREE_GB", "NICE_LEVEL", "DENY_HOSTS",
    "PROTECTED_SERVICE_GLOBS", "ALLOW_MLOCK", "TUNE_APPLY", "TUNE_NVME_GEN3", "TUNE_GOVERNOR", "TUNE_SWAPPINESS",
    "TUNE_READAHEAD_KB", "TUNE_MEMLOCK", "BUILD_JOBS", "HF_LIMIT_RATE", "PIN_BUDGET_BYTES",
)
MTP_26B = "mtp-gemma-4-26B-A4B-it-Q4_0.gguf"
Q3KM_FILE = "gemma-4-26B-A4B-it-UD-Q3_K_M.gguf"
FITS_MAPPED_FLAGS = "-nr -lm mmap+mlock -t 4 -c 4096 -fa on -ctk q8_0 -ctv q8_0"
STREAM_FLAGS = "-nr -lm mmap -t 4 -c 4096 -ctk q8_0 -ctv q8_0 -b 256 -ub 64"
BPT_Q3KM = 2399894648 + 644548608  # catalog nonexpert_bytes_per_token + expert_bytes_per_token_worst
ARGV_ECHO = "#!/usr/bin/env bash\nprintf 'FAKE-ARGV:'\nprintf ' [%s]' \"$@\"\nprintf '\\n'\nexit \"${FAKE_RC:-0}\"\n"


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
    env["PIGEMMA_LLAMA_BIN"] = _p(tmp_path / "bin")
    env["PIGEMMA_LOCK_FILE"] = _p(tmp_path / "bench.lock")
    env["LC_ALL"] = env.get("LC_ALL") or "C.UTF-8"
    if fake:
        env["PIGEMMA_FAKE_PROC"] = _p(fake if isinstance(fake, Path) else FIX / fake)
    env.update(extra)
    return env


def run(bash: str, args: list[str], env: dict[str, str], *, timeout: float = 240) -> subprocess.CompletedProcess:
    return subprocess.run([bash, SCRIPT, *args], cwd=REPO, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def _out(tmp_path: Path) -> list[str]:
    return ["--out", _p(tmp_path / "results"), "--calibration-out", _p(tmp_path / "calibration.json")]


def fake_root(tmp_path: Path, base: str = "pi_proc_16gb", *, mem_total_kb: int | None = None,
              mem_available_kb: int | None = None) -> Path:
    """Copy of a fake system root with MemTotal / MemAvailable rewritten."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    dst = tmp_path / f"proc_{base}"
    shutil.copytree(FIX / base, dst)
    mi = dst / "meminfo"
    text = mi.read_text(encoding="utf-8")
    if mem_total_kb is not None:
        text = re.sub(r"(?m)^MemTotal:\s+\d+ kB$", f"MemTotal:       {mem_total_kb} kB", text)
    if mem_available_kb is not None:
        text = re.sub(r"(?m)^MemAvailable:\s+\d+ kB$", f"MemAvailable:   {mem_available_kb} kB", text)
    mi.write_bytes(text.encode("utf-8"))
    return dst


def write_exe(path: Path, body: str) -> Path:
    path.write_bytes(body.encode("utf-8"))
    path.chmod(0o755)
    return path


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def kv(stdout: str) -> dict[str, str]:
    """NAME='value' lines (shlex-quoted) -> dict."""
    out: dict[str, str] = {}
    for line in stdout.splitlines():
        m = re.match(r"^([A-Z][A-Z0-9_]*)=(.*)$", line)
        if m:
            words = shlex.split(m.group(2)) if m.group(2) else [""]
            out[m.group(1)] = words[0] if words else ""
    return out


def _load(line: str) -> dict[str, Any]:
    p = Path(line)
    return json.loads((p if p.is_absolute() else REPO / p).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------ embedded python helper
def helper_source() -> str:
    text = (REPO / SCRIPT).read_text(encoding="utf-8")
    marker = "<<'PIGEMMA_PY_EOF'\n"
    start = text.index(marker) + len(marker)
    end = text.index("\nPIGEMMA_PY_EOF\n", start)
    return text[start:end + 1]


@pytest.fixture(scope="module")
def helper(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("pi05_helper") / "pi05_helper.py"
    p.write_text(helper_source(), encoding="utf-8")
    return p


def run_helper(helper: Path, *args: str, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PI05_")}
    env.update(PYTHONPATH=str(REPO), PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    env.update(env_extra or {})
    return subprocess.run([sys.executable, str(helper), *args], cwd=REPO, env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=120)


def envelope(path: Path, **fields: Any) -> Path:
    """A valid measured E8 envelope (pigemma.results) for the 26b-ud-q3_k_m, written to path."""
    from pigemma import results as rs
    base: dict[str, Any] = dict(
        model={"id": "26b-ud-q3_k_m", "file": Q3KM_FILE, "quant": "UD-Q3_K_M", "bytes": 12728497888,
               "sha_prefix": None},
        variant="norepack+kvq8", flags=["-nr", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"], threads=4,
        n_ctx=96, tg={"tok_s": 2.1, "sd": None, "n": 1})
    base.update(fields)
    base = {k: v for k, v in base.items() if v is not None}
    doc = rs.new_result("E8", "measured", **base)
    path.write_text(rs.dumps_result(doc), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------- basics
def test_bash_syntax(bash: str) -> None:
    r = subprocess.run([bash, "-n", SCRIPT], cwd=REPO, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr


def test_lf_line_endings_and_shebang() -> None:
    data = (REPO / SCRIPT).read_bytes()
    assert b"\r" not in data
    assert data.startswith(b"#!/usr/bin/env bash\n")
    assert b"set -euo pipefail" in data


def test_usage_line_matches_manifest(bash: str, tmp_path: Path) -> None:
    manifest = json.loads((REPO / "docs/design/manifest.json").read_text(encoding="utf-8"))
    entry = next(f for f in manifest["files"] if f["path"] == SCRIPT)
    m = re.match(r"`([^`]+)`", entry["spec"])
    assert m is not None
    usage = m.group(1)
    text = (REPO / SCRIPT).read_text(encoding="utf-8")
    assert f'PI05_USAGE_LINE="{usage}"' in text
    r = run(bash, ["--help"], _env(tmp_path, fake=None))
    assert r.returncode == 0
    assert usage in r.stdout and "Exit codes" in r.stdout and "--calibration-out" in r.stdout


@pytest.mark.parametrize("args,needle", [
    ([], "no subcommand"),
    (["bench", "--bogus"], "unknown option"),
    (["bench", "chat"], "more than one subcommand"),
    (["bench", "--model"], "needs a value"),
    (["frobnicate"], "unexpected argument"),
    (["chat", "--variants", "norepack"], "not used by 'chat'"),
    (["bench", "--draft", "auto"], "not used by 'bench'"),
    (["stream-bench", "--port", "9000"], "not used by 'stream-bench'"),
    (["server", "-p", "hello"], "server takes no -p"),
    (["chat", "--spec-type", "draft-mtp"], "--spec-type needs --draft"),
    (["server", "--port", "70000"], "--port must be 1..65535"),
    (["chat", "-t", "2,3"], "only for bench"),
    (["bench", "-r", "0"], "-r/--reps must be an integer >= 1"),
    (["bench", "--pp", "0", "--tg", "0"], "nothing to measure"),
    (["bench", "-p", "64", "--pp", "64"], "give the prompt tokens once"),
    (["stream-bench", "-p", "x", "-f", "y.txt"], "either -p TEXT or -f FILE"),
    (["stream-bench", "-n", "0"], "-n must be an integer >= 1"),
    (["bench", "--experiment", "E9"], "--experiment must be E0..E8"),
    (["bench", "--idle-check-s", "soon"], "--idle-check-s must be"),
])
def test_usage_errors_exit_1(bash: str, tmp_path: Path, args: list[str], needle: str) -> None:
    r = run(bash, args, _env(tmp_path, fake=None), timeout=60)
    assert r.returncode == 1, (r.stdout, r.stderr)
    assert needle in r.stderr
    assert "/llama-" not in r.stdout  # nothing ran


def test_version(bash: str, tmp_path: Path) -> None:
    r = run(bash, ["--version"], _env(tmp_path, fake=None), timeout=60)
    assert r.returncode == 0 and r.stdout.startswith("pi/05_run.sh ")


# ------------------------------------------------------------------------------- helper: planning
def test_helper_plan_bench_variants(helper: Path) -> None:
    r = run_helper(helper, "plan-bench", f"--flags={FITS_MAPPED_FLAGS}",
                   "--variants=default,norepack,nommap,hybrid_negctrl,norepack+kvq8,mlock",
                   "--anon-ok=0", "--anon-reason=no room", "--mlock-ok=1")
    assert r.returncode == 0, r.stderr
    lines = [ln.split("\t") for ln in r.stdout.splitlines()]
    runs = [ln for ln in lines if ln[0] == "RUN"]
    refused = {ln[1]: ln[2] for ln in lines if ln[0] == "REFUSED"}
    # the verdict's -lm mmap+mlock and q8_0 KV become variant parts; duplicates collapse
    assert runs == [["RUN", "norepack+mlock+kvq8", ""]]
    assert set(refused) == {"default", "nommap", "hybrid_negctrl", "mlock"}
    assert all(v == "no room" for v in refused.values())
    # mlock not allowed: -lm mmap instead of the mlock part
    r = run_helper(helper, "plan-bench", f"--flags={FITS_MAPPED_FLAGS}", "--variants=norepack", "--anon-ok=0",
                   "--mlock-ok=0")
    assert r.stdout.splitlines() == ["RUN\tnorepack+kvq8\t-lm mmap"]
    # streaming flags: KV q8_0 without -fa is not the kvq8 variant; batch sizes are kept
    r = run_helper(helper, "plan-bench", f"--flags={STREAM_FLAGS}", "--variants=norepack", "--mlock-ok=1",
                   "--extra=--poll 0")
    assert r.stdout.splitlines() == ["RUN\tnorepack\t-lm mmap -ctk q8_0 -ctv q8_0 -b 256 -ub 64 --poll 0"]
    # anonymous copy allowed: default runs without --repack 0
    r = run_helper(helper, "plan-bench", "--flags=-lm mmap -t 4 -c 4096 -fa on -ctk q8_0 -ctv q8_0",
                   "--variants=default,norepack", "--anon-ok=1", "--mlock-ok=1")
    assert r.stdout.splitlines() == ["RUN\tkvq8\t-lm mmap", "RUN\tnorepack+kvq8\t-lm mmap"]
    # --no-verdict-flags: only the variant
    r = run_helper(helper, "plan-bench", f"--flags={FITS_MAPPED_FLAGS}", "--variants=norepack",
                   "--no-verdict-flags=1")
    assert r.stdout.splitlines() == ["RUN\tnorepack\t"]
    for bad in ("turbo", "default+norepack", "+"):
        r = run_helper(helper, "plan-bench", "--flags=", f"--variants={bad}")
        assert r.returncode == 1 and "unknown or malformed variant" in r.stderr, bad


def test_helper_plan_run_and_split(helper: Path) -> None:
    v = kv(run_helper(helper, "plan-run", f"--flags={FITS_MAPPED_FLAGS}").stdout)
    assert v == {"PLAN_VARIANT": "norepack+mlock+kvq8", "PLAN_REST": "", "PLAN_THREADS": "4", "PLAN_CTX": "4096"}
    v = kv(run_helper(helper, "plan-run", f"--flags={STREAM_FLAGS}", "--extra=--api-key 'a b'").stdout)
    assert v["PLAN_VARIANT"] == "norepack"
    assert shlex.split(v["PLAN_REST"]) == ["-lm", "mmap", "-ctk", "q8_0", "-ctv", "q8_0", "-b", "256", "-ub", "64",
                                          "--api-key", "a b"]
    v = kv(run_helper(helper, "plan-run", "--flags=-lm none -t 2 -c 512").stdout)
    assert v["PLAN_VARIANT"] == "nommap" and v["PLAN_THREADS"] == "2" and v["PLAN_CTX"] == "512"
    r = run_helper(helper, "split", "--text=-nr --api-key 'a b' --host 0.0.0.0")
    assert shlex.split(r.stdout) == ["-nr", "--api-key", "a b", "--host", "0.0.0.0"]
    r = run_helper(helper, "split", "--text=-p 'unterminated")
    assert r.returncode == 1


# ---------------------------------------------------------------------------- helper: calibration
def test_helper_calibrate_measured_with_history(helper: Path, tmp_path: Path) -> None:
    from pigemma import feasibility as feas
    tg4 = envelope(tmp_path / "tg4.json", tg={"tok_s": 2.1, "sd": 0.01, "n": 3})
    tg2 = envelope(tmp_path / "tg2.json", threads=2, tg={"tok_s": 1.4, "sd": 0.01, "n": 3})
    pp4 = envelope(tmp_path / "pp4.json", tg=None, pp={"tok_s": 9.0, "sd": None, "n": 3})
    io_mem = tmp_path / "io_mem.json"
    io_mem.write_text(json.dumps({"memory": {"read_gbps_by_threads": {"1": 5.1, "4": 9.8},
                                             "copy_gbps": 7.0, "source": "measured"}}), encoding="utf-8")
    out = tmp_path / "pi" / "calibration.json"
    args = ["calibrate", "--mode=in_ram", f"--bpt={BPT_Q3KM}", "--threads=4", "--prefer-repack=0",
            f"--io-mem={io_mem}", f"--out={out}", "--run=bench", str(pp4), str(tg2), str(tg4)]
    r = run_helper(helper, *args, env_extra={"PI05_HOSTNAME": "pi5-test", "PI05_VERDICT": "FITS_REPACKED"})
    assert r.returncode == 0, r.stderr
    v = kv(r.stdout)
    eff = 2.1 * BPT_Q3KM / 9.8e9
    assert v["CAL_WRITTEN"] == "1" and v["CAL_KIND"] == "measured" and v["CAL_VALUE"] == f"{eff:.4f}"
    cal = json.loads(out.read_text(encoding="utf-8"))
    le = cal["llama_efficiency"]
    assert le["value"] == round(eff, 4) and le["from"] == [2.1, BPT_Q3KM, 9.8]
    assert le["source"].startswith("measured: pi/05_run.sh bench") and "tg4.json" in le["source"]
    assert cal["inputs"]["threads"] == 4 and cal["host"] == "pi5-test" and cal["history"] == []
    # the consumers read it
    val, src = feas.load_calibration(out)
    assert val == round(eff, 4) and src.startswith("measured")
    # a second calibration keeps the first one in history
    r = run_helper(helper, *args)
    assert r.returncode == 0, r.stderr
    cal2 = json.loads(out.read_text(encoding="utf-8"))
    assert len(cal2["history"]) == 1 and cal2["history"][0]["llama_efficiency"]["value"] == round(eff, 4)


def test_helper_calibrate_assumed_bandwidth_from_verdict(helper: Path, tmp_path: Path) -> None:
    tg = envelope(tmp_path / "tg.json")
    fz = tmp_path / "feasibility.json"
    fz.write_text(json.dumps({"device": {"mem_bw_gbps": {"value": 10.0, "source": "assumed: Pi 5 LPDDR4X"}}}),
                  encoding="utf-8")
    out = tmp_path / "cal.json"
    r = run_helper(helper, "calibrate", "--mode=in_ram", f"--bpt={BPT_Q3KM}", "--threads=4",
                   f"--io-mem={tmp_path / 'missing.json'}", f"--feasibility={fz}", f"--out={out}",
                   "--run=stream-bench", str(tg))
    assert r.returncode == 0, r.stderr
    assert kv(r.stdout)["CAL_KIND"] == "assumed"
    le = json.loads(out.read_text(encoding="utf-8"))["llama_efficiency"]
    assert le["source"].startswith("assumed:") and "run pi/03_bench_io_mem.sh" in le["source"]
    assert le["from"][2] == 10.0


@pytest.mark.parametrize("case,why", [
    ("stream_mode", "is not in-RAM"),
    ("storage_reads", "no measured tg tok/s from an in-RAM run"),
    ("out_of_range", "outside 0.05..1.5"),
    ("no_bpt", "bytes per token unknown"),
    ("failed_run", "no measured tg tok/s"),
])
def test_helper_calibrate_skips(helper: Path, tmp_path: Path, case: str, why: str) -> None:
    mode, bpt = "in_ram", BPT_Q3KM
    if case == "stream_mode":
        mode = "stream_moe"
        f = envelope(tmp_path / "e.json")
    elif case == "storage_reads":
        f = envelope(tmp_path / "e.json", read_bytes_per_token=0.1 * BPT_Q3KM)
    elif case == "out_of_range":
        f = envelope(tmp_path / "e.json", tg={"tok_s": 500.0, "sd": None, "n": 1})
    elif case == "no_bpt":
        bpt = 0
        f = envelope(tmp_path / "e.json")
    else:
        f = envelope(tmp_path / "e.json", tg=None, notes=["failed: llama-completion exit code 3"])
    io_mem = tmp_path / "io_mem.json"
    io_mem.write_text(json.dumps({"memory": {"read_gbps_by_threads": {"4": 9.8}}}), encoding="utf-8")
    out = tmp_path / "cal.json"
    r = run_helper(helper, "calibrate", f"--mode={mode}", f"--bpt={bpt}", "--threads=4", f"--io-mem={io_mem}",
                   f"--out={out}", str(f))
    assert r.returncode == 0, r.stderr
    v = kv(r.stdout)
    assert v["CAL_WRITTEN"] == "0" and why in v["CAL_REASON"]
    assert not out.exists()


# ------------------------------------------------------------------- helper: annotate / last-run
def test_helper_annotate_summary_and_last_run(helper: Path, tmp_path: Path) -> None:
    from pigemma import results as rs
    raw = tmp_path / "raw_stdout.json"
    raw.write_text(json.dumps([{"n_threads": 4, "avg_ts": 2.0}]), encoding="utf-8")
    bench_doc = envelope(tmp_path / "bench_tg.json", created_at="2026-10-05T12:00:00Z",
                         extra={"token_source": "random token ids (llama-bench)", "raw_stdout": str(raw)})
    comp_doc = envelope(tmp_path / "comp.json", read_bytes_per_token=1.5e6, majflt_per_token=3.0,
                        output_sample="<|channel>thought\nhmm<channel|>Experts are routed per token.",
                        buffer_split={"CPU_Mapped_mib": 12000.0})
    vm = tmp_path / "vmstat.log"
    vm.write_text(
        "procs -----------memory---------- ---swap-- -----io---- -system-- ------cpu-----\n"
        " r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st\n"
        " 1  0      0 900000  10000 200000    0    0   500    10  100  200  5  1 93  1  0\n"
        " 4  1      0 800000  10000 300000    0    0 90000     0 3000 4000 80  5  5 10  0\n"
        " 4  0      0 790000  10000 310000    0    0 70000     0 3000 4000 85  5  5  5  0\n", encoding="utf-8")
    raw_dir = tmp_path / "rawcopies"
    env = {"PI05_PROTECTED_OVERRIDE": "1", "PI05_HOSTNAME": "context-friend-1", "NICE_LEVEL": "10",
           "PI05_MLOCK_REMOVED": "1", "PI05_FORCED_NR": "1", "PI05_MODE": "stream_moe",
           "PI05_VERDICT": "STREAM_SLOW", "PI05_SUBCMD": "bench", "PI05_VERSION": "9.9",
           "PI05_ARGV_JSON": json.dumps(["bench", "--variants", "norepack"]), "PI05_GATE_NEED": "2077172088",
           "PI05_MEM_AVAILABLE": "16000000000", "PI05_GATE_RULE": "streaming", "PI05_BENCH_RAW_DIR": str(raw_dir),
           "PI05_THERMAL_BEFORE": "temp_c=41.2;throttled=0x0", "PI05_THERMAL_AFTER": "temp_c=55.0;throttled=0x0",
           "PI05_PINNED": "1", "PI05_PIN_PID": "4242", "PI05_PIN_BYTES": "5200000000",
           "PI05_MODEL_ID": "26b-ud-q3_k_m", "PI05_FLAGS_EFF": STREAM_FLAGS}
    r = run_helper(helper, "annotate", "--label=bench norepack", str(bench_doc), env_extra=env)
    assert r.returncode == 0, r.stderr
    r = run_helper(helper, "annotate", "--label=stream", f"--vmstat={vm}", str(comp_doc), env_extra=env)
    assert r.returncode == 0, r.stderr
    doc = json.loads(bench_doc.read_text(encoding="utf-8"))
    assert rs.check_result(doc)[0] == []
    pi = doc["extra"]["pi05"]
    assert pi["verdict"] == "STREAM_SLOW" and pi["mode"] == "stream_moe" and pi["argv"][0] == "bench"
    assert pi["protected_override"] is True and pi["mlock_removed"] is True and pi["forced_nr"] is True
    assert pi["ram_gate"] == {"need_bytes": 2077172088, "mem_available_bytes": 16000000000, "rule": "streaming"}
    assert pi["thermal"]["before"] == {"temp_c": 41.2, "throttled": "0x0"}
    assert pi["pin"] == {"pid": "4242", "vm_lck_bytes": 5200000000, "started_here": False}
    notes = " | ".join(doc["notes"])
    assert "PIGEMMA_I_KNOW=1" in notes and "random token ids" in notes and "mlock removed" in notes
    copies = sorted(raw_dir.glob("bench_26b-ud-q3_k_m_norepack+kvq8_*.json"))
    assert len(copies) == 1 and json.loads(copies[0].read_text(encoding="utf-8"))[0]["avg_ts"] == 2.0
    assert pi["raw_bench_json"].endswith(copies[0].name)
    # completion envelope: vmstat summary (first sample dropped), no raw copy
    doc2 = json.loads(comp_doc.read_text(encoding="utf-8"))
    vs = doc2["extra"]["pi05"]["vmstat"]
    assert vs["n_samples"] == 2 and vs["mean"]["bi"] == 80000 and vs["max"]["wa"] == 10
    assert "raw_bench_json" not in doc2["extra"]["pi05"]
    # annotating again does not duplicate notes
    run_helper(helper, "annotate", "--label=bench norepack", str(bench_doc), env_extra=env)
    assert json.loads(bench_doc.read_text(encoding="utf-8"))["notes"] == doc["notes"]
    # summary prints the visible text only (thinking channel stripped)
    r = run_helper(helper, "summary", "--label=EN", "--show-output=1", str(comp_doc))
    assert "Experts are routed per token." in r.stderr and "read 1.50 MB/token" in r.stderr
    assert "hmm" not in r.stderr.split("-----", 1)[1]
    # last-run
    lr1, lr2 = tmp_path / "last_run.json", tmp_path / "last_bench.json"
    r = run_helper(helper, "last-run", f"--out={lr1}", f"--out={lr2}", "--rc=1", "--labels=bench:a,stream",
                   str(bench_doc), str(comp_doc), env_extra=env)
    assert r.returncode == 0, r.stderr
    last = json.loads(lr1.read_text(encoding="utf-8"))
    assert lr2.read_text(encoding="utf-8") == lr1.read_text(encoding="utf-8")
    assert last["exit_code"] == 1 and last["subcommand"] == "bench" and last["pinned"] is True
    assert [m["label"] for m in last["metrics"]] == ["bench:a", "stream"]
    assert last["metrics"][0]["tg_tok_s"] == 2.1 and last["metrics"][1]["read_bytes_per_token"] == 1.5e6


# ------------------------------------------------------------------------ helper: model selection
def _feasibility(facts: Path, out: Path) -> None:
    r = subprocess.run([sys.executable, "-m", "pigemma.feasibility", "--facts", str(facts), "--json", str(out),
                        "--io-mem", "none", "--calibration", "none", "--quiet"], cwd=REPO, capture_output=True,
                       text=True, timeout=120, env={**os.environ, "PYTHONPATH": str(REPO)})
    assert out.is_file(), r.stderr


@pytest.fixture(scope="module")
def verdict_16gb(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """pigemma.feasibility on the 16 GB NVMe facts fixture (26b-q4 FITS_MAPPED, UD quants FITS_REPACKED)."""
    out = tmp_path_factory.mktemp("verdict") / "feasibility.json"
    _feasibility(FIX / "facts_pi5-16gb-nvme-gen3.json", out)
    return out


def _select(helper: Path, fz: Path, model_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    return run_helper(helper, "select", f"--feasibility={fz}", f"--catalog={REPO / 'pigemma/data/catalog.json'}",
                      f"--model-dir={model_dir}", "--mem-available=16000000000", *extra)


def test_helper_select(helper: Path, verdict_16gb: Path, tmp_path: Path) -> None:
    md = tmp_path / "models"
    md.mkdir()
    # nothing downloaded: the recommendation, marked missing
    r = _select(helper, verdict_16gb, md)
    v = kv(r.stdout)
    assert r.returncode == 0 and v["SEL_MODEL_ID"] == "26b-q4" and v["SEL_MODEL_EXISTS"] == "0"
    assert v["SEL_VERDICT"] == "FITS_MAPPED" and v["SEL_FITS_REPACKED"] == "0" and v["SEL_MODE"] == "in_ram"
    assert v["SEL_FLAGS"] == FITS_MAPPED_FLAGS and v["SEL_BYTES"] == "14439363584"
    assert v["SEL_NONEXPERT"] == "1577172088" and v["SEL_BPT"] == str(1577172088 + 802897920)
    # only a 3-bit quant downloaded: the best downloaded runnable row is used, with a note
    (md / Q3KM_FILE).write_bytes(b"GGUF" + b"\0" * 60)
    v = kv(_select(helper, verdict_16gb, md).stdout)
    assert v["SEL_MODEL_ID"] == "26b-ud-q3_k_m" and v["SEL_MODEL_EXISTS"] == "1"
    assert v["SEL_SIZE_MISMATCH"] == "1" and v["SEL_FILE_SIZE"] == "64"
    assert v["SEL_FLAGS"].startswith("-lm mmap -t 4") and v["SEL_FITS_REPACKED"] == "1"
    assert "not downloaded" in v["SEL_NOTES"]
    # QUANT_OVERRIDE matches with or without the UD- prefix, inside the recommended family
    v = kv(_select(helper, verdict_16gb, md, "--quant-override=q2_k_xl").stdout)
    assert v["SEL_MODEL_ID"] == "26b-ud-q2_k_xl" and "QUANT_OVERRIDE" in v["SEL_NOTES"]
    r = _select(helper, verdict_16gb, md, "--quant-override=Q9_Z")
    assert r.returncode == 3 and "matches no verdict row" in r.stderr
    # explicit catalog id, draft auto (the family's MTP sidecar, explicit spec type)
    v = kv(_select(helper, verdict_16gb, md, "--model=26b-ud-iq2_xxs", "--draft=auto").stdout)
    assert v["SEL_MODEL_ID"] == "26b-ud-iq2_xxs" and v["SEL_DRAFT_ID"] == "26b-mtp"
    assert v["SEL_DRAFT_PATH"].endswith(MTP_26B) and v["SEL_DRAFT_SPEC_TYPE"] == "draft-mtp"
    assert v["SEL_DRAFT_BYTES"] == "251939520" and v["SEL_DRAFT_EXISTS"] == "0"
    v = kv(_select(helper, verdict_16gb, md, "--model=26b-q4", "--draft=26b-dflash", "--spec-type=none").stdout)
    assert v["SEL_DRAFT_ID"] == "26b-dflash" and v["SEL_DRAFT_SPEC_TYPE"] == ""
    # refusals
    r = _select(helper, verdict_16gb, md, "--model=26b-mtp")
    assert r.returncode == 1 and "draft sidecar" in r.stderr
    r = _select(helper, verdict_16gb, md, f"--model={tmp_path / 'nope.gguf'}")
    assert r.returncode == 3 and "no such file" in r.stderr
    r = _select(helper, verdict_16gb, md, "--model=31b-q4")
    assert r.returncode == 3 and "not runnable" in r.stderr
    # an uncatalogued file that fits in RAM: FITS_MAPPED flags, quant guessed from the name
    tiny = tmp_path / "my-tiny-Q8_0.gguf"
    tiny.write_bytes(b"GGUF" + b"\0" * 1000)
    v = kv(_select(helper, verdict_16gb, md, f"--model={tiny}").stdout)
    assert v["SEL_VERDICT"] == "UNCATALOGUED" and v["SEL_FLAGS"] == FITS_MAPPED_FLAGS
    assert v["SEL_QUANT"] == "Q8_0" and v["SEL_BYTES"] == "1004" and v["SEL_BPT"] == "0"


def test_helper_select_reenables_disk_blocked_row_when_file_present(helper: Path, tmp_path: Path) -> None:
    md = tmp_path / "models"
    md.mkdir()
    (md / Q3KM_FILE).write_bytes(b"GGUF" + b"\0" * 60)
    facts = json.loads((FIX / "facts_pi5-16gb-nvme-gen3.json").read_text(encoding="utf-8"))
    facts.update(free_disk_bytes=1_000_000_000, model_dir=str(md))
    ff = tmp_path / "facts.json"
    ff.write_text(json.dumps(facts), encoding="utf-8")
    fz = tmp_path / "feasibility.json"
    _feasibility(ff, fz)
    rows = {r["model_id"]: r for r in json.loads(fz.read_text(encoding="utf-8"))["rows"]}
    assert rows["26b-ud-q3_k_m"]["verdict"] == "NO_DISK"
    r = _select(helper, fz, md, "--model=26b-ud-q3_k_m")
    v = kv(r.stdout)
    assert r.returncode == 0, r.stderr
    assert v["SEL_VERDICT"] == "FITS_REPACKED" and v["SEL_FLAGS"].startswith("-lm mmap")
    assert "already present" in v["SEL_NOTES"]


# --------------------------------------------------------------------------- dry runs (fake proc)
def test_dry_run_bench_16gb(bash: str, tmp_path: Path) -> None:
    r = run(bash, ["bench", "--dry-run", *_out(tmp_path), *HERMETIC], _env(tmp_path))
    assert r.returncode == 0, r.stderr
    cmds = [ln for ln in r.stdout.splitlines() if "llama-bench" in ln]
    assert len(cmds) == 1  # FITS_REPACKED is false for 26b-q4: only the -nr variant is planned
    assert "-t 2,3,4 -p 64 -n 32 -r 3 -o json --repack 0 -lm mmap+mlock -fa on -ctk q8_0 -ctv q8_0" in cmds[0]
    assert "model file not found" in r.stderr and "26b-q4" in r.stderr
    assert "dry run: nothing was run" in r.stderr
    assert not any((tmp_path / "results").iterdir())
    # -p/-n/-t spell the llama-bench test sizes and the thread list; counts above nproc (4) are dropped
    r = run(bash, ["bench", "--dry-run", "-p", "16", "-n", "8", "-r", "1", "-t", "1,4,8", *_out(tmp_path),
                   *HERMETIC], _env(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "-t 1,4 -p 16 -n 8 -r 1 -o json --repack 0" in r.stdout and "8 > nproc 4: dropped" in r.stderr
    # every requested variant needs the anonymous repack copy: nothing left, exit 3
    r = run(bash, ["bench", "--dry-run", "--variants", "default,nommap", *_out(tmp_path), *HERMETIC], _env(tmp_path))
    assert r.returncode == 3 and "no variant left to run" in r.stderr
    assert "FITS_REPACKED is false for 26b-q4" in r.stderr


def test_dry_run_stream_bench_chat_server(bash: str, tmp_path: Path) -> None:
    env = _env(tmp_path)
    r = run(bash, ["stream-bench", "--dry-run", *_out(tmp_path), *HERMETIC], env)
    assert r.returncode == 0, r.stderr
    cmds = [ln for ln in r.stdout.splitlines() if "llama-completion" in ln]
    assert len(cmds) == 2  # English stream-bench + Hebrew chat sample
    assert "-n 64 --temp 0 -t 4 -c 4096 -no-cnv --no-display-prompt -lv 4 -nr -lm mmap+mlock" in cmds[0]
    assert "-n 48 --temp 0" in cmds[1]
    r = run(bash, ["stream-bench", "--dry-run", "--no-he", "-n", "16", *_out(tmp_path), *HERMETIC], env)
    assert r.returncode == 0 and len([ln for ln in r.stdout.splitlines() if "llama-completion" in ln]) == 1
    assert "-n 16 --temp 0" in r.stdout
    r = run(bash, ["chat", "--dry-run", "-p", "Shalom, ma shlomcha?", "-n", "32", *HERMETIC], env)
    assert r.returncode == 0, r.stderr
    words = shlex.split(r.stdout.strip().splitlines()[-1])
    assert words[:3] == ["nice", "-n", "10"] and words[3].endswith("/llama-cli")
    assert words[words.index("-p") + 1] == "Shalom, ma shlomcha?" and words[-1] == "-st"
    assert "-nr" in words and words[words.index("-n", 4) + 1] == "32"
    r = run(bash, ["server", "--dry-run", "--draft", "auto", "--draft-n-max", "3", "--port", "18081",
                   "--flags", "--api-key 'k e y'", *HERMETIC], env)
    assert r.returncode == 0, r.stderr
    words = shlex.split(r.stdout.strip().splitlines()[-1])
    assert words[3].endswith("/llama-server")
    assert words[words.index("-md") + 1].endswith(MTP_26B)
    assert words[words.index("--spec-type") + 1] == "draft-mtp"
    assert words[words.index("--spec-draft-n-max") + 1] == "3"
    assert words[words.index("--port") + 1] == "18081" and words[words.index("--host") + 1] == "0.0.0.0"
    assert words[words.index("--api-key") + 1] == "k e y"
    assert "draft file not found" in r.stderr


def test_quant_override_selects_within_family(bash: str, tmp_path: Path) -> None:
    r = run(bash, ["stream-bench", "--dry-run", "--no-he", *HERMETIC], _env(tmp_path, QUANT_OVERRIDE="UD-Q3_K_M"))
    assert r.returncode == 0, r.stderr
    assert "model    : 26b-ud-q3_k_m" in r.stderr and Q3KM_FILE in r.stdout
    # FITS_REPACKED on an unprotected host: no -nr added
    assert "-nr" not in r.stdout.split(Q3KM_FILE, 1)[1]


def test_dry_run_pin_plans_pi06_first(bash: str, tmp_path: Path) -> None:
    r = run(bash, ["stream-bench", "--dry-run", "--no-he", "--pin", "--pin-arg=--budget-bytes",
                   "--pin-arg=5.2e9", *HERMETIC], _env(tmp_path))
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    pin = shlex.split(lines[0])
    assert pin[0] == "bash" and pin[1].endswith("pi/06_pin_hotset.sh") and pin[2] == "start"
    assert pin[pin.index("--model") + 1].endswith("gemma-4-26B_q4_0-it.gguf") and pin[-2:] == ["--budget-bytes",
                                                                                               "5.2e9"]
    assert "llama-completion" in lines[1]


def test_slow_verdict_is_accepted(bash: str, tmp_path: Path) -> None:
    # 31B on the 16 GB profile: every runnable row is NEVER_GOOD_SPEED -> feasibility exit 2, allowed here
    r = run(bash, ["bench", "--dry-run", "--want", "31b", *_out(tmp_path), *HERMETIC], _env(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "only slow streaming is possible (exit 2) - continuing" in r.stderr
    assert "verdict  : NEVER_GOOD_SPEED" in r.stderr and "model    : 31b-" in r.stderr
    assert "-o json --repack 0" in r.stdout


# ---------------------------------------------------------------------------------- refusals
def test_protected_host_refused_without_override(bash: str, tmp_path: Path) -> None:
    r = run(bash, ["stream-bench", "--dry-run", *HERMETIC], _env(tmp_path, "pi_proc_cf1_16gb"))
    assert r.returncode == 4
    assert "protected host 'context-friend-1'" in r.stderr and r.stdout == ""
    assert not (tmp_path / "out" / "feasibility.json").exists()  # refused before any verdict


def test_protected_host_override_forces_nr_and_drops_mlock(bash: str, tmp_path: Path) -> None:
    env = _env(tmp_path, "pi_proc_cf1_16gb", PIGEMMA_I_KNOW="1", QUANT_OVERRIDE="UD-Q3_K_M")
    r = run(bash, ["stream-bench", "--dry-run", "--no-he", *HERMETIC], env)
    assert r.returncode == 0, r.stderr
    assert "PROTECTED HOST OVERRIDE" in r.stderr and "-nr forced" in r.stderr
    cmd = next(ln for ln in r.stdout.splitlines() if "llama-completion" in ln)
    assert " -nr" in cmd and "mlock" not in cmd
    # 26b-q4 (FITS_MAPPED, -lm mmap+mlock in the verdict): mlock removed from the effective flags
    env26 = _env(tmp_path, "pi_proc_cf1_16gb", PIGEMMA_I_KNOW="1")
    r = run(bash, ["chat", "--dry-run", "-p", "hi", *HERMETIC], env26)
    assert r.returncode == 0, r.stderr
    assert "mlock" not in r.stdout and "-lm mmap" in r.stdout and "mlock removed" in r.stderr
    # the repack (anonymous copy) variant is refused on a protected host even when FITS_REPACKED ...
    r = run(bash, ["bench", "--dry-run", "--variants", "default", *_out(tmp_path), *HERMETIC], env)
    assert r.returncode == 3 and "protected host (PIGEMMA_I_KNOW=1)" in r.stderr
    # ... unless --allow-repack
    r = run(bash, ["bench", "--dry-run", "--variants", "default", "--allow-repack", *_out(tmp_path), *HERMETIC], env)
    assert r.returncode == 0, r.stderr
    cmd = next(ln for ln in r.stdout.splitlines() if "llama-bench" in ln)
    assert "--repack 0" not in cmd and "mlock" not in cmd
    # --pin needs mlock: refused there
    r = run(bash, ["stream-bench", "--dry-run", "--pin", *HERMETIC], env)
    assert r.returncode == 4 and "--pin locks model pages with mlock" in r.stderr


def test_not_viable_and_busy_hosts(bash: str, tmp_path: Path) -> None:
    r = run(bash, ["bench", "--dry-run", *HERMETIC], _env(tmp_path, "pi_proc_4gb", PIGEMMA_I_KNOW="1"))
    assert r.returncode == 3 and "nothing viable" in r.stderr
    r = run(bash, ["bench", "--dry-run", *HERMETIC], _env(tmp_path, PIGEMMA_MAX_LOADAVG="0.01"))
    assert r.returncode == 4 and "busy host: loadavg1" in r.stderr


def test_ram_gate_fits_mode(bash: str, tmp_path: Path) -> None:
    root = fake_root(tmp_path, mem_available_kb=9_000_000)  # 9.2 GB available, 26b-q4 needs 14.44 + 0.5
    r = run(bash, ["bench", "--dry-run", *HERMETIC], _env(tmp_path, root))
    assert r.returncode == 3, r.stderr
    assert "not enough available RAM for 26b-q4" in r.stderr
    assert "fits mode: file bytes 14.44 GB + 0.5 GB" in r.stderr


def test_ram_gate_streaming_mode(bash: str, tmp_path: Path) -> None:
    # an 8 GB board streams the 26B experts; the gate needs non-expert bytes + 0.5 GB = 2.08 GB
    root = fake_root(tmp_path / "a", mem_total_kb=8_245_000, mem_available_kb=1_500_000)
    r = run(bash, ["stream-bench", "--dry-run", *HERMETIC], _env(tmp_path, root))
    assert r.returncode == 3, r.stderr
    assert "26b-q4 (stream_moe)" in r.stderr and "streaming: non-expert bytes 1.58 GB + 0.5 GB" in r.stderr
    root2 = fake_root(tmp_path / "b", mem_total_kb=8_245_000, mem_available_kb=7_500_000)
    r = run(bash, ["stream-bench", "--dry-run", "--no-he", *HERMETIC], _env(tmp_path, root2))
    assert r.returncode == 0, r.stderr
    assert "RAM gate OK" in r.stderr and "-b 256 -ub 64" in r.stdout and "-lm mmap" in r.stdout


def test_missing_model_binary_and_draft_without_dry_run(bash: str, tmp_path: Path) -> None:
    env = _env(tmp_path)
    r = run(bash, ["stream-bench", *_out(tmp_path), *HERMETIC], env)
    assert r.returncode == 3 and "model file not found" in r.stderr and "pi/04_fetch_model.sh" in r.stderr
    tiny = tmp_path / "models" / "tiny-Q4_0.gguf"
    tiny.write_bytes(b"GGUF" + b"\0" * 4092)
    r = run(bash, ["stream-bench", "--model", _p(tiny), *_out(tmp_path), *HERMETIC], env)
    assert r.returncode == 3 and "llama-completion not found" in r.stderr and "pi/02_build_llama.sh" in r.stderr
    write_exe(tmp_path / "bin" / "llama-server", ARGV_ECHO)
    r = run(bash, ["server", "--model", _p(tiny), "--draft", _p(tmp_path / "nope-mtp.gguf"), *HERMETIC], env)
    assert r.returncode == 3 and "--draft" in r.stderr and "no such file" in r.stderr
    r = run(bash, ["bench", "--model", "26b-q4", *_out(tmp_path), *HERMETIC], env)
    assert r.returncode == 3 and "model file not found" in r.stderr
    assert list((tmp_path / "results").iterdir()) == []


# ------------------------------------------------------------ chat / server with test doubles
def test_chat_and_server_with_fake_binaries(bash: str, tmp_path: Path) -> None:
    env = _env(tmp_path)
    tiny = tmp_path / "models" / "tiny-Q4_0.gguf"
    tiny.write_bytes(b"GGUF" + b"\0" * 4092)
    write_exe(tmp_path / "bin" / "llama-cli", ARGV_ECHO)
    write_exe(tmp_path / "bin" / "llama-server", ARGV_ECHO)
    r = run(bash, ["chat", "--model", _p(tiny), "-p", "Hello there", "--flags", "--seed 7 --api-key 'a b'",
                   *HERMETIC], env)
    assert r.returncode == 0, r.stderr
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("FAKE-ARGV:"))
    assert "[-m] [" in line and "tiny-Q4_0.gguf]" in line
    assert "[-nr]" in line and "[-p] [Hello there] [-st]" in line and "[--api-key] [a b]" in line
    assert "llama-cli exited with 0" in r.stderr
    # the llama-cli exit code is passed through
    r = run(bash, ["chat", "--model", _p(tiny), "-p", "x", *HERMETIC], {**env, "FAKE_RC": "5"})
    assert r.returncode == 5
    port = free_port()
    r = run(bash, ["server", "--model", _p(tiny), "--host", "127.0.0.1", "--port", str(port),
                   "--out", _p(tmp_path / "results" / "srv"), *HERMETIC], env)
    assert r.returncode == 0, r.stderr
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("FAKE-ARGV:"))
    assert f"[--host] [127.0.0.1] [--port] [{port}]" in line and "[-nr]" in line
    assert "llama-server exited with 0" in r.stderr
    assert not (tmp_path / "bench.lock").exists()  # the lock was held for the server's lifetime only
    assert not (tmp_path / "results" / "srv").exists()  # chat / server write no results directory


def test_server_refuses_port_in_use_and_chat_refuses_held_lock(bash: str, tmp_path: Path) -> None:
    env = _env(tmp_path)
    tiny = tmp_path / "models" / "tiny-Q4_0.gguf"
    tiny.write_bytes(b"GGUF" + b"\0" * 4092)
    write_exe(tmp_path / "bin" / "llama-server", ARGV_ECHO)
    write_exe(tmp_path / "bin" / "llama-cli", ARGV_ECHO)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(4)
        port = s.getsockname()[1]
        r = run(bash, ["server", "--model", _p(tiny), "--host", "127.0.0.1", "--port", str(port), *HERMETIC], env)
    assert r.returncode == 4 and f"port {port} is already in use" in r.stderr
    assert "FAKE-ARGV" not in r.stdout
    # a bench lock held by another live model-loading job (this pytest process) refuses chat (exit 4)
    from pigemma import paths
    holder = subprocess.Popen([sys.executable, "-m", "pigemma.paths", "lock", "--hold", "--min-free-gb", "0",
                               "--lock-file", str(tmp_path / "bench.lock"), "--", sys.executable, "-c",
                               "import time; time.sleep(60)"], cwd=REPO,
                              env={**os.environ, "PYTHONPATH": str(REPO)}, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            if paths.lock_status(lock_path=tmp_path / "bench.lock")["state"] == "held":
                break
            import time
            time.sleep(0.1)
        else:
            pytest.fail("the helper process never took the bench lock")
        r = run(bash, ["chat", "--model", _p(tiny), "-p", "x", *HERMETIC], env)
        assert r.returncode == 4 and "bench lock" in r.stderr and "FAKE-ARGV" not in r.stdout
    finally:
        holder.terminate()
        holder.wait(timeout=30)


# ----------------------------------------------------- bench / stream-bench end to end (Linux)
TINY_ID = "tiny-moe-test"
TINY_FILE = "tiny-moe-q4_0.gguf"
TINY_NONEXPERT_TOK, TINY_EXPERT_TOK = 200_000, 20_000  # bytes/token: tg x 220 kB / 10 GB/s ~ 0.5


def _tiny_catalog(tmp_path: Path, size: int) -> Path:
    cat = json.loads((REPO / "pigemma/data/catalog.json").read_text(encoding="utf-8"))
    entry = {
        "id": TINY_ID, "family": "26b", "model": "tiny-test", "kind": "moe", "quant": "Q4_0", "qat": False,
        "repo": "test/tiny", "revision": None, "file": TINY_FILE, "local_path": None, "bytes": size,
        "bytes_exact": True, "estimated": False, "nonexpert_bytes": size // 2, "expert_bytes": size - size // 2,
        "expert_unit_bytes": TINY_EXPERT_TOK, "expert_bytes_per_token_worst": TINY_EXPERT_TOK,
        "nonexpert_bytes_per_token": TINY_NONEXPERT_TOK, "n_layer": 1, "n_moe_layers": 1, "n_expert": 4,
        "n_expert_used": 1, "drafts": [], "sources": {"bytes": "assumed: test fixture"},
    }
    cat["models"].append(entry)
    p = tmp_path / "catalog.json"
    p.write_text(json.dumps(cat, indent=1), encoding="utf-8")
    return p


@pytest.mark.linux
def test_bench_and_stream_bench_end_to_end_linux(bash: str, tmp_path: Path) -> None:
    from pigemma import feasibility as feas
    from pigemma import results as rs
    size = 4096
    cat = _tiny_catalog(tmp_path, size)
    io_mem = tmp_path / "io_mem.json"
    io_mem.write_text(json.dumps({"memory": {"read_gbps_by_threads": {"1": 6.0, "4": 10.0}}}), encoding="utf-8")
    env = _env(tmp_path, PIGEMMA_CATALOG=str(cat))
    (tmp_path / "models" / TINY_FILE).write_bytes(b"GGUF" + b"\0" * (size - 4))
    for tool, fake in (("llama-bench", "fake_llama_bench.py"), ("llama-completion", "fake_llama_completion.py")):
        write_exe(tmp_path / "bin" / tool, f'#!/bin/sh\nexec "{sys.executable}" "{FIX / "bench" / fake}" "$@"\n')
    common = ["--want", TINY_ID, "--idle-check-s", "0", "--io-mem", str(io_mem), *_out(tmp_path), *HERMETIC,
              "--feasibility-arg=--catalog", f"--feasibility-arg={cat}"]
    r = run(bash, ["bench", "--bench-threads", "2,4", "--variants", "norepack", *common], env)
    assert r.returncode == 0, r.stderr
    lines = [ln for ln in r.stdout.splitlines() if ln.endswith(".json")]
    assert len(lines) == 4  # (pp, tg) x threads 2, 4
    docs = [_load(ln) for ln in lines]
    for d in docs:
        assert rs.check_result(d)[0] == [] and d["experiment"] == "E8" and d["kind"] == "measured"
        assert d["variant"] == "norepack+kvq8" and d["extra"]["pi05"]["subcommand"] == "bench"
        assert d["extra"]["pi05"]["verdict"] == "FITS_REPACKED" and d["model"]["file"].endswith(TINY_FILE)
    # the raw llama-bench JSON is kept next to the envelopes (named after the envelope's model id)
    raws = sorted((tmp_path / "results").glob(f"bench_{docs[0]['model']['id']}_norepack+kvq8_*.json"))
    assert len(raws) == 1 and isinstance(json.loads(raws[0].read_text(encoding="utf-8")), list)
    cal = json.loads((tmp_path / "calibration.json").read_text(encoding="utf-8"))
    tg4 = next(d["tg"]["tok_s"] for d in docs if d.get("tg") and d["threads"] == 4)
    bpt = TINY_NONEXPERT_TOK + TINY_EXPERT_TOK
    assert cal["llama_efficiency"]["value"] == round(tg4 * bpt / 10e9, 4)
    assert cal["llama_efficiency"]["source"].startswith("measured")
    assert feas.load_calibration(tmp_path / "calibration.json")[1].startswith("measured")
    last = json.loads((tmp_path / "out" / "last_bench.json").read_text(encoding="utf-8"))
    assert last["exit_code"] == 0 and len(last["results"]) == 4 and last["model"]["id"] == TINY_ID
    # stream-bench: English completion + Hebrew chat sample, both annotated, outputs printed
    r = run(bash, ["stream-bench", "-n", "16", "--he-n", "8", "--no-calibrate", *common], env)
    assert r.returncode == 0, r.stderr
    lines = [ln for ln in r.stdout.splitlines() if ln.endswith(".json")]
    assert len(lines) == 2
    en, he = (_load(ln) for ln in lines)
    assert en["tg"]["tok_s"] > 0 and en["read_bytes_per_token"] is not None
    assert en["extra"]["pi05"]["subcommand"] == "stream-bench" and he["extra"]["pi05"]["label"].startswith("HE")
    assert "Experts are chosen per token" in r.stderr  # the generated text is printed
    last = json.loads((tmp_path / "out" / "last_stream-bench.json").read_text(encoding="utf-8"))
    assert [m["label"] for m in last["metrics"]] == ["stream-bench", "he-sample"]
    # a failing llama-bench still writes its failure envelopes and exits 1
    r = run(bash, ["bench", "--bench-threads", "2,4", "--variants", "norepack", *common],
            {**env, "FAKE_RC": "1", "FAKE_BENCH_JSON": str(FIX / "bench" / "llama_bench_fail.stdout.txt")})
    assert r.returncode == 1, r.stderr
    failed = [_load(ln) for ln in r.stdout.splitlines() if ln.endswith(".json")]
    assert failed and all(d["notes"][0].startswith("failed") for d in failed)
