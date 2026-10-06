"""Tests for pi/00_feasibility.sh (read-only fact collection + verdict table + recommendation).

Manifest test field (files[] 'pi/00_feasibility.sh'):
  * fixture-driven: PIGEMMA_FAKE_PROC 4 GB fixture -> exit 3/4 with the table printed; 16 GB fixture -> exit 0;
  * `bash -n`;
  * also runs inside the Docker image (generic Linux: storage 'unknown' tolerated) -> integration test below.
Plus: the protected-host banner and the PIGEMMA_I_KNOW override, the mlock policy applied to the flags shown
for this host, the pi/04 disk-rule line (context-friend-1: 26B Q4_0 does not fit its microSD -> UD-Q3_K_M),
the container fixture (exit 2), --facts on a busy host (exit 4), --want / --json - / --sim / --report, usage
errors, "writes only its outputs" and the rule that pi/00 is never weaker than the bash host gate.

Every run is hermetic: PIGEMMA_* and pi_config variables are stripped from the inherited environment,
PIGEMMA_OUT and MODEL_DIR point into tmp_path, and --io-mem none --calibration none keep measured files in
results/pi/ (if any) out of the verdict. bash is Git Bash on Windows (conftest `bash` fixture, never WSL).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures"
SCRIPT = "pi/00_feasibility.sh"
HERMETIC = ("--io-mem", "none", "--calibration", "none")
RECOMMENDATION_PREFIX = "Pi 5 16 GB + NVMe Gen3: 26B-A4B fits in RAM (-nr)"

#: pi/pi_config.env variables (stripped from the inherited environment so the tests are hermetic).
CONFIG_VARS = (
    "PIGEMMA_HOME", "MODEL_DIR", "LLAMA_DIR", "LLAMA_COMMIT", "PIGEMMA_OUT", "THREADS", "CTX", "QUANT_OVERRIDE",
    "OS_RESERVE_GIB", "MIN_FREE_DISK_FACTOR", "KEEP_FREE_GB", "NICE_LEVEL", "DENY_HOSTS",
    "PROTECTED_SERVICE_GLOBS", "ALLOW_MLOCK", "TUNE_APPLY", "TUNE_NVME_GEN3", "TUNE_GOVERNOR", "TUNE_SWAPPINESS",
    "TUNE_READAHEAD_KB", "TUNE_MEMLOCK", "BUILD_JOBS", "HF_LIMIT_RATE", "PIN_BUDGET_BYTES",
)


def _p(path: Path | str) -> str:
    """A path as bash (Git Bash or Linux) accepts it."""
    return Path(path).as_posix()


def _env(tmp_path: Path, fake: str | Path | None = None, **extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("PIGEMMA_") and not k.startswith("PG00_") and k not in CONFIG_VARS and k != "PY"}
    env["PIGEMMA_PYTHON"] = _p(sys.executable)
    env["PIGEMMA_OUT"] = _p(tmp_path / "out")
    env["MODEL_DIR"] = _p(tmp_path / "models")  # never the desktop's real models/ (presence changes NO_DISK)
    env["LC_ALL"] = env.get("LC_ALL") or "C.UTF-8"
    if fake is not None:
        env["PIGEMMA_FAKE_PROC"] = _p(fake if isinstance(fake, Path) else FIX / fake)
    env.update(extra)
    return env


def run00(bash: str, env: dict[str, str], *args: str, timeout: float = 240) -> subprocess.CompletedProcess:
    return subprocess.run([bash, SCRIPT, *args], cwd=REPO, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def _out(tmp_path: Path) -> Path:
    return tmp_path / "out"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(fz: dict) -> dict[str, dict]:
    return {r["model_id"]: r for r in fz["rows"]}


def _tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(f.relative_to(root).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def _assert_common_report(stdout: str) -> None:
    assert stdout.startswith("pi-gemma4 feasibility report - pi/00_feasibility.sh ")
    assert "READ-ONLY" in stdout.splitlines()[0]
    for section in ("==== host facts (read-only) ====", "==== verdict table:", "==== summary ===="):
        assert section in stdout
    # host facts the spec names
    for label in ("ram ", "storage ", "free disk", "kernel", "cpu flags", "HAT / PCIe", "protected host",
                  "page size", "read_ahead_kb"):
        assert label in stdout, label
    # the verdict table from pigemma.feasibility, with projected tok/s and its source
    assert "model            quant       size GB  verdict" in stdout
    assert "(assumed)" in stdout or "(simulated)" in stdout or "(measured in)" in stdout
    # the fixed recommendation line is the report's hardware line
    hw = [ln for ln in stdout.splitlines() if ln.strip().startswith("hardware")]
    assert hw and RECOMMENDATION_PREFIX in hw[-1]
    assert "31B dense: never good speed on any Pi 5" in hw[-1]


# ------------------------------------------------------------------------------------------ lint
def test_bash_syntax(bash: str) -> None:
    r = subprocess.run([bash, "-n", SCRIPT], cwd=REPO, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr


def test_script_shape() -> None:
    raw = (REPO / SCRIPT).read_bytes()
    assert b"\r\n" not in raw, "pi scripts must use LF line endings"
    text = raw.decode("utf-8")
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text
    assert '. "$PI_DIR/lib/common.sh"' in text
    # read-only: no call of the gate that exits before the report, nothing that changes the host. Checked on
    # code only (comments dropped, quoted strings blanked: messages may mention these words).
    code = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        s = re.sub(r'"[^"]*"', '""', s)
        s = re.sub(r"'[^']*'", "''", s)
        code.append(s)
    joined = "\n".join(code)
    for pat in (r"\bassert_safe_host\b", r"\bassert_not_busy\b",
                r"\bsystemctl\s+(stop|restart|start|disable|enable|kill)\b", r"\b(pkill|killall)\b", r"\bkill\s+-",
                r"\bapt(-get)?\s+(install|remove|purge)\b", r"\bsysctl\s+-w\b", r"\breboot\b", r"\bshutdown\b",
                r"\bmlock\b", r"\bsudo\b", r">>?\s*/(boot|etc|sys|proc)/"):
        assert not re.search(pat, joined), pat


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_shellcheck() -> None:  # pragma: no cover - shellcheck is not installed on the desktop
    r = subprocess.run(["shellcheck", "-x", SCRIPT], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


# ------------------------------------------------------------------------------------ fixtures
def test_4gb_protected_host_refuses_with_table(bash: str, tmp_path: Path) -> None:
    root_digest = _tree_digest(FIX / "pi_proc_4gb")
    r = run00(bash, _env(tmp_path, "pi_proc_4gb"), *HERMETIC)
    assert r.returncode == 4, r.stdout + r.stderr
    _assert_common_report(r.stdout)
    assert "PROTECTED HOST 'context-friend-1'" in r.stdout
    assert "EVERY OTHER pi/*.sh SCRIPT REFUSES ON THIS HOST (exit 4)" in r.stdout
    assert "PIGEMMA_I_KNOW=1" in r.stdout
    assert "contextfriend-box.service" in r.stdout
    assert "HAT: AI HAT+" in r.stdout and "no help for 25-30B models" in r.stdout
    # the rows are still printed (informational) and the 4 GB microSD rows are not viable
    for mid in ("26b-q4", "31b-q4", "26b-ud-iq2_xxs"):
        assert f"\n{mid} " in r.stdout
    assert "exit 4 = REFUSED" in r.stdout
    out = _out(tmp_path)
    facts = _json(out / "facts.json")
    assert facts["ram_total_kb"] == 4039000 and facts["storage"]["type"] == "microsd"
    assert facts["page_size"] == 16384
    fz = _json(out / "feasibility.json")
    assert fz["exit_code"] == 4 and fz["host_refused"] is True and fz["recommended"] is None
    assert (out / "feasibility.txt").read_text(encoding="utf-8") == r.stdout
    assert _tree_digest(FIX / "pi_proc_4gb") == root_digest  # the fake root is only read


def test_4gb_with_override_is_not_viable(bash: str, tmp_path: Path) -> None:
    r = run00(bash, _env(tmp_path, "pi_proc_4gb", PIGEMMA_I_KNOW="1"), *HERMETIC)
    assert r.returncode == 3, r.stdout + r.stderr
    _assert_common_report(r.stdout)
    assert "OVERRIDE ACTIVE (PIGEMMA_I_KNOW=1)" in r.stdout
    assert "Rules still in force" in r.stdout
    assert "exit 3 = NOT VIABLE" in r.stdout
    fz = _json(_out(tmp_path) / "feasibility.json")
    assert fz["exit_code"] == 3 and fz["recommended"] is None
    moe = [row for row in fz["rows"] if row["family"] == "26b"]
    assert moe and all(row["verdict"] == "NOT_VIABLE" for row in moe)
    assert all(row["verdict"] == "NEVER_GOOD_SPEED" for row in fz["rows"] if row["family"] == "31b")


def test_16gb_nvme_fits_mapped(bash: str, tmp_path: Path) -> None:
    r = run00(bash, _env(tmp_path, "pi_proc_16gb"), *HERMETIC)
    assert r.returncode == 0, r.stdout + r.stderr
    _assert_common_report(r.stdout)
    assert "PROTECTED HOST" not in r.stdout and "BUSY HOST" not in r.stdout
    assert "asimddp: yes" in r.stdout and "i8mm: no" in r.stdout and "sve: no" in r.stdout
    assert "PCIe link 8.0 GT/s PCIe x1 = Gen3" in r.stdout
    assert "read_ahead_kb 2048" in r.stdout
    assert "protected host  no" in r.stdout
    assert "exit 0 = OK" in r.stdout
    assert "recommended     26b-q4 (Q4_0, 14.44 GB) FITS_MAPPED" in r.stdout
    assert "llama flags     -nr -lm mmap+mlock -t 4 -c 4096" in r.stdout
    assert "on this host    the same flags (mlock allowed here)" in r.stdout
    assert "disk (pi/04)    OK:" in r.stdout
    fz = _json(_out(tmp_path) / "feasibility.json")
    rec = fz["recommended"]
    assert rec["model_id"] == "26b-q4" and rec["verdict"] == "FITS_MAPPED" and rec["fits_repacked"] is False
    assert rec["flags"].startswith("-nr -lm mmap+mlock")
    assert _rows(fz)["31b-q4"]["verdict"] == "NEVER_GOOD_SPEED"


def test_cf1_override_drops_mlock_and_names_a_quant_that_fits_the_sd(bash: str, tmp_path: Path) -> None:
    r = run00(bash, _env(tmp_path, "pi_proc_cf1_16gb", PIGEMMA_I_KNOW="1"), *HERMETIC)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OVERRIDE ACTIVE" in r.stdout
    flags_here = [ln for ln in r.stdout.splitlines() if ln.startswith("  on this host")]
    assert flags_here and "mlock" not in flags_here[0].split("(")[0]
    assert "-nr -lm mmap -t 4" in flags_here[0] and "mlock removed" in flags_here[0]
    # FACTS: the 26B Q4_0 fits in RAM with -nr but not on the 15.5 GB-free microSD (keep >= 2 GB free on /)
    assert "NOT ENOUGH: 26b-q4 needs" in r.stdout
    assert "pi/04_fetch_model.sh --model 26b --quant UD-Q3_K_M" in r.stdout
    fz = _json(_out(tmp_path) / "feasibility.json")
    assert fz["recommended"]["model_id"] == "26b-q4"  # the verdict itself is unchanged (exit code = verdict)


def test_cf1_without_override_refuses(bash: str, tmp_path: Path) -> None:
    r = run00(bash, _env(tmp_path, "pi_proc_cf1_16gb"), *HERMETIC)
    assert r.returncode == 4, r.stdout + r.stderr
    assert "PROTECTED HOST 'context-friend-1'" in r.stdout
    assert "every other pi/*.sh script refuses on this host (exit 4)" in r.stdout


def test_container_unknown_storage_is_tolerated(bash: str, tmp_path: Path) -> None:
    r = run00(bash, _env(tmp_path, "pi_proc_container"), *HERMETIC)
    assert r.returncode == 2, r.stdout + r.stderr
    _assert_common_report(r.stdout)
    assert "storage         unknown" in r.stdout
    assert "storage type unknown" in r.stdout
    assert "no boot config" in r.stdout
    assert "cgroup memory limit 8.59 GB" in r.stdout
    assert "running services could not be listed" in r.stderr
    assert "exit 2 = SLOW ONLY" in r.stdout


def test_facts_file_busy_host(bash: str, tmp_path: Path) -> None:
    facts = FIX / "facts_pi5-8gb-nvme-busy.json"
    r = run00(bash, _env(tmp_path), "--facts", _p(facts), *HERMETIC)
    assert r.returncode == 4, r.stdout + r.stderr
    assert "BUSY HOST 'pi5-lab-busy'" in r.stdout
    assert "loadavg1 3.60 > 2.00" in r.stdout
    assert "host busy" in r.stdout
    assert f"facts from {_p(facts)}" in r.stdout
    out = _out(tmp_path)
    assert not (out / "facts.json").exists()  # --facts: nothing collected
    assert _json(out / "feasibility.json")["busy"] is True


def test_want_31b_json_to_stdout(bash: str, tmp_path: Path) -> None:
    r = run00(bash, _env(tmp_path, "pi_proc_16gb"), "--want", "31b", "--json", "-", *HERMETIC)
    fz = json.loads(r.stdout)
    assert r.returncode == fz["exit_code"] == 2, r.stderr
    assert {row["family"] for row in fz["rows"]} == {"31b"}
    assert all(row["verdict"] == "NEVER_GOOD_SPEED" for row in fz["rows"])
    assert "==== verdict table:" in r.stderr and "==== summary ====" in r.stderr
    assert "verdict stdout" in r.stderr
    assert not (_out(tmp_path) / "feasibility.json").exists()


def test_sim_replaces_the_assumed_curve(bash: str, tmp_path: Path) -> None:
    sim = {"trace": "traces/26b/all.moetrace", "map": "results/map26.json", "runs": [
        {"policy": "lru", "budget_bytes": cap * 3_345_408, "capacity_items": cap, "profile": None,
         "steady": {"hit_rate": hit}, "per_layer_hit_rate": [hit] * 30}
        for cap, hit in ((0, 0.0), (1000, 0.90), (2000, 0.97), (3840, 1.0))]}
    sp = tmp_path / "sim.json"
    sp.write_text(json.dumps(sim), encoding="utf-8")
    facts = FIX / "facts_pi5-8gb-usb3.json"
    r = run00(bash, _env(tmp_path), "--facts", _p(facts), "--sim", _p(sp), *HERMETIC)
    assert r.returncode in (0, 2), r.stdout + r.stderr
    fz = _json(_out(tmp_path) / "feasibility.json")
    assert fz["hit_source"] == "simulated" and fz["exit_code"] == r.returncode
    assert "sim.json LRU (simulated)" in r.stdout
    assert "(simulated)" in r.stdout  # the table's tok/s source column


def test_report_none_and_custom_paths(bash: str, tmp_path: Path) -> None:
    jp = tmp_path / "custom" / "verdict.json"
    fp = tmp_path / "custom" / "facts16.json"
    r = run00(bash, _env(tmp_path, "pi_proc_16gb"), "--json", _p(jp), "--facts-out", _p(fp), "--report", "none",
              *HERMETIC)
    assert r.returncode == 0, r.stderr
    assert _json(jp)["exit_code"] == 0 and _json(fp)["hostname"] == "pi5-lab-16g"
    out = _out(tmp_path)
    assert not (out / "feasibility.txt").exists() and not (out / "feasibility.json").exists()


def test_writes_only_its_outputs(bash: str, tmp_path: Path) -> None:
    r = run00(bash, _env(tmp_path, "pi_proc_16gb"), *HERMETIC)
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in _out(tmp_path).iterdir()) == ["facts.json", "feasibility.json", "feasibility.txt"]
    assert not (tmp_path / "models").exists()


def test_never_weaker_than_the_bash_host_gate(bash: str, tmp_path: Path) -> None:
    # a glob that only matches the unit name WITHOUT '.service': the bash gate (used by every other pi script)
    # refuses, pigemma.feasibility's fnmatch on the full name does not -> pi/00 must still exit 4
    r = run00(bash, _env(tmp_path, "pi_proc_16gb", PROTECTED_SERVICE_GLOBS="friendly-context"), *HERMETIC)
    assert r.returncode == 4, r.stdout + r.stderr
    assert "raised to exit 4" in r.stdout
    assert "PROTECTED HOST 'pi5-lab-16g'" in r.stdout
    assert _json(_out(tmp_path) / "feasibility.json")["exit_code"] == 0


@pytest.mark.parametrize("args", [
    ("--bogus",),
    ("--want",),
    ("--want", "26b;rm"),
    ("--sim", "/nonexistent/sim.json"),
    ("--facts", "/nonexistent/facts.json"),
    ("--io-mem", "/nonexistent/io_mem.json"),
    ("--facts-out", "-"),
])
def test_usage_errors_exit_1(bash: str, tmp_path: Path, args: tuple[str, ...]) -> None:
    r = run00(bash, _env(tmp_path, "pi_proc_16gb"), *args)
    assert r.returncode == 1, r.stdout + r.stderr
    assert not (_out(tmp_path) / "feasibility.json").exists()


def test_help_and_version(bash: str, tmp_path: Path) -> None:
    r = run00(bash, _env(tmp_path), "--help")
    assert r.returncode == 0
    for word in ("--want", "--json", "--sim", "--facts", "Exit codes", "PIGEMMA_I_KNOW"):
        assert word in r.stdout
    r = run00(bash, _env(tmp_path), "--version")
    assert r.returncode == 0 and r.stdout.startswith("pi/00_feasibility.sh ")


def test_feasibility_error_exits_1(bash: str, tmp_path: Path) -> None:
    r = run00(bash, _env(tmp_path, "pi_proc_16gb"), "--want", "70b", *HERMETIC)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "exit 1 = ERROR" in r.stdout
    assert not (_out(tmp_path) / "feasibility.json").exists()


# ----------------------------------------------------------------------------------- integration
@pytest.mark.integration
def test_runs_inside_the_docker_build_image(bash: str, docker: str, tmp_path: Path) -> None:
    r = subprocess.run(
        [bash, "pc/docker_run.sh", "--no-tty", "--no-stdin", "--env", "PIGEMMA_OUT=/tmp/pg00",
         "--", "bash", SCRIPT, *HERMETIC, "--report", "none"],
        cwd=REPO, env=_env(tmp_path), capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=600)
    assert r.returncode in (0, 2, 3), r.stdout + r.stderr
    _assert_common_report(r.stdout)
    assert "storage         unknown" in r.stdout  # bind mount / overlay: tolerated, not a crash
    assert "container: yes" in r.stdout
