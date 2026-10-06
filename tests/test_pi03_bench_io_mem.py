"""Tests for pi/03_bench_io_mem.sh (storage + memory measurement -> results/pi/io_mem.json).

Manifest test field (files[] 'pi/03_bench_io_mem.sh'): `bash -n` (also covered by the sweep in
tests/test_pi_scripts.py); fio job strings validated with `fio --parse-only` in Docker (integration test
below: `--check-jobs`); the merge of an io_mem.json into the device profile (tests/test_catalog.py with a
fixture, and here with files the script itself produced from the checked-in fio / membw captures).

Unit tests run without fio, membw or a Pi:
  * `--from-raw DIR` rebuilds io_mem.json from raw outputs (tests/fixtures/fio_sample.json, membw_sample.json,
    a synthetic GNU dd line) -> the data_formats keys, values equal to the raw numbers, the device naming,
    the derived (projected) numbers, and pigemma.feasibility.load_catalog() merging them as 'measured';
  * `--dry-run` with PIGEMMA_FAKE_PROC fixtures -> the plan (16 KiB page -> fio bs 3,358,720 B), the
    protected-host refusal (exit 4) and its PIGEMMA_I_KNOW override, the free-disk refusal (exit 3);
  * a real measurement under a fake root is refused; usage errors exit 1.
Integration (`pytest -m integration`, Docker): `--check-jobs` (fio --parse-only of every job) and one short
real run in the build image.
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
FIX = REPO / "tests" / "fixtures"
SCRIPT = "pi/03_bench_io_mem.sh"
CATALOG = REPO / "pigemma" / "data" / "catalog.json"

#: Config variables of pi/pi_config.env (stripped from the inherited environment so tests are hermetic).
CONFIG_VARS = (
    "PIGEMMA_HOME", "MODEL_DIR", "LLAMA_DIR", "LLAMA_COMMIT", "PIGEMMA_OUT", "THREADS", "CTX", "QUANT_OVERRIDE",
    "OS_RESERVE_GIB", "MIN_FREE_DISK_FACTOR", "KEEP_FREE_GB", "NICE_LEVEL", "DENY_HOSTS",
    "PROTECTED_SERVICE_GLOBS", "ALLOW_MLOCK", "TUNE_APPLY", "TUNE_NVME_GEN3", "TUNE_GOVERNOR", "TUNE_SWAPPINESS",
    "TUNE_READAHEAD_KB", "TUNE_MEMLOCK", "BUILD_JOBS", "HF_LIMIT_RATE", "PIN_BUDGET_BYTES",
)
#: data_formats 'io_mem.json' storage keys.
STORAGE_KEYS = ("seq_read_mbps_qd1", "seq_read_mbps_qd8", "rand_read_3mb_mbps_qd1", "rand_read_3mb_mbps_qd4",
                "rand_read_16k_iops", "access_ms")
JOB_OF = {"seq_read_mbps_qd1": "seq_read_qd1", "seq_read_mbps_qd8": "seq_read_qd8",
          "rand_read_3mb_mbps_qd1": "rand_read_3mb_qd1", "rand_read_3mb_mbps_qd4": "rand_read_3mb_qd4"}
DD_LINE = ("2048+0 records in\n2048+0 records out\n"
           "2147483648 bytes (2.1 GB, 2.0 GiB) copied, 23.4425 s, 91.6 MB/s\n")


# ------------------------------------------------------------------------------------------- helpers
def _p(path: Path | str) -> str:
    """A path as Git Bash / Linux bash accepts it (forward slashes, drive letter kept)."""
    return Path(path).as_posix()


def _env(fake: str | None = None, **extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("PIGEMMA_") and k not in CONFIG_VARS and k != "PY"}
    env["PIGEMMA_PYTHON"] = _p(sys.executable)
    env["PIGEMMA_QUIET"] = "1"
    env["LC_ALL"] = env.get("LC_ALL") or "C.UTF-8"
    env["MODEL_DIR"] = "/srv/pigemma-test/models"
    if fake:
        env["PIGEMMA_FAKE_PROC"] = _p(FIX / fake)
    env.update(extra)
    return env


def _run(bash: str, args: list[str], env: dict[str, str], *, timeout: float = 240) -> subprocess.CompletedProcess:
    return subprocess.run([bash, SCRIPT, *args], cwd=REPO, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def _fio_fixture_jobs() -> dict[str, dict]:
    data = json.loads((FIX / "fio_sample.json").read_text(encoding="utf-8"))
    return {j["jobname"]: j for j in data["jobs"]}


def _raw_from_fixtures(tmp: Path, *, membw: bool = True) -> Path:
    raw = tmp / "raw"
    raw.mkdir(parents=True)
    shutil.copy(FIX / "fio_sample.json", raw / "fio_sample.json")
    if membw:
        shutil.copy(FIX / "membw_sample.json", raw / "membw.json")
    return raw


def _catalog_model(mid: str) -> dict:
    cat = json.loads(CATALOG.read_text(encoding="utf-8"))
    return next(m for m in cat["models"] if m["id"] == mid)


def _snapshot(d: Path) -> dict[str, bytes]:
    return {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


# ------------------------------------------------------------------------------------------ CLI basics
def test_script_is_lf_and_executable_text() -> None:
    data = (REPO / SCRIPT).read_bytes()
    assert b"\r" not in data, "CR line endings would break bash on the Pi"
    assert data.startswith(b"#!/usr/bin/env bash\n")


def test_help_and_version(bash: str) -> None:
    r = _run(bash, ["--help"], _env())
    assert r.returncode == 0, r.stderr
    for flag in ("--file", "--size-mb", "--seconds", "--quick", "--from-raw", "--check-jobs", "--dry-run"):
        assert flag in r.stdout
    v = _run(bash, ["--version"], _env())
    assert v.returncode == 0 and v.stdout.startswith("pi/03_bench_io_mem.sh ")


@pytest.mark.parametrize("args", [["--bogus"], ["--seconds", "0"], ["--size-mb", "abc"], ["--skip-io", "--skip-mem"],
                                  ["--threads", "1,,2"], ["--quick=1"], ["--from-raw", "x", "--dry-run"]])
def test_usage_errors_exit_1(bash: str, args: list[str]) -> None:
    r = _run(bash, args, _env())
    assert r.returncode == 1, (r.stdout, r.stderr)
    assert "usage" in r.stderr.lower()


# ------------------------------------------------------------------------------------------ --from-raw
def test_from_raw_fixture_builds_mergeable_io_mem(bash: str, tmp_path: Path) -> None:
    raw = _raw_from_fixtures(tmp_path)
    before = _snapshot(raw)
    out = tmp_path / "res" / "io_mem.json"
    r = _run(bash, ["--from-raw", _p(raw), "--out", _p(out)], _env("pi_proc_16gb"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert _snapshot(raw) == before, "--from-raw must never write into the raw directory"
    doc = json.loads(out.read_text(encoding="utf-8"))

    # data_formats keys
    for k in ("host", "device", "collected_at", "storage", "memory"):
        assert k in doc
    assert doc["kind"] == "measured" and doc["complete"] is True
    assert doc["host"] == "pi5-lab-16g"
    assert doc["device"] == "pi5-16gb-nvme-gen3"  # same RAM tier + storage class as the fake root
    assert set(doc["storage"]) == set(STORAGE_KEYS)
    for k, e in doc["storage"].items():
        assert e["source"] == "measured" and e["cmd"] and e["unit"], k

    jobs = _fio_fixture_jobs()
    for metric, job in JOB_OF.items():
        assert doc["storage"][metric]["value"] == pytest.approx(jobs[job]["read"]["bw_bytes"] / 1e6, abs=1e-3)
    r16 = jobs["rand_read_16k_qd1"]["read"]
    assert doc["storage"]["rand_read_16k_iops"]["value"] == pytest.approx(r16["iops"], abs=0.01)
    assert doc["storage"]["access_ms"]["value"] == pytest.approx(r16["lat_ns"]["mean"] / 1e6, abs=1e-4)

    mem = json.loads((FIX / "membw_sample.json").read_text(encoding="utf-8"))
    by_t = {str(x["threads"]): x["read_gbps"] for x in mem["results"]}
    assert doc["memory"]["read_gbps_by_threads"] == by_t
    assert doc["memory"]["copy_gbps"] == max(x["copy_gbps"] for x in mem["results"])
    assert doc["memory"]["triad_gbps"] == max(x["triad_gbps"] for x in mem["results"])
    assert doc["memory"]["source"] == "measured"

    # derived numbers are projections, efficiency assumed without calibration.json
    d = doc["derived"]
    assert d["kind"] == "projected"
    assert d["llama_efficiency"]["source"].startswith("assumed")
    m26 = _catalog_model("26b-q4")
    bpt = m26["nonexpert_bytes_per_token"] + m26["expert_bytes_per_token_worst"]
    assert d["in_ram_ceiling"]["26b-q4"]["bytes_per_token"] == bpt
    assert d["in_ram_ceiling"]["26b-q4"]["ceiling_tok_s"] == pytest.approx(max(by_t.values()) * 1e9 * 0.65 / bpt,
                                                                           abs=1e-3)
    s = d["streaming"]["26b-q4"]
    assert s["n_miss_per_token_worst"] == m26["n_moe_layers"] * m26["n_expert_used"]
    assert 0.0 <= s["min_hit_rate"]["1.0"] <= s["min_hit_rate"]["2.0"] <= 1.0
    assert "31b-q4" in d["in_ram_ceiling"] and "31b-q4" not in d["streaming"]

    # the report names the merge and shows measured values
    assert "MEASURED" in r.stdout and "PROJECTED" in r.stdout and "(measured)" in r.stdout

    feas = pytest.importorskip("pigemma.feasibility")
    cat_before = CATALOG.read_bytes()
    merged = feas.load_catalog(CATALOG, io_mem=out)
    dev = next(x for x in merged["devices"] if x["name"] == "pi5-16gb-nvme-gen3")
    assert dev["storage"]["seq_read_mbps"]["value"] == pytest.approx(
        max(doc["storage"]["seq_read_mbps_qd1"]["value"], doc["storage"]["seq_read_mbps_qd8"]["value"]))
    assert dev["storage"]["rand_read_3mb_mbps"]["value"] == doc["storage"]["rand_read_3mb_mbps_qd1"]["value"]
    assert dev["mem_bw_gbps"]["value"] == max(by_t.values())
    for fld in (dev["storage"]["seq_read_mbps"], dev["storage"]["rand_read_3mb_mbps"], dev["storage"]["access_ms"],
                dev["mem_bw_gbps"]):
        assert fld["source"].startswith("measured")
    assert CATALOG.read_bytes() == cat_before


def test_from_raw_dd_fallback_is_partial_and_flagged(bash: str, tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "dd_seq_read_qd1.txt").write_text(DD_LINE, encoding="utf-8")
    (raw / "run.tsv").write_text(
        "method.seq_read_qd1\tdd-direct\n"
        "cmd.seq_read_qd1\tdd if=/home/pi/pi-gemma4/models/x.gguf of=/dev/null bs=1M count=2048 iflag=direct\n"
        "fallback\tdd: fio is not installed (sudo apt-get install -y fio)\n"
        "skip.rand_read_3mb_mbps_qd1\tfio is not installed\n"
        "skip.memory\tmembw not built\n", encoding="utf-8", newline="\n")
    facts = json.loads((FIX / "facts_pi5-16gb-microsd-real.json").read_text(encoding="utf-8"))
    (raw / "facts.json").write_text(json.dumps(facts), encoding="utf-8")
    out = tmp_path / "io_mem.json"
    r = _run(bash, ["--from-raw", _p(raw), "--out", _p(out)], _env())
    assert r.returncode == 0, r.stdout + r.stderr
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["complete"] is False
    assert list(doc["storage"]) == ["seq_read_mbps_qd1"]
    e = doc["storage"]["seq_read_mbps_qd1"]
    assert e["method"] == "dd-direct" and e["value"] == pytest.approx(2147483648 / 23.4425 / 1e6, abs=1e-3)
    assert any("dd fallback" in f for f in e["flags"])
    assert "memory" not in doc and doc["skipped"]["memory"] == "membw not built"
    assert doc["skipped"]["rand_read_3mb_mbps_qd1"] == "fio is not installed"
    for k in STORAGE_KEYS[1:]:
        assert k in doc["skipped"]
    # context-friend-1 facts: 16 GB tier + microSD, Pi 5 kernel -> synthesized device name
    assert doc["host"] == "context-friend-1" and doc["device"] == "pi5-16gb-microsd"

    feas = pytest.importorskip("pigemma.feasibility")
    merged = feas.load_catalog(CATALOG, io_mem=out)
    dev = next(x for x in merged["devices"] if x["name"] == "pi5-16gb-microsd")
    assert dev["storage"]["seq_read_mbps"]["source"].startswith("measured")
    assert dev["storage"]["rand_read_3mb_mbps"]["source"].startswith("assumed")  # not measured -> stays assumed
    assert dev["mem_bw_gbps"]["source"].startswith("assumed")


def test_from_raw_explicit_device_and_calibration(bash: str, tmp_path: Path) -> None:
    raw = _raw_from_fixtures(tmp_path, membw=True)
    out_dir = tmp_path / "pi"
    out_dir.mkdir()
    (out_dir / "calibration.json").write_text(
        json.dumps({"llama_efficiency": {"value": 0.5, "source": "measured: pi/05 test fixture"}}), encoding="utf-8")
    out = out_dir / "io_mem.json"
    r = _run(bash, ["--from-raw", _p(raw), "--out", _p(out), "--device", "pi5-8gb-nvme-gen2"], _env("pi_proc_16gb"))
    assert r.returncode == 0, r.stdout + r.stderr
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["device"] == "pi5-8gb-nvme-gen2" and doc["device_source"].startswith("explicit")
    assert doc["derived"]["llama_efficiency"] == {"value": 0.5, "source": "measured: pi/05 test fixture"}
    feas = pytest.importorskip("pigemma.feasibility")
    merged = feas.load_catalog(CATALOG, io_mem=out)
    dev = next(x for x in merged["devices"] if x["name"] == "pi5-8gb-nvme-gen2")
    assert dev.get("synthesized") is True and dev["storage"]["access_ms"]["source"].startswith("measured")


def test_from_raw_nothing_measured_exits_3(bash: str, tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "io_mem.json"
    r = _run(bash, ["--from-raw", _p(raw), "--out", _p(out)], _env("pi_proc_16gb"))
    assert r.returncode == 3, r.stdout + r.stderr
    assert not out.exists()
    r2 = _run(bash, ["--from-raw", _p(tmp_path / "missing"), "--out", _p(out)], _env("pi_proc_16gb"))
    assert r2.returncode == 3


# ------------------------------------------------------------------------------------------ --dry-run
def test_dry_run_plan_on_16gb_fixture(bash: str, tmp_path: Path) -> None:
    out = tmp_path / "io_mem.json"
    r = _run(bash, ["--dry-run", "--quick", "--out", _p(out)], _env("pi_proc_16gb"))
    assert r.returncode == 0, r.stdout + r.stderr
    plan = [ln for ln in r.stdout.splitlines() if ln.startswith("plan: ")]
    assert plan, r.stdout
    text = "\n".join(plan)
    # 3,345,408 B expert slab rounded up to the fixture's 16 KiB page
    assert "expert slab 3345408 B" in text and "fio bs 3358720 B (page 16384)" in text
    assert "(new, 536870912 bytes)" in text  # --quick -> 512 MiB test file (no GGUF under MODEL_DIR)
    if shutil.which("fio"):
        for job in ("seq_read_qd1", "seq_read_qd8", "rand_read_3mb_qd1", "rand_read_3mb_qd4", "rand_read_16k_qd1"):
            assert f"--name={job}" in text
    else:
        assert "dd fallback" in text
    assert not out.exists() and not (tmp_path / "raw").exists()


def test_dry_run_emulated_defaults_to_pi_emu(bash: str) -> None:
    r = _run(bash, ["--dry-run", "--quick"], _env("pi_proc_16gb", PIGEMMA_EMULATED="1"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "results/pi-emu/io_mem.json" in r.stdout


def test_dry_run_protected_host_refused_unless_override(bash: str) -> None:
    r = _run(bash, ["--dry-run"], _env("pi_proc_cf1_16gb"))
    assert r.returncode == 4, r.stdout + r.stderr
    assert "REFUSED" in r.stderr and "plan:" not in r.stdout
    ok = _run(bash, ["--dry-run"], _env("pi_proc_cf1_16gb", PIGEMMA_I_KNOW="1"))
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "PROTECTED HOST OVERRIDE" in ok.stderr


def test_dry_run_refuses_when_disk_too_small(bash: str, tmp_path: Path) -> None:
    fake = tmp_path / "fake"
    shutil.copytree(FIX / "pi_proc_16gb", fake)
    (fake / "df.txt").write_text(
        "Filesystem 1-blocks Used Available Capacity Mounted on\n"
        "/dev/nvme0n1p1 500107862016 499107862016 1000000000 99% /mnt/nvme\n", encoding="utf-8", newline="\n")
    env = _env(None, PIGEMMA_FAKE_PROC=_p(fake))
    r = _run(bash, ["--dry-run", "--file", "new"], env)  # 2048 MiB x 1.2 > 1 GB free
    assert r.returncode == 3, r.stdout + r.stderr
    assert "not enough free disk" in r.stderr
    small = _run(bash, ["--dry-run", "--file", "new", "--size-mb", "512"], env)  # 0.64 GB <= 1 GB
    assert small.returncode == 0, small.stdout + small.stderr


def test_real_measurement_refused_under_fake_root(bash: str, tmp_path: Path) -> None:
    r = _run(bash, ["--quick", "--out", _p(tmp_path / "io_mem.json")], _env("pi_proc_16gb"))
    assert r.returncode == 1, r.stdout + r.stderr
    assert "PIGEMMA_FAKE_PROC" in r.stderr
    assert not (tmp_path / "io_mem.json").exists()


# ------------------------------------------------------------------------------------- integration
def _docker_run(bash: str, args: list[str], *, timeout: float = 600) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PIGEMMA_")}
    return subprocess.run([bash, "pc/docker_run.sh", "--no-stdin", "--env", "PIGEMMA_QUIET=1", *args], cwd=REPO,
                          env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=timeout)


@pytest.mark.integration
def test_fio_job_strings_parse_in_docker(bash: str, docker: str) -> None:
    r = _docker_run(bash, ["--", "bash", SCRIPT, "--check-jobs"])
    assert r.returncode == 0, r.stdout + r.stderr
    ok = [ln for ln in r.stdout.splitlines() if ln.startswith("OK ")]
    assert len(ok) == 7, r.stdout  # layout, probe and the five measurement jobs


@pytest.mark.integration
def test_short_real_run_in_docker(bash: str, docker: str) -> None:
    rel = f"pi/out/pytest-p03-{uuid.uuid4().hex[:8]}"
    out_dir = REPO / rel
    try:
        args = ["--env", "PIGEMMA_TOOLS_BIN=/work/tools/build-linux-x86/bin", "--", "bash", SCRIPT,
                "--file", "new", "--dir", "/tmp/p03", "--size-mb", "64", "--seconds", "1",
                "--membw-size-mb", "32", "--membw-reps", "2", "--out", f"/work/{rel}/io_mem.json",
                "--no-busy-check"]  # functional check only: other jobs on the desktop must not make it flaky
        r = _docker_run(bash, args)
        assert r.returncode == 0, r.stdout + r.stderr
        doc = json.loads((out_dir / "io_mem.json").read_text(encoding="utf-8"))
        assert set(doc["storage"]) == set(STORAGE_KEYS)
        assert doc["test_file"]["mode"] == "created-and-deleted"
        if (REPO / "tools/build-linux-x86/bin/membw").is_file():
            assert doc["memory"]["source"] == "measured"
        raws = list((out_dir / "raw").glob("io_mem-*/run.tsv"))
        assert len(raws) == 1
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
